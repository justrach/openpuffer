//! turbopuffer-compatible HTTP API served on top of the local HNSW engine.
//!
//! Endpoints (mirror https://turbopuffer.com/docs):
//!   POST /v2/namespaces/{ns}        write: upsert_columns or upsert_rows
//!                                   (columns other than id/vector are
//!                                   attributes), deletes [ids],
//!                                   delete_by_filter, distance_metric
//!                                   (cosine_distance, euclidean_squared,
//!                                   dot_product; fixed by the first write)
//!   POST /v2/namespaces/{ns}/query  ANN query (rank_by ["vector","ANN",[...]],
//!                                   top_k/limit, filters, include_attributes;
//!                                   consistency ignored)
//!   GET  /v2/namespaces/{ns}        stats
//!   DELETE /v2/namespaces/{ns}      drop namespace
//!
//! Responses use the same shapes: {"rows":[{"id":N,"$distance":D,...attrs}],...}
//!
//! A deleted document leaves its row in the graph as a tombstone
//! (doc_ids[row] == deleted_doc) so the graph stays connected; searches walk
//! through it but never return it.

const std = @import("std");
const builtin = @import("builtin");
const hnsw_mod = @import("hnsw.zig");
const s3_mod = @import("s3.zig");
const persist_mod = @import("persist.zig");
const iouring = @import("iouring_sock.zig");
const ns_mod = @import("namespace.zig");
const attrs_mod = @import("attrs.zig");

const Hnsw = hnsw_mod.Hnsw(void);

/// doc_ids value of a deleted row. Ids come from JSON integers checked to be
/// non-negative, so no document can have it.
const deleted_doc: u64 = std.math.maxInt(u64);

const Namespace = struct {
    name: []const u8,
    dim: usize = 0,
    index: Hnsw,
    /// local hnsw id -> external document id
    doc_ids: std.ArrayList(u64) = .empty,
    /// last persisted sequence number (snapshot or WAL)
    wal_seq: u64 = 0,
    /// WAL segments written since last snapshot
    pending_wal: u64 = 0,
    io: std.Io,
    lock: std.Io.RwLock = .init,
    /// Serializes id_map get/put among writers. Queries do not take this.
    map_mu: std.Io.Mutex = .init,
    /// Serializes back-edge splices among writers. Readers never take this.
    splice_mu: std.Io.Mutex = .init,
    /// Publishes that have not finished `spliceBackEdges`. Snapshot waits
    /// for zero so WAL/snapshot never persist a half-spliced node.
    unspliced: u32 = 0,
    /// external document id -> local hnsw id (upserts)
    id_map: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// Attribute columns by local id. Changed only under the exclusive lock.
    attrs: attrs_mod.Store,
    /// Tombstoned rows (doc_ids[row] == deleted_doc).
    dead: usize = 0,
    /// The first write fixes the distance metric; later writes must match.
    metric_fixed: bool = false,

    fn init(alloc: std.mem.Allocator, name: []const u8, io: std.Io) Namespace {
        return .{ .name = name, .index = Hnsw.init(alloc, 0, .{}), .io = io, .attrs = attrs_mod.Store.init(alloc) };
    }

    fn metric(ns: *const Namespace) hnsw_mod.Metric {
        return ns.index.opts.metric;
    }
};

const WalJob = struct {
    ns: *Namespace,
    body: []u8,
    finished: bool = false,
    result: ?anyerror = null,
};

const PersistWorker = struct {
    alloc: std.mem.Allocator,
    store: *persist_mod.Store,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    s3_mu: std.Io.Mutex = .init,
    queue: std.ArrayList(*WalJob) = .empty,
    shutdown: bool = false,
    thread: ?std.Thread = null,
    compact_threshold: u64 = wal_compact_threshold,

    fn start(self: *PersistWorker) !void {
        self.thread = try std.Thread.spawn(.{}, run, .{self});
    }

    fn commit(self: *PersistWorker, job: *WalJob) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        try self.queue.append(self.alloc, job);
        self.cv.signal(self.io);
        while (!job.finished) self.cv.waitUncancelable(self.io, &self.mutex);
        if (job.result) |e| return e;
    }

    fn run(self: *PersistWorker) void {
        while (true) {
            self.mutex.lockUncancelable(self.io);
            while (self.queue.items.len == 0 and !self.shutdown) {
                self.cv.waitUncancelable(self.io, &self.mutex);
            }
            if (self.shutdown and self.queue.items.len == 0) {
                self.mutex.unlock(self.io);
                return;
            }
            const jobs = self.queue.toOwnedSlice(self.alloc) catch {
                self.mutex.unlock(self.io);
                continue;
            };
            self.queue = .empty;
            self.mutex.unlock(self.io);

            self.flush(jobs) catch |e| {
                self.mutex.lockUncancelable(self.io);
                for (jobs) |j| {
                    j.result = e;
                    j.finished = true;
                }
                self.cv.broadcast(self.io);
                self.mutex.unlock(self.io);
                self.alloc.free(jobs);
                continue;
            };

            self.mutex.lockUncancelable(self.io);
            for (jobs) |j| {
                j.finished = true;
            }
            self.cv.broadcast(self.io);
            self.mutex.unlock(self.io);
            self.alloc.free(jobs);
        }
    }

    fn flush(self: *PersistWorker, jobs: []*WalJob) !void {
        const used = try self.alloc.alloc(bool, jobs.len);
        defer self.alloc.free(used);
        @memset(used, false);
        for (jobs, 0..) |job, i| {
            if (used[i]) continue;
            var count: usize = 0;
            for (jobs, 0..) |j, k| {
                if (!used[k] and j.ns == job.ns) count += 1;
            }
            const group = try self.alloc.alloc(*WalJob, count);
            defer self.alloc.free(group);
            var g: usize = 0;
            for (jobs, 0..) |j, k| {
                if (!used[k] and j.ns == job.ns) {
                    used[k] = true;
                    group[g] = j;
                    g += 1;
                }
            }
            try self.flushGroup(job.ns, group);
        }
    }

    fn flushGroup(self: *PersistWorker, ns: *Namespace, group: []*WalJob) !void {
        var merged: std.Io.Writer.Allocating = .init(self.alloc);
        defer merged.deinit();
        const w = &merged.writer;
        try writeWalOpen(w, ns.metric());
        var first = true;
        for (group) |job| {
            const inner = walDocsInner(job.body);
            if (inner.len == 0) continue;
            if (!first) try w.writeAll(",");
            first = false;
            try w.writeAll(inner);
        }
        try w.writeAll("]}");

        ns.lock.lockUncancelable(ns.io);
        ns.wal_seq += 1;
        const seq = ns.wal_seq;
        ns.pending_wal += 1;
        const do_snap = ns.pending_wal >= self.compact_threshold;
        ns.lock.unlock(ns.io);

        self.s3_mu.lockUncancelable(self.io);
        defer self.s3_mu.unlock(self.io);
        try self.store.appendWal(ns.name, seq, merged.written());

        for (group) |job| {
            self.alloc.free(job.body);
            job.body = &.{};
        }

        // snapshot after the WAL PUT so waiters can be unblocked first;
        // we still do it in this flush so the graph cut is seq-consistent.
        if (do_snap) {
            if (snapshotNamespace(self.store, ns)) |_| {
                self.store.deleteWalUpTo(ns.name, seq);
            } else |e| {
                // The WAL is the only durable copy until a snapshot lands, so
                // it stays. (Deleting it after a failed snapshot PUT lost
                // every document since the previous snapshot.) Re-arm so the
                // next flush for this namespace tries the snapshot again.
                std.debug.print("snapshot of '{s}' failed ({any}); keeping its WAL and retrying on the next flush\n", .{ ns.name, e });
                ns.lock.lockUncancelable(ns.io);
                ns.pending_wal = @max(ns.pending_wal, self.compact_threshold - 1);
                ns.lock.unlock(ns.io);
            }
        }
    }
};

/// `{"docs":[` or, outside cosine, `{"metric":"...","docs":[` so replay
/// rebuilds the namespace with the metric it was written with.
fn writeWalOpen(w: *std.Io.Writer, metric: hnsw_mod.Metric) !void {
    if (metric == .cosine) return w.writeAll("{\"docs\":[");
    try w.print("{{\"metric\":\"{s}\",\"docs\":[", .{metric.name()});
}

fn walDocsInner(body: []const u8) []const u8 {
    const prefix = "{\"docs\":[";
    const suffix = "]}";
    if (std.mem.startsWith(u8, body, prefix) and std.mem.endsWith(u8, body, suffix)) {
        return body[prefix.len .. body.len - suffix.len];
    }
    return body;
}

/// Monotonic stopwatch (same shape as main.zig's).
const Sw = struct {
    t0: std.Io.Timestamp,
    fn start(io: std.Io) Sw {
        return .{ .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    fn readNs(self: *const Sw, io: std.Io) u64 {
        const d = self.t0.durationTo(std.Io.Timestamp.now(io, .awake));
        return @intCast(@max(0, d.nanoseconds));
    }
};

const Registry = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    namespaces: std.StringHashMapUnmanaged(*Namespace) = .empty,
    /// null when running without object-storage persistence
    store: ?*persist_mod.Store = null,
    /// WAL upload lanes, each with its own S3 connection. A namespace always
    /// maps to the same lane, so its segments stay in order, while different
    /// namespaces upload in parallel instead of queueing behind one PUT.
    lanes: []*PersistWorker = &.{},
    /// Namespaces whose recovery failed. They are neither served nor
    /// writable until a restart recovers them: a half-loaded index hides data,
    /// and a write would reuse WAL sequence numbers or compact over the
    /// snapshot, destroying what is still in object storage.
    failed: std.StringHashMapUnmanaged(void) = .empty,

    fn laneFor(self: *Registry, name: []const u8) ?*PersistWorker {
        if (self.lanes.len == 0) return null;
        return self.lanes[std.hash.Wyhash.hash(0, name) % self.lanes.len];
    }

    fn isFailed(self: *Registry, name: []const u8) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.failed.contains(name);
    }

    /// Drop a partially recovered namespace and refuse it until the healer
    /// (or a restart) recovers it.
    fn quarantine(self: *Registry, name: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.namespaces.fetchRemove(name)) |kv| freeNamespace(self.alloc, kv.value);
        const owned = self.alloc.dupe(u8, name) catch return;
        self.failed.put(self.alloc, owned, {}) catch self.alloc.free(owned);
    }

    /// Owned copies of the quarantined names (caller frees each and the slice).
    fn failedNames(self: *Registry, alloc: std.mem.Allocator) ![][]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const names = try alloc.alloc([]u8, self.failed.count());
        var it = self.failed.keyIterator();
        var i: usize = 0;
        errdefer {
            for (names[0..i]) |n| alloc.free(n);
            alloc.free(names);
        }
        while (it.next()) |k| : (i += 1) names[i] = try alloc.dupe(u8, k.*);
        return names;
    }

    /// Publish a fully reloaded namespace and lift its quarantine in one step,
    /// so no request ever sees it half-loaded.
    fn publishHealed(self: *Registry, ns: *Namespace) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        try self.namespaces.put(self.alloc, ns.name, ns);
        if (self.failed.fetchRemove(ns.name)) |kv| self.alloc.free(kv.key);
    }

    fn getOrCreate(self: *Registry, name: []const u8) !*Namespace {
        try ns_mod.validate(name);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.failed.contains(name)) return error.NamespaceUnavailable;
        if (self.namespaces.get(name)) |ns| return ns;
        const ns = try self.alloc.create(Namespace);
        ns.* = Namespace.init(self.alloc, try self.alloc.dupe(u8, name), self.io);
        try self.namespaces.put(self.alloc, ns.name, ns);
        return ns;
    }

    fn get(self: *Registry, name: []const u8) ?*Namespace {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.namespaces.get(name);
    }
};

pub const Options = struct {
    port: u16 = 8080,
    ef: u32 = 128,
    /// Exact-rerank width as a multiple of k. 4 is the engine default.
    rerank_mult: usize = 4,
    /// null = ncpu. Override with --workers / OPENPUFFER_WORKERS.
    workers: ?usize = null,
    s3_cfg: ?s3_mod.Config = null,
    /// Parallel namespace loads at startup (OPENPUFFER_RECOVERY_WORKERS).
    recovery_workers: usize = default_recovery_workers,
    /// WAL upload lanes (OPENPUFFER_PERSIST_LANES).
    persist_lanes: usize = default_persist_lanes,
    /// First retry delay for quarantined namespaces (OPENPUFFER_HEAL_INITIAL_MS).
    heal_initial_ms: u64 = 1000,
};

/// Auto-compaction threshold: snapshot after this many WAL segments.
const wal_compact_threshold: u64 = 8;

/// WAL upload lanes (`OPENPUFFER_PERSIST_LANES`). Each lane is one thread
/// with one kept-alive S3 connection.
const default_persist_lanes: usize = 8;

pub fn serve(alloc: std.mem.Allocator, io: std.Io, o: Options, out: *std.Io.Writer) !void {
    var registry = Registry{ .alloc = alloc, .io = io };

    if (o.s3_cfg) |cfg| {
        const t_all = Sw.start(io);
        const s3_client = try alloc.create(s3_mod.Client);
        s3_client.* = s3_mod.Client.init(alloc, io, cfg);
        const store = try alloc.create(persist_mod.Store);
        store.* = persist_mod.Store.init(alloc, s3_client);
        registry.store = store;
        // If the bucket can't even be listed, refuse to start. Serving empty
        // would accept writes at WAL sequence 1 and compact over every
        // namespace's real snapshot.
        const recovered = recoverAll(alloc, io, &registry, store, out, o.recovery_workers) catch |e| {
            try out.print("FATAL: recovery could not list object storage ({any}); not serving, so nothing is overwritten\n", .{e});
            try out.flush();
            return e;
        };
        if (recovered > 0) {
            try out.print("recovery total: {d:.1}ms\n", .{@as(f64, @floatFromInt(t_all.readNs(io))) / 1e6});
            try out.flush();
        }
        if (registry.failed.count() > 0) {
            const healer = try alloc.create(Healer);
            healer.* = .{ .alloc = alloc, .io = io, .registry = &registry, .cfg = cfg, .initial_ms = o.heal_initial_ms };
            const t = try std.Thread.spawn(.{}, Healer.run, .{healer});
            t.detach();
        }
        const n_lanes = @max(1, @min(o.persist_lanes, 64));
        const lanes = try alloc.alloc(*PersistWorker, n_lanes);
        for (lanes) |*lane| {
            const lane_client = try alloc.create(s3_mod.Client);
            lane_client.* = s3_mod.Client.init(alloc, io, cfg);
            const lane_store = try alloc.create(persist_mod.Store);
            lane_store.* = persist_mod.Store.init(alloc, lane_client);
            lane.* = try alloc.create(PersistWorker);
            lane.*.* = .{ .alloc = alloc, .store = lane_store, .io = io };
            try lane.*.start();
        }
        registry.lanes = lanes;
    }

    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", o.port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    try out.print("openpuffer serving turbopuffer-compatible API on http://127.0.0.1:{d} (ef={d} rerank_mult={d})\n", .{ o.port, o.ef, o.rerank_mult });
    if (registry.store != null) try out.print("persistence: s3 bucket '{s}' (group-commit WAL)\n", .{o.s3_cfg.?.bucket});
    try out.flush();

    if (builtin.os.tag == .linux) {
        serveIoUring(alloc, io, listener.socket.handle, &registry, o, out) catch |e| {
            try out.print("io_uring serve failed ({any}); falling back to thread-per-conn\n", .{e});
            try out.flush();
            try serveThreaded(alloc, io, &listener, &registry, o, out);
        };
    } else {
        try serveThreaded(alloc, io, &listener, &registry, o, out);
    }
}

fn workerCount(o: Options) usize {
    // One worker per core: extra threads oversubscribe a memory-bound ANN
    // walk and raise concurrent p50 without adding QPS.
    if (o.workers) |n| return @min(64, @max(1, n));
    const n = std.Thread.getCpuCount() catch 4;
    return @min(16, @max(1, n));
}

const FdQueue = struct {
    io: std.Io,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    slots: [256]std.posix.fd_t = undefined,
    head: usize = 0,
    tail: usize = 0,
    count: usize = 0,

    fn push(self: *FdQueue, fd: std.posix.fd_t) void {
        self.mu.lockUncancelable(self.io);
        while (self.count == self.slots.len) {
            self.cv.waitUncancelable(self.io, &self.mu);
        }
        self.slots[self.tail] = fd;
        self.tail = (self.tail + 1) % self.slots.len;
        self.count += 1;
        self.cv.signal(self.io);
        self.mu.unlock(self.io);
    }

    fn tryPush(self: *FdQueue, fd: std.posix.fd_t) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.count == self.slots.len) return false;
        self.slots[self.tail] = fd;
        self.tail = (self.tail + 1) % self.slots.len;
        self.count += 1;
        self.cv.signal(self.io);
        return true;
    }

    fn pop(self: *FdQueue) std.posix.fd_t {
        self.mu.lockUncancelable(self.io);
        while (self.count == 0) {
            self.cv.waitUncancelable(self.io, &self.mu);
        }
        const fd = self.slots[self.head];
        self.head = (self.head + 1) % self.slots.len;
        self.count -= 1;
        self.cv.signal(self.io);
        self.mu.unlock(self.io);
        return fd;
    }

    fn isEmpty(self: *FdQueue) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return self.count == 0;
    }
};

fn listenPending(listen_fd: std.posix.fd_t) bool {
    var pfd = [1]std.os.linux.pollfd{.{
        .fd = listen_fd,
        .events = std.os.linux.POLL.IN,
        .revents = 0,
    }};
    const n = std.os.linux.poll(&pfd, 1, 0);
    return n > 0 and pfd[0].revents & std.os.linux.POLL.IN != 0;
}

fn drainListenBacklog(listen_fd: std.posix.fd_t, queue: *FdQueue) void {
    while (listenPending(listen_fd)) {
        const res = std.os.linux.accept4(listen_fd, null, null, std.os.linux.SOCK.CLOEXEC);
        if (std.os.linux.errno(res) != .SUCCESS) return;
        const fd: std.posix.fd_t = @intCast(res);
        iouring.enableTcpNoDelay(fd);
        queue.push(fd);
    }
}

const PoolCtx = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    o: Options,
    queue: *FdQueue,
    listen_fd: std.posix.fd_t,
};

fn startWorkerPool(alloc: std.mem.Allocator, io: std.Io, registry: *Registry, o: Options, listen_fd: std.posix.fd_t) !*FdQueue {
    const queue = try alloc.create(FdQueue);
    queue.* = .{ .io = io };
    const ctx = try alloc.create(PoolCtx);
    ctx.* = .{ .alloc = alloc, .io = io, .registry = registry, .o = o, .queue = queue, .listen_fd = listen_fd };
    const n = workerCount(ctx.o);
    for (0..n) |_| {
        const t = try std.Thread.spawn(.{ .stack_size = 512 * 1024 }, poolWorker, .{ctx});
        t.detach();
    }
    return queue;
}

fn poolWorker(ctx: *PoolCtx) void {
    var store: std.ArrayList(u8) = .empty;
    defer store.deinit(ctx.alloc);
    store.resize(ctx.alloc, 1 << 16) catch return;
    var arena_state = std.heap.ArenaAllocator.init(ctx.alloc);
    defer arena_state.deinit();
    var xfer = ConnXfer{};
    while (true) {
        const fd = ctx.queue.pop();
        const end = handleIoUringConn(&xfer, fd, ctx.alloc, ctx.io, ctx.registry, ctx.o, &store, &arena_state, ctx.queue, ctx.listen_fd) catch .close;
        if (end == .close) _ = std.os.linux.close(fd);
    }
}

fn serveThreaded(
    alloc: std.mem.Allocator,
    io: std.Io,
    listener: *std.Io.net.Server,
    registry: *Registry,
    o: Options,
    out: *std.Io.Writer,
) !void {
    var conn_sem = std.Io.Semaphore{ .permits = 32 };
    while (true) {
        const stream = listener.accept(io) catch |e| {
            try out.print("accept error: {any}\n", .{e});
            continue;
        };
        iouring.enableTcpNoDelay(stream.socket.handle);
        conn_sem.waitUncancelable(io);
        const t = std.Thread.spawn(.{ .stack_size = 256 * 1024 }, handleConnectionThread, .{ alloc, io, stream, registry, o, &conn_sem }) catch {
            conn_sem.post(io);
            handleConnection(alloc, io, stream, registry, o) catch {};
            continue;
        };
        t.detach();
    }
}

fn serveIoUring(
    alloc: std.mem.Allocator,
    io: std.Io,
    listen_fd: std.posix.fd_t,
    registry: *Registry,
    o: Options,
    out: *std.Io.Writer,
) !void {
    _ = std.os.linux.listen(listen_fd, 1024);
    const queue = try startWorkerPool(alloc, io, registry, o, listen_fd);
    try out.print("linux serve: io_uring accept + {d} keep-alive workers (fair rotate), ef={d} rerank_mult={d}, TCP_NODELAY\n", .{ workerCount(o), o.ef, o.rerank_mult });
    try out.flush();
    var ring = try iouring.Ring.init();
    defer ring.deinit();
    while (true) {
        const fd = ring.accept(listen_fd) catch |e| {
            try out.print("io_uring accept error: {any}\n", .{e});
            try out.flush();
            continue;
        };
        iouring.enableTcpNoDelay(fd);
        // Never handle on the accept thread: a keep-alive query would pin
        // accept and leave upserts sitting in the listen backlog until they
        // time out (~3s). Always queue; workers rotate after each request.
        drainListenBacklog(listen_fd, queue);
        queue.push(fd);
    }
}

const ConnXfer = struct {
    ring: ?*iouring.Ring = null,

    fn recv(self: *ConnXfer, fd: std.posix.fd_t, buf: []u8) !usize {
        if (self.ring) |r| return r.recv(fd, buf);
        return iouring.posixRecv(fd, buf);
    }

    fn sendAll(self: *ConnXfer, fd: std.posix.fd_t, data: []const u8) !void {
        if (self.ring) |r| return r.sendAll(fd, data);
        return iouring.posixSendAll(fd, data);
    }
};

const ConnEnd = enum { close, requeued };

fn handleIoUringConn(
    xfer: *ConnXfer,
    fd: std.posix.fd_t,
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    o: Options,
    store: *std.ArrayList(u8),
    arena_state: *std.heap.ArenaAllocator,
    queue: ?*FdQueue,
    listen_fd: ?std.posix.fd_t,
) !ConnEnd {
    var n: usize = 0;
    while (true) {
        while (iouring.completeHttpRequest(store.items[0..n]) == 0) {
            if (n == store.items.len) {
                const cap = store.items.len;
                if (cap >= 8 << 20) return error.RequestTooLarge;
                try store.resize(alloc, cap * 2);
            }
            const got = try xfer.recv(fd, store.items[n..]);
            if (got == 0) return .close;
            n += got;
        }
        const used = iouring.completeHttpRequest(store.items[0..n]);
        const raw = try iouring.parseHttpRequest(store.items[0..used]);
        const keepalive = !iouring.wantsClose(store.items[0..used]);

        _ = arena_state.reset(.retain_capacity);
        const arena = arena_state.allocator();
        var res = Responder{ .raw_alloc = arena };
        dispatch(arena, io, raw.method, raw.path, raw.body, registry, o, &res) catch {
            res.raw_status = .internal_server_error;
            res.raw_body = "{\"error\":\"internal\"}";
        };
        const status_n: u16 = @backingInt(res.raw_status);
        const phrase = res.raw_status.phrase() orelse "OK";
        const wire = try iouring.formatResponse(arena, status_n, phrase, res.raw_body, keepalive);
        try xfer.sendAll(fd, wire);

        if (!keepalive) return .close;
        // One request per pop. Requeue the keep-alive fd so a waiting
        // insert/update is not stuck behind this client's next recv.
        if (queue) |q| {
            if (listen_fd) |lfd| drainListenBacklog(lfd, q);
            if (q.tryPush(fd)) return .requeued;
        }
        if (used < n) {
            const rest = n - used;
            std.mem.copyForwards(u8, store.items[0..rest], store.items[used..n]);
            n = rest;
            continue;
        }
        n = 0;
    }
}

fn handleConnectionThread(
    alloc: std.mem.Allocator,
    io: std.Io,
    stream: std.Io.net.Stream,
    registry: *Registry,
    o: Options,
    sem: *std.Io.Semaphore,
) void {
    defer sem.post(io);
    handleConnection(alloc, io, stream, registry, o) catch {};
}

fn handleConnection(
    alloc: std.mem.Allocator,
    io: std.Io,
    stream: std.Io.net.Stream,
    registry: *Registry,
    o: Options,
) !void {
    defer stream.close(io);
    var read_buf: [1 << 16]u8 = undefined;
    var write_buf: [1 << 16]u8 = undefined;
    var reader = stream.reader(io, &read_buf);
    var writer = stream.writer(io, &write_buf);
    var server = std.http.Server.init(&reader.interface, &writer.interface);

    while (true) {
        var req = server.receiveHead() catch break;
        var arena_state = std.heap.ArenaAllocator.init(alloc);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        handleRequest(arena, io, &req, registry, o) catch |e| {
            var err_res = Responder{ .req = &req };
            respondJson(&err_res, .internal_server_error, "{\"error\":\"internal\"}") catch {};
            return e;
        };
    }
}

const NsPlan = struct {
    name: []const u8,
    has_bin: bool = false,
    has_json: bool = false,
};

fn freeNamespace(alloc: std.mem.Allocator, ns: *Namespace) void {
    ns.index.deinit();
    ns.id_map.deinit(alloc);
    ns.doc_ids.deinit(alloc);
    ns.attrs.deinit();
    alloc.free(ns.name);
    alloc.destroy(ns);
}

/// Background retry for quarantined namespaces. Each attempt reloads the
/// namespace into a fresh, unpublished `Namespace` and swaps it in only once
/// it is complete, so a store outage during startup heals without a restart
/// and no request ever reads a half-loaded index. Backoff doubles from
/// `Options.heal_initial_ms` to 30 s; the thread exits once nothing is
/// quarantined.
const Healer = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    cfg: s3_mod.Config,
    initial_ms: u64,

    const max_delay_ms: u64 = 30_000;

    fn run(h: *Healer) void {
        var client = s3_mod.Client.init(h.alloc, h.io, h.cfg);
        defer client.deinit();
        var store = persist_mod.Store.init(h.alloc, &client);
        var log_buf: [4096]u8 = undefined;
        var log_file = std.Io.File.stderr();
        var log = log_file.writer(h.io, &log_buf);
        var log_mu: std.Io.Mutex = .init;

        var delay_ms: u64 = @max(1, h.initial_ms);
        var round: u32 = 0;
        while (true) {
            h.io.sleep(.{ .nanoseconds = delay_ms * std.time.ns_per_ms }, .awake) catch {};
            round += 1;
            const names = h.registry.failedNames(h.alloc) catch continue;
            defer {
                for (names) |n| h.alloc.free(n);
                h.alloc.free(names);
            }
            if (names.len == 0) return;
            for (names) |name| {
                h.healOne(&store, name, &log.interface, &log_mu) catch |e| {
                    log.interface.print("  heal '{s}' attempt {d} failed: {any}; retrying in {d} ms\n", .{ name, round, e, @min(delay_ms * 2, max_delay_ms) }) catch {};
                    log.interface.flush() catch {};
                    continue;
                };
                log.interface.print("healed '{s}' on attempt {d}: serving and accepting writes again\n", .{ name, round }) catch {};
                log.interface.flush() catch {};
            }
            delay_ms = @min(delay_ms * 2, max_delay_ms);
        }
    }

    fn healOne(h: *Healer, store: *persist_mod.Store, name: []const u8, log: *std.Io.Writer, log_mu: *std.Io.Mutex) !void {
        const prefix = try std.fmt.allocPrint(h.alloc, "openpuffer/{s}/", .{name});
        defer h.alloc.free(prefix);
        const keys = try store.client.listKeys(prefix);
        defer {
            for (keys) |k| h.alloc.free(k);
            h.alloc.free(keys);
        }
        var plans: std.ArrayList(NsPlan) = .empty;
        defer plans.deinit(h.alloc);
        try plansFromKeys(h.alloc, keys, &plans);
        var plan: NsPlan = .{ .name = name };
        for (plans.items) |p| {
            if (std.mem.eql(u8, p.name, name)) plan = p;
        }

        const ns = try h.alloc.create(Namespace);
        ns.* = Namespace.init(h.alloc, h.alloc.dupe(u8, name) catch |e| {
            h.alloc.destroy(ns);
            return e;
        }, h.io);
        var job = RecoverJob{
            .alloc = h.alloc,
            .io = h.io,
            .registry = h.registry,
            .keys = keys,
            .plan = plan,
            .out = log,
            .out_mu = log_mu,
        };
        recoverInto(&job, store, ns) catch |e| {
            freeNamespace(h.alloc, ns);
            return e;
        };
        h.registry.publishHealed(ns) catch |e| {
            freeNamespace(h.alloc, ns);
            return e;
        };
    }
};

/// Group listed keys `openpuffer/{ns}/...` into one plan per namespace.
fn plansFromKeys(alloc: std.mem.Allocator, keys: []const []const u8, plans: *std.ArrayList(NsPlan)) !void {
    for (keys) |k| {
        if (!std.mem.startsWith(u8, k, "openpuffer/")) continue;
        const tail = k["openpuffer/".len..];
        const slash = std.mem.indexOfScalar(u8, tail, '/') orelse continue;
        const name = tail[0..slash];
        if (!ns_mod.isValid(name)) continue;
        const rest = tail[slash + 1 ..];
        var found: ?*NsPlan = null;
        for (plans.items) |*p| {
            if (std.mem.eql(u8, p.name, name)) {
                found = p;
                break;
            }
        }
        const plan = found orelse blk: {
            try plans.append(alloc, .{ .name = name });
            break :blk &plans.items[plans.items.len - 1];
        };
        if (std.mem.eql(u8, rest, "snapshot.bin")) plan.has_bin = true;
        if (std.mem.eql(u8, rest, "snapshot.json")) plan.has_json = true;
    }
}

const RecoverJob = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    keys: []const []const u8,
    plan: NsPlan,
    out: *std.Io.Writer,
    out_mu: *std.Io.Mutex,
    ok: bool = false,
};

/// Parallel recovery is capped: one thread (and S3 connection) per namespace
/// opened hundreds of connections at once and drew resets from the store.
/// `Options.recovery_workers` (OPENPUFFER_RECOVERY_WORKERS) overrides.
const default_recovery_workers: usize = 32;

const RecoverPool = struct {
    jobs: []RecoverJob,
    next: std.atomic.Value(usize) = .init(0),
    cfg: s3_mod.Config,

    /// One worker: one S3 client (kept-alive connection) for every namespace it takes.
    fn work(pool: *RecoverPool) void {
        const first = pool.jobs[0];
        var client = s3_mod.Client.init(first.alloc, first.io, pool.cfg);
        defer client.deinit();
        var store = persist_mod.Store.init(first.alloc, &client);
        while (true) {
            const i = pool.next.fetchAdd(1, .monotonic);
            if (i >= pool.jobs.len) return;
            recoverOneNs(&pool.jobs[i], &store);
        }
    }
};

/// Load every namespace persisted under the store prefix: snapshot first,
/// then replay WAL segments newer than it; compact WALs into a fresh snapshot.
/// Snapshot/WAL fetches for different namespaces run in parallel; the initial
/// LIST is reused so we never 404-probe snapshot.bin or re-LIST WAL prefixes.
fn recoverAll(
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    store: *persist_mod.Store,
    out: *std.Io.Writer,
    workers: usize,
) !usize {
    const keys = try store.client.listKeys("openpuffer/");
    defer {
        for (keys) |k| alloc.free(k);
        alloc.free(keys);
    }

    var plans: std.ArrayList(NsPlan) = .empty;
    defer plans.deinit(alloc);
    try plansFromKeys(alloc, keys, &plans);

    var t_all = Sw.start(io);
    var out_mu: std.Io.Mutex = .init;
    const jobs = try alloc.alloc(RecoverJob, plans.items.len);
    defer alloc.free(jobs);
    for (plans.items, 0..) |plan, i| {
        jobs[i] = .{
            .alloc = alloc,
            .io = io,
            .registry = registry,
            .keys = keys,
            .plan = plan,
            .out = out,
            .out_mu = &out_mu,
        };
    }
    if (jobs.len == 0) return 0;

    var pool = RecoverPool{ .jobs = jobs, .cfg = store.client.cfg };
    const n_workers = @max(1, @min(workers, jobs.len));
    var threads: std.ArrayList(std.Thread) = .empty;
    defer threads.deinit(alloc);
    for (0..n_workers) |_| {
        const t = std.Thread.spawn(.{}, RecoverPool.work, .{&pool}) catch break;
        try threads.append(alloc, t);
    }
    if (threads.items.len == 0) RecoverPool.work(&pool);
    for (threads.items) |t| t.join();

    var failed: usize = 0;
    for (jobs) |job| {
        if (!job.ok) {
            failed += 1;
            registry.quarantine(job.plan.name);
        }
    }
    try out.print("recovered {d} namespace(s) from object storage in {d:.1}ms with {d} worker(s)\n", .{
        jobs.len - failed, @as(f64, @floatFromInt(t_all.readNs(io))) / 1e6, threads.items.len,
    });
    if (failed > 0) {
        try out.print("WARNING: {d} namespace(s) failed to recover and are quarantined (503 until a background retry recovers them)\n", .{failed});
    }
    return jobs.len - failed;
}

fn recoverOneNs(job: *RecoverJob, store: *persist_mod.Store) void {
    const ns = job.registry.getOrCreate(job.plan.name) catch |e| {
        job.out_mu.lockUncancelable(job.io);
        defer job.out_mu.unlock(job.io);
        job.out.print("  recover '{s}' failed: {any}\n", .{ job.plan.name, e }) catch {};
        job.out.flush() catch {};
        return;
    };
    recoverInto(job, store, ns) catch |e| {
        job.out_mu.lockUncancelable(job.io);
        defer job.out_mu.unlock(job.io);
        job.out.print("  recover '{s}' failed: {any}\n", .{ job.plan.name, e }) catch {};
        job.out.flush() catch {};
        return;
    };
    job.ok = true;
}

/// Load one namespace's snapshot and WAL tail from object storage into `ns`.
/// On error `ns` may be partially loaded and must not be served.
fn recoverInto(job: *RecoverJob, store: *persist_mod.Store, ns: *Namespace) !void {
    const alloc = job.alloc;
    const io = job.io;
    const name = job.plan.name;

    var last_seq: u64 = 0;
    var t_fetch = Sw.start(io);
    if (try store.getSnapshotKnown(name, job.plan.has_bin, job.plan.has_json)) |snap| {
        defer alloc.free(snap.bytes);
        const t_snap_get = t_fetch.readNs(io);
        var t_parse = Sw.start(io);
        ns.dim = snap.dim;
        ns.index.dim = snap.dim;
        ns.wal_seq = snap.seq;
        last_seq = snap.seq;

        var t_parse_done: u64 = 0;
        var t_build_done: u64 = 0;
        switch (snap.kind) {
            .binary => {
                const hdr = try persist_mod.parseBinaryHeader(snap.bytes);
                t_parse_done = t_parse.readNs(io);
                var t_build = Sw.start(io);
                try loadBinarySnapshot(ns, job.registry.alloc, alloc, snap.bytes, hdr);
                t_build_done = t_build.readNs(io);
            },
            .json => {
                const parsed = try std.json.parseFromSlice(std.json.Value, alloc, snap.bytes, .{});
                t_parse_done = t_parse.readNs(io);
                var t_build = Sw.start(io);
                const docs = parsed.value.object.get("docs").?.array.items;
                for (docs) |doc| {
                    const pair = doc.array.items;
                    const id: u64 = @intCast(pair[0].integer);
                    const arr = pair[1].array;
                    const v = try alloc.alloc(f32, arr.items.len);
                    for (arr.items, 0..) |x, i| v[i] = switch (x) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    };
                    try upsertDoc(ns, job.registry.alloc, id, v);
                }
                parsed.deinit();
                t_build_done = t_build.readNs(io);
                snapshotNamespace(store, ns) catch {};
            },
        }
        job.out_mu.lockUncancelable(io);
        defer job.out_mu.unlock(io);
        try job.out.print("  recovered '{s}': {s} seq={d} ({d} docs) [fetch={d:.1}ms parse={d:.1}ms build={d:.1}ms]\n", .{ name, @tagName(snap.kind), snap.seq, ns.doc_ids.items.len, @as(f64, @floatFromInt(t_snap_get)) / 1e6, @as(f64, @floatFromInt(t_parse_done)) / 1e6, @as(f64, @floatFromInt(t_build_done)) / 1e6 });
        try job.out.flush();
    }

    const wal_seqs = try persist_mod.Store.walSeqsFromKeys(alloc, job.keys, name, last_seq);
    defer alloc.free(wal_seqs);
    var t_walget: u64 = 0;
    var t_walparse: u64 = 0;
    var replayed: usize = 0;
    for (wal_seqs) |seq| {
        var tg = Sw.start(io);
        const body = try store.getWalBody(name, seq);
        t_walget += tg.readNs(io);
        defer alloc.free(body);
        var tp2 = Sw.start(io);
        _ = try replayWalBody(ns, job.registry.alloc, alloc, body);
        t_walparse += tp2.readNs(io);
        replayed += 1;
        ns.wal_seq = seq;
        ns.pending_wal += 1;
    }

    store.deleteWalListed(name, job.keys, last_seq);

    if (replayed > 0) {
        var tc = Sw.start(io);
        try snapshotNamespace(store, ns);
        store.deleteWalListed(name, job.keys, ns.wal_seq);
        const t_compact = tc.readNs(io);
        job.out_mu.lockUncancelable(io);
        defer job.out_mu.unlock(io);
        try job.out.print("  replayed {d} WAL segments for '{s}' [walget={d:.1}ms walparse+build={d:.1}ms compact={d:.1}ms]\n", .{ replayed, name, @as(f64, @floatFromInt(t_walget)) / 1e6, @as(f64, @floatFromInt(t_walparse)) / 1e6, @as(f64, @floatFromInt(t_compact)) / 1e6 });
        try job.out.flush();
    }
}

fn waitNoUnspliced(ns: *Namespace) void {
    while (@atomicLoad(u32, &ns.unspliced, .acquire) != 0) {
        std.Thread.yield() catch {};
    }
}

/// Write a full binary snapshot of `ns` (graph + vectors) and reset pending-WAL.
/// Exclusive + unspliced==0 so the cut is after every publish's back-edges.
fn snapshotNamespace(store: *persist_mod.Store, ns: *Namespace) !void {
    while (true) {
        waitNoUnspliced(ns);
        ns.lock.lockUncancelable(ns.io);
        if (@atomicLoad(u32, &ns.unspliced, .acquire) == 0) break;
        ns.lock.unlock(ns.io);
    }
    const ext = snapshotExt(store.alloc, ns) catch |e| {
        ns.lock.unlock(ns.io);
        return e;
    };
    defer store.alloc.free(ext);
    const body = persist_mod.Store.buildBinarySnapshotExt(
        store.alloc,
        ns.wal_seq,
        ns.dim,
        ns.doc_ids.items,
        ext,
        &ns.index,
    ) catch |e| {
        ns.lock.unlock(ns.io);
        return e;
    };
    ns.pending_wal = 0;
    ns.lock.unlock(ns.io);
    defer store.alloc.free(body);
    try store.putSnapshotBin(ns.name, body);
}

/// Snapshot extension: metric and attribute columns, as JSON. Empty for a
/// cosine namespace without attributes, which keeps the original format.
/// `{"metric":"...","attrs":{"name":[v_row0, v_row1, ...], ...}}`
fn snapshotExt(alloc: std.mem.Allocator, ns: *const Namespace) ![]u8 {
    if (ns.metric() == .cosine and ns.attrs.isEmpty()) return alloc.alloc(u8, 0);
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const w = &out.writer;
    try w.print("{{\"metric\":\"{s}\",\"attrs\":{{", .{ns.metric().name()});
    for (ns.attrs.names.items, 0..) |name, col| {
        if (col > 0) try w.writeAll(",");
        try attrs_mod.writeJsonString(w, name);
        try w.writeAll(":[");
        for (0..ns.doc_ids.items.len) |row| {
            if (row > 0) try w.writeAll(",");
            try ns.attrs.writeJson(w, ns.attrs.get(col, @intCast(row)));
        }
        try w.writeAll("]");
    }
    try w.writeAll("}}");
    return out.toOwnedSlice();
}

/// Fill an empty `ns` from a binary snapshot (doc ids, tombstones,
/// extension, graph).
fn loadBinarySnapshot(ns: *Namespace, persist: std.mem.Allocator, scratch: std.mem.Allocator, bytes: []const u8, hdr: persist_mod.BinaryHeader) !void {
    try ns.doc_ids.ensureTotalCapacity(persist, hdr.n);
    try ns.id_map.ensureTotalCapacity(persist, @intCast(hdr.n));
    var di: usize = 0;
    while (di < hdr.n) : (di += 1) {
        const id = std.mem.readInt(u64, bytes[hdr.doc_ids_off + di * 8 ..][0..8], .little);
        ns.doc_ids.appendAssumeCapacity(id);
        if (id == deleted_doc) {
            ns.dead += 1;
        } else {
            ns.id_map.putAssumeCapacity(id, @intCast(di));
        }
    }
    if (hdr.ext_len != 0) try loadSnapshotExt(ns, scratch, bytes[hdr.ext_off..][0..hdr.ext_len]);
    try ns.index.load(bytes[hdr.graph_off..]);
    ns.metric_fixed = true;
}

fn loadSnapshotExt(ns: *Namespace, alloc: std.mem.Allocator, ext: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, ext, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadSnapshot;
    if (parsed.value.object.get("metric")) |m| {
        if (m != .string) return error.BadSnapshot;
        ns.index.opts.metric = hnsw_mod.Metric.parse(m.string) orelse return error.BadSnapshot;
    }
    const cols = parsed.value.object.get("attrs") orelse return;
    if (cols != .object) return error.BadSnapshot;
    var it = cols.object.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .array) return error.BadSnapshot;
        const col = try ns.attrs.ensureColumn(kv.key_ptr.*);
        for (kv.value_ptr.array.items, 0..) |v, row| {
            try ns.attrs.set(col, @intCast(row), try ns.attrs.fromJson(v));
        }
    }
}

const Responder = struct {
    req: ?*std.http.Server.Request = null,
    raw_alloc: ?std.mem.Allocator = null,
    raw_status: std.http.Status = .ok,
    raw_body: []const u8 = "",
};

fn takeNamespace(rest: []const u8, suffix: []const u8) ns_mod.Error![]const u8 {
    const name = if (suffix.len != 0 and std.mem.endsWith(u8, rest, suffix))
        rest[0 .. rest.len - suffix.len]
    else
        rest;
    try ns_mod.validate(name);
    return name;
}

fn respondInvalidNamespace(res: *Responder, raw: []const u8) !void {
    var esc_buf: [256]u8 = undefined;
    const esc = ns_mod.jsonEscape(raw, &esc_buf);
    var body_buf: [320]u8 = undefined;
    const body = std.fmt.bufPrint(&body_buf, "{{\"error\":\"invalid namespace\",\"namespace\":\"{s}\"}}", .{esc}) catch
        "{\"error\":\"invalid namespace\"}";
    return respondJson(res, .bad_request, body);
}

fn respondJson(res: *Responder, status: std.http.Status, body: []const u8) !void {
    if (res.req) |req| {
        try req.respond(body, .{
            .status = status,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
        return;
    }
    res.raw_status = status;
    if (res.raw_alloc) |a| {
        res.raw_body = try a.dupe(u8, body);
    } else {
        res.raw_body = body;
    }
}

/// Tight scanner for `{"rank_by":["vector","ANN",[f,f,...]],"top_k":N}`.
/// Avoids building a 1536-node JSON AST on the query hot path.
fn parseU32Field(body: []const u8, key: []const u8) ?u32 {
    const p = std.mem.indexOf(u8, body, key) orelse return null;
    var j = p + key.len;
    while (j < body.len and body[j] != ':') : (j += 1) {}
    if (j < body.len) j += 1;
    while (j < body.len and (body[j] == ' ' or body[j] == '\t')) : (j += 1) {}
    const ns = j;
    while (j < body.len and body[j] >= '0' and body[j] <= '9') : (j += 1) {}
    if (j == ns) return null;
    return std.fmt.parseInt(u32, body[ns..j], 10) catch null;
}

fn parseAnnQuery(body: []const u8, alloc: std.mem.Allocator) !struct { vec: []f32, top_k: usize, ef: ?u32, rerank_mult: ?usize } {
    const ann = std.mem.indexOf(u8, body, "\"ANN\"") orelse return error.BadQuery;
    var i: usize = ann + 5;
    while (i < body.len and body[i] != '[') : (i += 1) {}
    if (i >= body.len) return error.BadQuery;
    i += 1;
    var vec: std.ArrayList(f32) = .empty;
    while (i < body.len) {
        while (i < body.len and (body[i] == ' ' or body[i] == '\t' or body[i] == '\n' or body[i] == '\r' or body[i] == ',')) : (i += 1) {}
        if (i >= body.len) return error.BadQuery;
        if (body[i] == ']') break;
        const start = i;
        if (body[i] == '+' or body[i] == '-') i += 1;
        var saw_digit = false;
        while (i < body.len) {
            const c = body[i];
            if (c >= '0' and c <= '9') {
                saw_digit = true;
                i += 1;
            } else if (c == '.' or c == 'e' or c == 'E') {
                i += 1;
            } else if ((c == '+' or c == '-') and i > start and (body[i - 1] == 'e' or body[i - 1] == 'E')) {
                i += 1;
            } else break;
        }
        if (!saw_digit) return error.BadQuery;
        try vec.append(alloc, try std.fmt.parseFloat(f32, body[start..i]));
    }
    var top_k: usize = 10;
    if (parseU32Field(body, "\"top_k\"") orelse parseU32Field(body, "\"limit\"")) |n| {
        top_k = @max(1, n);
    }
    const ef = parseU32Field(body, "\"ef_search\"") orelse parseU32Field(body, "\"ef\"");
    const rerank_mult = parseU32Field(body, "\"rerank_mult\"") orelse parseU32Field(body, "\"rerank\"");
    return .{
        .vec = try vec.toOwnedSlice(alloc),
        .top_k = top_k,
        .ef = ef,
        .rerank_mult = if (rerank_mult) |v| @as(usize, v) else null,
    };
}

fn handleRequest(
    alloc: std.mem.Allocator,
    io: std.Io,
    req: *std.http.Server.Request,
    registry: *Registry,
    o: Options,
) !void {
    // head.target points into the connection's read buffer; the body read
    // below overwrites it, so copy it out first.
    const path = try alloc.dupe(u8, req.head.target);
    const method = req.head.method;
    var body_read_buf: [1 << 16]u8 = undefined;

    var body: []const u8 = "";
    if (method == .POST) {
        if (req.head.transfer_encoding == .none and req.head.content_length == null) {
            req.server.reader.state = .ready;
        } else {
            var body_reader = req.readerExpectNone(&body_read_buf);
            var body_writer: std.Io.Writer.Allocating = .init(alloc);
            _ = try body_reader.streamRemaining(&body_writer.writer);
            body = body_writer.written();
        }
    }
    const method_s: []const u8 = switch (method) {
        .GET => "GET",
        .POST => "POST",
        .DELETE => "DELETE",
        else => "OTHER",
    };
    var res = Responder{ .req = req };
    return dispatch(alloc, io, method_s, path, body, registry, o, &res);
}

fn dispatch(
    alloc: std.mem.Allocator,
    io: std.Io,
    method: []const u8,
    path: []const u8,
    body: []const u8,
    registry: *Registry,
    o: Options,
    res: *Responder,
) !void {
    const prefix = "/v2/namespaces/";
    if (!std.mem.startsWith(u8, path, prefix)) {
        var ebuf: [512]u8 = undefined;
        const emsg = try std.fmt.bufPrint(&ebuf, "{{\"error\":\"unknown route\",\"target\":\"{s}\"}}", .{path});
        return respondJson(res, .not_found, emsg);
    }
    const rest = path[prefix.len..];

    if (std.mem.endsWith(u8, rest, "/snapshot") and std.mem.eql(u8, method, "POST")) {
        const ns_name = takeNamespace(rest, "/snapshot") catch |e| switch (e) {
            error.InvalidNamespace => return respondInvalidNamespace(res, rest[0 .. rest.len - "/snapshot".len]),
        };
        if (registry.isFailed(ns_name)) return respondJson(res, .service_unavailable, "{\"error\":\"namespace failed to load from object storage; quarantined until a background retry recovers it\"}");
        const ns = registry.get(ns_name) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        if (registry.laneFor(ns.name)) |lane| {
            // On the namespace's own lane, serialized with its WAL uploads.
            var t = Sw.start(io);
            lane.s3_mu.lockUncancelable(lane.io);
            defer lane.s3_mu.unlock(lane.io);
            try snapshotNamespace(lane.store, ns);
            lane.store.deleteWalUpTo(ns.name, ns.wal_seq);
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"snapshot_ms\":{d:.2}}}", .{@as(f64, @floatFromInt(t.readNs(io))) / 1e6});
            return respondJson(res, .ok, msg);
        }
        return respondJson(res, .bad_request, "{\"error\":\"persistence not configured\"}");
    }

    if (std.mem.endsWith(u8, rest, "/query") and std.mem.eql(u8, method, "POST")) {
        const ns_name = takeNamespace(rest, "/query") catch |e| switch (e) {
            error.InvalidNamespace => return respondInvalidNamespace(res, rest[0 .. rest.len - "/query".len]),
        };
        if (registry.isFailed(ns_name)) return respondJson(res, .service_unavailable, "{\"error\":\"namespace failed to load from object storage; quarantined until a background retry recovers it\"}");
        const ns = registry.get(ns_name) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        return handleQuery(alloc, res, ns, o, body);
    }

    if (std.mem.eql(u8, method, "POST")) {
        const ns_name = takeNamespace(rest, "") catch |e| switch (e) {
            error.InvalidNamespace => return respondInvalidNamespace(res, rest),
        };
        const ns = registry.getOrCreate(ns_name) catch |e| switch (e) {
            error.NamespaceUnavailable => return respondJson(res, .service_unavailable, "{\"error\":\"namespace failed to load from object storage; quarantined until a background retry recovers it\"}"),
            else => return e,
        };
        return handleWrite(alloc, registry.alloc, res, ns, body, registry);
    }

    if (std.mem.eql(u8, method, "GET")) {
        const ns_name = takeNamespace(rest, "") catch |e| switch (e) {
            error.InvalidNamespace => return respondInvalidNamespace(res, rest),
        };
        if (registry.isFailed(ns_name)) return respondJson(res, .service_unavailable, "{\"error\":\"namespace failed to load from object storage; quarantined until a background retry recovers it\"}");
        const ns = registry.get(ns_name) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        ns.lock.lockSharedUncancelable(ns.io);
        defer ns.lock.unlockShared(ns.io);
        var buf: [256]u8 = undefined;
        var esc_buf: [160]u8 = undefined;
        const esc = ns_mod.jsonEscape(ns.name, &esc_buf);
        const s = try std.fmt.bufPrint(&buf, "{{\"namespace\":\"{s}\",\"dim\":{d},\"count\":{d},\"metric\":\"{s}\"}}", .{ esc, ns.dim, ns.index.len() - ns.dead, ns.metric().name() });
        return respondJson(res, .ok, s);
    }

    if (std.mem.eql(u8, method, "DELETE")) {
        const ns_name = takeNamespace(rest, "") catch |e| switch (e) {
            error.InvalidNamespace => return respondInvalidNamespace(res, rest),
        };
        registry.mutex.lockUncancelable(registry.io);
        const removed = registry.namespaces.fetchRemove(ns_name);
        registry.mutex.unlock(registry.io);
        if (removed) |kv| {
            kv.value.lock.lockUncancelable(kv.value.io);
            kv.value.lock.unlock(kv.value.io);
            freeNamespace(registry.alloc, kv.value);
        }
        return respondJson(res, .ok, "{\"ok\":true}");
    }

    return respondJson(res, .method_not_allowed, "{\"error\":\"method not allowed\"}");
}

fn upsertDoc(ns: *Namespace, persist: std.mem.Allocator, id: u64, vec: []const f32) !void {
    if (ns.id_map.get(id)) |local| {
        try ns.index.update(local, vec);
        return;
    }
    const local = try ns.index.insert(vec);
    try ns.doc_ids.append(persist, id);
    try ns.id_map.put(persist, id, local);
}

/// Room for one more published row without reallocating anything a
/// concurrent reader or writer may hold a pointer into.
fn hasWriteRoom(ns: *const Namespace) bool {
    return ns.index.hasInsertRoom(1) and ns.doc_ids.capacity > ns.doc_ids.items.len and
        ns.id_map.available > 0 and ns.attrs.rows > ns.doc_ids.items.len;
}

/// Grow the packed slabs, doc_ids and id_map. The exclusive lock keeps out
/// queries, in-place updates and publishers (they all hold it shared);
/// map_mu keeps out id_map lookups.
fn growWriteRoom(ns: *Namespace, persist: std.mem.Allocator) !void {
    ns.lock.lockUncancelable(ns.io);
    defer ns.lock.unlock(ns.io);
    ns.map_mu.lockUncancelable(ns.io);
    defer ns.map_mu.unlock(ns.io);
    if (hasWriteRoom(ns)) return;
    try ns.index.ensureInsertRoom(1024);
    try ns.doc_ids.ensureTotalCapacity(persist, ns.doc_ids.items.len + 1024);
    try ns.id_map.ensureUnusedCapacity(persist, 1024);
    try ns.attrs.reserveRows(ns.doc_ids.capacity);
}

/// Replace an existing document's vector and attributes. With attributes
/// in play this takes the exclusive lock, so no query sees the new vector
/// with the old attributes (or a half-written value).
fn updateDoc(ns: *Namespace, local: u32, vec: []const f32, attrs: []const AttrSet) !void {
    if (attrs.len == 0 and ns.attrs.isEmpty()) {
        ns.lock.lockSharedUncancelable(ns.io);
        defer ns.lock.unlockShared(ns.io);
        return ns.index.update(local, vec);
    }
    ns.lock.lockUncancelable(ns.io);
    defer ns.lock.unlock(ns.io);
    if (local < ns.doc_ids.items.len and ns.doc_ids.items[local] == deleted_doc) return; // deleted meanwhile
    try ns.index.update(local, vec);
    ns.attrs.clearRow(local);
    for (attrs) |a| try ns.attrs.set(a.col, local, a.val);
}

/// Concurrent upsert. Every step that touches the packed slabs holds the
/// namespace lock at least shared, so `growWriteRoom` (exclusive) can never
/// reallocate them under an in-place update or a publish. (Publishing used
/// to run under splice_mu alone and could append into a slab that another
/// writer was reallocating; concurrent bulk writes crashed.)
fn upsertDocFair(ns: *Namespace, persist: std.mem.Allocator, scratch: std.mem.Allocator, id: u64, vec: []const f32, attrs: []const AttrSet) !void {
    ns.map_mu.lockUncancelable(ns.io);
    const existing = ns.id_map.get(id);
    ns.map_mu.unlock(ns.io);
    if (existing) |local| return updateDoc(ns, local, vec, attrs);
    if (ns.index.entryPoint() == null) {
        ns.lock.lockUncancelable(ns.io);
        defer ns.lock.unlock(ns.io);
        if (ns.index.entryPoint() == null) {
            ns.map_mu.lockUncancelable(ns.io);
            defer ns.map_mu.unlock(ns.io);
            try upsertDoc(ns, persist, id, vec);
            const local = ns.id_map.get(id).?;
            ns.attrs.clearRow(local);
            for (attrs) |a| try ns.attrs.set(a.col, local, a.val);
            return;
        }
    }
    if (try insertDocFair(ns, persist, scratch, id, vec, attrs)) |local| return updateDoc(ns, local, vec, attrs);
}

/// Insert path of `upsertDocFair`. Returns the row instead when another
/// writer inserted `id` first (the caller then updates it).
fn insertDocFair(ns: *Namespace, persist: std.mem.Allocator, scratch: std.mem.Allocator, id: u64, vec: []const f32, attrs: []const AttrSet) !?u32 {
    ns.splice_mu.lockUncancelable(ns.io);
    const level = ns.index.nextLevel();
    ns.splice_mu.unlock(ns.io);

    ns.lock.lockSharedUncancelable(ns.io);
    var plan = ns.index.planInsert(vec, level, scratch) catch |e| {
        ns.lock.unlockShared(ns.io);
        return e;
    };
    ns.lock.unlockShared(ns.io);

    ns.splice_mu.lockUncancelable(ns.io);
    defer ns.splice_mu.unlock(ns.io);
    // Publishers are serialized by splice_mu, so room seen here stays ours.
    while (true) {
        ns.lock.lockSharedUncancelable(ns.io);
        if (hasWriteRoom(ns)) break;
        ns.lock.unlockShared(ns.io);
        growWriteRoom(ns, persist) catch |e| {
            plan.discard(ns.index.allocator);
            return e;
        };
    }
    // Held through the splice: queries also hold it shared and keep walking.
    defer ns.lock.unlockShared(ns.io);

    ns.map_mu.lockUncancelable(ns.io);
    if (ns.id_map.get(id)) |local| {
        ns.map_mu.unlock(ns.io);
        plan.discard(ns.index.allocator);
        return local;
    }
    ns.map_mu.unlock(ns.io);

    const local = ns.index.publishInsert(&plan) catch |e| {
        plan.discard(ns.index.allocator);
        return e;
    };
    // Queries see a row only once doc_ids covers it, so its attributes are
    // in place first. A fresh row's values are all null already.
    for (attrs) |a| ns.attrs.setReserved(a.col, local, a.val);
    ns.doc_ids.appendAssumeCapacity(id);
    ns.map_mu.lockUncancelable(ns.io);
    ns.id_map.putAssumeCapacity(id, local);
    ns.map_mu.unlock(ns.io);
    ns.index.publishEntry(local, plan.level);
    _ = @atomicRmw(u32, &ns.unspliced, .Add, 1, .acq_rel);
    defer _ = @atomicRmw(u32, &ns.unspliced, .Sub, 1, .acq_rel);
    try ns.index.spliceBackEdges(&plan, local);
    return null;
}

/// Replay one WAL segment. Entries are `[id,[vec]]` and `[id,[vec],{attrs}]`
/// (upserts) and `[id,null]` (deletes), in write order.
fn replayWalBody(ns: *Namespace, persist: std.mem.Allocator, alloc: std.mem.Allocator, body: []const u8) !usize {
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadWal;
    if (parsed.value.object.get("metric")) |m| {
        if (m != .string) return error.BadWal;
        const metric = hnsw_mod.Metric.parse(m.string) orelse return error.BadWal;
        if (ns.index.len() == 0) ns.index.opts.metric = metric else if (metric != ns.metric()) return error.BadWal;
    }
    const docs = parsed.value.object.get("docs") orelse return error.BadWal;
    if (docs != .array) return error.BadWal;
    for (docs.array.items) |doc| {
        if (doc != .array or doc.array.items.len < 2 or doc.array.items[0] != .integer) return error.BadWal;
        const entry = doc.array.items;
        const id: u64 = std.math.cast(u64, entry[0].integer) orelse return error.BadWal;
        if (entry[1] == .null) {
            deleteDocLocked(ns, id);
            continue;
        }
        if (entry[1] != .array) return error.BadWal;
        if (ns.dim == 0) {
            ns.dim = entry[1].array.items.len;
            ns.index.dim = ns.dim;
        }
        const v = try vecFromJson(entry[1].array, alloc);
        try upsertDoc(ns, persist, id, v);
        const local = ns.id_map.get(id).?;
        ns.attrs.clearRow(local);
        if (entry.len > 2) {
            if (entry[2] != .object) return error.BadWal;
            var it = entry[2].object.iterator();
            while (it.next()) |kv| {
                const col = try ns.attrs.ensureColumn(kv.key_ptr.*);
                try ns.attrs.set(col, local, try ns.attrs.fromJson(kv.value_ptr.*));
            }
        }
    }
    ns.metric_fixed = true;
    return docs.array.items.len;
}

/// Tombstone `id`. Caller holds the exclusive lock (or owns `ns` outright).
fn deleteDocLocked(ns: *Namespace, id: u64) void {
    const kv = ns.id_map.fetchRemove(id) orelse return;
    ns.doc_ids.items[kv.value] = deleted_doc;
    ns.attrs.clearRow(kv.value);
    ns.dead += 1;
}

fn vecFromJson(arr: std.json.Array, alloc: std.mem.Allocator) ![]f32 {
    const v = try alloc.alloc(f32, arr.items.len);
    for (arr.items, 0..) |x, i| {
        v[i] = switch (x) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadVector,
        };
    }
    return v;
}

const AttrIn = struct { name: []const u8, val: std.json.Value };
/// An attribute resolved against the namespace's store (column created,
/// string interned) under the exclusive lock, so applying it never allocates.
const AttrSet = struct { col: usize, val: attrs_mod.Value };
const UpsertIn = struct { id: u64, vec: []f32, attrs: []const AttrIn, resolved: []AttrSet = &.{} };

/// Why a write body was rejected (400).
const BadWrite = struct { msg: []const u8 };

fn docId(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |n| std.math.cast(u64, n),
        else => null,
    };
}

fn isAttrValue(v: std.json.Value) bool {
    return switch (v) {
        .null, .bool, .integer, .float, .number_string, .string => true,
        .array, .object => false,
    };
}

fn isReservedColumn(name: []const u8) bool {
    return std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "vector") or
        std.mem.eql(u8, name, "attributes") or (name.len > 0 and name[0] == '$');
}

/// Parse upsert_columns / upsert_rows into `out`. Returns a message on bad input.
fn parseUpserts(alloc: std.mem.Allocator, root: std.json.ObjectMap, out: *std.ArrayList(UpsertIn)) !?BadWrite {
    if (root.get("upsert_columns")) |cols_v| {
        if (cols_v != .object) return .{ .msg = "upsert_columns must be an object" };
        const cols = cols_v.object;
        const ids_v = cols.get("id") orelse return .{ .msg = "id column required" };
        const vecs_v = cols.get("vector") orelse return .{ .msg = "vector column required" };
        if (ids_v != .array or vecs_v != .array) return .{ .msg = "columns must be arrays" };
        const n = ids_v.array.items.len;
        if (vecs_v.array.items.len != n) return .{ .msg = "every column must have one value per id" };
        var names: std.ArrayList([]const u8) = .empty;
        var values: std.ArrayList([]std.json.Value) = .empty;
        var it = cols.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "vector")) continue;
            if (isReservedColumn(name)) return .{ .msg = "attribute names must not be reserved (id, vector, attributes, $...)" };
            if (kv.value_ptr.* != .array or kv.value_ptr.array.items.len != n) return .{ .msg = "every column must have one value per id" };
            try names.append(alloc, name);
            try values.append(alloc, kv.value_ptr.array.items);
        }
        for (0..n) |i| {
            const id = docId(ids_v.array.items[i]) orelse return .{ .msg = "ids must be non-negative integers" };
            const vec_v = vecs_v.array.items[i];
            if (vec_v != .array) return .{ .msg = "vector rows must be arrays" };
            const row_attrs = try alloc.alloc(AttrIn, names.items.len);
            for (names.items, values.items, row_attrs) |name, vals, *slot| {
                if (!isAttrValue(vals[i])) return .{ .msg = "attribute values must be strings, numbers, booleans or null" };
                slot.* = .{ .name = name, .val = vals[i] };
            }
            const vec = vecFromJson(vec_v.array, alloc) catch return .{ .msg = "vector values must be numbers" };
            try out.append(alloc, .{ .id = id, .vec = vec, .attrs = row_attrs });
        }
    }
    if (root.get("upsert_rows")) |rows_v| {
        if (rows_v != .array) return .{ .msg = "upsert_rows must be an array" };
        for (rows_v.array.items) |row| {
            if (row != .object) return .{ .msg = "rows must be objects" };
            const robj = row.object;
            const id = docId(robj.get("id") orelse return .{ .msg = "row missing id" }) orelse
                return .{ .msg = "ids must be non-negative integers" };
            const vec_v = robj.get("vector") orelse return .{ .msg = "row missing vector" };
            if (vec_v != .array) return .{ .msg = "vector must be an array" };
            var row_attrs: std.ArrayList(AttrIn) = .empty;
            var it = robj.iterator();
            while (it.next()) |kv| {
                const name = kv.key_ptr.*;
                if (std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "vector")) continue;
                if (isReservedColumn(name)) return .{ .msg = "attribute names must not be reserved (id, vector, attributes, $...)" };
                if (!isAttrValue(kv.value_ptr.*)) return .{ .msg = "attribute values must be strings, numbers, booleans or null" };
                try row_attrs.append(alloc, .{ .name = name, .val = kv.value_ptr.* });
            }
            const vec = vecFromJson(vec_v.array, alloc) catch return .{ .msg = "vector values must be numbers" };
            try out.append(alloc, .{ .id = id, .vec = vec, .attrs = row_attrs.items });
        }
    }
    return null;
}

fn writeJsonValue(w: *std.Io.Writer, v: std.json.Value) !void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |n| try w.print("{d}", .{n}),
        .float => |f| try w.print("{d}", .{f}),
        .number_string => |str| try w.writeAll(str),
        .string => |str| try attrs_mod.writeJsonString(w, str),
        .array, .object => unreachable, // rejected by isAttrValue
    }
}

fn respondBadWrite(res: *Responder, alloc: std.mem.Allocator, msg: []const u8) !void {
    var out: std.Io.Writer.Allocating = .init(alloc);
    try out.writer.writeAll("{\"error\":");
    try attrs_mod.writeJsonString(&out.writer, msg);
    try out.writer.writeAll("}");
    return respondJson(res, .bad_request, out.written());
}

/// Write order within one request: delete_by_filter, then upserts, then
/// deletes. The WAL records the same order, with filter matches expanded to
/// ids, so replay is deterministic.
fn handleWrite(
    arena: std.mem.Allocator,
    persist: std.mem.Allocator,
    res: *Responder,
    ns: *Namespace,
    body: []const u8,
    registry: *Registry,
) !void {
    // NOTE: body already consumed by caller
    const alloc = arena;
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch
        return respondJson(res, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();
    if (parsed.value != .object) return respondJson(res, .bad_request, "{\"error\":\"expected object\"}");
    const root = parsed.value.object;

    var metric: ?hnsw_mod.Metric = null;
    if (root.get("distance_metric")) |dm| {
        if (dm != .string) return respondBadWrite(res, alloc, "distance_metric must be a string");
        metric = hnsw_mod.Metric.parse(dm.string) orelse
            return respondBadWrite(res, alloc, "distance_metric must be cosine_distance, euclidean_squared or dot_product");
    }

    var ups: std.ArrayList(UpsertIn) = .empty;
    if (try parseUpserts(alloc, root, &ups)) |bad| return respondBadWrite(res, alloc, bad.msg);

    var deletes: std.ArrayList(u64) = .empty;
    if (root.get("deletes")) |dv| {
        if (dv != .array) return respondBadWrite(res, alloc, "deletes must be an array of ids");
        for (dv.array.items) |v| try deletes.append(alloc, docId(v) orelse return respondBadWrite(res, alloc, "ids must be non-negative integers"));
    }
    const del_filter = root.get("delete_by_filter");
    if (ups.items.len == 0 and deletes.items.len == 0 and del_filter == null) {
        if (root.get("upsert_columns") != null or root.get("upsert_rows") != null) return respondJson(res, .ok, "{\"ok\":true}");
        return respondBadWrite(res, alloc, "missing upsert_columns, upsert_rows, deletes or delete_by_filter");
    }

    {
        ns.lock.lockUncancelable(ns.io);
        defer ns.lock.unlock(ns.io);
        if (ups.items.len != 0) {
            if (ns.dim == 0) {
                ns.dim = ups.items[0].vec.len;
                ns.index.dim = ns.dim;
            }
            for (ups.items) |u| {
                if (u.vec.len != ns.dim) return respondJson(res, .bad_request, "{\"error\":\"dimension mismatch\"}");
            }
        }
        if (metric) |m| {
            if (!ns.metric_fixed and ns.index.len() == 0) {
                ns.index.opts.metric = m;
            } else if (m != ns.metric()) {
                var buf: [160]u8 = undefined;
                const msg = try std.fmt.bufPrint(&buf, "distance_metric is {s} for this namespace", .{ns.metric().name()});
                return respondBadWrite(res, alloc, msg);
            }
        }
        if (ups.items.len != 0) ns.metric_fixed = true;
        for (ups.items) |*u| {
            u.resolved = try alloc.alloc(AttrSet, u.attrs.len);
            for (u.attrs, u.resolved) |a, *r| r.* = .{
                .col = try ns.attrs.ensureColumn(a.name),
                .val = ns.attrs.fromJson(a.val) catch |e| switch (e) {
                    error.BadAttribute, error.BadFilter => unreachable, // rejected by isAttrValue
                    error.OutOfMemory => return e,
                },
            };
        }
    }
    // Reject bad vectors before applying anything: a failure halfway through
    // would leave applied rows that the WAL never records.
    for (ups.items) |u| {
        hnsw_mod.checkVector(ns.metric(), u.vec) catch |e| return respondBadWrite(res, alloc, switch (e) {
            error.ZeroNormVector => "a zero vector has no cosine direction",
            else => "vector values must be finite numbers",
        });
    }

    var filter_deletes: std.ArrayList(u64) = .empty;
    if (del_filter) |f| {
        ns.lock.lockSharedUncancelable(ns.io);
        defer ns.lock.unlockShared(ns.io);
        const n = @min(ns.index.len(), ns.doc_ids.items.len);
        var bits = ns.attrs.eval(f, ns.doc_ids.items[0..n], n, alloc) catch |e| switch (e) {
            error.BadFilter, error.BadAttribute => return respondBadWrite(res, alloc, "delete_by_filter is not a valid filter"),
            error.OutOfMemory => return e,
        };
        var it = bits.iterator(.{});
        while (it.next()) |row| {
            if (ns.doc_ids.items[row] != deleted_doc) try filter_deletes.append(alloc, ns.doc_ids.items[row]);
        }
    }

    try applyDeletes(ns, filter_deletes.items);

    // Updates: vec_gen seqlock, no namespace lock. Inserts: shared
    // neighbor search, exclusive only if slabs need to grow, then
    // splice_mu publish + shared back-edge splice.
    for (ups.items) |u| try upsertDocFair(ns, persist, alloc, u.id, u.vec, u.resolved);

    try applyDeletes(ns, deletes.items);

    // durable WAL: group-commit via persist worker when configured
    if (registry.lanes.len != 0 or registry.store != null) {
        var wb: std.Io.Writer.Allocating = .init(persist);
        const ww = &wb.writer;
        try ww.writeAll("{\"docs\":[");
        var first = true;
        for (filter_deletes.items) |id| {
            if (!first) try ww.writeAll(",");
            first = false;
            try ww.print("[{d},null]", .{id});
        }
        for (ups.items) |u| {
            if (!first) try ww.writeAll(",");
            first = false;
            try ww.print("[{d},[", .{u.id});
            for (u.vec, 0..) |x, j| {
                if (j > 0) try ww.writeAll(",");
                try ww.print("{d}", .{x});
            }
            try ww.writeAll("]");
            if (u.attrs.len != 0) {
                try ww.writeAll(",{");
                for (u.attrs, 0..) |a, j| {
                    if (j > 0) try ww.writeAll(",");
                    try attrs_mod.writeJsonString(ww, a.name);
                    try ww.writeAll(":");
                    try writeJsonValue(ww, a.val);
                }
                try ww.writeAll("}");
            }
            try ww.writeAll("]");
        }
        for (deletes.items) |id| {
            if (!first) try ww.writeAll(",");
            first = false;
            try ww.print("[{d},null]", .{id});
        }
        try ww.writeAll("]}");
        const wal_body = try wb.toOwnedSlice();
        if (registry.laneFor(ns.name)) |pw| {
            var job = WalJob{ .ns = ns, .body = wal_body };
            pw.commit(&job) catch |e| {
                persist.free(job.body);
                return e;
            };
        } else if (registry.store) |store| {
            defer persist.free(wal_body);
            var full: std.Io.Writer.Allocating = .init(persist);
            defer full.deinit();
            try writeWalOpen(&full.writer, ns.metric());
            try full.writer.writeAll(walDocsInner(wal_body));
            try full.writer.writeAll("]}");
            ns.lock.lockUncancelable(ns.io);
            ns.wal_seq += 1;
            const seq = ns.wal_seq;
            ns.lock.unlock(ns.io);
            try store.appendWal(ns.name, seq, full.written());
        }
    }

    return respondJson(res, .ok, "{\"ok\":true}");
}

/// Same lock order as `upsertDocFair`: splice_mu, then the namespace lock,
/// then map_mu.
fn lockForRowChanges(ns: *Namespace) void {
    ns.splice_mu.lockUncancelable(ns.io);
    ns.lock.lockUncancelable(ns.io);
    ns.map_mu.lockUncancelable(ns.io);
}

fn unlockRowChanges(ns: *Namespace) void {
    ns.map_mu.unlock(ns.io);
    ns.lock.unlock(ns.io);
    ns.splice_mu.unlock(ns.io);
}

fn applyDeletes(ns: *Namespace, ids: []const u64) !void {
    if (ids.len == 0) return;
    lockForRowChanges(ns);
    defer unlockRowChanges(ns);
    for (ids) |id| deleteDocLocked(ns, id);
}

/// Accepts rows whose document is live.
const LiveRows = struct {
    doc_ids: []const u64,
    pub inline fn ok(l: LiveRows, id: u32) bool {
        return id < l.doc_ids.len and l.doc_ids[id] != deleted_doc;
    }
};

/// What a query asked to get back besides id and distance.
const Include = union(enum) { none, all, names: []const []const u8 };

fn parseInclude(alloc: std.mem.Allocator, body: []const u8) !?Include {
    const span = attrs_mod.valueSpan(body, "\"include_attributes\"") orelse return .none;
    const v = (std.json.parseFromSlice(std.json.Value, alloc, span, .{}) catch return null).value;
    switch (v) {
        .bool => |b| return if (b) .all else .none,
        .null => return .none,
        .array => |arr| {
            const names = try alloc.alloc([]const u8, arr.items.len);
            for (arr.items, names) |item, *n| {
                if (item != .string) return null;
                n.* = item.string;
            }
            return .{ .names = names };
        },
        else => return null,
    }
}

fn handleQuery(
    alloc: std.mem.Allocator,
    res: *Responder,
    ns: *Namespace,
    o: Options,
    body: []const u8,
) !void {
    const spec = parseAnnQuery(body, alloc) catch
        return respondJson(res, .bad_request, "{\"error\":\"rank_by must be [field,'ANN',vector]\"}");
    const query_vec = spec.vec;
    const top_k = spec.top_k;
    var filter: ?std.json.Value = null;
    if (attrs_mod.valueSpan(body, "\"filters\"")) |span| {
        const f = (std.json.parseFromSlice(std.json.Value, alloc, span, .{}) catch
            return respondJson(res, .bad_request, "{\"error\":\"filters must be JSON\"}")).value;
        if (f != .null) filter = f;
    }
    const include = try parseInclude(alloc, body) orelse
        return respondJson(res, .bad_request, "{\"error\":\"include_attributes must be true, false or a list of names\"}");

    ns.lock.lockSharedUncancelable(ns.io);
    defer ns.lock.unlockShared(ns.io);
    if (ns.index.entryPoint() == null) return respondJson(res, .ok, "{\"rows\":[]}");
    if (query_vec.len != ns.dim) {
        var dbuf: [128]u8 = undefined;
        const dmsg = try std.fmt.bufPrint(&dbuf, "{{\"error\":\"dimension mismatch\",\"expected\":{d},\"got\":{d}}}", .{ ns.dim, query_vec.len });
        return respondJson(res, .bad_request, dmsg);
    }

    const ef = spec.ef orelse o.ef;
    const rerank_mult = spec.rerank_mult orelse o.rerank_mult;
    // Rows past doc_ids are mid-publish; leave them out.
    const n = @min(ns.index.len(), ns.doc_ids.items.len);
    const live = LiveRows{ .doc_ids = ns.doc_ids.items[0..n] };
    const results: []const hnsw_mod.SearchResult = blk: {
        if (filter) |f| {
            var bits = ns.attrs.eval(f, live.doc_ids, n, alloc) catch |e| switch (e) {
                error.BadFilter, error.BadAttribute => return respondJson(res, .bad_request, "{\"error\":\"filters must be [attr, op, value], [\\\"And\\\"|\\\"Or\\\", [filters]] or [\\\"Not\\\", filter]\"}"),
                error.OutOfMemory => return e,
            };
            if (ns.dead != 0) {
                for (live.doc_ids, 0..) |d, row| {
                    if (d == deleted_doc) bits.unset(row);
                }
            }
            // Few matches: exact scan of just those rows. Many: graph walk
            // that only accepts matches.
            const matches = bits.count();
            if (matches == 0) break :blk &.{};
            if (matches <= ns.index.scanLimit(ef)) break :blk try ns.index.searchScan(query_vec, top_k, rerank_mult, &bits, alloc);
            break :blk try ns.index.searchAccept(query_vec, top_k, ef, rerank_mult, Hnsw.AcceptBits{ .bits = &bits }, alloc);
        }
        if (ns.dead != 0 or n < ns.index.len()) break :blk try ns.index.searchAccept(query_vec, top_k, ef, rerank_mult, live, alloc);
        break :blk try ns.index.searchAdvanced(query_vec, top_k, ef, rerank_mult, alloc);
    };

    var out: std.Io.Writer.Allocating = .init(alloc);
    const w = &out.writer;
    try w.writeAll("{\"rows\":[");
    for (results, 0..) |r, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("{{\"id\":{d},\"$distance\":{d}", .{ ns.doc_ids.items[r.id], r.distance });
        switch (include) {
            .none => {},
            .all => try ns.attrs.writeRowFields(w, r.id, null),
            .names => |names| try ns.attrs.writeRowFields(w, r.id, names),
        }
        try w.writeAll("}");
    }
    try w.writeAll("],\"usage\":{}}");
    return respondJson(res, .ok, out.written());
}

test "parseAnnQuery reads vector and top_k" {
    const body = "{\"rank_by\":[\"vector\",\"ANN\",[1.5,-2,3e-1]],\"top_k\":7}";
    const spec = try parseAnnQuery(body, std.testing.allocator);
    defer std.testing.allocator.free(spec.vec);
    try std.testing.expectEqual(@as(usize, 3), spec.vec.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), spec.vec[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), spec.vec[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), spec.vec[2], 1e-6);
    try std.testing.expectEqual(@as(usize, 7), spec.top_k);
    try std.testing.expectEqual(@as(?u32, null), spec.ef);
    try std.testing.expectEqual(@as(?usize, null), spec.rerank_mult);
}

test "parseAnnQuery reads optional ef" {
    const body = "{\"rank_by\":[\"vector\",\"ANN\",[1]],\"top_k\":3,\"ef\":64}";
    const spec = try parseAnnQuery(body, std.testing.allocator);
    defer std.testing.allocator.free(spec.vec);
    try std.testing.expectEqual(@as(usize, 3), spec.top_k);
    try std.testing.expectEqual(@as(?u32, 64), spec.ef);
    try std.testing.expectEqual(@as(?usize, null), spec.rerank_mult);
}

test "recovery listing skips invalid object-key namespaces" {
    const keys = [_][]const u8{
        "openpuffer/../snapshot.bin",
        "openpuffer/wal/snapshot.bin",
        "openpuffer/codedb/nested/snapshot.bin",
        "openpuffer/codedb/snapshot.bin",
    };
    var kept: usize = 0;
    var last: []const u8 = "";
    for (keys) |k| {
        if (!std.mem.startsWith(u8, k, "openpuffer/")) continue;
        const tail = k["openpuffer/".len..];
        const slash = std.mem.indexOfScalar(u8, tail, '/') orelse continue;
        const name = tail[0..slash];
        if (!ns_mod.isValid(name)) continue;
        if (std.mem.eql(u8, tail[slash + 1 ..], "snapshot.bin")) {
            kept += 1;
            last = name;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), kept);
    try std.testing.expectEqualStrings("codedb", last);
}

test "takeNamespace rejects nested and empty names" {
    try std.testing.expectError(error.InvalidNamespace, takeNamespace("codedb/nested", ""));
    try std.testing.expectError(error.InvalidNamespace, takeNamespace("codedb/nested/query", "/query"));
    try std.testing.expectError(error.InvalidNamespace, takeNamespace("", ""));
    try std.testing.expectError(error.InvalidNamespace, takeNamespace(".", ""));
    try std.testing.expectError(error.InvalidNamespace, takeNamespace("a\nb/query", "/query"));
    try std.testing.expectEqualStrings("codedb", try takeNamespace("codedb", ""));
    try std.testing.expectEqualStrings("codedb", try takeNamespace("codedb/query", "/query"));
    try std.testing.expectEqualStrings("openpuffer-bench-1", try takeNamespace("openpuffer-bench-1/snapshot", "/snapshot"));
}

test "parseAnnQuery reads optional rerank_mult" {
    const body = "{\"rank_by\":[\"vector\",\"ANN\",[1]],\"top_k\":3,\"ef\":64,\"rerank_mult\":8}";
    const spec = try parseAnnQuery(body, std.testing.allocator);
    defer std.testing.allocator.free(spec.vec);
    try std.testing.expectEqual(@as(?u32, 64), spec.ef);
    try std.testing.expectEqual(@as(?usize, 8), spec.rerank_mult);
}

fn testCall(reg: *Registry, alloc: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !struct { status: std.http.Status, body: []const u8 } {
    var res = Responder{ .raw_alloc = alloc };
    try dispatch(alloc, std.testing.io, method, path, body, reg, .{}, &res);
    return .{ .status = res.raw_status, .body = res.raw_body };
}

fn expectContains(hay: []const u8, needle: []const u8, want: bool) !void {
    if ((std.mem.indexOf(u8, hay, needle) != null) != want) {
        std.debug.print("expected {s}'{s}' in {s}\n", .{ if (want) "" else "no ", needle, hay });
        return error.TestUnexpectedResult;
    }
}

test "attributes, filters, deletes and delete_by_filter" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var reg = Registry{ .alloc = a, .io = std.testing.io };
    const ns = "/v2/namespaces/mem";

    var r = try testCall(&reg, a, "POST", ns,
        \\{"upsert_columns":{"id":[1,2,3,4],"vector":[[1,0],[0.9,0.1],[0.8,0.2],[0,1]],
        \\ "kind":["fact","todo","fact",null],"pri":[1,2,3.5,4]},"distance_metric":"cosine_distance"}
    );
    try std.testing.expectEqual(std.http.Status.ok, r.status);

    r = try testCall(&reg, a, "POST", ns ++ "/query",
        \\{"rank_by":["vector","ANN",[1,0]],"top_k":10,"filters":["kind","Eq","fact"],"include_attributes":true}
    );
    try std.testing.expectEqual(std.http.Status.ok, r.status);
    try expectContains(r.body, "{\"id\":1,", true);
    try expectContains(r.body, "{\"id\":3,", true);
    try expectContains(r.body, "{\"id\":2,", false);
    try expectContains(r.body, "\"kind\":\"fact\",\"pri\":1}", true);

    r = try testCall(&reg, a, "POST", ns ++ "/query",
        \\{"rank_by":["vector","ANN",[1,0]],"top_k":10,"filters":["And",[["pri","Gte",2],["Not",["kind","Eq",null]]]],"include_attributes":["pri"]}
    );
    try expectContains(r.body, "{\"id\":2,", true);
    try expectContains(r.body, "{\"id\":3,", true);
    try expectContains(r.body, "{\"id\":4,", false);
    try expectContains(r.body, "\"kind\"", false);

    // delete one id, then everything matching a filter
    r = try testCall(&reg, a, "POST", ns, "{\"deletes\":[1]}");
    try std.testing.expectEqual(std.http.Status.ok, r.status);
    r = try testCall(&reg, a, "POST", ns ++ "/query", "{\"rank_by\":[\"vector\",\"ANN\",[1,0]],\"top_k\":10}");
    try expectContains(r.body, "{\"id\":1,", false);
    try expectContains(r.body, "{\"id\":2,", true);
    r = try testCall(&reg, a, "POST", ns, "{\"delete_by_filter\":[\"kind\",\"Eq\",\"todo\"]}");
    try std.testing.expectEqual(std.http.Status.ok, r.status);
    r = try testCall(&reg, a, "GET", ns, "");
    try expectContains(r.body, "\"count\":2", true);

    // an upsert replaces the document: attributes not given are cleared
    r = try testCall(&reg, a, "POST", ns, "{\"upsert_rows\":[{\"id\":3,\"vector\":[0.7,0.3]}]}");
    try std.testing.expectEqual(std.http.Status.ok, r.status);
    r = try testCall(&reg, a, "POST", ns ++ "/query",
        \\{"rank_by":["vector","ANN",[1,0]],"top_k":10,"filters":["kind","Eq","fact"]}
    );
    try std.testing.expectEqualStrings("{\"rows\":[],\"usage\":{}}", r.body);
    // a deleted id can come back
    r = try testCall(&reg, a, "POST", ns, "{\"upsert_rows\":[{\"id\":1,\"vector\":[1,0],\"kind\":\"fact\"}]}");
    r = try testCall(&reg, a, "POST", ns ++ "/query",
        \\{"rank_by":["vector","ANN",[1,0]],"top_k":10,"filters":["kind","In",["fact"]]}
    );
    try expectContains(r.body, "{\"id\":1,", true);

    // rejected writes and queries
    const bad = [_][]const u8{
        "{\"upsert_rows\":[{\"id\":9,\"vector\":[1,0]}],\"distance_metric\":\"euclidean_squared\"}",
        "{\"upsert_rows\":[{\"id\":-1,\"vector\":[1,0]}]}",
        "{\"upsert_rows\":[{\"id\":9,\"vector\":[0,0]}]}",
        "{\"upsert_columns\":{\"id\":[9,10],\"vector\":[[1,0]]}}",
        "{\"upsert_rows\":[{\"id\":9,\"vector\":[1,0],\"tags\":[\"x\"]}]}",
        "{\"upsert_rows\":[{\"id\":9,\"vector\":[1,0],\"$distance\":1}]}",
        "{\"delete_by_filter\":[\"kind\",\"Like\",\"x\"]}",
        "{\"distance_metric\":\"manhattan\",\"deletes\":[1]}",
    };
    for (bad) |b| {
        r = try testCall(&reg, a, "POST", ns, b);
        std.testing.expectEqual(std.http.Status.bad_request, r.status) catch |e| {
            std.debug.print("accepted: {s}\n", .{b});
            return e;
        };
    }
    r = try testCall(&reg, a, "POST", ns ++ "/query", "{\"rank_by\":[\"vector\",\"ANN\",[1,0]],\"filters\":[\"kind\"]}");
    try std.testing.expectEqual(std.http.Status.bad_request, r.status);
    r = try testCall(&reg, a, "GET", ns, "");
    try expectContains(r.body, "\"count\":3", true);
}

test "euclidean namespaces use the metric for ranking and distances" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var reg = Registry{ .alloc = a, .io = std.testing.io };
    var r = try testCall(&reg, a, "POST", "/v2/namespaces/l2",
        \\{"upsert_columns":{"id":[1,2],"vector":[[10,0],[1,0]]},"distance_metric":"euclidean_squared"}
    );
    try std.testing.expectEqual(std.http.Status.ok, r.status);
    // by cosine both are equally close to [2,0]; by squared L2, id 2 is (1 vs 64)
    r = try testCall(&reg, a, "POST", "/v2/namespaces/l2/query", "{\"rank_by\":[\"vector\",\"ANN\",[2,0]],\"top_k\":2}");
    try std.testing.expectEqualStrings("{\"rows\":[{\"id\":2,\"$distance\":1},{\"id\":1,\"$distance\":64}],\"usage\":{}}", r.body);
    r = try testCall(&reg, a, "GET", "/v2/namespaces/l2", "");
    try expectContains(r.body, "\"metric\":\"euclidean_squared\"", true);
}

test "WAL replay and snapshots keep attributes, tombstones and the metric" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var ns = Namespace.init(a, "m", std.testing.io);
    const wal =
        \\{"metric":"euclidean_squared","docs":[[5,[1,0],{"kind":"a","n":2}],[6,[3,4]],[5,null],[7,[0,2],{"kind":"b \"q\""}]]}
    ;
    _ = try replayWalBody(&ns, a, a, wal);
    try std.testing.expectEqual(hnsw_mod.Metric.euclidean_squared, ns.metric());
    try std.testing.expectEqual(@as(usize, 1), ns.dead);
    try std.testing.expect(ns.id_map.get(5) == null);
    const col = ns.attrs.column("kind").?;
    try std.testing.expectEqualStrings("b \"q\"", ns.attrs.strings.items[ns.attrs.get(col, ns.id_map.get(7).?).str]);
    try std.testing.expect(ns.attrs.get(col, 0) == .null); // deleted row's attributes are gone

    const ext = try snapshotExt(a, &ns);
    const body = try persist_mod.Store.buildBinarySnapshotExt(a, 9, ns.dim, ns.doc_ids.items, ext, &ns.index);
    const hdr = try persist_mod.parseBinaryHeader(body);
    try std.testing.expectEqual(persist_mod.SNAP_VERSION_EXT, hdr.version);
    var back = Namespace.init(a, "m", std.testing.io);
    back.dim = hdr.dim;
    back.index.dim = hdr.dim;
    try loadBinarySnapshot(&back, a, a, body, hdr);
    try std.testing.expectEqual(hnsw_mod.Metric.euclidean_squared, back.metric());
    try std.testing.expectEqual(@as(usize, 1), back.dead);
    try std.testing.expectEqual(@as(?u32, 2), back.id_map.get(7));
    const bcol = back.attrs.column("kind").?;
    try std.testing.expectEqualStrings("b \"q\"", back.attrs.strings.items[back.attrs.get(bcol, 2).str]);
    // the column survives; its only value belonged to the deleted row
    try std.testing.expect(back.attrs.get(back.attrs.column("n").?, 0) == .null);

    // a cosine namespace without attributes still writes the original format
    var plain = Namespace.init(a, "p", std.testing.io);
    _ = try replayWalBody(&plain, a, a, "{\"docs\":[[1,[1,0]]]}");
    try std.testing.expectEqual(@as(usize, 0), (try snapshotExt(a, &plain)).len);
}

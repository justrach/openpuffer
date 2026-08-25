//! turbopuffer-compatible HTTP API served on top of the local HNSW engine.
//!
//! Endpoints (mirror https://turbopuffer.com/docs):
//!   POST /v2/namespaces/{ns}        write   (upsert_columns or upsert_rows,
//!                                            distance_metric: cosine_distance)
//!   POST /v2/namespaces/{ns}/query  ANN query (rank_by ["vector","ANN",[...]],
//!                                            top_k/limit, consistency ignored)
//!   GET  /v2/namespaces/{ns}        stats
//!   DELETE /v2/namespaces/{ns}      drop namespace
//!
//! Responses use the same shapes: {"rows":[{"id":N,"$distance":D}],...}

const std = @import("std");
const builtin = @import("builtin");
const hnsw_mod = @import("hnsw.zig");
const s3_mod = @import("s3.zig");
const persist_mod = @import("persist.zig");
const iouring = @import("iouring_sock.zig");

const Hnsw = hnsw_mod.Hnsw(void);

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
    /// external document id -> local hnsw id (upserts)
    id_map: std.AutoHashMapUnmanaged(u64, u32) = .empty,

    fn init(alloc: std.mem.Allocator, name: []const u8, io: std.Io) Namespace {
        return .{ .name = name, .index = Hnsw.init(alloc, 0, .{}), .io = io };
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
        try w.writeAll("{\"docs\":[");
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
            snapshotNamespace(self.store, ns) catch {};
            self.store.deleteWalUpTo(ns.name, seq);
        }
    }
};

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
    persist: ?*PersistWorker = null,

    fn getOrCreate(self: *Registry, name: []const u8) !*Namespace {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
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
};

/// Auto-compaction threshold: snapshot after this many WAL segments.
const wal_compact_threshold: u64 = 8;

pub fn serve(alloc: std.mem.Allocator, io: std.Io, o: Options, out: *std.Io.Writer) !void {
    var registry = Registry{ .alloc = alloc, .io = io };

    if (o.s3_cfg) |cfg| {
        const t_all = Sw.start(io);
        const s3_client = try alloc.create(s3_mod.Client);
        s3_client.* = s3_mod.Client.init(alloc, io, cfg);
        const store = try alloc.create(persist_mod.Store);
        store.* = persist_mod.Store.init(alloc, s3_client);
        registry.store = store;
        const recovered = recoverAll(alloc, io, &registry, store, out) catch |e| blk: {
            try out.print("recovery failed ({any}); serving empty (store unreachable?)\n", .{e});
            try out.flush();
            break :blk @as(usize, 0);
        };
        if (recovered > 0) {
            try out.print("recovery total: {d:.1}ms\n", .{@as(f64, @floatFromInt(t_all.readNs(io))) / 1e6});
            try out.flush();
        }
        const worker = try alloc.create(PersistWorker);
        worker.* = .{ .alloc = alloc, .store = store, .io = io };
        try worker.start();
        registry.persist = worker;
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
};

fn startWorkerPool(alloc: std.mem.Allocator, io: std.Io, registry: *Registry, o: Options) !*FdQueue {
    const queue = try alloc.create(FdQueue);
    queue.* = .{ .io = io };
    const ctx = try alloc.create(PoolCtx);
    ctx.* = .{ .alloc = alloc, .io = io, .registry = registry, .o = o, .queue = queue };
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
        handleIoUringConn(&xfer, fd, ctx.alloc, ctx.io, ctx.registry, ctx.o, &store, &arena_state) catch {};
        _ = std.os.linux.close(fd);
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
    const queue = try startWorkerPool(alloc, io, registry, o);
    try out.print("linux serve: io_uring accept + {d} keep-alive workers, ef={d} rerank_mult={d}, TCP_NODELAY\n", .{ workerCount(o), o.ef, o.rerank_mult });
    try out.flush();
    var ring = try iouring.Ring.init();
    defer ring.deinit();
    var store: std.ArrayList(u8) = .empty;
    defer store.deinit(alloc);
    try store.resize(alloc, 1 << 16);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    var xfer = ConnXfer{};
    while (true) {
        const fd = ring.accept(listen_fd) catch |e| {
            try out.print("io_uring accept error: {any}\n", .{e});
            try out.flush();
            continue;
        };
        iouring.enableTcpNoDelay(fd);
        // If more clients are already in the listen backlog, hand everyone
        // (including this fd) to the pool so ANN work runs on all cores.
        // A lone connection is handled inline to avoid a futex wake (~0.3ms).
        drainListenBacklog(listen_fd, queue);
        if (queue.isEmpty()) {
            handleIoUringConn(&xfer, fd, alloc, io, registry, o, &store, &arena_state) catch {};
            _ = std.os.linux.close(fd);
        } else {
            queue.push(fd);
        }
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

fn handleIoUringConn(
    xfer: *ConnXfer,
    fd: std.posix.fd_t,
    alloc: std.mem.Allocator,
    io: std.Io,
    registry: *Registry,
    o: Options,
    store: *std.ArrayList(u8),
    arena_state: *std.heap.ArenaAllocator,
) !void {
    var n: usize = 0;
    while (true) {
        while (iouring.completeHttpRequest(store.items[0..n]) == 0) {
            if (n == store.items.len) {
                const cap = store.items.len;
                if (cap >= 8 << 20) return error.RequestTooLarge;
                try store.resize(alloc, cap * 2);
            }
            const got = try xfer.recv(fd, store.items[n..]);
            if (got == 0) return;
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
        const status_n: u16 = @intFromEnum(res.raw_status);
        const phrase = res.raw_status.phrase() orelse "OK";
        const wire = try iouring.formatResponse(arena, status_n, phrase, res.raw_body, keepalive);
        try xfer.sendAll(fd, wire);

        if (!keepalive) return;
        if (used < n) {
            const rest = n - used;
            std.mem.copyForwards(u8, store.items[0..rest], store.items[used..n]);
            n = rest;
        } else {
            n = 0;
        }
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

const RecoverJob = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    cfg: s3_mod.Config,
    registry: *Registry,
    keys: []const []const u8,
    plan: NsPlan,
    out: *std.Io.Writer,
    out_mu: *std.Io.Mutex,
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
) !usize {
    const keys = try store.client.listKeys("openpuffer/");
    defer {
        for (keys) |k| alloc.free(k);
        alloc.free(keys);
    }

    var plans: std.ArrayList(NsPlan) = .empty;
    defer plans.deinit(alloc);
    for (keys) |k| {
        if (!std.mem.startsWith(u8, k, "openpuffer/")) continue;
        const tail = k["openpuffer/".len..];
        const slash = std.mem.indexOfScalar(u8, tail, '/') orelse continue;
        const name = tail[0..slash];
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

    var t_all = Sw.start(io);
    var out_mu: std.Io.Mutex = .init;
    const jobs = try alloc.alloc(RecoverJob, plans.items.len);
    defer alloc.free(jobs);
    var threads: std.ArrayList(std.Thread) = .empty;
    defer threads.deinit(alloc);
    for (plans.items, 0..) |plan, i| {
        jobs[i] = .{
            .alloc = alloc,
            .io = io,
            .cfg = store.client.cfg,
            .registry = registry,
            .keys = keys,
            .plan = plan,
            .out = out,
            .out_mu = &out_mu,
        };
        const t = std.Thread.spawn(.{}, recoverOneNs, .{&jobs[i]}) catch {
            recoverOneNs(&jobs[i]);
            continue;
        };
        try threads.append(alloc, t);
    }
    for (threads.items) |t| t.join();

    if (plans.items.len > 0) {
        try out.print("recovered {d} namespace(s) from object storage in {d:.1}ms\n", .{
            plans.items.len, @as(f64, @floatFromInt(t_all.readNs(io))) / 1e6,
        });
    }
    return plans.items.len;
}

fn recoverOneNs(job: *RecoverJob) void {
    recoverOneNsInner(job) catch |e| {
        job.out_mu.lockUncancelable(job.io);
        defer job.out_mu.unlock(job.io);
        job.out.print("  recover '{s}' failed: {any}\n", .{ job.plan.name, e }) catch {};
        job.out.flush() catch {};
    };
}

fn recoverOneNsInner(job: *RecoverJob) !void {
    const alloc = job.alloc;
    const io = job.io;
    const name = job.plan.name;
    var client = s3_mod.Client.init(alloc, io, job.cfg);
    defer client.deinit();
    var store = persist_mod.Store.init(alloc, &client);
    const ns = try job.registry.getOrCreate(name);

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
                try ns.doc_ids.ensureTotalCapacity(job.registry.alloc, hdr.n);
                try ns.id_map.ensureTotalCapacity(job.registry.alloc, @intCast(hdr.n));
                var di: usize = 0;
                while (di < hdr.n) : (di += 1) {
                    const id = std.mem.readInt(u64, snap.bytes[hdr.doc_ids_off + di * 8 ..][0..8], .little);
                    ns.doc_ids.appendAssumeCapacity(id);
                    ns.id_map.putAssumeCapacity(id, @intCast(di));
                }
                try ns.index.load(snap.bytes[hdr.graph_off..]);
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
                snapshotNamespace(&store, ns) catch {};
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
        try snapshotNamespace(&store, ns);
        store.deleteWalListed(name, job.keys, ns.wal_seq);
        const t_compact = tc.readNs(io);
        job.out_mu.lockUncancelable(io);
        defer job.out_mu.unlock(io);
        try job.out.print("  replayed {d} WAL segments for '{s}' [walget={d:.1}ms walparse+build={d:.1}ms compact={d:.1}ms]\n", .{ replayed, name, @as(f64, @floatFromInt(t_walget)) / 1e6, @as(f64, @floatFromInt(t_walparse)) / 1e6, @as(f64, @floatFromInt(t_compact)) / 1e6 });
        try job.out.flush();
    }
}

/// Write a full binary snapshot of `ns` (graph + vectors) and reset pending-WAL.
fn snapshotNamespace(store: *persist_mod.Store, ns: *Namespace) !void {
    ns.lock.lockUncancelable(ns.io);
    const body = persist_mod.Store.buildBinarySnapshot(
        store.alloc,
        ns.wal_seq,
        ns.dim,
        ns.doc_ids.items,
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

const Responder = struct {
    req: ?*std.http.Server.Request = null,
    raw_alloc: ?std.mem.Allocator = null,
    raw_status: std.http.Status = .ok,
    raw_body: []const u8 = "",
};

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
        const ns_name = rest[0 .. rest.len - "/snapshot".len];
        const ns = registry.get(ns_name) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        if (registry.store) |store| {
            var t = Sw.start(io);
            if (registry.persist) |pw| pw.s3_mu.lockUncancelable(pw.io);
            defer if (registry.persist) |pw| pw.s3_mu.unlock(pw.io);
            try snapshotNamespace(store, ns);
            store.deleteWalUpTo(ns.name, ns.wal_seq);
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"snapshot_ms\":{d:.2}}}", .{@as(f64, @floatFromInt(t.readNs(io))) / 1e6});
            return respondJson(res, .ok, msg);
        }
        return respondJson(res, .bad_request, "{\"error\":\"persistence not configured\"}");
    }

    if (std.mem.endsWith(u8, rest, "/query") and std.mem.eql(u8, method, "POST")) {
        const ns_name = rest[0 .. rest.len - "/query".len];
        const ns = registry.get(ns_name) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        return handleQuery(alloc, res, ns, o, body);
    }

    if (std.mem.eql(u8, method, "POST")) {
        const ns = try registry.getOrCreate(rest);
        return handleWrite(alloc, registry.alloc, res, ns, body, registry);
    }

    if (std.mem.eql(u8, method, "GET")) {
        const ns = registry.get(rest) orelse
            return respondJson(res, .not_found, "{\"error\":\"namespace not found\"}");
        ns.lock.lockSharedUncancelable(ns.io);
        defer ns.lock.unlockShared(ns.io);
        var buf: [256]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{{\"namespace\":\"{s}\",\"dim\":{d},\"count\":{d}}}", .{ ns.name, ns.dim, ns.index.len() });
        return respondJson(res, .ok, s);
    }

    if (std.mem.eql(u8, method, "DELETE")) {
        registry.mutex.lockUncancelable(registry.io);
        const removed = registry.namespaces.fetchRemove(rest);
        registry.mutex.unlock(registry.io);
        if (removed) |kv| {
            kv.value.lock.lockUncancelable(kv.value.io);
            kv.value.index.deinit();
            kv.value.id_map.deinit(registry.alloc);
            kv.value.lock.unlock(kv.value.io);
            registry.alloc.destroy(kv.value);
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

fn replayWalBody(ns: *Namespace, persist: std.mem.Allocator, alloc: std.mem.Allocator, body: []const u8) !usize {
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    const docs = parsed.value.object.get("docs") orelse return error.BadWal;
    if (docs != .array) return error.BadWal;
    for (docs.array.items) |doc| {
        const pair = doc.array.items;
        if (ns.dim == 0) {
            ns.dim = pair[1].array.items.len;
            ns.index.dim = ns.dim;
        }
        const id: u64 = @intCast(pair[0].integer);
        const arr = pair[1].array;
        const v = try alloc.alloc(f32, arr.items.len);
        for (arr.items, 0..) |x, i| v[i] = switch (x) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => 0,
        };
        try upsertDoc(ns, persist, id, v);
    }
    return docs.array.items.len;
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

    if (root.get("distance_metric")) |dm| {
        if (dm != .string or !std.mem.eql(u8, dm.string, "cosine_distance")) {
            return respondJson(res, .bad_request, "{\"error\":\"only cosine_distance supported\"}");
        }
    }

    const IdVec = struct { id: u64, vec: []f32 };
    var batch: std.ArrayList(IdVec) = .empty;

    if (root.get("upsert_columns")) |cols_v| {
        if (cols_v != .object) return respondJson(res, .bad_request, "{\"error\":\"upsert_columns must be object\"}");
        const cols = cols_v.object;
        const ids_v = cols.get("id") orelse return respondJson(res, .bad_request, "{\"error\":\"id column required\"}");
        const vecs_v = cols.get("vector") orelse return respondJson(res, .bad_request, "{\"error\":\"vector column required\"}");
        if (ids_v != .array or vecs_v != .array) return respondJson(res, .bad_request, "{\"error\":\"columns must be arrays\"}");
        for (ids_v.array.items, vecs_v.array.items) |id_v, vec_v| {
            if (vec_v != .array) return respondJson(res, .bad_request, "{\"error\":\"vector rows must be arrays\"}");
            const id: u64 = switch (id_v) {
                .integer => |n| @intCast(n),
                else => return respondJson(res, .bad_request, "{\"error\":\"ids must be integers\"}"),
            };
            try batch.append(alloc, .{ .id = id, .vec = try vecFromJson(vec_v.array, alloc) });
        }
    } else if (root.get("upsert_rows")) |rows_v| {
        if (rows_v != .array) return respondJson(res, .bad_request, "{\"error\":\"upsert_rows must be array\"}");
        for (rows_v.array.items) |row| {
            if (row != .object) return respondJson(res, .bad_request, "{\"error\":\"rows must be objects\"}");
            const robj = row.object;
            const id_v = robj.get("id") orelse return respondJson(res, .bad_request, "{\"error\":\"row missing id\"}");
            const vec_v = robj.get("vector") orelse return respondJson(res, .bad_request, "{\"error\":\"row missing vector\"}");
            if (vec_v != .array) return respondJson(res, .bad_request, "{\"error\":\"vector must be array\"}");
            const id: u64 = switch (id_v) {
                .integer => |n| @intCast(n),
                else => return respondJson(res, .bad_request, "{\"error\":\"ids must be integers\"}"),
            };
            try batch.append(alloc, .{ .id = id, .vec = try vecFromJson(vec_v.array, alloc) });
        }
    } else {
        return respondJson(res, .bad_request, "{\"error\":\"missing upsert_columns/upsert_rows\"}");
    }

    if (batch.items.len == 0) return respondJson(res, .ok, "{\"ok\":true}");

    {
        ns.lock.lockUncancelable(ns.io);
        defer ns.lock.unlock(ns.io);
        if (ns.dim == 0) {
            ns.dim = batch.items[0].vec.len;
            ns.index.dim = ns.dim;
        }
        for (batch.items) |iv| {
            if (iv.vec.len != ns.dim) return respondJson(res, .bad_request, "{\"error\":\"dimension mismatch\"}");
            try upsertDoc(ns, persist, iv.id, iv.vec);
        }
    }

    // durable WAL: group-commit via persist worker when configured
    if (registry.persist != null or registry.store != null) {
        var wb: std.Io.Writer.Allocating = .init(persist);
        const ww = &wb.writer;
        try ww.writeAll("{\"docs\":[");
        for (batch.items, 0..) |iv, i| {
            if (i > 0) try ww.writeAll(",");
            try ww.print("[{d},[", .{iv.id});
            for (iv.vec, 0..) |x, j| {
                if (j > 0) try ww.writeAll(",");
                try ww.print("{d}", .{x});
            }
            try ww.writeAll("]]");
        }
        try ww.writeAll("]}");
        const wal_body = try wb.toOwnedSlice();
        if (registry.persist) |pw| {
            var job = WalJob{ .ns = ns, .body = wal_body };
            pw.commit(&job) catch |e| {
                persist.free(job.body);
                return e;
            };
        } else if (registry.store) |store| {
            defer persist.free(wal_body);
            ns.lock.lockUncancelable(ns.io);
            ns.wal_seq += 1;
            const seq = ns.wal_seq;
            ns.lock.unlock(ns.io);
            try store.appendWal(ns.name, seq, wal_body);
        }
    }

    return respondJson(res, .ok, "{\"ok\":true}");
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

    ns.lock.lockSharedUncancelable(ns.io);
    defer ns.lock.unlockShared(ns.io);
    if (ns.index.entry_point == null) return respondJson(res, .ok, "{\"rows\":[]}");
    if (query_vec.len != ns.dim) {
        var dbuf: [128]u8 = undefined;
        const dmsg = try std.fmt.bufPrint(&dbuf, "{{\"error\":\"dimension mismatch\",\"expected\":{d},\"got\":{d}}}", .{ ns.dim, query_vec.len });
        return respondJson(res, .bad_request, dmsg);
    }

    const ef = spec.ef orelse o.ef;
    const rerank_mult = spec.rerank_mult orelse o.rerank_mult;
    const results = try ns.index.searchAdvanced(query_vec, top_k, ef, rerank_mult, alloc);

    var out: std.Io.Writer.Allocating = .init(alloc);
    const w = &out.writer;
    try w.writeAll("{\"rows\":[");
    for (results, 0..) |r, i| {
        if (i > 0) try w.writeAll(",");
        const doc_id: u64 = if (r.id < ns.doc_ids.items.len) ns.doc_ids.items[r.id] else r.id;
        try w.print("{{\"id\":{d},\"$distance\":{d}}}", .{ doc_id, r.distance });
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

test "parseAnnQuery reads optional rerank_mult" {
    const body = "{\"rank_by\":[\"vector\",\"ANN\",[1]],\"top_k\":3,\"ef\":64,\"rerank_mult\":8}";
    const spec = try parseAnnQuery(body, std.testing.allocator);
    defer std.testing.allocator.free(spec.vec);
    try std.testing.expectEqual(@as(?u32, 64), spec.ef);
    try std.testing.expectEqual(@as(?usize, 8), spec.rerank_mult);
}

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
const hnsw_mod = @import("hnsw.zig");
const s3_mod = @import("s3.zig");
const persist_mod = @import("persist.zig");

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
    ef: u32 = 256,
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
    try out.print("openpuffer serving turbopuffer-compatible API on http://127.0.0.1:{d}\n", .{o.port});
    if (registry.store != null) try out.print("persistence: s3 bucket '{s}' (group-commit WAL)\n", .{o.s3_cfg.?.bucket});
    try out.flush();

    var conn_sem = std.Io.Semaphore{ .permits = 32 };
    while (true) {
        const stream = listener.accept(io) catch |e| {
            try out.print("accept error: {any}\n", .{e});
            continue;
        };
        conn_sem.waitUncancelable(io);
        const t = std.Thread.spawn(.{}, handleConnectionThread, .{ alloc, io, stream, &registry, o, &conn_sem }) catch {
            conn_sem.post(io);
            handleConnection(alloc, io, stream, &registry, o) catch {};
            continue;
        };
        t.detach();
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
            respondJson(&req, .internal_server_error, "{\"error\":\"internal\"}") catch {};
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

fn respondJson(req: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    try req.respond(body, .{
        .status = status,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/json" },
        },
    });
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

    // route: /v2/namespaces[/{ns}[/query]]
    const prefix = "/v2/namespaces/";
    if (!std.mem.startsWith(u8, path, prefix)) {
        var ebuf: [512]u8 = undefined;
        const emsg = try std.fmt.bufPrint(&ebuf, "{{\"error\":\"unknown route\",\"target\":\"{s}\"}}", .{path});
        return respondJson(req, .not_found, emsg);
    }
    const rest = path[prefix.len..];

    // Body handling: std 0.17-dev's discardBody asserts when responding to a
    // POST whose head has neither transfer_encoding nor content_length (curl
    // sends exactly that for `-X POST` with no body). Mark those consumed up
    // front; drain real bodies through a properly-sized reader buffer.
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
    if (std.mem.endsWith(u8, rest, "/snapshot") and method == .POST) {
        const ns_name = rest[0 .. rest.len - "/snapshot".len];
        const ns = registry.get(ns_name) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        if (registry.store) |store| {
            var t = Sw.start(io);
            if (registry.persist) |pw| pw.s3_mu.lockUncancelable(pw.io);
            defer if (registry.persist) |pw| pw.s3_mu.unlock(pw.io);
            try snapshotNamespace(store, ns);
            store.deleteWalUpTo(ns.name, ns.wal_seq);
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"snapshot_ms\":{d:.2}}}", .{@as(f64, @floatFromInt(t.readNs(io))) / 1e6});
            return respondJson(req, .ok, msg);
        }
        return respondJson(req, .bad_request, "{\"error\":\"persistence not configured\"}");
    }

    if (std.mem.endsWith(u8, rest, "/query") and method == .POST) {
        const ns_name = rest[0 .. rest.len - "/query".len];
        const ns = registry.get(ns_name) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        return handleQuery(alloc, req, ns, o, body);
    }

    if (method == .POST) {
        const ns = try registry.getOrCreate(rest);
        return handleWrite(alloc, registry.alloc, req, ns, body, registry);
    }

    if (method == .GET) {
        const ns = registry.get(rest) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        ns.lock.lockSharedUncancelable(ns.io);
        defer ns.lock.unlockShared(ns.io);
        var buf: [256]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{{\"namespace\":\"{s}\",\"dim\":{d},\"count\":{d}}}", .{ ns.name, ns.dim, ns.index.len() });
        return respondJson(req, .ok, s);
    }

    if (method == .DELETE) {
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
        return respondJson(req, .ok, "{\"ok\":true}");
    }

    return respondJson(req, .method_not_allowed, "{\"error\":\"method not allowed\"}");
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
    req: *std.http.Server.Request,
    ns: *Namespace,
    body: []const u8,
    registry: *Registry,
) !void {
    // NOTE: body already consumed by caller
    const alloc = arena;
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch
        return respondJson(req, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();
    if (parsed.value != .object) return respondJson(req, .bad_request, "{\"error\":\"expected object\"}");
    const root = parsed.value.object;

    if (root.get("distance_metric")) |dm| {
        if (dm != .string or !std.mem.eql(u8, dm.string, "cosine_distance")) {
            return respondJson(req, .bad_request, "{\"error\":\"only cosine_distance supported\"}");
        }
    }

    const IdVec = struct { id: u64, vec: []f32 };
    var batch: std.ArrayList(IdVec) = .empty;

    if (root.get("upsert_columns")) |cols_v| {
        if (cols_v != .object) return respondJson(req, .bad_request, "{\"error\":\"upsert_columns must be object\"}");
        const cols = cols_v.object;
        const ids_v = cols.get("id") orelse return respondJson(req, .bad_request, "{\"error\":\"id column required\"}");
        const vecs_v = cols.get("vector") orelse return respondJson(req, .bad_request, "{\"error\":\"vector column required\"}");
        if (ids_v != .array or vecs_v != .array) return respondJson(req, .bad_request, "{\"error\":\"columns must be arrays\"}");
        for (ids_v.array.items, vecs_v.array.items) |id_v, vec_v| {
            if (vec_v != .array) return respondJson(req, .bad_request, "{\"error\":\"vector rows must be arrays\"}");
            const id: u64 = switch (id_v) {
                .integer => |n| @intCast(n),
                else => return respondJson(req, .bad_request, "{\"error\":\"ids must be integers\"}"),
            };
            try batch.append(alloc, .{ .id = id, .vec = try vecFromJson(vec_v.array, alloc) });
        }
    } else if (root.get("upsert_rows")) |rows_v| {
        if (rows_v != .array) return respondJson(req, .bad_request, "{\"error\":\"upsert_rows must be array\"}");
        for (rows_v.array.items) |row| {
            if (row != .object) return respondJson(req, .bad_request, "{\"error\":\"rows must be objects\"}");
            const robj = row.object;
            const id_v = robj.get("id") orelse return respondJson(req, .bad_request, "{\"error\":\"row missing id\"}");
            const vec_v = robj.get("vector") orelse return respondJson(req, .bad_request, "{\"error\":\"row missing vector\"}");
            if (vec_v != .array) return respondJson(req, .bad_request, "{\"error\":\"vector must be array\"}");
            const id: u64 = switch (id_v) {
                .integer => |n| @intCast(n),
                else => return respondJson(req, .bad_request, "{\"error\":\"ids must be integers\"}"),
            };
            try batch.append(alloc, .{ .id = id, .vec = try vecFromJson(vec_v.array, alloc) });
        }
    } else {
        return respondJson(req, .bad_request, "{\"error\":\"missing upsert_columns/upsert_rows\"}");
    }

    if (batch.items.len == 0) return respondJson(req, .ok, "{\"ok\":true}");

    {
        ns.lock.lockUncancelable(ns.io);
        defer ns.lock.unlock(ns.io);
        if (ns.dim == 0) {
            ns.dim = batch.items[0].vec.len;
            ns.index.dim = ns.dim;
        }
        for (batch.items) |iv| {
            if (iv.vec.len != ns.dim) return respondJson(req, .bad_request, "{\"error\":\"dimension mismatch\"}");
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

    return respondJson(req, .ok, "{\"ok\":true}");
}

fn handleQuery(
    alloc: std.mem.Allocator,
    req: *std.http.Server.Request,
    ns: *Namespace,
    o: Options,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch
        return respondJson(req, .bad_request, "{\"error\":\"invalid json\"}");
    defer parsed.deinit();
    if (parsed.value != .object) return respondJson(req, .bad_request, "{\"error\":\"expected object\"}");
    const qobj = parsed.value.object;

    const rb = qobj.get("rank_by") orelse return respondJson(req, .bad_request, "{\"error\":\"rank_by required\"}");
    if (rb != .array or rb.array.items.len < 3) return respondJson(req, .bad_request, "{\"error\":\"rank_by must be [field,'ANN',vector]\"}");
    const field = rb.array.items[0];
    const fnv = rb.array.items[1];
    if (field != .string or !std.mem.eql(u8, field.string, "vector")) return respondJson(req, .bad_request, "{\"error\":\"only vector ANN ranking supported\"}");
    if (fnv != .string or !std.mem.eql(u8, fnv.string, "ANN")) return respondJson(req, .bad_request, "{\"error\":\"only ANN function supported\"}");
    const vec_v = rb.array.items[2];
    if (vec_v != .array) return respondJson(req, .bad_request, "{\"error\":\"ANN vector must be array\"}");
    const query_vec = try vecFromJson(vec_v.array, alloc);

    var top_k: usize = 10;
    if (qobj.get("top_k")) |t| {
        if (t == .integer) top_k = @intCast(@max(1, t.integer));
    } else if (qobj.get("limit")) |l| {
        if (l == .integer) top_k = @intCast(@max(1, l.integer));
    }

    ns.lock.lockSharedUncancelable(ns.io);
    defer ns.lock.unlockShared(ns.io);
    if (ns.index.entry_point == null) return respondJson(req, .ok, "{\"rows\":[]}");
    if (query_vec.len != ns.dim) {
        var dbuf: [128]u8 = undefined;
        const dmsg = try std.fmt.bufPrint(&dbuf, "{{\"error\":\"dimension mismatch\",\"expected\":{d},\"got\":{d}}}", .{ ns.dim, query_vec.len });
        return respondJson(req, .bad_request, dmsg);
    }

    const results = try ns.index.search(query_vec, top_k, o.ef, alloc);

    var out: std.Io.Writer.Allocating = .init(alloc);
    const w = &out.writer;
    try w.writeAll("{\"rows\":[");
    for (results, 0..) |r, i| {
        if (i > 0) try w.writeAll(",");
        const doc_id: u64 = if (r.id < ns.doc_ids.items.len) ns.doc_ids.items[r.id] else r.id;
        try w.print("{{\"id\":{d},\"$distance\":{d}}}", .{ doc_id, r.distance });
    }
    try w.writeAll("],\"usage\":{}}");
    return respondJson(req, .ok, out.written());
}

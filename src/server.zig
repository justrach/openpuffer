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

    fn init(alloc: std.mem.Allocator, name: []const u8) Namespace {
        return .{ .name = name, .index = Hnsw.init(alloc, 0, .{}) };
    }
};

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
    namespaces: std.StringHashMapUnmanaged(*Namespace) = .empty,
    /// null when running without object-storage persistence
    store: ?*persist_mod.Store = null,

    fn getOrCreate(self: *Registry, name: []const u8) !*Namespace {
        if (self.namespaces.get(name)) |ns| return ns;
        const ns = try self.alloc.create(Namespace);
        ns.* = Namespace.init(self.alloc, try self.alloc.dupe(u8, name));
        try self.namespaces.put(self.alloc, ns.name, ns);
        return ns;
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
    var registry = Registry{ .alloc = alloc };

    if (o.s3_cfg) |cfg| {
        const t_all = Sw.start(io);
        const s3_client = try alloc.create(s3_mod.Client);
        s3_client.* = s3_mod.Client.init(alloc, io, cfg);
        const store = try alloc.create(persist_mod.Store);
        store.* = persist_mod.Store.init(alloc, s3_client);
        registry.store = store;
        const recovered = try recoverAll(alloc, io, &registry, store, out);
        if (recovered > 0) {
            try out.print("recovery total: {d:.1}ms\n", .{@as(f64, @floatFromInt(t_all.readNs(io))) / 1e6});
            try out.flush();
        }
    }

    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", o.port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    try out.print("openpuffer serving turbopuffer-compatible API on http://127.0.0.1:{d}\n", .{o.port});
    if (registry.store != null) try out.print("persistence: s3 bucket '{s}'\n", .{o.s3_cfg.?.bucket});
    try out.flush();

    while (true) {
        const stream = listener.accept(io) catch |e| {
            try out.print("accept error: {any}\n", .{e});
            continue;
        };
        // connection-level failures shouldn't kill the server
        handleConnection(alloc, io, stream, &registry, o) catch {};
    }
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

/// Load every namespace persisted under the store prefix: snapshot first,
/// then replay WAL segments newer than it; compact WALs into a fresh snapshot.
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

    // collect distinct namespace names
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(alloc);
    for (keys) |k| {
        if (!std.mem.startsWith(u8, k, "openpuffer/")) continue;
        const tail = k["openpuffer/".len..];
        const slash = std.mem.indexOfScalar(u8, tail, '/') orelse continue;
        const name = tail[0..slash];
        var dup = false;
        for (names.items) |n| {
            if (std.mem.eql(u8, n, name)) {
                dup = true;
                break;
            }
        }
        if (!dup) try names.append(alloc, name);
    }

    var t_all = Sw.start(io);
    for (names.items) |name| {
        const ns = try registry.getOrCreate(name);
        var last_seq: u64 = 0;

        if (try store.getSnapshot(name)) |snap| {
            ns.dim = snap.dim;
            ns.index.dim = snap.dim;
            last_seq = snap.seq;

            // parse docs [[id,[...]],...]
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, snap.docs_json, .{});
            const docs = parsed.value.object.get("docs").?.array.items;
            for (docs) |doc| {
                const pair = doc.array.items;
                const id: u64 = @intCast(pair[0].integer);
                const arr = pair[1].array;
                const v = try alloc.alloc(f32, arr.items.len);
                for (arr.items, 0..) |x, i| v[i] = @floatCast(x.float);
                _ = try ns.index.insert(v);
                try ns.doc_ids.append(registry.alloc, id);
            }
            parsed.deinit();
            try out.print("  recovered '{s}': snapshot seq={d} ({d} docs)\n", .{ name, snap.seq, ns.doc_ids.items.len });
        }

        // replay newer WAL segments
        const wal_seqs = try store.listWal(name, last_seq);
        defer alloc.free(wal_seqs);
        var replayed: usize = 0;
        for (wal_seqs) |seq| {
            const body = try store.getWalBody(name, seq);
            defer alloc.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
            defer parsed.deinit();
            const docs = parsed.value.object.get("docs").?.array.items;
            for (docs) |doc| {
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
                _ = try ns.index.insert(v);
                try ns.doc_ids.append(registry.alloc, id);
            }
            replayed += 1;
            ns.wal_seq = seq;
            ns.pending_wal += 1;
        }

        // compact: fold everything into a fresh snapshot and drop consumed WALs
        if (replayed > 0) {
            try snapshotNamespace(store, ns);
            for (wal_seqs) |seq| store.deleteWal(name, seq) catch {};
            try out.print("  replayed {d} WAL segments for '{s}'\n", .{ replayed, name });
        }
    }
    if (names.items.len > 0) {
        try out.print("recovered {d} namespace(s) from object storage in {d:.1}ms\n", .{
            names.items.len, @as(f64, @floatFromInt(t_all.readNs(io))) / 1e6,
        });
    }
    return names.items.len;
}

/// Write a full snapshot of `ns` and reset its pending-WAL counter.
fn snapshotNamespace(store: *persist_mod.Store, ns: *Namespace) !void {
    const body = try persist_mod.Store.buildSnapshot(
        store.alloc,
        ns.wal_seq,
        ns.dim,
        ns.doc_ids.items,
        ns.index.vectors.items,
    );
    defer store.alloc.free(body);
    const key = try std.fmt.allocPrint(store.alloc, "openpuffer/{s}/snapshot.json", .{ns.name});
    defer store.alloc.free(key);
    try store.client.putObject(key, body);
    ns.pending_wal = 0;
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
        const ns = registry.namespaces.get(ns_name) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        if (registry.store) |store| {
            var t = Sw.start(io);
            try snapshotNamespace(store, ns);
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"snapshot_ms\":{d:.2}}}", .{@as(f64, @floatFromInt(t.readNs(io))) / 1e6});
            return respondJson(req, .ok, msg);
        }
        return respondJson(req, .bad_request, "{\"error\":\"persistence not configured\"}");
    }

    if (std.mem.endsWith(u8, rest, "/query") and method == .POST) {
        const ns_name = rest[0 .. rest.len - "/query".len];
        const ns = registry.namespaces.get(ns_name) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        return handleQuery(alloc, req, ns, o, body);
    }

    if (method == .POST) {
        const ns = try registry.getOrCreate(rest);
        return handleWrite(alloc, registry.alloc, req, ns, body, registry.store);
    }

    if (method == .GET) {
        const ns = registry.namespaces.get(rest) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        var buf: [256]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{{\"namespace\":\"{s}\",\"dim\":{d},\"count\":{d}}}", .{ ns.name, ns.dim, ns.index.len() });
        return respondJson(req, .ok, s);
    }

    if (method == .DELETE) {
        if (registry.namespaces.fetchRemove(rest)) |kv| {
            kv.value.index.deinit();
            registry.alloc.destroy(kv.value);
        }
        return respondJson(req, .ok, "{\"ok\":true}");
    }

    return respondJson(req, .method_not_allowed, "{\"error\":\"method not allowed\"}");
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
    registry_store: ?*persist_mod.Store,
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

    if (ns.dim == 0) {
        ns.dim = batch.items[0].vec.len;
        ns.index.dim = ns.dim;
    }
    for (batch.items) |iv| {
        if (iv.vec.len != ns.dim) return respondJson(req, .bad_request, "{\"error\":\"dimension mismatch\"}");
        const local = try ns.index.insert(iv.vec);
        try ns.doc_ids.append(persist, iv.id);
        if (local + 1 != ns.doc_ids.items.len) return respondJson(req, .internal_server_error, "{\"error\":\"id bookkeeping desync\"}");
    }

    // durable WAL append before acknowledging the write
    if (registry_store) |store| {
        ns.wal_seq += 1;
        ns.pending_wal += 1;
        var wb: std.Io.Writer.Allocating = .init(arena);
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
        try store.appendWal(ns.name, ns.wal_seq, wb.written());
        // auto-compaction: fold WAL segments into a snapshot periodically
        if (ns.pending_wal >= wal_compact_threshold) {
            try snapshotNamespace(store, ns);
            const consumed = store.listWal(ns.name, ns.wal_seq) catch &[_]u64{};
            for (consumed) |seq| store.deleteWal(ns.name, seq) catch {};
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

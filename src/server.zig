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

const Hnsw = hnsw_mod.Hnsw(void);

const Namespace = struct {
    name: []const u8,
    dim: usize = 0,
    index: Hnsw,
    /// local hnsw id -> external document id
    doc_ids: std.ArrayList(u64) = .empty,

    fn init(alloc: std.mem.Allocator, name: []const u8) Namespace {
        return .{ .name = name, .index = Hnsw.init(alloc, 0, .{}) };
    }
};

const Registry = struct {
    alloc: std.mem.Allocator,
    namespaces: std.StringHashMapUnmanaged(*Namespace) = .empty,

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
};

pub fn serve(alloc: std.mem.Allocator, io: std.Io, o: Options, out: *std.Io.Writer) !void {
    var registry = Registry{ .alloc = alloc };

    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", o.port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    try out.print("zvec serving turbopuffer-compatible API on http://127.0.0.1:{d}\n", .{o.port});
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

        handleRequest(arena, &req, registry, o) catch |e| {
            respondJson(&req, .internal_server_error, "{\"error\":\"internal\"}") catch {};
            return e;
        };
    }
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
    req: *std.http.Server.Request,
    registry: *Registry,
    o: Options,
) !void {
    // head.target points into the connection's read buffer; the body read
    // below overwrites it, so copy it out first.
    const path = try alloc.dupe(u8, req.head.target);
    const method = req.head.method;

    // read full body
    var body_reader = req.readerExpectNone(&.{});
    var body_writer: std.Io.Writer.Allocating = .init(alloc);
    _ = try body_reader.streamRemaining(&body_writer.writer);
    const body = body_writer.written();

    // route: /v2/namespaces[/{ns}[/query]]
    const prefix = "/v2/namespaces/";
    if (!std.mem.startsWith(u8, path, prefix)) {
        var ebuf: [512]u8 = undefined;
        const emsg = try std.fmt.bufPrint(&ebuf, "{{\"error\":\"unknown route\",\"target\":\"{s}\"}}", .{path});
        return respondJson(req, .not_found, emsg);
    }
    const rest = path[prefix.len..];

    if (std.mem.endsWith(u8, rest, "/query") and method == .POST) {
        const ns_name = rest[0 .. rest.len - "/query".len];
        const ns = registry.namespaces.get(ns_name) orelse
            return respondJson(req, .not_found, "{\"error\":\"namespace not found\"}");
        return handleQuery(alloc, req, ns, o, body);
    }

    if (method == .POST) {
        const ns = try registry.getOrCreate(rest);
        return handleWrite(alloc, registry.alloc, req, ns, body);
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
) !void {
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

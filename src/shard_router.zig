//! Zig/io_uring shard router in front of N independent `openpuffer serve` children.
//!
//! Same FNV-1a contract as `tools/shard_key.py`:
//!   utf-8(namespace) + NUL + utf-8(decimal_id)   then  key % N
//! Writes go to exactly one child. Queries scatter-gather and merge by `$distance`.
//! Children are reached over one keep-alive TCP_NODELAY socket each
//! (mutex-shared across router workers so idle keep-alives cannot pin
//! every child serve thread).

const std = @import("std");
const builtin = @import("builtin");
const iouring = @import("iouring_sock.zig");
const rss = @import("rss.zig");

pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x00000100000001b3;

pub const ShardBy = enum { doc, namespace };

pub const Options = struct {
    port: u16 = 8800,
    shards: usize = 1,
    shard_port_base: ?u16 = null,
    ef: u32 = 128,
    workers: ?usize = null,
    router_workers: ?usize = null,
    shard_by: ShardBy = .doc,
    binary: []const u8 = "./zig-out/bin/openpuffer",
    spawn_children: bool = true,
};

pub fn fnv1a64(data: []const u8) u64 {
    var h: u64 = FNV_OFFSET;
    for (data) |b| {
        h ^= b;
        h *%= FNV_PRIME;
    }
    return h;
}

pub fn isDigits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| {
        if (c < '0' or c > '9') return false;
    }
    return true;
}

pub fn canonicalId(alloc: std.mem.Allocator, value: std.json.Value) ![]u8 {
    switch (value) {
        .bool => return error.BadId,
        .integer => |n| {
            if (n < 0) return error.BadId;
            return std.fmt.allocPrint(alloc, "{d}", .{n});
        },
        .string => |s| {
            const t = std.mem.trim(u8, s, " \t\r\n");
            if (isDigits(t)) {
                const n = std.fmt.parseInt(u64, t, 10) catch return alloc.dupe(u8, t);
                return std.fmt.allocPrint(alloc, "{d}", .{n});
            }
            return alloc.dupe(u8, t);
        },
        else => return error.BadId,
    }
}

pub fn shardKeyBytes(alloc: std.mem.Allocator, namespace: []const u8, doc_id: std.json.Value, shard_by: ShardBy) ![]u8 {
    if (shard_by == .namespace) return alloc.dupe(u8, namespace);
    const id = try canonicalId(alloc, doc_id);
    defer alloc.free(id);
    const out = try alloc.alloc(u8, namespace.len + 1 + id.len);
    @memcpy(out[0..namespace.len], namespace);
    out[namespace.len] = 0;
    @memcpy(out[namespace.len + 1 ..], id);
    return out;
}

pub fn shardFor(alloc: std.mem.Allocator, namespace: []const u8, doc_id: std.json.Value, n_shards: usize, shard_by: ShardBy) !usize {
    if (n_shards == 0) return error.BadShardCount;
    const key = try shardKeyBytes(alloc, namespace, doc_id, shard_by);
    defer alloc.free(key);
    return @intCast(fnv1a64(key) % n_shards);
}

pub fn parseTopK(body: []const u8, alloc: std.mem.Allocator) usize {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return 10;
    defer parsed.deinit();
    if (parsed.value != .object) return 10;
    for ([_][]const u8{ "top_k", "limit" }) |key| {
        if (parsed.value.object.get(key)) |v| {
            switch (v) {
                .integer => |n| if (n > 0) return @intCast(n),
                else => {},
            }
        }
    }
    return 10;
}

const MergeRow = struct {
    id_int: ?i64,
    id_str: []const u8,
    distance: f64,
    raw: std.json.Value,
};

fn jsonAsF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 1e9,
        else => 1e9,
    };
}

fn idLess(a: MergeRow, b: MergeRow) bool {
    if (a.id_int != null and b.id_int != null) return a.id_int.? < b.id_int.?;
    if (a.id_int != null and b.id_int == null) return true;
    if (a.id_int == null and b.id_int != null) return false;
    return std.mem.lessThan(u8, a.id_str, b.id_str);
}

fn idEql(a: MergeRow, b: MergeRow) bool {
    if (a.id_int != null and b.id_int != null) return a.id_int.? == b.id_int.?;
    if (a.id_int == null and b.id_int == null) return std.mem.eql(u8, a.id_str, b.id_str);
    return false;
}

fn rowFromJson(v: std.json.Value) ?MergeRow {
    if (v != .object) return null;
    const id_v = v.object.get("id") orelse return null;
    var row = MergeRow{ .id_int = null, .id_str = "", .distance = 1e9, .raw = v };
    switch (id_v) {
        .integer => |n| {
            row.id_int = n;
            row.id_str = "";
        },
        .string => |s| {
            row.id_str = s;
            if (isDigits(s)) row.id_int = std.fmt.parseInt(i64, s, 10) catch null;
        },
        else => return null,
    }
    if (v.object.get("$distance")) |d| row.distance = jsonAsF64(d);
    return row;
}

pub fn mergeRows(alloc: std.mem.Allocator, shard_bodies: []const []const u8, top_k: usize) ![]u8 {
    var rows: std.ArrayList(MergeRow) = .empty;
    defer rows.deinit(alloc);
    var parsed_hold: std.ArrayList(std.json.Parsed(std.json.Value)) = .empty;
    defer {
        for (parsed_hold.items) |*p| p.deinit();
        parsed_hold.deinit(alloc);
    }

    for (shard_bodies) |body| {
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch continue;
        try parsed_hold.append(alloc, parsed);
        if (parsed.value != .object) continue;
        const rows_v = parsed.value.object.get("rows") orelse continue;
        if (rows_v != .array) continue;
        for (rows_v.array.items) |item| {
            if (rowFromJson(item)) |r| try rows.append(alloc, r);
        }
    }

    std.mem.sort(MergeRow, rows.items, {}, struct {
        fn lt(_: void, a: MergeRow, b: MergeRow) bool {
            if (a.distance < b.distance) return true;
            if (a.distance > b.distance) return false;
            return idLess(a, b);
        }
    }.lt);

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.writeAll("{\"rows\":[");
    var written: usize = 0;
    var i: usize = 0;
    while (i < rows.items.len and written < top_k) : (i += 1) {
        const r = rows.items[i];
        var dup = false;
        var j: usize = 0;
        while (j < i) : (j += 1) {
            if (idEql(rows.items[j], r)) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        if (written > 0) try out.writer.writeAll(",");
        try std.json.Stringify.value(r.raw, .{}, &out.writer);
        written += 1;
    }
    try out.writer.writeAll("],\"usage\":{}}");
    return out.toOwnedSlice();
}

const Slice = struct {
    shard: usize,
    body: []u8,
};

fn writeObjectExcept(w: *std.Io.Writer, root: std.json.ObjectMap, skip_a: []const u8, skip_b: []const u8) !bool {
    var first = true;
    var it = root.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, skip_a) or std.mem.eql(u8, kv.key_ptr.*, skip_b)) continue;
        if (!first) try w.writeAll(",");
        first = false;
        try std.json.Stringify.value(kv.key_ptr.*, .{}, w);
        try w.writeAll(":");
        try std.json.Stringify.value(kv.value_ptr.*, .{}, w);
    }
    return !first;
}

pub fn splitUpsert(alloc: std.mem.Allocator, namespace: []const u8, body: []const u8, n_shards: usize, shard_by: ShardBy) ![]Slice {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.BadJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadJson;
    const root = parsed.value.object;

    var buckets = try alloc.alloc(std.ArrayList(usize), n_shards);
    for (buckets) |*b| b.* = .empty;
    defer {
        for (buckets) |*b| b.deinit(alloc);
        alloc.free(buckets);
    }

    if (root.get("upsert_columns")) |cols_v| {
        if (cols_v != .object) return error.BadJson;
        const cols = cols_v.object;
        const ids_v = cols.get("id") orelse return error.BadJson;
        const vecs_v = cols.get("vector") orelse return error.BadJson;
        if (ids_v != .array or vecs_v != .array) return error.BadJson;
        if (ids_v.array.items.len != vecs_v.array.items.len) return error.BadJson;
        for (ids_v.array.items, 0..) |id_v, i| {
            const sid = try shardFor(alloc, namespace, id_v, n_shards, shard_by);
            try buckets[sid].append(alloc, i);
        }
        var out: std.ArrayList(Slice) = .empty;
        for (buckets, 0..) |b, sid| {
            if (b.items.len == 0) continue;
            var w: std.Io.Writer.Allocating = .init(alloc);
            try w.writer.writeAll("{");
            const had = try writeObjectExcept(&w.writer, root, "upsert_columns", "upsert_rows");
            if (had) try w.writer.writeAll(",");
            try w.writer.writeAll("\"upsert_columns\":{");
            var cfirst = true;
            var cit = cols.iterator();
            while (cit.next()) |ckv| {
                if (!cfirst) try w.writer.writeAll(",");
                cfirst = false;
                try std.json.Stringify.value(ckv.key_ptr.*, .{}, &w.writer);
                try w.writer.writeAll(":");
                if (ckv.value_ptr.* == .array) {
                    try w.writer.writeAll("[");
                    for (b.items, 0..) |ri, k| {
                        if (k > 0) try w.writer.writeAll(",");
                        if (ri < ckv.value_ptr.*.array.items.len) {
                            try std.json.Stringify.value(ckv.value_ptr.*.array.items[ri], .{}, &w.writer);
                        } else {
                            try w.writer.writeAll("null");
                        }
                    }
                    try w.writer.writeAll("]");
                } else {
                    try std.json.Stringify.value(ckv.value_ptr.*, .{}, &w.writer);
                }
            }
            try w.writer.writeAll("}}");
            try out.append(alloc, .{ .shard = sid, .body = try w.toOwnedSlice() });
        }
        return out.toOwnedSlice(alloc);
    }

    if (root.get("upsert_rows")) |rows_v| {
        if (rows_v != .array) return error.BadJson;
        for (rows_v.array.items, 0..) |row, i| {
            if (row != .object) return error.BadJson;
            const id_v = row.object.get("id") orelse return error.BadJson;
            const sid = try shardFor(alloc, namespace, id_v, n_shards, shard_by);
            try buckets[sid].append(alloc, i);
        }
        var out: std.ArrayList(Slice) = .empty;
        for (buckets, 0..) |b, sid| {
            if (b.items.len == 0) continue;
            var w: std.Io.Writer.Allocating = .init(alloc);
            try w.writer.writeAll("{");
            const had = try writeObjectExcept(&w.writer, root, "upsert_columns", "upsert_rows");
            if (had) try w.writer.writeAll(",");
            try w.writer.writeAll("\"upsert_rows\":[");
            for (b.items, 0..) |ri, k| {
                if (k > 0) try w.writer.writeAll(",");
                try std.json.Stringify.value(rows_v.array.items[ri], .{}, &w.writer);
            }
            try w.writer.writeAll("]}");
            try out.append(alloc, .{ .shard = sid, .body = try w.toOwnedSlice() });
        }
        return out.toOwnedSlice(alloc);
    }

    return error.MissingUpsert;
}

fn rssMiB(io: std.Io, pid: std.posix.pid_t) ?u64 {
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/status", .{pid}) catch return null;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var iov = [_][]u8{buf[0..]};
    const n = file.readStreaming(io, &iov) catch return null;
    const s = rss.parseLinuxStatus(buf[0..n]) orelse return null;
    return s.current_kib / 1024;
}

fn tcpConnectLoopback(port: u16) !std.posix.fd_t {
    if (!iouring.is_linux) return error.ConnectFailed;
    const sock_r = std.os.linux.socket(
        std.os.linux.AF.INET,
        std.os.linux.SOCK.STREAM | std.os.linux.SOCK.CLOEXEC,
        std.os.linux.IPPROTO.TCP,
    );
    if (std.os.linux.errno(sock_r) != .SUCCESS) return error.ConnectFailed;
    const fd: std.posix.fd_t = @intCast(sock_r);
    var sa = std.os.linux.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = std.mem.nativeToBig(u32, 0x7f000001),
    };
    const cr = std.os.linux.connect(fd, @ptrCast(&sa), @sizeOf(std.os.linux.sockaddr.in));
    if (std.os.linux.errno(cr) != .SUCCESS) {
        _ = std.os.linux.close(fd);
        return error.ConnectFailed;
    }
    iouring.enableTcpNoDelay(fd);
    return fd;
}

fn closeFd(fd: std.posix.fd_t) void {
    if (iouring.is_linux) _ = std.os.linux.close(fd);
}

const ChildConn = struct {
    port: u16,
    io: std.Io,
    fd: ?std.posix.fd_t = null,
    mu: std.Io.Mutex = .init,

    fn close(self: *ChildConn) void {
        if (self.fd) |fd| {
            closeFd(fd);
            self.fd = null;
        }
    }

    fn ensure(self: *ChildConn) !void {
        if (self.fd != null) return;
        self.fd = try tcpConnectLoopback(self.port);
    }

    fn request(self: *ChildConn, alloc: std.mem.Allocator, method: []const u8, path: []const u8, body: ?[]const u8) !struct { status: u16, body: []u8 } {
        // One keep-alive socket per child, shared across router workers.
        // Per-worker sockets pin child serve workers in recv() on idle
        // keep-alives (serve requeues after one request), so a third
        // connection is never read.
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        var last: ?anyerror = null;
        var attempt: u8 = 0;
        while (attempt < 3) : (attempt += 1) {
            self.ensure() catch |e| {
                last = e;
                self.close();
                continue;
            };
            const fd = self.fd.?;
            const wire = iouring.formatRequest(alloc, method, path, body) catch |e| {
                last = e;
                continue;
            };
            defer alloc.free(wire);
            iouring.posixSendAll(fd, wire) catch {
                last = error.SendFailed;
                self.close();
                continue;
            };
            var store: std.ArrayList(u8) = .empty;
            store.resize(alloc, 1 << 15) catch {
                last = error.OutOfMemory;
                self.close();
                continue;
            };
            var n: usize = 0;
            const ok = blk: {
                while (iouring.completeHttpResponse(store.items[0..n]) == 0) {
                    if (n == store.items.len) {
                        if (store.items.len >= 8 << 20) break :blk false;
                        store.resize(alloc, store.items.len * 2) catch break :blk false;
                    }
                    const got = iouring.posixRecv(fd, store.items[n..]) catch break :blk false;
                    if (got == 0) break :blk false;
                    n += got;
                }
                break :blk true;
            };
            if (!ok) {
                store.deinit(alloc);
                last = error.RecvFailed;
                self.close();
                continue;
            }
            const used = iouring.completeHttpResponse(store.items[0..n]);
            const status = iouring.parseHttpStatus(store.items[0..used]) catch {
                store.deinit(alloc);
                last = error.BadResponse;
                self.close();
                continue;
            };
            const sep = std.mem.indexOf(u8, store.items[0..used], "\r\n\r\n") orelse {
                store.deinit(alloc);
                last = error.BadResponse;
                self.close();
                continue;
            };
            const copy = alloc.dupe(u8, store.items[sep + 4 .. used]) catch {
                store.deinit(alloc);
                last = error.OutOfMemory;
                continue;
            };
            store.deinit(alloc);
            // Don't reuse a child socket after GET/DELETE. serve requeues the
            // keep-alive fd after one request; the next POST can sit unread.
            if (std.mem.eql(u8, method, "GET") or std.mem.eql(u8, method, "DELETE")) {
                self.close();
            }
            return .{ .status = status, .body = copy };
        }
        return last orelse error.ChildUnreachable;
    }
};

const Child = struct {
    port: u16,
    proc: ?std.process.Child = null,
    pid: ?std.posix.pid_t = null,
};

const Cluster = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    o: Options,
    ports: []u16,
    children: []Child,
    shard_by: ShardBy,

    fn queryTargets(self: *Cluster, alloc: std.mem.Allocator, namespace: []const u8) ![]usize {
        if (self.shard_by == .namespace) {
            const sid = try shardFor(alloc, namespace, .{ .integer = 0 }, self.ports.len, .namespace);
            const one = try alloc.alloc(usize, 1);
            one[0] = sid;
            return one;
        }
        return allTargets(alloc, self.ports.len);
    }

    fn rssTotal(self: *Cluster) u64 {
        var sum: u64 = 0;
        for (self.children) |c| {
            if (c.pid) |pid| {
                if (rssMiB(self.io, pid)) |m| sum += m;
            }
        }
        return sum;
    }
};

const FanJob = struct {
    conn: *ChildConn,
    alloc: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    body: ?[]const u8,
    status: u16 = 0,
    resp: []u8 = &.{},
    err: ?anyerror = null,
};

fn fanOne(job: *FanJob) void {
    const r = job.conn.request(job.alloc, job.method, job.path, job.body) catch |e| {
        job.err = e;
        return;
    };
    job.status = r.status;
    job.resp = r.body;
}

fn fanout(
    conns: []ChildConn,
    alloc: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    body: ?[]const u8,
    targets: []const usize,
) ![]FanJob {
    const jobs = try alloc.alloc(FanJob, targets.len);
    for (jobs, targets) |*job, idx| {
        job.* = .{
            .conn = &conns[idx],
            .alloc = alloc,
            .method = method,
            .path = path,
            .body = body,
        };
    }
    if (jobs.len == 1) {
        fanOne(&jobs[0]);
        return jobs;
    }
    var threads = try alloc.alloc(std.Thread, jobs.len);
    defer alloc.free(threads);
    for (jobs, 0..) |*job, i| {
        threads[i] = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, fanOne, .{job});
    }
    for (threads) |t| t.join();
    return jobs;
}

fn urlDecode(alloc: std.mem.Allocator, src: []const u8) ![]u8 {
    var out = try alloc.alloc(u8, src.len);
    var i: usize = 0;
    var j: usize = 0;
    while (i < src.len) {
        if (src[i] == '%' and i + 2 < src.len) {
            if (std.fmt.parseInt(u8, src[i + 1 .. i + 3], 16)) |b| {
                out[j] = b;
                i += 3;
                j += 1;
                continue;
            } else |_| {}
        }
        out[j] = src[i];
        i += 1;
        j += 1;
    }
    return out[0..j];
}

const RouteOut = struct {
    status: u16,
    reason: []const u8,
    body: []const u8,
};

fn reasonOf(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        else => "OK",
    };
}

fn allTargets(alloc: std.mem.Allocator, n: usize) ![]usize {
    const t = try alloc.alloc(usize, n);
    for (t, 0..) |*x, i| x.* = i;
    return t;
}

fn handleRoute(
    cluster: *Cluster,
    conns: []ChildConn,
    alloc: std.mem.Allocator,
    method: []const u8,
    path_raw: []const u8,
    body: []const u8,
) !RouteOut {
    const path = try urlDecode(alloc, path_raw);
    const qpos = std.mem.indexOfScalar(u8, path, '?') orelse path.len;
    const path_only = path[0..qpos];

    if (std.mem.eql(u8, path_only, "/health") or std.mem.eql(u8, path_only, "/")) {
        var w: std.Io.Writer.Allocating = .init(alloc);
        try w.writer.writeAll("{\"ok\":true,\"impl\":\"zig-iouring\",\"shards\":");
        try w.writer.print("{d}", .{cluster.ports.len});
        try w.writer.writeAll(",\"shard_by\":\"");
        try w.writer.writeAll(@tagName(cluster.shard_by));
        try w.writer.writeAll("\",\"hash\":\"fnv1a64(namespace + NUL + decimal_id) % N  (doc mode)\",\"children\":[");
        for (cluster.children, 0..) |c, i| {
            if (i > 0) try w.writer.writeAll(",");
            const alive = if (c.proc) |p| p.id != null else true;
            try w.writer.print("{{\"index\":{d},\"port\":{d}", .{ i, c.port });
            if (c.pid) |pid| {
                try w.writer.print(",\"pid\":{d}", .{pid});
                if (rssMiB(cluster.io, pid)) |m| try w.writer.print(",\"rss_mib\":{d}", .{m});
            }
            try w.writer.print(",\"alive\":{}", .{alive});
            try w.writer.writeAll("}");
        }
        try w.writer.print("],\"rss_mib\":{d}}}", .{cluster.rssTotal()});
        return .{ .status = 200, .reason = "OK", .body = try w.toOwnedSlice() };
    }

    const prefix = "/v2/namespaces/";
    if (!std.mem.startsWith(u8, path_only, prefix)) {
        return .{ .status = 404, .reason = "Not Found", .body = "{\"error\":\"unknown route\"}" };
    }
    const rest = path_only[prefix.len..];
    var meth_buf: [8]u8 = undefined;
    const meth = std.ascii.upperString(&meth_buf, method);

    if (std.mem.endsWith(u8, rest, "/query") and std.mem.eql(u8, meth, "POST")) {
        const ns = rest[0 .. rest.len - "/query".len];
        const top_k = parseTopK(body, alloc);
        const targets = try cluster.queryTargets(alloc, ns);
        var path_buf: [1024]u8 = undefined;
        const child_path = try std.fmt.bufPrint(&path_buf, "{s}{s}/query", .{ prefix, ns });
        const jobs = try fanout(conns, alloc, "POST", child_path, body, targets);
        var ok_bodies: std.ArrayList([]const u8) = .empty;
        var any_ok = false;
        var errors: usize = 0;
        for (jobs) |j| {
            if (j.err != null) {
                errors += 1;
                continue;
            }
            if (j.status == 404) continue;
            if (j.status != 200) {
                errors += 1;
                continue;
            }
            any_ok = true;
            try ok_bodies.append(alloc, j.resp);
        }
        if (errors > 0 and !any_ok) {
            return .{ .status = 502, .reason = "Bad Gateway", .body = "{\"error\":\"all shards failed\"}" };
        }
        if (!any_ok) {
            return .{ .status = 404, .reason = "Not Found", .body = "{\"error\":\"namespace not found\"}" };
        }
        const merged = try mergeRows(alloc, ok_bodies.items, top_k);
        return .{ .status = 200, .reason = "OK", .body = merged };
    }

    if (std.mem.endsWith(u8, rest, "/snapshot") and std.mem.eql(u8, meth, "POST")) {
        const ns = rest[0 .. rest.len - "/snapshot".len];
        var path_buf: [1024]u8 = undefined;
        const child_path = try std.fmt.bufPrint(&path_buf, "{s}{s}/snapshot", .{ prefix, ns });
        _ = try fanout(conns, alloc, "POST", child_path, if (body.len > 0) body else "", try allTargets(alloc, cluster.ports.len));
        return .{ .status = 200, .reason = "OK", .body = "{\"ok\":true}" };
    }

    if (std.mem.eql(u8, meth, "POST")) {
        const ns = rest;
        const slices = splitUpsert(alloc, ns, body, cluster.ports.len, cluster.shard_by) catch |e| switch (e) {
            error.BadJson, error.MissingUpsert, error.BadId, error.BadShardCount => {
                return .{ .status = 400, .reason = "Bad Request", .body = "{\"error\":\"invalid upsert\"}" };
            },
            else => return e,
        };
        if (slices.len == 0) {
            return .{ .status = 200, .reason = "OK", .body = "{\"ok\":true}" };
        }
        var path_buf: [1024]u8 = undefined;
        const child_path = try std.fmt.bufPrint(&path_buf, "{s}{s}", .{ prefix, ns });
        if (slices.len == 1) {
            const r = conns[slices[0].shard].request(alloc, "POST", child_path, slices[0].body) catch {
                return .{ .status = 502, .reason = "Bad Gateway", .body = "{\"error\":\"shard write failed\"}" };
            };
            if (r.status != 200) return .{ .status = r.status, .reason = reasonOf(r.status), .body = r.body };
            return .{ .status = 200, .reason = "OK", .body = "{\"ok\":true}" };
        }
        const jobs = try alloc.alloc(FanJob, slices.len);
        const threads = try alloc.alloc(std.Thread, slices.len);
        for (slices, 0..) |sl, i| {
            jobs[i] = .{
                .conn = &conns[sl.shard],
                .alloc = alloc,
                .method = "POST",
                .path = child_path,
                .body = sl.body,
            };
            threads[i] = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, fanOne, .{&jobs[i]});
        }
        for (threads) |t| t.join();
        for (jobs) |j| {
            if (j.err != null or j.status != 200) {
                if (j.err == null and j.resp.len > 0) {
                    return .{ .status = j.status, .reason = reasonOf(j.status), .body = j.resp };
                }
                return .{ .status = 502, .reason = "Bad Gateway", .body = "{\"error\":\"shard write failed\"}" };
            }
        }
        return .{ .status = 200, .reason = "OK", .body = "{\"ok\":true}" };
    }

    if (std.mem.eql(u8, meth, "GET")) {
        const ns = rest;
        const targets = try cluster.queryTargets(alloc, ns);
        var path_buf: [1024]u8 = undefined;
        const child_path = try std.fmt.bufPrint(&path_buf, "{s}{s}", .{ prefix, ns });
        const jobs = try fanout(conns, alloc, "GET", child_path, null, targets);
        var count: i64 = 0;
        var dim: i64 = 0;
        var any_ok = false;
        for (jobs) |j| {
            if (j.err != null) continue;
            if (j.status == 404) continue;
            if (j.status != 200) return .{ .status = j.status, .reason = reasonOf(j.status), .body = j.resp };
            any_ok = true;
            const parsed = std.json.parseFromSlice(std.json.Value, alloc, j.resp, .{}) catch continue;
            if (parsed.value == .object) {
                if (parsed.value.object.get("count")) |c| switch (c) {
                    .integer => |n| count += n,
                    else => {},
                };
                if (dim == 0) {
                    if (parsed.value.object.get("dim")) |d| switch (d) {
                        .integer => |n| dim = n,
                        else => {},
                    };
                }
            }
        }
        if (!any_ok) return .{ .status = 404, .reason = "Not Found", .body = "{\"error\":\"namespace not found\"}" };
        var buf: [256]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{{\"namespace\":\"{s}\",\"dim\":{d},\"count\":{d}}}", .{ ns, dim, count });
        return .{ .status = 200, .reason = "OK", .body = try alloc.dupe(u8, s) };
    }

    if (std.mem.eql(u8, meth, "DELETE")) {
        const ns = rest;
        var path_buf: [1024]u8 = undefined;
        const child_path = try std.fmt.bufPrint(&path_buf, "{s}{s}", .{ prefix, ns });
        _ = try fanout(conns, alloc, "DELETE", child_path, null, try allTargets(alloc, cluster.ports.len));
        var buf: [64]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{{\"ok\":true,\"shards\":{d}}}", .{cluster.ports.len});
        return .{ .status = 200, .reason = "OK", .body = try alloc.dupe(u8, s) };
    }

    return .{ .status = 405, .reason = "Method Not Allowed", .body = "{\"error\":\"method not allowed\"}" };
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
    cluster: *Cluster,
    queue: *FdQueue,
    listen_fd: std.posix.fd_t,
    conns: []ChildConn,
};

fn routerWorkerCount(o: Options) usize {
    if (o.router_workers) |n| return @min(64, @max(1, n));
    const n = std.Thread.getCpuCount() catch 4;
    return @min(16, @max(1, n));
}

fn childWorkerCount(o: Options) usize {
    if (o.workers) |n| return n;
    return if (o.shards > 1) 2 else 1;
}

fn poolWorker(ctx: *PoolCtx) void {
    var store: std.ArrayList(u8) = .empty;
    defer store.deinit(ctx.alloc);
    store.resize(ctx.alloc, 1 << 16) catch return;
    var arena_state = std.heap.ArenaAllocator.init(ctx.alloc);
    defer arena_state.deinit();

    while (true) {
        const fd = ctx.queue.pop();
        const end = handleClient(fd, ctx, &store, &arena_state, ctx.conns) catch .close;
        if (end == .close) closeFd(fd);
    }
}

const ConnEnd = enum { close, requeued };

fn handleClient(
    fd: std.posix.fd_t,
    ctx: *PoolCtx,
    store: *std.ArrayList(u8),
    arena_state: *std.heap.ArenaAllocator,
    conns: []ChildConn,
) !ConnEnd {
    var n: usize = 0;
    while (true) {
        while (iouring.completeHttpRequest(store.items[0..n]) == 0) {
            if (n == store.items.len) {
                if (store.items.len >= 8 << 20) return error.RequestTooLarge;
                try store.resize(ctx.alloc, store.items.len * 2);
            }
            const got = iouring.posixRecv(fd, store.items[n..]) catch return .close;
            if (got == 0) return .close;
            n += got;
        }
        const used = iouring.completeHttpRequest(store.items[0..n]);
        const raw = iouring.parseHttpRequest(store.items[0..used]) catch return .close;
        const keepalive = !iouring.wantsClose(store.items[0..used]);

        _ = arena_state.reset(.retain_capacity);
        const arena = arena_state.allocator();
        const out = handleRoute(ctx.cluster, conns, arena, raw.method, raw.path, raw.body) catch RouteOut{
            .status = 500,
            .reason = "Internal Server Error",
            .body = "{\"error\":\"internal\"}",
        };
        const wire = try iouring.formatResponse(arena, out.status, out.reason, out.body, keepalive);
        iouring.posixSendAll(fd, wire) catch return .close;

        if (!keepalive) return .close;
        drainListenBacklog(ctx.listen_fd, ctx.queue);
        if (ctx.queue.tryPush(fd)) return .requeued;
        if (used < n) {
            const rest = n - used;
            std.mem.copyForwards(u8, store.items[0..rest], store.items[used..n]);
            n = rest;
            continue;
        }
        n = 0;
    }
}

fn spawnChildren(cluster: *Cluster, out: *std.Io.Writer) !void {
    const workers = childWorkerCount(cluster.o);
    var ef_buf: [16]u8 = undefined;
    const ef_s = try std.fmt.bufPrint(&ef_buf, "{d}", .{cluster.o.ef});
    var w_buf: [16]u8 = undefined;
    const w_s = try std.fmt.bufPrint(&w_buf, "{d}", .{workers});
    var port_bufs: [64][16]u8 = undefined;
    if (cluster.ports.len > port_bufs.len) return error.TooManyShards;

    for (cluster.children, 0..) |*ch, i| {
        const port_s = try std.fmt.bufPrint(&port_bufs[i], "{d}", .{ch.port});
        const argv = [_][]const u8{ cluster.o.binary, "serve", "--port", port_s, "--ef", ef_s, "--workers", w_s };
        const child = try std.process.spawn(cluster.io, .{
            .argv = &argv,
            .stdout = .ignore,
            .stderr = .ignore,
        });
        ch.proc = child;
        ch.pid = child.id;
        try out.print("shard[{d}] pid={?d} port={d} workers={d}\n", .{ i, ch.pid, ch.port, workers });
        try out.flush();
    }
    cluster.io.sleep(.{ .nanoseconds = 80_000_000 }, .awake) catch {};
    for (cluster.children, 0..) |ch, i| {
        if (ch.pid) |pid| {
            var path_buf: [48]u8 = undefined;
            const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/stat", .{pid}) catch return error.ChildDied;
            const st = std.Io.Dir.openFileAbsolute(cluster.io, path, .{}) catch {
                try out.print("shard[{d}] pid={d} died immediately (port {d} busy?)\n", .{ i, pid, ch.port });
                try out.flush();
                return error.ChildDied;
            };
            st.close(cluster.io);
        }
    }
}

fn waitReady(cluster: *Cluster, timeout_ns: i128) !void {
    const start = std.Io.Timestamp.now(cluster.io, .awake);
    while (true) {
        var pending: usize = 0;
        for (cluster.ports) |port| {
            if (tcpConnectLoopback(port)) |fd| {
                closeFd(fd);
            } else |_| {
                pending += 1;
            }
        }
        if (pending == 0) return;
        const elapsed = start.durationTo(std.Io.Timestamp.now(cluster.io, .awake));
        if (elapsed.nanoseconds >= timeout_ns) return error.ShardsNotReady;
        cluster.io.sleep(.{ .nanoseconds = 50_000_000 }, .awake) catch {};
    }
}

fn stopChildren(cluster: *Cluster) void {
    for (cluster.children) |*ch| {
        if (ch.proc) |*p| {
            p.kill(cluster.io);
            _ = p.wait(cluster.io) catch {};
            ch.proc = null;
        }
    }
}

pub fn serve(alloc: std.mem.Allocator, io: std.Io, o: Options, out: *std.Io.Writer) !void {
    if (o.shards < 1) return error.BadShardCount;
    const base: u16 = o.shard_port_base orelse (o.port + 1);
    const ports = try alloc.alloc(u16, o.shards);
    const children = try alloc.alloc(Child, o.shards);
    for (0..o.shards) |i| {
        const p: u16 = @intCast(base + i);
        if (p == o.port) return error.PortCollision;
        ports[i] = p;
        children[i] = .{ .port = p };
    }

    var cluster = Cluster{
        .alloc = alloc,
        .io = io,
        .o = o,
        .ports = ports,
        .children = children,
        .shard_by = o.shard_by,
    };

    if (o.spawn_children) {
        try spawnChildren(&cluster, out);
        waitReady(&cluster, 15_000_000_000) catch |e| {
            try out.print("shards not ready: {any}\n", .{e});
            try out.flush();
            stopChildren(&cluster);
            return e;
        };
    }

    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", o.port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    try out.print(
        "openpuffer shard router (zig/io_uring) on http://127.0.0.1:{d}  shards={d} by={s} children=",
        .{ o.port, o.shards, @tagName(o.shard_by) },
    );
    for (ports, 0..) |p, i| {
        if (i > 0) try out.writeAll(",");
        try out.print("{d}", .{p});
    }
    try out.writeAll("\n");
    try out.flush();

    if (builtin.os.tag != .linux) {
        try out.writeAll("shard-router requires linux io_uring\n");
        stopChildren(&cluster);
        return error.Unsupported;
    }
    serveIoUring(alloc, io, listener.socket.handle, &cluster, out) catch |e| {
        try out.print("io_uring router failed ({any})\n", .{e});
        try out.flush();
        stopChildren(&cluster);
        return e;
    };
}

fn serveIoUring(alloc: std.mem.Allocator, io: std.Io, listen_fd: std.posix.fd_t, cluster: *Cluster, out: *std.Io.Writer) !void {
    _ = std.os.linux.listen(listen_fd, 1024);
    const queue = try alloc.create(FdQueue);
    queue.* = .{ .io = io };
    const conns = try alloc.alloc(ChildConn, cluster.ports.len);
    for (conns, cluster.ports) |*c, p| c.* = .{ .port = p, .io = io };
    const ctx = try alloc.create(PoolCtx);
    ctx.* = .{ .alloc = alloc, .io = io, .cluster = cluster, .queue = queue, .listen_fd = listen_fd, .conns = conns };
    const n = routerWorkerCount(cluster.o);
    for (0..n) |_| {
        const t = try std.Thread.spawn(.{ .stack_size = 512 * 1024 }, poolWorker, .{ctx});
        t.detach();
    }
    try out.print("linux shard-router: io_uring accept + {d} keep-alive workers, child keep-alive + TCP_NODELAY\n", .{n});
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
        drainListenBacklog(listen_fd, queue);
        queue.push(fd);
    }
}

fn jsonInt(n: i64) std.json.Value {
    return .{ .integer = n };
}

test "fnv1a64 empty is offset" {
    try std.testing.expectEqual(FNV_OFFSET, fnv1a64(""));
}

test "shard_for matches tools/shard_key.py fixtures" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 3), try shardFor(a, "serve-bench", jsonInt(0), 4, .doc));
    try std.testing.expectEqual(@as(usize, 0), try shardFor(a, "serve-bench", jsonInt(1), 4, .doc));
    try std.testing.expectEqual(@as(usize, 1), try shardFor(a, "serve-bench", jsonInt(2), 4, .doc));
    try std.testing.expectEqual(@as(usize, 2), try shardFor(a, "serve-bench", jsonInt(3), 4, .doc));
    try std.testing.expectEqual(try shardFor(a, "ns", jsonInt(42), 4, .doc), try shardFor(a, "ns", .{ .string = "42" }, 4, .doc));
    try std.testing.expectEqual(try shardFor(a, "ns", jsonInt(42), 4, .doc), try shardFor(a, "ns", .{ .string = "042" }, 4, .doc));
    try std.testing.expect(try shardFor(a, "a", jsonInt(1), 8, .doc) != try shardFor(a, "b", jsonInt(1), 8, .doc));
    try std.testing.expectEqual(try shardFor(a, "tenant", jsonInt(1), 4, .namespace), try shardFor(a, "tenant", jsonInt(999), 4, .namespace));
    const k1 = try shardKeyBytes(a, "", jsonInt(1), .doc);
    defer a.free(k1);
    const k2 = try shardKeyBytes(a, "1", .{ .string = "" }, .doc);
    defer a.free(k2);
    try std.testing.expect(!std.mem.eql(u8, k1, k2));
    var seen = [4]bool{ false, false, false, false };
    var i: i64 = 0;
    while (i < 64) : (i += 1) {
        seen[try shardFor(a, "serve-bench", jsonInt(i), 4, .doc)] = true;
    }
    try std.testing.expect(seen[0] and seen[1] and seen[2] and seen[3]);
}

test "merge_rows lower distance wins and de-dups" {
    const a = std.testing.allocator;
    const s0 = "{\"rows\":[{\"id\":1,\"$distance\":0.2},{\"id\":2,\"$distance\":0.4}]}";
    const s1 = "{\"rows\":[{\"id\":3,\"$distance\":0.1},{\"id\":1,\"$distance\":0.9}]}";
    const bodies = [_][]const u8{ s0, s1 };
    const out = try mergeRows(a, &bodies, 2);
    defer a.free(out);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, out, .{});
    defer parsed.deinit();
    const rows = parsed.value.object.get("rows").?.array;
    try std.testing.expectEqual(@as(usize, 2), rows.items.len);
    try std.testing.expectEqual(@as(i64, 3), rows.items[0].object.get("id").?.integer);
    try std.testing.expectEqual(@as(i64, 1), rows.items[1].object.get("id").?.integer);
}

test "split_upsert fixtures from shard_key.selftest" {
    const a = std.testing.allocator;
    const body = "{\"upsert_columns\":{\"id\":[0,1,2,3],\"vector\":[[0.0],[1.0],[2.0],[3.0]]},\"distance_metric\":\"cosine_distance\"}";
    const parts = try splitUpsert(a, "serve-bench", body, 4, .doc);
    defer {
        for (parts) |p| a.free(p.body);
        a.free(parts);
    }
    try std.testing.expectEqual(@as(usize, 4), parts.len);
    var seen: usize = 0;
    for (parts) |p| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, p.body, .{});
        defer parsed.deinit();
        const ids = parsed.value.object.get("upsert_columns").?.object.get("id").?.array;
        try std.testing.expectEqual(@as(usize, 1), ids.items.len);
        const id = ids.items[0].integer;
        if (id == 0) try std.testing.expectEqual(@as(usize, 3), p.shard);
        if (id == 1) try std.testing.expectEqual(@as(usize, 0), p.shard);
        if (id == 2) try std.testing.expectEqual(@as(usize, 1), p.shard);
        if (id == 3) try std.testing.expectEqual(@as(usize, 2), p.shard);
        seen += 1;
        try std.testing.expectEqualStrings("cosine_distance", parsed.value.object.get("distance_metric").?.string);
    }
    try std.testing.expectEqual(@as(usize, 4), seen);
}

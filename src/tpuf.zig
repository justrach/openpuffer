//! Minimal turbopuffer v2 REST client: upsert_columns write + ANN query.

const std = @import("std");

pub const Config = struct {
    api_key: []const u8,
    region: []const u8 = "gcp-us-central1",
    /// when set, requests go to this base URL instead of turbopuffer cloud
    /// (drop-in mode against a local `openpuffer serve` instance).
    endpoint: ?[]const u8 = null,
};

pub const QueryHit = struct { id: u64, distance: f64 };

const Timer = struct {
    io: std.Io,
    t0: std.Io.Timestamp,
    fn start(io: std.Io) Timer {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    fn readNs(self: *const Timer) u64 {
        const d = self.t0.durationTo(std.Io.Timestamp.now(self.io, .awake));
        return @intCast(@max(0, d.nanoseconds));
    }
};

pub const Client = struct {
    cfg: Config,
    client: std.http.Client,
    io: std.Io,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, cfg: Config) Client {
        return .{
            .cfg = cfg,
            .client = .{ .allocator = alloc, .io = io },
            .io = io,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Client) void {
        self.client.deinit();
    }

    fn authHeader(self: *Client, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "Bearer {s}", .{self.cfg.api_key}) catch "Bearer ";
    }

    fn url(self: *Client, namespace: []const u8, suffix: []const u8, buf: []u8) ![]u8 {
        if (self.cfg.endpoint) |base| {
            return std.fmt.bufPrint(buf, "{s}/v2/namespaces/{s}{s}", .{ base, namespace, suffix });
        }
        return std.fmt.bufPrint(buf, "https://{s}.turbopuffer.com/v2/namespaces/{s}{s}", .{
            self.cfg.region, namespace, suffix,
        });
    }

    /// Upsert a batch of vectors (row-major, n rows of dim) with ids.
    pub fn upsert(self: *Client, namespace: []const u8, ids: []const u64, rows: []const []const f32) !void {
        const dim = if (rows.len > 0) rows[0].len else return;
        var body: std.Io.Writer.Allocating = .init(self.alloc);
        defer body.deinit();
        const w = &body.writer;
        try w.writeAll("{\"upsert_columns\":{\"id\":[");
        for (ids, 0..) |id, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{d}", .{id});
        }
        try w.writeAll("],\"vector\":[");
        for (0..ids.len) |i| {
            if (i > 0) try w.writeAll(",");
            try w.writeAll("[");
            for (0..dim) |j| {
                if (j > 0) try w.writeAll(",");
                try w.print("{d}", .{rows[i][j]});
            }
            try w.writeAll("]");
        }
        try w.writeAll("]},\"distance_metric\":\"cosine_distance\",\"encoding\":null}");

        var url_buf: [512]u8 = undefined;
        const u = try self.url(namespace, "", &url_buf);
        var auth_buf: [600]u8 = undefined;
        var resp: std.Io.Writer.Allocating = .init(self.alloc);
        defer resp.deinit();
        const result = try self.client.fetch(.{
            .location = .{ .url = u },
            .method = .POST,
            .payload = body.written(),
            .extra_headers = &.{
                .{ .name = "authorization", .value = self.authHeader(&auth_buf) },
                .{ .name = "content-type", .value = "application/json" },
            },
            .response_writer = &resp.writer,
            .keep_alive = false,
        });
        if (result.status != .ok and result.status != .multi_status) {
            std.debug.print("tpuf write error {d}: {s}\n", .{ @intFromEnum(result.status), resp.written() });
            return error.TpufWriteFailed;
        }
    }

    pub const AnnQuery = struct {
        vector: []const f32,
        top_k: usize,
        ef: usize = 200,
        consistency: []const u8 = "strong",
    };

    pub const QueryResult = struct { count: usize, latency_ns: u64 };

    /// ANN query; returns hit count and round-trip latency.
    pub fn query(self: *Client, namespace: []const u8, q: AnnQuery, out_hits: []QueryHit) !QueryResult {
        var body: std.Io.Writer.Allocating = .init(self.alloc);
        defer body.deinit();
        const w = &body.writer;
        try w.print("{{\"rank_by\":[\"vector\",\"ANN\",[", .{});
        for (q.vector, 0..) |x, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{d}", .{x});
        }
        try w.print("]],\"top_k\":{d},\"consistency\":{{\"level\":\"{s}\"}}}}", .{
            q.top_k, q.consistency,
        });

        var url_buf: [512]u8 = undefined;
        const u = try self.url(namespace, "/query", &url_buf);
        var auth_buf: [600]u8 = undefined;

        var timer = Timer.start(self.io);
        var resp: std.Io.Writer.Allocating = .init(self.alloc);
        defer resp.deinit();
        const result = try self.client.fetch(.{
            .location = .{ .url = u },
            .method = .POST,
            .payload = body.written(),
            .extra_headers = &.{
                .{ .name = "authorization", .value = self.authHeader(&auth_buf) },
                .{ .name = "content-type", .value = "application/json" },
            },
            .response_writer = &resp.writer,
        });
        const elapsed = timer.readNs();

        if (result.status != .ok) {
            std.debug.print("tpuf query error {d}: {s}\n", .{ @intFromEnum(result.status), resp.written() });
            return error.TpufQueryFailed;
        }

        // parse {"rows":[{"id":...,"dist":...}]}
        const parsed = try std.json.parseFromSlice(std.json.Value, self.alloc, resp.written(), .{});
        defer parsed.deinit();
        var count: usize = 0;
        if (parsed.value == .object) {
            if (parsed.value.object.get("rows")) |rows| {
                for (rows.array.items) |row| {
                    if (count >= out_hits.len) break;
                    const obj = row.object;
                    const id: u64 = switch (obj.get("id").?) {
                        .integer => |n| @intCast(n),
                        else => 0,
                    };
                    const dist: f64 = blk: {
                        if (obj.get("$distance")) |d| break :blk d.float;
                        break :blk -1;
                    };
                    out_hits[count] = .{ .id = id, .distance = dist };
                    count += 1;
                }
            }
        }
        _ = &timer;
        return .{ .count = count, .latency_ns = elapsed };
    }
};

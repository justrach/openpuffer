//! Gemini embedding client (gemini-embedding-2) via batchEmbedContents.

const std = @import("std");

pub const Config = struct {
    api_key: []const u8,
    model: []const u8 = "gemini-embedding-2",
    dim: usize = 1536, // MRL-truncated output dimensionality
    base_url: []const u8 = "https://generativelanguage.googleapis.com/v1beta",
};

pub const Embedder = struct {
    cfg: Config,
    client: std.http.Client,
    io: std.Io,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, cfg: Config) Embedder {
        return .{
            .cfg = cfg,
            .client = .{ .allocator = alloc, .io = io },
            .io = io,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Embedder) void {
        self.client.deinit();
    }

    fn post(self: *Embedder, url: []const u8, body: []const u8, out: *std.Io.Writer.Allocating) !std.http.Status {
        const result = try self.client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/json" },
            },
            .response_writer = &out.writer,
            .keep_alive = false,
        });
        return result.status;
    }

    /// Embed one batch of texts. Caller frees each returned slice and the list.
    pub fn embedBatch(self: *Embedder, texts: []const []const u8, task_type: []const u8) ![][]f32 {
        var body: std.Io.Writer.Allocating = .init(self.alloc);
        defer body.deinit();
        const w = &body.writer;
        try w.print(
            "{{\"requests\":[",
            .{},
        );
        for (texts, 0..) |t, i| {
            if (i > 0) try w.writeAll(",");
            try w.print(
                "{{\"model\":\"models/{s}\",\"taskType\":\"{s}\",\"outputDimensionality\":{d},\"content\":{{\"parts\":[{{\"text\":",
                .{ self.cfg.model, task_type, self.cfg.dim },
            );
            try std.json.Stringify.value(t, .{}, w);
            try w.writeAll("}]}"); // close part, parts, content
            try w.writeAll("}"); // close request object
        }
        try w.writeAll("]}");

        var url_buf: [512]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buf, "{s}/models/{s}:batchEmbedContents?key={s}", .{
            self.cfg.base_url, self.cfg.model, self.cfg.api_key,
        });

        var resp: std.Io.Writer.Allocating = .init(self.alloc);
        defer resp.deinit();
        const status = try self.post(url, body.written(), &resp);
        if (status != .ok) {
            std.debug.print("gemini error {d}: {s}\n", .{ @intFromEnum(status), resp.written() });
            return error.EmbeddingRequestFailed;
        }
        return parseBatchResponse(self.alloc, resp.written(), self.cfg.dim);
    }

    fn parseBatchResponse(alloc: std.mem.Allocator, text: []const u8, expected_dim: usize) ![][]f32 {
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, text, .{});
        defer parsed.deinit();
        const embeddings = parsed.value.object.get("embeddings") orelse return error.BadResponse;
        var out: [][]f32 = try alloc.alloc([]f32, embeddings.array.items.len);
        errdefer alloc.free(out);
        for (embeddings.array.items, 0..) |item, i| {
            const values = item.object.get("values") orelse return error.BadResponse;
            const row = try alloc.alloc(f32, values.array.items.len);
            for (values.array.items, 0..) |v, j| {
                row[j] = switch (v) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    else => return error.BadResponse,
                };
            }
            if (row.len != expected_dim) return error.DimensionMismatch;
            vec_normalize(row);
            out[i] = row;
        }
        return out;
    }
};

fn vec_normalize(v: []f32) void {
    var sum: f32 = 0;
    for (v) |x| sum += x * x;
    const n = @sqrt(sum);
    if (n > 0) {
        for (v) |*x| x.* /= n;
    }
}

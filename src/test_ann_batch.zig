const std = @import("std");
const op = @import("openpuffer");
const a = std.testing.allocator;

fn expectSame(index: *op.Hnsw(void), query: []const f32) !void {
    index.opts.batch_neighbor_scoring = false;
    const scalar = try index.searchAdvanced(query, 12, 48, 4, a);
    defer a.free(scalar);
    index.opts.batch_neighbor_scoring = true;
    const batch = try index.searchAdvanced(query, 12, 48, 4, a);
    defer a.free(batch);
    try std.testing.expectEqual(scalar.len, batch.len);
    for (scalar, batch) |x, y| {
        try std.testing.expectEqual(x.id, y.id);
        try std.testing.expectEqual(@as(u32, @bitCast(x.distance)), @as(u32, @bitCast(y.distance)));
    }
}

test "ann batch: integer dot products exactly match scalar including SIMD tails" {
    var random = std.Random.DefaultPrng.init(0x4241544348);
    const rng = random.random();
    var query: [1537]i8 = undefined;
    var rows: [4][1537]i8 = undefined;
    for (&query) |*v| v.* = rng.int(i8);
    for (&rows) |*row| for (row) |*v| {
        v.* = rng.int(i8);
    };
    for ([_]usize{ 0, 1, 15, 16, 17, 63, 64, 65, 511, 512, 513, 1536, 1537 }) |dim| {
        const vectors = [4][]const i8{ rows[0][0..dim], rows[1][0..dim], rows[2][0..dim], rows[3][0..dim] };
        const sums = op.vector.dotI8Batch4(query[0..dim], vectors);
        for (vectors, sums) |row, value| try std.testing.expectEqual(op.dotI8(query[0..dim], row), value);
    }
}

test "ann batch: heap and mmap preserve IDs, distance bits and tie order" {
    var random = std.Random.DefaultPrng.init(0x48534e57);
    const rng = random.random();
    for ([_]usize{ 63, 64, 65, 512, 513, 1536 }) |dim| {
        for ([_]bool{ true, false }) |store_f32| {
            var index = op.Hnsw(void).init(a, dim, .{ .seed = 13, .store_f32 = store_f32 });
            defer index.deinit();
            const rows = try a.alloc(f32, 96 * dim);
            defer a.free(rows);
            for (rows) |*v| v.* = rng.float(f32) * 2 - 1;
            for (0..96) |i| {
                // Identical vectors exercise int8 and exact-score heap ties.
                if (i > 0 and i % 5 == 0) @memcpy(rows[i * dim ..][0..dim], rows[(i - 1) * dim ..][0..dim]);
                _ = try index.insert(rows[i * dim ..][0..dim]);
            }
            for (0..12) |i| try expectSame(&index, rows[i * dim ..][0..dim]);
            if (@import("builtin").os.tag == .windows) continue;
            var tmp = std.testing.tmpDir(.{});
            defer tmp.cleanup();
            var buffer: [std.fs.max_path_bytes]u8 = undefined;
            const directory = buffer[0..try tmp.dir.realPathFile(std.testing.io, ".", &buffer)];
            const path = try std.fmt.allocPrint(a, "{s}/batch.hmls", .{directory});
            defer a.free(path);
            try index.writeSlabs(path);
            var mapped = op.Hnsw(void).init(a, dim, .{});
            defer mapped.deinit();
            try mapped.loadMmap(path);
            for (0..12) |i| try expectSame(&mapped, rows[i * dim ..][0..dim]);
            // Subsequent in-place vector updates retain per-vector generations.
            for (0..8) |i| {
                for (rows[0..dim]) |*v| v.* = rng.float(f32) * 2 - 1;
                try mapped.update(@intCast(i), rows[0..dim]);
                try expectSame(&mapped, rows[0..dim]);
            }
        }
    }
}

test "ann batch: filtered traversal preserves accepted results for every metric" {
    const Even = struct {
        pub fn ok(_: @This(), id: u32) bool {
            return id % 2 == 0;
        }
    };
    for ([_]op.Metric{ .cosine, .euclidean_squared, .dot_product }) |metric| {
        var index = op.Hnsw(void).init(a, 65, .{ .seed = 29, .metric = metric });
        defer index.deinit();
        var random = std.Random.DefaultPrng.init(719);
        var row: [65]f32 = undefined;
        for (0..96) |_| {
            for (&row) |*v| v.* = random.random().float(f32) * 2 - 1;
            _ = try index.insert(&row);
        }
        for (0..12) |_| {
            for (&row) |*v| v.* = random.random().float(f32) * 2 - 1;
            index.opts.batch_neighbor_scoring = false;
            const scalar = try index.searchAccept(&row, 12, 48, 4, Even{}, a);
            defer a.free(scalar);
            index.opts.batch_neighbor_scoring = true;
            const batch = try index.searchAccept(&row, 12, 48, 4, Even{}, a);
            defer a.free(batch);
            try std.testing.expectEqual(scalar.len, batch.len);
            for (scalar, batch) |x, y| {
                try std.testing.expect((Even{}).ok(y.id));
                try std.testing.expectEqual(x.id, y.id);
                try std.testing.expectEqual(@as(u32, @bitCast(x.distance)), @as(u32, @bitCast(y.distance)));
            }
        }
    }
}

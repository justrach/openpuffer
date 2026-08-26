//! Compile-time proof that a dependent can `@import("openpuffer")`
//! without pulling HTTP serve into the graph.

const std = @import("std");
const builtin = @import("builtin");
const openpuffer = @import("openpuffer");

test "dependent imports openpuffer module" {
    const Index = openpuffer.Hnsw(void);
    var idx = Index.init(std.testing.allocator, 2, .{ .store_f32 = false });
    defer idx.deinit();

    _ = try idx.insert(&.{ 1, 0 });
    _ = try idx.insert(&.{ 0, 1 });
    try std.testing.expectEqual(@as(usize, 2), idx.len());
    try std.testing.expect(!idx.hasStoredF32());

    const hits = try idx.search(&.{ 0.95, 0.05 }, 1, 8, std.testing.allocator);
    defer std.testing.allocator.free(hits);
    try std.testing.expectEqual(@as(u32, 0), hits[0].id);

    var v = [_]f32{ 3, 4 };
    openpuffer.normalize(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 1), openpuffer.l2Norm(&v), 1e-6);
}

test "dependent loadMmap reopens writeSlabs snapshot" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    const Index = openpuffer.Hnsw(void);
    const dim = 8;
    var idx = Index.init(std.testing.allocator, dim, .{});
    defer idx.deinit();

    var planted = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    planted[1] = 1;
    const id = try idx.insert(&planted);
    const path = "/tmp/openpuffer-lib-mmap.slabs";
    try idx.writeSlabs(path);

    var loaded = Index.init(std.testing.allocator, dim, .{});
    defer loaded.deinit();
    try loaded.loadMmap(path);
    try std.testing.expect(loaded.isMmapBacked());
    try std.testing.expectEqual(idx.len(), loaded.len());
    const hits = try loaded.search(&planted, 1, 8, std.testing.allocator);
    defer std.testing.allocator.free(hits);
    try std.testing.expectEqual(id, hits[0].id);
}

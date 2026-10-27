//! Paired full-dimensional ANN execution experiment: no embedding/model calls.
const std = @import("std");
const op = @import("openpuffer");
// Native macOS RSS sampling for this experiment, without exporting another
// OpenPuffer API or assigning a shared source file to two Zig modules.
const rss = struct {
    const Info = extern struct {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_seconds: i32,
        user_microseconds: i32,
        system_seconds: i32,
        system_microseconds: i32,
        policy: i32,
        suspend_count: i32,
    };
    extern "c" fn mach_task_self() u32;
    extern "c" fn task_info(task: u32, flavor: c_int, info: *anyopaque, count: *u32) c_int;
    fn currentKiB() ?u64 {
        if (comptime @import("builtin").os.tag != .macos) return null;
        var info: Info = std.mem.zeroes(Info);
        var count: u32 = @sizeOf(Info) / @sizeOf(u32);
        if (task_info(mach_task_self(), 20, &info, &count) != 0) return null;
        return info.resident_size / 1024;
    }
};
const a = std.heap.c_allocator;
const Index = op.Hnsw(void);
const query_count = 128;

pub fn main(init: std.process.Init.Minimal) !void {
    _ = init;
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try synthetic(io, "clustered-512", 512, 4096, true, 24, 48, 2);
    try synthetic(io, "uniform-512", 512, 4096, false, 24, 48, 2);
    try synthetic(io, "clustered-1536", 1536, 4096, true, 10, 128, 4);
}

fn synthetic(io: std.Io, name: []const u8, dim: usize, count: usize, clustered: bool, k: usize, ef: u32, rerank: usize) !void {
    var random = std.Random.DefaultPrng.init(0x414e4e + dim + @intFromBool(clustered));
    const rng = random.random();
    var index = Index.init(a, dim, .{ .seed = 131 });
    defer index.deinit();
    const centers = try a.alloc(f32, 64 * dim);
    defer a.free(centers);
    for (centers) |*v| v.* = rng.float(f32) * 2 - 1;
    for (0..64) |i| try op.vector.normalizeChecked(centers[i * dim ..][0..dim]);
    const row = try a.alloc(f32, dim);
    defer a.free(row);
    const build_start = benchmarkNanoseconds();
    for (0..count) |i| {
        for (row, 0..) |*v, d| v.* = if (clustered) centers[(i % 64) * dim + d] + (rng.float(f32) * 2 - 1) * 0.04 else rng.float(f32) * 2 - 1;
        _ = try index.insert(row);
    }
    std.debug.print("fixture={s} count={d} dim={d} build_ms={d:.2}\n", .{ name, count, dim, @as(f64, @floatFromInt(benchmarkNanoseconds() - build_start)) / 1e6 });
    const queries = try a.alloc(f32, query_count * dim);
    defer a.free(queries);
    for (0..query_count) |i| for (queries[i * dim ..][0..dim], 0..) |*v, d| {
        v.* = if (clustered) centers[(i % 64) * dim + d] + (rng.float(f32) * 2 - 1) * 0.04 else rng.float(f32) * 2 - 1;
    };
    try compare(&index, queries, name, "heap", k, ef, rerank);
    const path = try std.fmt.allocPrint(a, ".zig-cache/ann-batch-{d}.hmls", .{benchmarkNanoseconds()});
    defer a.free(path);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    try index.writeSlabs(path);
    var mapped = Index.init(a, dim, .{});
    defer mapped.deinit();
    try mapped.loadMmap(path);
    try compare(&mapped, queries, name, "mmap", k, ef, rerank);
}

fn compare(index: *Index, queries: []const f32, fixture: []const u8, storage: []const u8, k: usize, ef: u32, rerank: usize) !void {
    var recall_hits: usize = 0;
    const exact = try a.alloc(op.SearchResult, k);
    defer a.free(exact);
    const nq = try a.alloc(f32, index.dim);
    defer a.free(nq);
    for (0..query_count) |qi| {
        const query = queries[qi * index.dim ..][0..index.dim];
        index.opts.batch_neighbor_scoring = false;
        const baseline = try index.searchAdvanced(query, k, ef, rerank, a);
        defer a.free(baseline);
        index.opts.batch_neighbor_scoring = true;
        const candidate = try index.searchAdvanced(query, k, ef, rerank, a);
        defer a.free(candidate);
        if (baseline.len != candidate.len) return error.ResultLengthMismatch;
        for (baseline, candidate) |x, y| {
            if (x.id != y.id or @as(u32, @bitCast(x.distance)) != @as(u32, @bitCast(y.distance))) return error.ResultParityFailed;
        }
        @memcpy(nq, query);
        try op.vector.normalizeChecked(nq);
        var len: usize = 0;
        for (0..index.len()) |id| {
            const distance = op.cosineDistance(nq, index.vectorConst(@intCast(id)));
            if (len == k and distance >= exact[k - 1].distance) continue;
            var position = @min(len, k - 1);
            while (position > 0 and distance < exact[position - 1].distance) : (position -= 1) {
                exact[position] = exact[position - 1];
            }
            exact[position] = .{ .id = @intCast(id), .distance = distance };
            len = @min(k, len + 1);
        }
        for (baseline) |hit| for (exact[0..len]) |gold| {
            if (hit.id == gold.id) {
                recall_hits += 1;
                break;
            }
        };
    }
    std.debug.print("fixture={s} storage={s} queries={d} k={d} ef={d} rerank={d} parity=all recall={d:.6}\n", .{ fixture, storage, query_count, k, ef, rerank, @as(f64, @floatFromInt(recall_hits)) / @as(f64, @floatFromInt(query_count * k)) });
    const times = try a.alloc(u64, query_count * 3);
    defer a.free(times);
    var checksum: u64 = 0;
    for (0..7) |round| {
        for (0..2) |turn| {
            const batched = (round + turn) % 2 != 0;
            index.opts.batch_neighbor_scoring = batched;
            for (0..3) |pass| for (0..query_count) |qi| {
                const start = benchmarkNanoseconds();
                const hits = try index.searchAdvanced(queries[qi * index.dim ..][0..index.dim], k, ef, rerank, a);
                times[pass * query_count + qi] = benchmarkNanoseconds() - start;
                for (hits) |hit| checksum +%= hit.id +% @as(u32, @bitCast(hit.distance));
                a.free(hits);
            };
            std.mem.sort(u64, times, {}, std.sort.asc(u64));
            std.debug.print("fixture={s} storage={s} round={d} implementation={s} p50_ns={d} p95_ns={d} rss_kib={d}\n", .{ fixture, storage, round, if (batched) "batch4" else "scalar", times[times.len / 2], times[times.len * 95 / 100], rss.currentKiB() orelse 0 });
        }
    }
    std.debug.print("fixture={s} storage={s} checksum={d}\n", .{ fixture, storage, checksum });
}

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;

fn benchmarkNanoseconds() u64 {
    if (comptime @import("builtin").os.tag == .macos) {
        return clock_gettime_nsec_np(8); // Darwin CLOCK_UPTIME_RAW.
    }
    var ts: Timespec = undefined;
    if (clock_gettime(1, &ts) != 0) @panic("monotonic clock failed");
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Timespec = extern struct { sec: isize, nsec: isize };
extern "c" fn clock_gettime(clock: c_int, ts: *Timespec) c_int;

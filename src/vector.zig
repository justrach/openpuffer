//! Core vector math: cosine distance over f32 vectors.

const std = @import("std");
const builtin = @import("builtin");

const has_avx512_vnni = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni);

pub fn dot(a: []const f32, b: []const f32) f64 {
    const lane = 16;
    var acc: @Vector(lane, f32) = @splat(0);
    var i: usize = 0;
    while (i + lane <= a.len) : (i += lane) {
        const va: @Vector(lane, f32) = a[i..][0..lane].*;
        const vb: @Vector(lane, f32) = b[i..][0..lane].*;
        acc += va * vb;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return @floatCast(sum);
}

/// Cosine distance in [0, 2]. For normalized vectors this is 1 - dot.
pub fn cosineDistance(a: []const f32, b: []const f32) f32 {
    return @max(0.0, 1.0 - @as(f32, @floatCast(dot(a, b))));
}

pub fn l2Norm(v: []const f32) f32 {
    return @sqrt(@as(f32, @floatCast(dot(v, v))));
}

pub fn normalize(v: []f32) void {
    var n = l2Norm(v);
    if (n == 0) n = 1;
    for (v) |*x| x.* /= n;
}

/// 64-lane i8→i32 widen-and-multiply (E013). Used when AVX-512 VNNI
/// is not in the compile target. Integer math is bit-exact.
fn dotI8Widen(a: []const i8, b: []const i8) i32 {
    const lane = 64;
    var acc: @Vector(lane, i32) = @splat(0);
    var i: usize = 0;
    while (i + lane <= a.len) : (i += lane) {
        const va: @Vector(lane, i8) = a[i..][0..lane].*;
        const vb: @Vector(lane, i8) = b[i..][0..lane].*;
        const wide_a: @Vector(lane, i32) = @intCast(va);
        const wide_b: @Vector(lane, i32) = @intCast(vb);
        acc += wide_a * wide_b;
    }
    var sum: i32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += @as(i32, a[i]) * @as(i32, b[i]);
    return sum;
}

/// Vectorized int8 dot product for quantized distance evaluation.
/// On AVX-512 VNNI hosts, one `vpdpbusd` does 64 u8×i8 products into
/// 16 i32 lanes (one zmm acc; E019 dual 64-lane accs were a discard).
/// Signed i8×i8 is recovered by XOR-0x80 on `a` and subtracting 128*sum(b).
pub fn dotI8(a: []const i8, b: []const i8) i32 {
    if (comptime has_avx512_vnni) {
        const I = struct {
            extern fn @"llvm.x86.avx512.vpdpbusd.512"(
                src1: @Vector(16, i32),
                src2: @Vector(16, i32),
                src3: @Vector(16, i32),
            ) @Vector(16, i32);

            fn run(xs: []const i8, ys: []const i8) i32 {
                const lane = 64;
                var acc: @Vector(16, i32) = @splat(0);
                var corr: i32 = 0;
                var i: usize = 0;
                const xor80: @Vector(lane, u8) = @splat(0x80);
                while (i + lane <= xs.len) : (i += lane) {
                    const va: @Vector(lane, i8) = xs[i..][0..lane].*;
                    const vb: @Vector(lane, i8) = ys[i..][0..lane].*;
                    const a_off: @Vector(lane, u8) = @as(@Vector(lane, u8), @bitCast(va)) ^ xor80;
                    acc = @"llvm.x86.avx512.vpdpbusd.512"(acc, @bitCast(a_off), @bitCast(vb));
                    const vb16: @Vector(lane, i16) = @intCast(vb);
                    corr += @reduce(.Add, vb16);
                }
                var sum: i32 = @reduce(.Add, acc) - corr * 128;
                while (i < xs.len) : (i += 1) sum += @as(i32, xs[i]) * @as(i32, ys[i]);
                return sum;
            }
        };
        return I.run(a, b);
    }
    return dotI8Widen(a, b);
}

test "cosine basics" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 1, 0, 0 };
    const c = [_]f32{ 0, 1, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0), cosineDistance(&a, &b), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), cosineDistance(&a, &c), 1e-6);
}

test "dotI8 matches scalar, including signed extremes" {
    try std.testing.expectEqual(@as(i32, 0), dotI8(&.{}, &.{}));
    try std.testing.expectEqual(@as(i32, 2 - 3 + 0 - 127), dotI8(&.{ 1, -1, 0, 127 }, &.{ 2, 3, 9, -1 }));

    const lo64: [64]i8 = @splat(-128);
    const hi64: [64]i8 = @splat(127);
    const want64: i32 = @as(i32, -128) * @as(i32, 127) * 64;
    try std.testing.expectEqual(want64, dotI8(&lo64, &hi64));
    try std.testing.expectEqual(want64, dotI8Widen(&lo64, &hi64));

    const hi67: [67]i8 = @splat(127);
    const lo67: [67]i8 = @splat(-128);
    const want67: i32 = @as(i32, 127) * @as(i32, -128) * 67;
    try std.testing.expectEqual(want67, dotI8(&hi67, &lo67));
    try std.testing.expectEqual(want67, dotI8Widen(&hi67, &lo67));

    var a: [1536]i8 = undefined;
    var b: [1536]i8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);
    const r = prng.random();
    for (&a) |*x| x.* = r.intRangeAtMost(i8, -128, 127);
    for (&b) |*x| x.* = r.intRangeAtMost(i8, -128, 127);
    var want: i32 = 0;
    for (a, b) |x, y| want += @as(i32, x) * @as(i32, y);
    try std.testing.expectEqual(want, dotI8(&a, &b));
    try std.testing.expectEqual(want, dotI8Widen(&a, &b));
}

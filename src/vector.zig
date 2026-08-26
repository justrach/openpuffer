//! Core vector math: cosine distance over f32 vectors.

const std = @import("std");

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

/// Vectorized int8 dot product for quantized distance evaluation.
/// 64 i8 lanes = 512 bits, matching AVX-512 register width. Integer
/// products are still widened to i32 (max |product| = 16129).
pub fn dotI8(a: []const i8, b: []const i8) i32 {
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

/// Suffix bound: `suffix[i] = Σ_{j≥i} |q[j]| * 127`. Length must be `q.len + 1`.
pub fn i8SuffixBound(q: []const i8, suffix: []i32) void {
    std.debug.assert(suffix.len == q.len + 1);
    suffix[q.len] = 0;
    var i = q.len;
    while (i > 0) {
        i -= 1;
        const mag: i32 = @intCast(@abs(q[i]));
        suffix[i] = suffix[i + 1] + mag * 127;
    }
}

/// Same integer math as `dotI8`, but returns `null` when even the remaining
/// worst-case product cannot reach `need` (needed for cosine d < w).
pub fn dotI8Abandon(a: []const i8, b: []const i8, suffix: []const i32, need: f64) ?i32 {
    std.debug.assert(a.len == b.len);
    std.debug.assert(suffix.len >= a.len + 1);
    const lane = 64;
    const check_every: usize = 256;
    var acc: @Vector(lane, i32) = @splat(0);
    var i: usize = 0;
    var since: usize = 0;
    while (i + lane <= a.len) : (i += lane) {
        const va: @Vector(lane, i8) = a[i..][0..lane].*;
        const vb: @Vector(lane, i8) = b[i..][0..lane].*;
        const wide_a: @Vector(lane, i32) = @intCast(va);
        const wide_b: @Vector(lane, i32) = @intCast(vb);
        acc += wide_a * wide_b;
        since += lane;
        if (since >= check_every) {
            since = 0;
            const partial = @reduce(.Add, acc);
            const max_dot = @as(f64, @floatFromInt(partial)) + @as(f64, @floatFromInt(suffix[i + lane]));
            if (max_dot < need) return null;
        }
    }
    var sum: i32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += @as(i32, a[i]) * @as(i32, b[i]);
    return sum;
}

test "dotI8Abandon matches dotI8 when the bound cannot fire" {
    var a: [320]i8 = undefined;
    var b: [320]i8 = undefined;
    for (&a, 0..) |*x, i| x.* = @intCast(@as(i32, @intCast(i % 17)) - 8);
    for (&b, 0..) |*x, i| x.* = @intCast(@as(i32, @intCast(i % 13)) - 6);
    var suf: [321]i32 = undefined;
    i8SuffixBound(&a, &suf);
    const full = dotI8(&a, &b);
    const maybe = dotI8Abandon(&a, &b, &suf, -1.0e300);
    try std.testing.expectEqual(full, maybe.?);
}

test "dotI8Abandon returns null when remaining dims cannot reach need" {
    var a: [320]i8 = undefined;
    var b: [320]i8 = undefined;
    @memset(&a, 1);
    @memset(&b, 1);
    var suf: [321]i32 = undefined;
    i8SuffixBound(&a, &suf);
    try std.testing.expect(dotI8Abandon(&a, &b, &suf, 1.0e12) == null);
}

test "cosine basics" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 1, 0, 0 };
    const c = [_]f32{ 0, 1, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0), cosineDistance(&a, &b), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), cosineDistance(&a, &c), 1e-6);
}

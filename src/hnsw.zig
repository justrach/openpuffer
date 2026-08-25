//! HNSW (Hierarchical Navigable Small World) ANN index, in-memory.
//! Cosine distance; vectors are L2-normalized on insert so
//! cosine distance == 1 - dot, which keeps the inner loop tight.

const std = @import("std");
const vecmath = @import("vector.zig");

pub const Options = struct {
    m: u32 = 16,
    m0_factor: u32 = 2,
    ef_construction: u32 = 100,
    ml: f64 = 1.0 / @log(16.0),
    seed: u64 = 0x9e3779b97f4a7c15,
};

pub const SearchResult = struct { id: u32, distance: f32 };

pub fn Hnsw(comptime D: type) type {
    _ = D;
    return struct {
        const Self = @This();
        pub const Candidate = struct { id: u32, d: f32 };

        allocator: std.mem.Allocator,
        dim: usize,
        opts: Options,
        rng: std.Random.DefaultPrng,

        vectors: std.ArrayList([]f32) = .empty,
        /// int8-quantized copies (symmetric per-vector scale) used to cut
        /// memory traffic during graph traversal at high dimensionality.
        qvecs: std.ArrayList([]i8) = .empty,
        qscales: std.ArrayList(f32) = .empty,
        levels: std.ArrayList(u32) = .empty,
        /// node_links[id] is a per-node list of per-layer neighbor lists.
        node_links: std.ArrayList([]std.ArrayList(u32)) = .empty,
        entry_point: ?u32 = null,
        max_level: u32 = 0,

        pub fn init(allocator: std.mem.Allocator, dim: usize, opts: Options) Self {
            return .{
                .allocator = allocator,
                .dim = dim,
                .opts = opts,
                .rng = std.Random.DefaultPrng.init(opts.seed),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.vectors.items) |v| self.allocator.free(v);
            self.vectors.deinit(self.allocator);
            for (self.qvecs.items) |v| self.allocator.free(v);
            self.qvecs.deinit(self.allocator);
            self.qscales.deinit(self.allocator);
            self.levels.deinit(self.allocator);
            for (self.node_links.items) |layers| {
                for (layers) |*l| l.deinit(self.allocator);
                self.allocator.free(layers);
            }
            self.node_links.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.vectors.items.len;
        }

        fn randomLevel(self: *Self) u32 {
            var r = self.rng.random().float(f64);
            if (r <= 0) r = std.math.floatMin(f64);
            // standard: level = floor(-ln(uniform(0,1)) * mL)
            return @intFromFloat(std.math.floor(-std.math.log(f64, std.math.e, r) * self.opts.ml));
        }

        inline fn distTo(self: *const Self, q: []const f32, id: u32) f32 {
            return vecmath.cosineDistance(q, self.vectors.items[id]);
        }

        inline fn distQ(self: *const Self, qq: []const i8, qs: f32, id: u32) f32 {
            // stored vectors are unit-norm; a ~= aq * scale, so
            // cos(a,b) ~= dot(aq,bq) * qs * scale_b
            const approx = @as(f64, @floatFromInt(vecmath.dotI8(qq, self.qvecs.items[id]))) *
                qs * self.qscales.items[id];
            return @max(0.0, 1.0 - @as(f32, @floatCast(approx)));
        }

        /// Distance contexts let searchLayer run on either exact or quantized math.
        pub const F32Dist = struct {
            s: *Self,
            q: []const f32,
            pub inline fn dist(d: @This(), id: u32) f32 {
                return d.s.distTo(d.q, id);
            }
        };
        pub const QDist = struct {
            s: *Self,
            qq: []const i8,
            qs: f32,
            pub inline fn dist(d: @This(), id: u32) f32 {
                return d.s.distQ(d.qq, d.qs, id);
            }
        };

        /// greedy search on a single layer; `ef` bounds result set size.
        fn searchLayer(
            self: *Self,
            dist_ctx: anytype,
            entry_points: []const Candidate,
            ef: usize,
            layer: u32,
            visited: *std.DynamicBitSetUnmanaged,
            alloc: std.mem.Allocator,
        ) !std.ArrayList(Candidate) {
            // visited state is per-layer: upper-layer expansions must not
            // suppress re-expansion at lower layers (their adjacency differs).
            visited.setRangeValue(.{ .start = 0, .end = visited.capacity() }, false);
            // candidates: min-heap by distance (use PriorityQueue)
            const CandCtx = struct {
                pub fn compare(_: void, a: Candidate, b: Candidate) std.math.Order {
                    return std.math.order(a.d, b.d);
                }
            };

            var candidates = std.PriorityQueue(Candidate, void, CandCtx.compare).initContext({});
            defer candidates.deinit(alloc);
            try candidates.ensureTotalCapacity(alloc, ef + 64);
            // results: max-heap by distance (store negated via reverse order)
            const RevCtx = struct {
                pub fn compare(_: void, a: Candidate, b: Candidate) std.math.Order {
                    return std.math.order(b.d, a.d);
                }
            };
            var results = std.PriorityQueue(Candidate, void, RevCtx.compare).initContext({});
            defer results.deinit(alloc);

            for (entry_points) |c| {
                visited.set(c.id);
                try candidates.push(alloc, c);
                try results.push(alloc, c);
            }

            while (candidates.count() > 0) {
                const c = candidates.pop().?;
                const worst = if (results.count() > 0) results.peek().?.d else c.d;
                if (c.d > worst and results.count() >= ef) break;

                if (layer >= self.node_links.items[c.id].len) {
                    std.debug.print("VIOLATION c.id={d} its_layers={d} layer={d} max_level={d} ep={any} ep_layers={d}\n", .{ c.id, self.node_links.items[c.id].len, layer, self.max_level, self.entry_point, if (self.entry_point) |e| self.node_links.items[e].len else 0 });
                    @panic("hnsw invariant broken");
                }
                const neighbors = self.node_links.items[c.id][layer].items;
                for (neighbors, 0..) |nid, ni| {
                    // one-ahead software prefetch: the next neighbor's int8
                    // vector starts its memory fetch while we dot this one.
                    // Skip already-visited neighbors (wasted prefetches).
                    if (ni + 1 < neighbors.len) {
                        const nxt = neighbors[ni + 1];
                        if (!visited.isSet(nxt)) {
                            @prefetch(self.qvecs.items[nxt].ptr, .{
                                .rw = .read,
                                .locality = 3,
                                .cache = .data,
                            });
                        }
                    }
                    if (visited.isSet(nid)) continue;
                    visited.set(nid);
                    const d = dist_ctx.dist(nid);
                    const w = if (results.count() > 0) results.peek().?.d else std.math.floatMax(f32);
                    if (results.count() < ef or d < w) {
                        try candidates.push(alloc, .{ .id = nid, .d = d });
                        try results.push(alloc, .{ .id = nid, .d = d });
                        if (results.count() > ef) _ = results.pop();
                    }
                }
            }

            var out: std.ArrayList(Candidate) = .empty;
            try out.ensureTotalCapacity(alloc, results.count());
            while (results.pop()) |r| out.appendAssumeCapacity(r);
            std.mem.sort(Candidate, out.items, {}, struct {
                fn cmp(_: void, a: Candidate, b: Candidate) bool {
                    return a.d < b.d;
                }
            }.cmp);
            return out;
        }

        fn selectNeighborsHeuristic(
            self: *Self,
            base: []const Candidate,
            max_m: u32,
            alloc: std.mem.Allocator,
        ) ![]Candidate {
            var selected: std.ArrayList(Candidate) = .empty;
            errdefer selected.deinit(alloc);
            try selected.ensureTotalCapacity(alloc, max_m);
            // keepPruned: fill leftover slots with closest discarded candidates
            var pruned: std.ArrayList(Candidate) = .empty;
            defer pruned.deinit(alloc);
            outer: for (base) |c| {
                if (selected.items.len >= max_m) {
                    try pruned.append(alloc, c);
                    continue;
                }
                for (selected.items) |s| {
                    if (self.distTo(self.vectors.items[c.id], s.id) < c.d) {
                        try pruned.append(alloc, c);
                        continue :outer;
                    }
                }
                selected.appendAssumeCapacity(c);
            }
            var pi: usize = 0;
            while (selected.items.len < max_m and pi < pruned.items.len) : (pi += 1) {
                try selected.append(alloc, pruned.items[pi]);
            }
            return selected.toOwnedSlice(alloc);
        }

        /// Replace the vector at `id` in place (requantize; graph links stay).
        pub fn update(self: *Self, id: u32, vector: []const f32) !void {
            if (id >= self.vectors.items.len) return error.NotFound;
            if (vector.len != self.dim) return error.DimensionMismatch;
            const copy = self.vectors.items[id];
            @memcpy(copy, vector);
            vecmath.normalize(copy);
            var amax: f32 = 0;
            for (copy) |x| amax = @max(amax, @abs(x));
            const scale = amax / 127.0;
            const q8 = self.qvecs.items[id];
            if (scale == 0) {
                @memset(q8, 0);
            } else {
                for (copy, 0..) |x, di| q8[di] = @intFromFloat(std.math.clamp(@round(x / scale), -127.0, 127.0));
            }
            self.qscales.items[id] = scale;
        }

        /// Insert a vector (copies it). Returns its id.
        pub fn insert(self: *Self, vector: []const f32) !u32 {
            if (vector.len != self.dim) return error.DimensionMismatch;
            const alloc = self.allocator;
            const id: u32 = @intCast(self.vectors.items.len);

            const copy = try alloc.alloc(f32, self.dim);
            @memcpy(copy, vector);
            vecmath.normalize(copy);
            try self.vectors.append(alloc, copy);

            // int8-quantized copy for fast traversal
            var amax: f32 = 0;
            for (copy) |x| amax = @max(amax, @abs(x));
            const scale = amax / 127.0;
            const q8 = try alloc.alloc(i8, self.dim);
            if (scale == 0) {
                @memset(q8, 0);
            } else {
                for (copy, 0..) |x, di| q8[di] = @intFromFloat(std.math.clamp(@round(x / scale), -127.0, 127.0));
            }
            try self.qvecs.append(alloc, q8);
            try self.qscales.append(alloc, scale);

            const level = self.randomLevel();
            try self.levels.append(alloc, level);

            const layers = try alloc.alloc(std.ArrayList(u32), level + 1);
            for (layers) |*l| l.* = .empty;
            try self.node_links.append(alloc, layers);

            const max_m0 = self.opts.m * self.opts.m0_factor;

            if (self.entry_point == null) {
                self.entry_point = id;
                self.max_level = level;
                return id;
            }

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, self.vectors.items.len);
            defer visited.deinit(alloc);

            var ep: [1]Candidate = .{.{ .id = self.entry_point.?, .d = self.distTo(copy, self.entry_point.?) }};
            var cur = self.max_level;
            while (cur > level) : (cur -= 1) {
                const res = try self.searchLayer(F32Dist{ .s = self, .q = copy }, &ep, 1, cur, &visited, alloc);
                var res_mut = res;
                defer res_mut.deinit(alloc);
                ep[0] = res_mut.items[0];
            }

            var eps: std.ArrayList(Candidate) = .empty;
            defer eps.deinit(alloc);
            try eps.append(alloc, ep[0]);

            const top_connect = @min(level, self.max_level);

            var l: i64 = @intCast(top_connect);
            while (l >= 0) : (l -= 1) {
                const layer: u32 = @intCast(l);
                const max_m: u32 = if (layer == 0) max_m0 else self.opts.m;
                var found = try self.searchLayer(F32Dist{ .s = self, .q = copy }, eps.items, self.opts.ef_construction, layer, &visited, alloc);
                defer found.deinit(alloc);

                const selected = try self.selectNeighborsHeuristic(found.items, max_m, alloc);
                defer alloc.free(selected);

                for (selected) |s| {
                    try self.node_links.items[id][layer].append(alloc, s.id);
                    const back = &self.node_links.items[s.id][layer];
                    try back.append(alloc, id);
                    if (back.items.len > max_m) {
                        // shrink: keep closest to s.id
                        var cands: std.ArrayList(Candidate) = .empty;
                        defer cands.deinit(alloc);
                        try cands.ensureTotalCapacity(alloc, back.items.len);
                        for (back.items) |n| {
                            cands.appendAssumeCapacity(.{ .id = n, .d = self.distTo(self.vectors.items[s.id], n) });
                        }
                        std.mem.sort(Candidate, cands.items, {}, struct {
                            fn cmp(_: void, a: Candidate, b: Candidate) bool {
                                return a.d < b.d;
                            }
                        }.cmp);
                        const pruned = try self.selectNeighborsHeuristic(cands.items, max_m, alloc);
                        defer alloc.free(pruned);
                        back.clearRetainingCapacity();
                        for (pruned) |p| try back.append(alloc, p.id);
                    }
                }
                eps.clearRetainingCapacity();
                for (found.items) |f| try eps.append(alloc, f);
            }

            if (level > self.max_level) {
                self.max_level = level;
                self.entry_point = id;
            }
            return id;
        }

        /// k-ANN search: graph traversal runs on int8 distances (4x less memory
        /// traffic at high dim), then the top `rerank_mult * k` candidates are
        /// reranked with exact f32 cosine so reported results keep full precision.
        pub fn search(
            self: *Self,
            query: []const f32,
            k: usize,
            ef_search: u32,
            alloc: std.mem.Allocator,
        ) ![]SearchResult {
            return self.searchAdvanced(query, k, ef_search, 4, alloc);
        }

        /// `rerank_mult` exposes the recall/latency knob; 1 disables rerank.
        pub fn searchAdvanced(
            self: *Self,
            query: []const f32,
            k: usize,
            ef_search: u32,
            rerank_mult: usize,
            alloc: std.mem.Allocator,
        ) ![]SearchResult {
            if (query.len != self.dim) return error.DimensionMismatch;
            const ep_id = self.entry_point orelse return alloc.alloc(SearchResult, 0);
            var norm_buf: [512]f32 = undefined;
            var heap_nq: ?[]f32 = null;
            defer if (heap_nq) |h| alloc.free(h);
            const nq: []const f32 = if (query.len <= norm_buf.len) blk: {
                @memcpy(norm_buf[0..query.len], query);
                vecmath.normalize(norm_buf[0..query.len]);
                break :blk norm_buf[0..query.len];
            } else blk: {
                heap_nq = try alloc.alloc(f32, query.len);
                @memcpy(heap_nq.?, query);
                vecmath.normalize(heap_nq.?);
                break :blk heap_nq.?;
            };

            // quantize the normalized query once; reuse the amax scan for qs
            var qbuf: [512]i8 = undefined;
            var amax: f32 = 0;
            for (nq) |x| amax = @max(amax, @abs(x));
            const scale = amax / 127.0;
            const qs: f32 = scale;
            var heap_qq: ?[]i8 = null;
            defer if (heap_qq) |h| alloc.free(h);
            const qq: []const i8 = blk: {
                const buf = if (nq.len <= qbuf.len) qbuf[0..nq.len] else blk2: {
                    heap_qq = try alloc.alloc(i8, nq.len);
                    break :blk2 heap_qq.?;
                };
                if (scale == 0) {
                    @memset(buf, 0);
                } else {
                    for (nq, 0..) |x, i| buf[i] = @intFromFloat(std.math.clamp(@round(x / scale), -127.0, 127.0));
                }
                break :blk buf;
            };
            const ctx = QDist{ .s = self, .qq = qq, .qs = qs };

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, self.vectors.items.len);
            defer visited.deinit(alloc);

            var ep: [1]Candidate = .{.{ .id = ep_id, .d = ctx.dist(ep_id) }};
            var cur = self.max_level;
            while (cur > 0) : (cur -= 1) {
                var res = try self.searchLayer(ctx, &ep, 1, cur, &visited, alloc);
                defer res.deinit(alloc);
                ep[0] = res.items[0];
            }

            var res = try self.searchLayer(ctx, &ep, @max(ef_search, k), 0, &visited, alloc);
            defer res.deinit(alloc);

            // exact rerank of the best candidates (already sorted by approx d)
            const rerank_n = @min(res.items.len, @max(k * rerank_mult, k));
            const Rerank = struct { id: u32, d: f32 };
            var rr: std.ArrayList(Rerank) = .empty;
            defer rr.deinit(alloc);
            try rr.ensureTotalCapacity(alloc, rerank_n);
            for (res.items[0..rerank_n]) |c| {
                rr.appendAssumeCapacity(.{ .id = c.id, .d = self.distTo(nq, c.id) });
            }
            std.mem.sort(Rerank, rr.items, {}, struct {
                fn lt(_: void, a: Rerank, b: Rerank) bool {
                    return a.d < b.d;
                }
            }.lt);

            const out = try alloc.alloc(SearchResult, @min(k, rr.items.len));
            for (out, 0..) |*o, i| o.* = .{ .id = rr.items[i].id, .distance = rr.items[i].d };
            return out;
        }

        pub const GRAPH_MAGIC: u32 = 0x57534E48; // 'HNSW' le
        pub const GRAPH_VERSION: u32 = 1;

        pub fn serializedSize(self: *const Self) usize {
            var n: usize = 48; // header
            const count = self.vectors.items.len;
            n += count * 4; // levels
            n += count * 4; // qscales
            n += count * self.dim * 4; // f32
            n += count * self.dim; // i8
            for (self.node_links.items) |layers| {
                n += 4; // n_layers
                for (layers) |l| n += 4 + l.items.len * 4;
            }
            return n;
        }

        /// Little-endian graph blob: header + levels + qscales + f32 + i8 + links.
        /// Does not include external document ids.
        pub fn serialize(self: *const Self, alloc: std.mem.Allocator) ![]u8 {
            const buf = try alloc.alloc(u8, self.serializedSize());
            errdefer alloc.free(buf);
            var i: usize = 0;
            const wr = struct {
                fn u32le(b: []u8, off: *usize, v: u32) void {
                    std.mem.writeInt(u32, b[off.* ..][0..4], v, .little);
                    off.* += 4;
                }
                fn u64le(b: []u8, off: *usize, v: u64) void {
                    std.mem.writeInt(u64, b[off.* ..][0..8], v, .little);
                    off.* += 8;
                }
            };
            wr.u32le(buf, &i, GRAPH_MAGIC);
            wr.u32le(buf, &i, GRAPH_VERSION);
            wr.u64le(buf, &i, self.dim);
            wr.u64le(buf, &i, self.vectors.items.len);
            wr.u32le(buf, &i, self.entry_point orelse std.math.maxInt(u32));
            wr.u32le(buf, &i, self.max_level);
            wr.u32le(buf, &i, self.opts.m);
            wr.u32le(buf, &i, self.opts.m0_factor);
            wr.u32le(buf, &i, self.opts.ef_construction);
            wr.u32le(buf, &i, 1); // flags: has_qvecs

            for (self.levels.items) |lv| wr.u32le(buf, &i, lv);
            for (self.qscales.items) |s| {
                const bits: u32 = @bitCast(s);
                wr.u32le(buf, &i, bits);
            }
            for (self.vectors.items) |v| {
                const bytes = std.mem.sliceAsBytes(v);
                @memcpy(buf[i..][0..bytes.len], bytes);
                i += bytes.len;
            }
            for (self.qvecs.items) |v| {
                @memcpy(buf[i..][0..v.len], std.mem.sliceAsBytes(v));
                i += v.len;
            }
            for (self.node_links.items) |layers| {
                wr.u32le(buf, &i, @intCast(layers.len));
                for (layers) |l| {
                    wr.u32le(buf, &i, @intCast(l.items.len));
                    for (l.items) |nid| wr.u32le(buf, &i, nid);
                }
            }
            if (i != buf.len) return error.SerializeSizeMismatch;
            return buf;
        }

        /// Populate an empty index from `serialize` output. No HNSW rebuild.
        pub fn load(self: *Self, bytes: []const u8) !void {
            if (self.vectors.items.len != 0) return error.NotEmpty;
            if (bytes.len < 48) return error.Truncated;
            var i: usize = 0;
            const rd = struct {
                fn u32le(b: []const u8, off: *usize) !u32 {
                    if (off.* + 4 > b.len) return error.Truncated;
                    const v = std.mem.readInt(u32, b[off.* ..][0..4], .little);
                    off.* += 4;
                    return v;
                }
                fn u64le(b: []const u8, off: *usize) !u64 {
                    if (off.* + 8 > b.len) return error.Truncated;
                    const v = std.mem.readInt(u64, b[off.* ..][0..8], .little);
                    off.* += 8;
                    return v;
                }
            };
            if (try rd.u32le(bytes, &i) != GRAPH_MAGIC) return error.BadMagic;
            if (try rd.u32le(bytes, &i) != GRAPH_VERSION) return error.UnsupportedVersion;
            const dim: usize = @intCast(try rd.u64le(bytes, &i));
            const n: usize = @intCast(try rd.u64le(bytes, &i));
            const ep_raw = try rd.u32le(bytes, &i);
            const max_level = try rd.u32le(bytes, &i);
            const m = try rd.u32le(bytes, &i);
            const m0 = try rd.u32le(bytes, &i);
            const efc = try rd.u32le(bytes, &i);
            const flags = try rd.u32le(bytes, &i);
            _ = flags;
            if (self.dim == 0) self.dim = dim;
            if (self.dim != dim) return error.DimensionMismatch;
            self.opts.m = m;
            self.opts.m0_factor = m0;
            self.opts.ef_construction = efc;
            self.max_level = max_level;
            self.entry_point = if (ep_raw == std.math.maxInt(u32)) null else ep_raw;

            const alloc = self.allocator;
            try self.levels.ensureTotalCapacity(alloc, n);
            try self.qscales.ensureTotalCapacity(alloc, n);
            try self.vectors.ensureTotalCapacity(alloc, n);
            try self.qvecs.ensureTotalCapacity(alloc, n);
            try self.node_links.ensureTotalCapacity(alloc, n);

            for (0..n) |_| self.levels.appendAssumeCapacity(try rd.u32le(bytes, &i));
            for (0..n) |_| {
                const bits = try rd.u32le(bytes, &i);
                self.qscales.appendAssumeCapacity(@bitCast(bits));
            }
            for (0..n) |_| {
                const nbytes = dim * @sizeOf(f32);
                if (i + nbytes > bytes.len) return error.Truncated;
                const copy = try alloc.alloc(f32, dim);
                @memcpy(std.mem.sliceAsBytes(copy), bytes[i..][0..nbytes]);
                i += nbytes;
                self.vectors.appendAssumeCapacity(copy);
            }
            for (0..n) |_| {
                if (i + dim > bytes.len) return error.Truncated;
                const copy = try alloc.alloc(i8, dim);
                @memcpy(std.mem.sliceAsBytes(copy), bytes[i..][0..dim]);
                i += dim;
                self.qvecs.appendAssumeCapacity(copy);
            }
            for (0..n) |_| {
                const n_layers: usize = @intCast(try rd.u32le(bytes, &i));
                const layers = try alloc.alloc(std.ArrayList(u32), n_layers);
                for (layers) |*l| l.* = .empty;
                errdefer {
                    for (layers) |*l| l.deinit(alloc);
                    alloc.free(layers);
                }
                for (layers) |*l| {
                    const deg: usize = @intCast(try rd.u32le(bytes, &i));
                    try l.ensureTotalCapacity(alloc, deg);
                    var k: usize = 0;
                    while (k < deg) : (k += 1) {
                        l.appendAssumeCapacity(try rd.u32le(bytes, &i));
                    }
                }
                self.node_links.appendAssumeCapacity(layers);
            }
            if (i != bytes.len) return error.TrailingBytes;
        }

    };
}

test "hnsw finds planted neighbor" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 16;
    var rng = std.Random.DefaultPrng.init(42);
    var index = Hnsw(void).init(alloc, dim, .{});
    defer index.deinit();

    var v: [dim]f32 = undefined;
    for (0..500) |_| {
        for (&v) |*x| x.* = rng.random().floatNorm(f32);
        _ = try index.insert(&v);
    }
    // plant a distinctive vector
    @memset(&v, 0);
    v[3] = 1;
    const target = try index.insert(&v);

    const results = try index.search(&v, 1, 64, alloc);
    try std.testing.expectEqual(target, results[0].id);
}

test "layer-0 connectivity and recall on clustered data" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 1536;
    const n = 400;
    const clusters = 8;
    var rng = std.Random.DefaultPrng.init(9);

    var centers: [clusters][dim]f32 = undefined;
    for (&centers) |*c| {
        for (c) |*x| x.* = rng.random().floatNorm(f32);
        vecmath.normalize(c);
    }

    var index = Hnsw(void).init(alloc, dim, .{});
    defer index.deinit();
    var v: [dim]f32 = undefined;
    for (0..n) |i| {
        const cl = i % clusters;
        for (&v, 0..) |_, d| v[d] = centers[cl][d] + 0.05 * rng.random().floatNorm(f32);
        _ = try index.insert(&v);
    }

    // BFS reachability from entry point at layer 0
    var visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, n);
    defer visited.deinit(alloc);
    var queue: std.ArrayList(u32) = .empty;
    try queue.append(alloc, index.entry_point.?);
    visited.set(index.entry_point.?);
    var head: usize = 0;
    while (head < queue.items.len) : (head += 1) {
        const id = queue.items[head];
        for (index.node_links.items[id][0].items) |nb| {
            if (!visited.isSet(nb)) {
                visited.set(nb);
                try queue.append(alloc, nb);
            }
        }
    }
    var total: usize = 0;
    var it = visited.iterator(.{});
    while (it.next()) |_| total += 1;
    std.debug.print("reachable {d}/{d} from entry\n", .{ total, n });

    var q: [dim]f32 = undefined;
    for (&q, 0..) |_, d| q[d] = centers[3][d] + 0.05 * rng.random().floatNorm(f32);
    const res = try index.search(&q, 10, 200, alloc);
    std.debug.print("returned {d} results, best d={d:.4}\n", .{ res.len, res[0].distance });

    // brute force top-10 over the same stored vectors
    const GT = struct { id: u32, d: f32 };
    var all: std.ArrayList(GT) = .empty;
    const nq = try alloc.dupe(f32, &q);
    vecmath.normalize(nq);
    for (index.vectors.items, 0..) |row, idx| {
        try all.append(alloc, .{ .id = @intCast(idx), .d = vecmath.cosineDistance(nq, row) });
    }
    std.mem.sort(GT, all.items, {}, struct {
        fn lt(_: void, a: GT, b: GT) bool {
            return a.d < b.d;
        }
    }.lt);
    var hits: usize = 0;
    for (res) |r| {
        for (all.items[0..10]) |g| {
            if (g.id == r.id) {
                hits += 1;
                break;
            }
        }
    }
    std.debug.print("recall@10 vs brute force: {d:.3} (ann best {d:.4} vs gt best {d:.4} id {d})\n", .{
        @as(f64, @floatFromInt(hits)) / 10.0, res[0].distance, all.items[0].d, all.items[0].id,
    });
}

test "hnsw update replaces vector without growing the index" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();
    var index = Hnsw(void).init(alloc, 4, .{});
    defer index.deinit();
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    const id = try index.insert(&a);
    _ = try index.insert(&b);
    try std.testing.expectEqual(@as(usize, 2), index.len());
    const flipped = [_]f32{ 0, 0, 0, 1 };
    try index.update(id, &flipped);
    try std.testing.expectEqual(@as(usize, 2), index.len());
    const res = try index.search(&flipped, 1, 8, alloc);
    try std.testing.expectEqual(id, res[0].id);
}

test "hnsw serialize/load preserves neighbors and search" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 32;
    var rng = std.Random.DefaultPrng.init(7);
    var index = Hnsw(void).init(alloc, dim, .{});
    defer index.deinit();

    var v: [dim]f32 = undefined;
    for (0..80) |_| {
        for (&v) |*x| x.* = rng.random().floatNorm(f32);
        _ = try index.insert(&v);
    }
    @memset(&v, 0);
    v[1] = 1;
    const planted = try index.insert(&v);
    const before = try index.search(&v, 5, 64, alloc);

    const blob = try index.serialize(alloc);
    defer alloc.free(blob);

    var loaded = Hnsw(void).init(alloc, dim, .{});
    defer loaded.deinit();
    try loaded.load(blob);

    try std.testing.expectEqual(index.len(), loaded.len());
    try std.testing.expectEqual(index.entry_point, loaded.entry_point);
    try std.testing.expectEqual(index.max_level, loaded.max_level);
    for (index.node_links.items, loaded.node_links.items) |a, b| {
        try std.testing.expectEqual(a.len, b.len);
        for (a, b) |al, bl| {
            try std.testing.expectEqualSlices(u32, al.items, bl.items);
        }
    }
    const after = try loaded.search(&v, 5, 64, alloc);
    try std.testing.expectEqual(planted, after[0].id);
    try std.testing.expectEqual(before[0].id, after[0].id);
    try std.testing.expectEqual(before.len, after.len);
    for (before, after) |b, a| {
        try std.testing.expectEqual(b.id, a.id);
        try std.testing.expectApproxEqAbs(b.distance, a.distance, 1e-6);
    }
}

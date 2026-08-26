//! HNSW (Hierarchical Navigable Small World) ANN index, in-memory.
//! Cosine distance; vectors are L2-normalized on insert so
//! cosine distance == 1 - dot, which keeps the inner loop tight.
//!
//! Memory layout is flat / mmap-shaped (one slab per array, not one malloc
//! per vector or per neighbor list). On-disk next step is append-only
//! segments: RecordID → {segment, offset, generation}. This file does not
//! implement NVMe/SPDK/ZNS; see experiments/log.md.

const std = @import("std");
const vecmath = @import("vector.zig");

pub const Options = struct {
    m: u32 = 16,
    m0_factor: u32 = 2,
    ef_construction: u32 = 100,
    ml: f64 = 1.0 / @log(16.0),
    seed: u64 = 0x9e3779b97f4a7c15,
    /// Keep a packed f32 copy of every vector for exact rerank / `update`.
    /// Default true so the scored 20k bench stays comparable. Set false
    /// (`--no-f32`) to drop ~4×dim bytes per row; search then uses int8 only.
    store_f32: bool = true,
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

        /// Packed f32 payload, stride `dim`. Replaces ArrayList([]f32).
        vectors_flat: std.ArrayList(f32) = .empty,
        /// Packed int8 payload, stride `dim`. Replaces ArrayList([]i8).
        qvecs_flat: std.ArrayList(i8) = .empty,
        qscales: std.ArrayList(f32) = .empty,
        levels: std.ArrayList(u32) = .empty,

        /// Layer-0 packed adjacency (CSR with fixed slack).
        /// l0_nbrs stride = m0()+1 so a splice can temporarily exceed M.
        l0_deg: std.ArrayList(u16) = .empty,
        l0_nbrs: std.ArrayList(u32) = .empty,

        /// Higher layers (1..level) packed the same way.
        /// hi_start[id] indexes the first higher-layer slot in hi_deg / hi_nbrs.
        hi_start: std.ArrayList(u32) = .empty,
        hi_deg: std.ArrayList(u16) = .empty,
        hi_nbrs: std.ArrayList(u32) = .empty,

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
            self.vectors_flat.deinit(self.allocator);
            self.qvecs_flat.deinit(self.allocator);
            self.qscales.deinit(self.allocator);
            self.levels.deinit(self.allocator);
            self.l0_deg.deinit(self.allocator);
            self.l0_nbrs.deinit(self.allocator);
            self.hi_start.deinit(self.allocator);
            self.hi_deg.deinit(self.allocator);
            self.hi_nbrs.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.qscales.items.len;
        }

        /// True when this index retains the f32 slab (exact rerank / snapshot f32).
        pub fn hasStoredF32(self: *const Self) bool {
            return self.opts.store_f32;
        }

        inline fn m0(self: *const Self) usize {
            return @as(usize, self.opts.m) * @as(usize, self.opts.m0_factor);
        }

        inline fn l0Stride(self: *const Self) usize {
            return self.m0() + 1;
        }

        inline fn hiStride(self: *const Self) usize {
            return @as(usize, self.opts.m) + 1;
        }

        /// Mutable f32 row. Server snapshot/WAL and `update` use this.
        /// Empty when `store_f32=false` (callers must check `hasStoredF32`).
        pub fn vector(self: *Self, id: u32) []f32 {
            if (!self.opts.store_f32 or self.vectors_flat.items.len == 0) return &.{};
            const off = @as(usize, id) * self.dim;
            return self.vectors_flat.items[off..][0..self.dim];
        }

        pub fn vectorConst(self: *const Self, id: u32) []const f32 {
            if (!self.opts.store_f32 or self.vectors_flat.items.len == 0) return &.{};
            const off = @as(usize, id) * self.dim;
            return self.vectors_flat.items[off..][0..self.dim];
        }

        pub fn qvec(self: *Self, id: u32) []i8 {
            const off = @as(usize, id) * self.dim;
            return self.qvecs_flat.items[off..][0..self.dim];
        }

        pub fn qvecConst(self: *const Self, id: u32) []const i8 {
            const off = @as(usize, id) * self.dim;
            return self.qvecs_flat.items[off..][0..self.dim];
        }

        pub fn layerCount(self: *const Self, id: u32) u32 {
            return self.levels.items[id] + 1;
        }

        pub fn neighbors(self: *const Self, id: u32, layer: u32) []const u32 {
            const deg = self.degree(id, layer);
            return self.neighborSlotsConst(id, layer)[0..deg];
        }

        fn degree(self: *const Self, id: u32, layer: u32) usize {
            if (layer == 0) return self.l0_deg.items[id];
            const slot = self.hi_start.items[id] + (layer - 1);
            return self.hi_deg.items[slot];
        }

        fn neighborSlotsConst(self: *const Self, id: u32, layer: u32) []const u32 {
            if (layer == 0) {
                const stride = self.l0Stride();
                const off = @as(usize, id) * stride;
                return self.l0_nbrs.items[off..][0..stride];
            }
            const slot = self.hi_start.items[id] + (layer - 1);
            const stride = self.hiStride();
            const off = slot * stride;
            return self.hi_nbrs.items[off..][0..stride];
        }

        fn neighborSlots(self: *Self, id: u32, layer: u32) []u32 {
            if (layer == 0) {
                const stride = self.l0Stride();
                const off = @as(usize, id) * stride;
                return self.l0_nbrs.items[off..][0..stride];
            }
            const slot = self.hi_start.items[id] + (layer - 1);
            const stride = self.hiStride();
            const off = slot * stride;
            return self.hi_nbrs.items[off..][0..stride];
        }

        fn setDegree(self: *Self, id: u32, layer: u32, deg: u16) void {
            if (layer == 0) {
                self.l0_deg.items[id] = deg;
            } else {
                const slot = self.hi_start.items[id] + (layer - 1);
                self.hi_deg.items[slot] = deg;
            }
        }

        fn pushNeighbor(self: *Self, id: u32, layer: u32, nid: u32) void {
            const deg = self.degree(id, layer);
            self.neighborSlots(id, layer)[deg] = nid;
            self.setDegree(id, layer, @intCast(deg + 1));
        }

        fn replaceNeighbors(self: *Self, id: u32, layer: u32, ids: []const u32) void {
            const slots = self.neighborSlots(id, layer);
            @memcpy(slots[0..ids.len], ids);
            self.setDegree(id, layer, @intCast(ids.len));
        }

        fn randomLevel(self: *Self) u32 {
            var r = self.rng.random().float(f64);
            if (r <= 0) r = std.math.floatMin(f64);
            // standard: level = floor(-ln(uniform(0,1)) * mL)
            return @intFromFloat(std.math.floor(-std.math.log(f64, std.math.e, r) * self.opts.ml));
        }

        pub fn nextLevel(self: *Self) u32 {
            return self.randomLevel();
        }

        inline fn distTo(self: *const Self, q: []const f32, id: u32) f32 {
            return vecmath.cosineDistance(q, self.vectorConst(id));
        }

        /// Pairwise distance used while splicing the graph. Exact when f32
        /// is stored; otherwise the same int8 score search already uses.
        inline fn distPair(self: *const Self, src: u32, dst: u32) f32 {
            if (self.opts.store_f32) return self.distTo(self.vectorConst(src), dst);
            return self.distQ(self.qvecConst(src), self.qscales.items[src], dst);
        }

        inline fn distQ(self: *const Self, qq: []const i8, qs: f32, id: u32) f32 {
            // stored vectors are unit-norm; a ~= aq * scale, so
            // cos(a,b) ~= dot(aq,bq) * qs * scale_b
            const approx = @as(f64, @floatFromInt(vecmath.dotI8(qq, self.qvecConst(id)))) *
                qs * self.qscales.items[id];
            return @max(0.0, 1.0 - @as(f32, @floatCast(approx)));
        }

        /// Distance contexts let searchLayer run on either exact or quantized math.
        pub const F32Dist = struct {
            s: *const Self,
            q: []const f32,
            pub inline fn dist(d: @This(), id: u32) f32 {
                return d.s.distTo(d.q, id);
            }
        };
        pub const QDist = struct {
            s: *const Self,
            qq: []const i8,
            qs: f32,
            pub inline fn dist(d: @This(), id: u32) f32 {
                return d.s.distQ(d.qq, d.qs, id);
            }
        };

        /// greedy search on a single layer; `ef` bounds result set size.
        fn searchLayer(
            self: *const Self,
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

                if (layer >= self.layerCount(c.id)) {
                    std.debug.print("VIOLATION c.id={d} its_layers={d} layer={d} max_level={d} ep={any} ep_layers={d}\n", .{ c.id, self.layerCount(c.id), layer, self.max_level, self.entry_point, if (self.entry_point) |e| self.layerCount(e) else 0 });
                    @panic("hnsw invariant broken");
                }
                const nbrs = self.neighbors(c.id, layer);
                for (nbrs, 0..) |nid, ni| {
                    // one-ahead software prefetch: the next neighbor's int8
                    // vector starts its memory fetch while we dot this one.
                    // Skip already-visited neighbors (wasted prefetches).
                    if (ni + 1 < nbrs.len) {
                        const nxt = nbrs[ni + 1];
                        if (!visited.isSet(nxt)) {
                            @prefetch(self.qvecConst(nxt).ptr, .{
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
            self: *const Self,
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
                    if (self.distPair(c.id, s.id) < c.d) {
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
        pub fn update(self: *Self, id: u32, vector_in: []const f32) !void {
            if (id >= self.len()) return error.NotFound;
            if (vector_in.len != self.dim) return error.DimensionMismatch;
            var stack: [2048]f32 = undefined;
            var heap_tmp: ?[]f32 = null;
            defer if (heap_tmp) |h| self.allocator.free(h);
            const copy: []f32 = if (self.opts.store_f32) self.vector(id) else if (self.dim <= stack.len)
                stack[0..self.dim]
            else blk: {
                heap_tmp = try self.allocator.alloc(f32, self.dim);
                break :blk heap_tmp.?;
            };
            @memcpy(copy, vector_in);
            vecmath.normalize(copy);
            var amax: f32 = 0;
            for (copy) |x| amax = @max(amax, @abs(x));
            const scale = amax / 127.0;
            const q8 = self.qvec(id);
            if (scale == 0) {
                @memset(q8, 0);
            } else {
                for (copy, 0..) |x, di| q8[di] = @intFromFloat(std.math.clamp(@round(x / scale), -127.0, 127.0));
            }
            self.qscales.items[id] = scale;
        }

        /// Neighbor picks for one insert. `copy`/`q8` are owned by the index
        /// allocator; `layers` slices are owned by `scratch`.
        pub const InsertPlan = struct {
            copy: []f32,
            q8: []i8,
            scale: f32,
            level: u32,
            /// layers[i] = selected neighbor ids at graph layer i
            layers: [][]u32,
            scratch: std.mem.Allocator,

            pub fn discard(self: InsertPlan, index_alloc: std.mem.Allocator) void {
                index_alloc.free(self.copy);
                index_alloc.free(self.q8);
                for (self.layers) |ids| self.scratch.free(ids);
                self.scratch.free(self.layers);
            }
        };

        fn quantizeOwned(self: *const Self, vector_in: []const f32) !struct { copy: []f32, q8: []i8, scale: f32 } {
            const alloc = self.allocator;
            const copy = try alloc.alloc(f32, self.dim);
            errdefer alloc.free(copy);
            @memcpy(copy, vector_in);
            vecmath.normalize(copy);
            var amax: f32 = 0;
            for (copy) |x| amax = @max(amax, @abs(x));
            const scale = amax / 127.0;
            const q8 = try alloc.alloc(i8, self.dim);
            errdefer alloc.free(q8);
            if (scale == 0) {
                @memset(q8, 0);
            } else {
                for (copy, 0..) |x, di| q8[di] = @intFromFloat(std.math.clamp(@round(x / scale), -127.0, 127.0));
            }
            return .{ .copy = copy, .q8 = q8, .scale = scale };
        }

        fn appendRecord(self: *Self, copy: []const f32, q8: []const i8, scale: f32, level: u32) !u32 {
            const alloc = self.allocator;
            const id: u32 = @intCast(self.len());
            if (self.opts.store_f32) {
                try self.vectors_flat.appendSlice(alloc, copy);
            }
            try self.qvecs_flat.appendSlice(alloc, q8);
            try self.qscales.append(alloc, scale);
            try self.levels.append(alloc, level);
            try self.l0_deg.append(alloc, 0);
            const l0_pad = try self.l0_nbrs.addManyAsSlice(alloc, self.l0Stride());
            @memset(l0_pad, 0);
            try self.hi_start.append(alloc, @intCast(self.hi_deg.items.len));
            if (level > 0) {
                const slots: usize = level;
                const degs = try self.hi_deg.addManyAsSlice(alloc, slots);
                @memset(degs, 0);
                const nbrs = try self.hi_nbrs.addManyAsSlice(alloc, slots * self.hiStride());
                @memset(nbrs, 0);
            }
            return id;
        }

        /// Read-only neighbor search for a new node. Caller must hold a shared
        /// (or exclusive) lock so the graph arrays are stable.
        pub fn planInsert(self: *const Self, vector_in: []const f32, level: u32, scratch: std.mem.Allocator) !InsertPlan {
            if (vector_in.len != self.dim) return error.DimensionMismatch;
            const q = try self.quantizeOwned(vector_in);
            errdefer {
                self.allocator.free(q.copy);
                self.allocator.free(q.q8);
            }
            const n_layers: usize = @as(usize, @min(level, self.max_level)) + 1;
            const layers = try scratch.alloc([]u32, n_layers);
            errdefer scratch.free(layers);
            for (layers) |*slot| slot.* = &.{};

            if (self.entry_point == null) {
                return .{ .copy = q.copy, .q8 = q.q8, .scale = q.scale, .level = level, .layers = layers, .scratch = scratch };
            }

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(scratch, self.len());
            defer visited.deinit(scratch);
            const qctx = QDist{ .s = self, .qq = q.q8, .qs = q.scale };
            var ep: [1]Candidate = .{.{ .id = self.entry_point.?, .d = qctx.dist(self.entry_point.?) }};
            var cur = self.max_level;
            while (cur > level) : (cur -= 1) {
                var res = try self.searchLayer(qctx, &ep, 1, cur, &visited, scratch);
                defer res.deinit(scratch);
                ep[0] = res.items[0];
            }
            var eps: std.ArrayList(Candidate) = .empty;
            defer eps.deinit(scratch);
            try eps.append(scratch, ep[0]);
            const max_m0: u32 = @intCast(self.m0());
            var li: i64 = @intCast(n_layers - 1);
            while (li >= 0) : (li -= 1) {
                const layer: u32 = @intCast(li);
                const max_m: u32 = if (layer == 0) max_m0 else self.opts.m;
                var found = try self.searchLayer(qctx, eps.items, self.opts.ef_construction, layer, &visited, scratch);
                defer found.deinit(scratch);
                const selected = try self.selectNeighborsHeuristic(found.items, max_m, scratch);
                defer scratch.free(selected);
                const ids = try scratch.alloc(u32, selected.len);
                for (selected, ids) |s, *out| out.* = s.id;
                layers[layer] = ids;
                eps.clearRetainingCapacity();
                for (found.items) |f| try eps.append(scratch, f);
            }
            return .{ .copy = q.copy, .q8 = q.q8, .scale = q.scale, .level = level, .layers = layers, .scratch = scratch };
        }

        /// Splice a planned node into the graph. Caller must hold exclusive lock.
        /// Copies `plan.copy` / `plan.q8` into the flat pools and frees them.
        pub fn commitInsert(self: *Self, plan: InsertPlan) !u32 {
            const alloc = self.allocator;
            const id = try self.appendRecord(plan.copy, plan.q8, plan.scale, plan.level);
            alloc.free(plan.copy);
            alloc.free(plan.q8);

            const max_m0: u32 = @intCast(self.m0());
            const connect_n = @min(plan.layers.len, self.layerCount(id));
            var layer: u32 = 0;
            while (layer < connect_n) : (layer += 1) {
                const max_m: u32 = if (layer == 0) max_m0 else self.opts.m;
                for (plan.layers[layer]) |nid| {
                    if (nid >= self.len() or layer >= self.layerCount(nid)) continue;
                    self.pushNeighbor(id, layer, nid);
                    self.pushNeighbor(nid, layer, id);
                    if (self.degree(nid, layer) > max_m) {
                        var cands: std.ArrayList(Candidate) = .empty;
                        defer cands.deinit(alloc);
                        const back = self.neighbors(nid, layer);
                        try cands.ensureTotalCapacity(alloc, back.len);
                        for (back) |n| {
                            cands.appendAssumeCapacity(.{ .id = n, .d = self.distPair(nid, n) });
                        }
                        std.mem.sort(Candidate, cands.items, {}, struct {
                            fn cmp(_: void, a: Candidate, b: Candidate) bool {
                                return a.d < b.d;
                            }
                        }.cmp);
                        const pruned = try self.selectNeighborsHeuristic(cands.items, max_m, alloc);
                        defer alloc.free(pruned);
                        var ids: std.ArrayList(u32) = .empty;
                        defer ids.deinit(alloc);
                        try ids.ensureTotalCapacity(alloc, pruned.len);
                        for (pruned) |p| ids.appendAssumeCapacity(p.id);
                        self.replaceNeighbors(nid, layer, ids.items);
                    }
                }
            }
            for (plan.layers) |ids| plan.scratch.free(ids);
            plan.scratch.free(plan.layers);

            if (self.entry_point == null or plan.level > self.max_level) {
                self.max_level = plan.level;
                self.entry_point = id;
            }
            return id;
        }

        /// Insert a vector (copies it). Returns its id.
        pub fn insert(self: *Self, vector_in: []const f32) !u32 {
            if (vector_in.len != self.dim) return error.DimensionMismatch;
            const level = self.randomLevel();
            const plan = try self.planInsert(vector_in, level, self.allocator);
            return self.commitInsert(plan);
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

        /// `rerank_mult` is the recall/latency knob (CLI `--rerank-mult`, default 4).
        /// 1 disables rerank (int8 order only). Raising it only helps if the true
        /// neighbors are already in the ef-sized candidate list.
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

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, self.len());
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

            // exact rerank only when the f32 slab is present. Without it,
            // reported order is the int8 traversal order (still valid ANN).
            if (!self.opts.store_f32) {
                const out_n = @min(k, res.items.len);
                const out = try alloc.alloc(SearchResult, out_n);
                for (out, 0..) |*o, i| o.* = .{ .id = res.items[i].id, .distance = res.items[i].d };
                return out;
            }

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
        /// Version 2: f32 slab is optional (FLAG_HAS_F32). Version 1 always has f32.
        pub const GRAPH_VERSION_OPT_F32: u32 = 2;
        pub const FLAG_HAS_QVECS: u32 = 1;
        pub const FLAG_HAS_F32: u32 = 2;

        pub fn serializedSize(self: *const Self) usize {
            var n: usize = 48; // header
            const count = self.len();
            n += count * 4; // levels
            n += count * 4; // qscales
            if (self.opts.store_f32) n += count * self.dim * 4; // f32
            n += count * self.dim; // i8
            var id: u32 = 0;
            while (id < count) : (id += 1) {
                const n_layers = self.layerCount(id);
                n += 4; // n_layers
                var layer: u32 = 0;
                while (layer < n_layers) : (layer += 1) {
                    n += 4 + self.degree(id, layer) * 4;
                }
            }
            return n;
        }

        /// Little-endian graph blob: header + levels + qscales + f32 + i8 + links.
        /// Does not include external document ids. On-disk format is unchanged.
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
            const version: u32 = if (self.opts.store_f32) GRAPH_VERSION else GRAPH_VERSION_OPT_F32;
            wr.u32le(buf, &i, version);
            wr.u64le(buf, &i, self.dim);
            wr.u64le(buf, &i, self.len());
            wr.u32le(buf, &i, self.entry_point orelse std.math.maxInt(u32));
            wr.u32le(buf, &i, self.max_level);
            wr.u32le(buf, &i, self.opts.m);
            wr.u32le(buf, &i, self.opts.m0_factor);
            wr.u32le(buf, &i, self.opts.ef_construction);
            var flags: u32 = FLAG_HAS_QVECS;
            if (self.opts.store_f32) flags |= FLAG_HAS_F32;
            wr.u32le(buf, &i, flags);

            for (self.levels.items) |lv| wr.u32le(buf, &i, lv);
            for (self.qscales.items) |s| {
                const bits: u32 = @bitCast(s);
                wr.u32le(buf, &i, bits);
            }
            if (self.opts.store_f32) {
                const fbytes = std.mem.sliceAsBytes(self.vectors_flat.items);
                @memcpy(buf[i..][0..fbytes.len], fbytes);
                i += fbytes.len;
            }
            const qbytes = std.mem.sliceAsBytes(self.qvecs_flat.items);
            @memcpy(buf[i..][0..qbytes.len], qbytes);
            i += qbytes.len;
            var id: u32 = 0;
            while (id < self.len()) : (id += 1) {
                const n_layers = self.layerCount(id);
                wr.u32le(buf, &i, n_layers);
                var layer: u32 = 0;
                while (layer < n_layers) : (layer += 1) {
                    const nbrs = self.neighbors(id, layer);
                    wr.u32le(buf, &i, @intCast(nbrs.len));
                    for (nbrs) |nid| wr.u32le(buf, &i, nid);
                }
            }
            if (i != buf.len) return error.SerializeSizeMismatch;
            return buf;
        }

        /// Populate an empty index from `serialize` output. No HNSW rebuild.
        pub fn load(self: *Self, bytes: []const u8) !void {
            if (self.len() != 0) return error.NotEmpty;
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
            const version = try rd.u32le(bytes, &i);
            if (version != GRAPH_VERSION and version != GRAPH_VERSION_OPT_F32) return error.UnsupportedVersion;
            const dim: usize = @intCast(try rd.u64le(bytes, &i));
            const n: usize = @intCast(try rd.u64le(bytes, &i));
            const ep_raw = try rd.u32le(bytes, &i);
            const max_level = try rd.u32le(bytes, &i);
            const m = try rd.u32le(bytes, &i);
            const m0_factor = try rd.u32le(bytes, &i);
            const efc = try rd.u32le(bytes, &i);
            const flags = try rd.u32le(bytes, &i);
            // v1 always stored f32; v2 honors FLAG_HAS_F32 so a no-f32 snapshot
            // (and a later WAL compact of that ns) cannot crash on a missing slab.
            const has_f32 = version == GRAPH_VERSION or (flags & FLAG_HAS_F32) != 0;
            if (self.dim == 0) self.dim = dim;
            if (self.dim != dim) return error.DimensionMismatch;
            self.opts.m = m;
            self.opts.m0_factor = m0_factor;
            self.opts.ef_construction = efc;
            self.opts.store_f32 = has_f32;
            self.max_level = max_level;
            self.entry_point = if (ep_raw == std.math.maxInt(u32)) null else ep_raw;

            const alloc = self.allocator;
            try self.levels.ensureTotalCapacity(alloc, n);
            try self.qscales.ensureTotalCapacity(alloc, n);
            if (has_f32) try self.vectors_flat.ensureTotalCapacity(alloc, n * dim);
            try self.qvecs_flat.ensureTotalCapacity(alloc, n * dim);
            try self.l0_deg.ensureTotalCapacity(alloc, n);
            try self.l0_nbrs.ensureTotalCapacity(alloc, n * self.l0Stride());
            try self.hi_start.ensureTotalCapacity(alloc, n);

            for (0..n) |_| self.levels.appendAssumeCapacity(try rd.u32le(bytes, &i));
            for (0..n) |_| {
                const bits = try rd.u32le(bytes, &i);
                self.qscales.appendAssumeCapacity(@bitCast(bits));
            }
            if (has_f32) {
                const fnbytes = n * dim * @sizeOf(f32);
                if (i + fnbytes > bytes.len) return error.Truncated;
                const fdest = try self.vectors_flat.addManyAsSlice(alloc, n * dim);
                @memcpy(std.mem.sliceAsBytes(fdest), bytes[i..][0..fnbytes]);
                i += fnbytes;
            }
            const qnbytes = n * dim;
            if (i + qnbytes > bytes.len) return error.Truncated;
            const qdest = try self.qvecs_flat.addManyAsSlice(alloc, n * dim);
            @memcpy(std.mem.sliceAsBytes(qdest), bytes[i..][0..qnbytes]);
            i += qnbytes;

            var id: u32 = 0;
            while (id < n) : (id += 1) {
                const n_layers: usize = @intCast(try rd.u32le(bytes, &i));
                const level: u32 = if (n_layers == 0) 0 else @intCast(n_layers - 1);
                if (id >= self.levels.items.len) return error.Truncated;
                self.levels.items[id] = level;
                self.l0_deg.appendAssumeCapacity(0);
                const l0_pad = try self.l0_nbrs.addManyAsSlice(alloc, self.l0Stride());
                @memset(l0_pad, 0);
                try self.hi_start.append(alloc, @intCast(self.hi_deg.items.len));
                if (level > 0) {
                    const slots: usize = level;
                    const degs = try self.hi_deg.addManyAsSlice(alloc, slots);
                    @memset(degs, 0);
                    const nbrs = try self.hi_nbrs.addManyAsSlice(alloc, slots * self.hiStride());
                    @memset(nbrs, 0);
                }
                var layer: u32 = 0;
                while (layer < n_layers) : (layer += 1) {
                    const deg: usize = @intCast(try rd.u32le(bytes, &i));
                    if (deg > self.neighborSlots(id, layer).len) return error.DegreeOverflow;
                    var k: usize = 0;
                    while (k < deg) : (k += 1) {
                        self.neighborSlots(id, layer)[k] = try rd.u32le(bytes, &i);
                    }
                    self.setDegree(id, layer, @intCast(deg));
                }
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
        for (index.neighbors(id, 0)) |nb| {
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
    var idx: u32 = 0;
    while (idx < index.len()) : (idx += 1) {
        try all.append(alloc, .{ .id = idx, .d = vecmath.cosineDistance(nq, index.vectorConst(idx)) });
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
    const recall = @as(f64, @floatFromInt(hits)) / 10.0;
    std.debug.print("recall@10 vs brute force: {d:.3} (ann best {d:.4} vs gt best {d:.4} id {d})\n", .{
        recall, res[0].distance, all.items[0].d, all.items[0].id,
    });
    try std.testing.expectEqual(@as(usize, 10), hits);
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
    var id: u32 = 0;
    while (id < index.len()) : (id += 1) {
        try std.testing.expectEqual(index.layerCount(id), loaded.layerCount(id));
        var layer: u32 = 0;
        while (layer < index.layerCount(id)) : (layer += 1) {
            try std.testing.expectEqualSlices(u32, index.neighbors(id, layer), loaded.neighbors(id, layer));
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

test "hnsw store_f32=false search still finds planted neighbor" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 16;
    var rng = std.Random.DefaultPrng.init(42);
    var index = Hnsw(void).init(alloc, dim, .{ .store_f32 = false });
    defer index.deinit();

    var v: [dim]f32 = undefined;
    for (0..500) |_| {
        for (&v) |*x| x.* = rng.random().floatNorm(f32);
        _ = try index.insert(&v);
    }
    try std.testing.expect(!index.hasStoredF32());
    try std.testing.expectEqual(@as(usize, 0), index.vectors_flat.items.len);
    try std.testing.expectEqual(@as(usize, 500), index.len());

    @memset(&v, 0);
    v[3] = 1;
    const target = try index.insert(&v);
    const results = try index.search(&v, 1, 64, alloc);
    try std.testing.expectEqual(target, results[0].id);
}

test "hnsw store_f32=false serialize/load and update" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 32;
    var rng = std.Random.DefaultPrng.init(7);
    var index = Hnsw(void).init(alloc, dim, .{ .store_f32 = false });
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
    try std.testing.expectEqual(index.serializedSize(), blob.len);
    try std.testing.expectEqual(Hnsw(void).GRAPH_VERSION_OPT_F32, std.mem.readInt(u32, blob[4..8], .little));
    try std.testing.expectEqual(@as(usize, 0), index.vectors_flat.items.len);

    var loaded = Hnsw(void).init(alloc, dim, .{});
    defer loaded.deinit();
    try loaded.load(blob);
    try std.testing.expect(!loaded.hasStoredF32());
    try std.testing.expectEqual(index.len(), loaded.len());
    try std.testing.expectEqual(index.entry_point, loaded.entry_point);
    try std.testing.expectEqual(@as(usize, 0), loaded.vectors_flat.items.len);
    var id: u32 = 0;
    while (id < index.len()) : (id += 1) {
        try std.testing.expectEqualSlices(u32, index.neighbors(id, 0), loaded.neighbors(id, 0));
    }

    const after = try loaded.search(&v, 5, 64, alloc);
    try std.testing.expectEqual(planted, after[0].id);
    try std.testing.expectEqual(before[0].id, after[0].id);

    var flip: [dim]f32 = undefined;
    @memset(&flip, 0);
    flip[4] = 1;
    try loaded.update(planted, &flip);
    const updated = try loaded.search(&flip, 1, 64, alloc);
    try std.testing.expectEqual(planted, updated[0].id);
}

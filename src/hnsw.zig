//! HNSW (Hierarchical Navigable Small World) ANN index, in-memory.
//! Cosine distance; vectors are L2-normalized on insert so
//! cosine distance == 1 - dot, which keeps the inner loop tight.
//!
//! Memory layout is flat / mmap-shaped (one slab per array, not one malloc
//! per vector or per neighbor list). Snapshots persist those slabs as one
//! HMLS file and can be reopened with POSIX mmap (MAP_PRIVATE) so RSS is
//! demand-paged — no copy-in of vectors_flat / qvecs_flat / packed CSR.
//!
//! Growth after mmap: in-memory append buffer (the existing ArrayLists).
//! Splices/updates on mapped nodes write MAP_PRIVATE COW pages; the snapshot
//! file is never mutated. Crash mid-insert leaves the file intact. Durable
//! persist is an atomic rewrite (tmp + fsync + rename).
//!
//! On-disk next step is append-only segments: RecordID → {segment, offset,
//! generation}. This file does not implement NVMe/SPDK/ZNS; see
//! experiments/log.md.

const std = @import("std");
const builtin = @import("builtin");
const vecmath = @import("vector.zig");
const rss = @import("rss.zig");
const posix = std.posix;
const linux = std.os.linux;

/// Persist envelope magic ('OPSN' le). Duplicated so this file does not
/// import persist.zig (persist already imports hnsw).
const SNAP_MAGIC_DUP: u32 = 0x4E53504F;

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

        /// MAP_PRIVATE view of a slab snapshot. ArrayLists above are the
        /// post-mmap append buffer (empty until the first insert after load).
        slab_map: ?SlabMap = null,

        /// Per-node generation for RCU-style neighbor snapshots.
        /// Even = stable; odd = a writer is replacing that node's adjacency.
        /// `pushNeighbor` does not bump this (write-then-release-degree is enough).
        /// Heap-sized to `len()` including mmap-backed ids (zeros on load).
        nbr_gen: std.ArrayList(u32) = .empty,
        /// Per-node seqlock for in-place vector updates. Distances retry
        /// when the generation is odd or changes mid-dot.
        vec_gen: std.ArrayList(u32) = .empty,

        /// Visible node count. `vectors_flat.items` may be ahead while a
        /// writer fills a new row; readers use this acquire-load.
        published: usize = 0,
        /// Sentinel `maxInt(u32)` = none. Atomic so publish need not take
        /// the namespace exclusive lock.
        entry_id: u32 = std.math.maxInt(u32),
        max_level: u32 = 0,

        pub const SLAB_MAGIC: u32 = 0x534C4D48; // 'HMLS' le
        pub const SLAB_VERSION: u32 = 1;
        pub const SLAB_HEADER_SIZE: usize = 256;

        /// Live mmap of one HMLS (or persist-envelope + HMLS) file.
        pub const SlabMap = struct {
            bytes: []align(std.heap.page_size_min) u8,
            n: usize,
            hi_slots: usize,
            levels: []u32,
            qscales: []f32,
            vectors: []f32,
            qvecs: []i8,
            l0_deg: []u16,
            l0_nbrs: []u32,
            hi_start: []u32,
            hi_deg: []u16,
            hi_nbrs: []u32,
        };

        pub fn entryPoint(self: *const Self) ?u32 {
            const v = @atomicLoad(u32, &self.entry_id, .acquire);
            return if (v == std.math.maxInt(u32)) null else v;
        }

        fn setEntryPoint(self: *Self, id: u32) void {
            @atomicStore(u32, &self.entry_id, id, .release);
        }

        pub fn getMaxLevel(self: *const Self) u32 {
            return @atomicLoad(u32, &self.max_level, .acquire);
        }

        fn setMaxLevel(self: *Self, lvl: u32) void {
            @atomicStore(u32, &self.max_level, lvl, .release);
        }

        pub fn init(allocator: std.mem.Allocator, dim: usize, opts: Options) Self {
            return .{
                .allocator = allocator,
                .dim = dim,
                .opts = opts,
                .rng = std.Random.DefaultPrng.init(opts.seed),
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.slab_map) |m| posix.munmap(m.bytes);
            self.slab_map = null;
            self.vectors_flat.deinit(self.allocator);
            self.qvecs_flat.deinit(self.allocator);
            self.qscales.deinit(self.allocator);
            self.levels.deinit(self.allocator);
            self.l0_deg.deinit(self.allocator);
            self.l0_nbrs.deinit(self.allocator);
            self.hi_start.deinit(self.allocator);
            self.hi_deg.deinit(self.allocator);
            self.hi_nbrs.deinit(self.allocator);
            self.nbr_gen.deinit(self.allocator);
            self.vec_gen.deinit(self.allocator);
            self.* = undefined;
        }

        inline fn mappedCount(self: *const Self) usize {
            return if (self.slab_map) |m| m.n else 0;
        }

        pub fn len(self: *const Self) usize {
            return @atomicLoad(usize, &self.published, .acquire);
        }

        /// True when this index retains the f32 slab (exact rerank / snapshot f32).
        pub fn hasStoredF32(self: *const Self) bool {
            return self.opts.store_f32;
        }

        pub fn isMmapBacked(self: *const Self) bool {
            return self.slab_map != null;
        }

        /// True when one more insert will not realloc any packed slab.
        pub fn hasInsertRoom(self: *const Self, extra: usize) bool {
            if (self.dim == 0) return false;
            const heap_n = self.qscales.items.len;
            const mapped = self.mappedCount();
            const total = mapped + heap_n + extra;
            const need_heap = heap_n + extra;
            if (self.opts.store_f32 and self.vectors_flat.capacity < need_heap * self.dim) return false;
            if (self.qvecs_flat.capacity < need_heap * self.dim) return false;
            if (self.qscales.capacity < need_heap) return false;
            if (self.levels.capacity < need_heap) return false;
            if (self.l0_deg.capacity < need_heap) return false;
            if (self.l0_nbrs.capacity < need_heap * self.l0Stride()) return false;
            if (self.hi_start.capacity < need_heap) return false;
            if (self.nbr_gen.capacity < total) return false;
            if (self.vec_gen.capacity < total) return false;
            const hi_extra = extra * 8;
            if (self.hi_deg.capacity < self.hi_deg.items.len + hi_extra) return false;
            if (self.hi_nbrs.capacity < self.hi_nbrs.items.len + hi_extra * self.hiStride()) return false;
            return true;
        }

        /// Grow slabs. Caller must hold exclusive so live readers cannot
        /// hold dangling `items` pointers across the realloc.
        pub fn ensureInsertRoom(self: *Self, extra: usize) !void {
            const alloc = self.allocator;
            const heap_n = self.qscales.items.len;
            const mapped = self.mappedCount();
            const total = mapped + heap_n + extra;
            const need_heap = heap_n + extra;
            if (self.opts.store_f32) {
                try self.vectors_flat.ensureTotalCapacity(alloc, need_heap * @max(self.dim, 1));
            }
            try self.qvecs_flat.ensureTotalCapacity(alloc, need_heap * @max(self.dim, 1));
            try self.qscales.ensureTotalCapacity(alloc, need_heap);
            try self.levels.ensureTotalCapacity(alloc, need_heap);
            try self.l0_deg.ensureTotalCapacity(alloc, need_heap);
            try self.l0_nbrs.ensureTotalCapacity(alloc, need_heap * self.l0Stride());
            try self.hi_start.ensureTotalCapacity(alloc, need_heap);
            try self.nbr_gen.ensureTotalCapacity(alloc, total);
            try self.vec_gen.ensureTotalCapacity(alloc, total);
            try self.hi_deg.ensureTotalCapacity(alloc, self.hi_deg.items.len + extra * 8);
            try self.hi_nbrs.ensureTotalCapacity(alloc, self.hi_nbrs.items.len + extra * 8 * self.hiStride());
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

        fn heapId(self: *const Self, id: u32) u32 {
            return id - @as(u32, @intCast(self.mappedCount()));
        }

        /// Mutable f32 row. Server snapshot/WAL and `update` use this.
        /// Empty when `store_f32=false` (callers must check `hasStoredF32`).
        pub fn vector(self: *Self, id: u32) []f32 {
            if (!self.opts.store_f32) return &.{};
            if (self.slab_map) |m| {
                if (id < m.n) {
                    if (m.vectors.len == 0) return &.{};
                    const off = @as(usize, id) * self.dim;
                    return m.vectors[off..][0..self.dim];
                }
                if (self.vectors_flat.items.len == 0) return &.{};
                const off = @as(usize, self.heapId(id)) * self.dim;
                return self.vectors_flat.items[off..][0..self.dim];
            }
            if (self.vectors_flat.items.len == 0) return &.{};
            const off = @as(usize, id) * self.dim;
            return self.vectors_flat.items[off..][0..self.dim];
        }

        pub fn vectorConst(self: *const Self, id: u32) []const f32 {
            if (!self.opts.store_f32) return &.{};
            if (self.slab_map) |m| {
                if (id < m.n) {
                    if (m.vectors.len == 0) return &.{};
                    const off = @as(usize, id) * self.dim;
                    return m.vectors[off..][0..self.dim];
                }
                if (self.vectors_flat.items.len == 0) return &.{};
                const off = @as(usize, self.heapId(id)) * self.dim;
                return self.vectors_flat.items[off..][0..self.dim];
            }
            if (self.vectors_flat.items.len == 0) return &.{};
            const off = @as(usize, id) * self.dim;
            return self.vectors_flat.items[off..][0..self.dim];
        }

        pub fn qvec(self: *Self, id: u32) []i8 {
            if (self.slab_map) |m| {
                if (id < m.n) {
                    const off = @as(usize, id) * self.dim;
                    return m.qvecs[off..][0..self.dim];
                }
                const off = @as(usize, self.heapId(id)) * self.dim;
                return self.qvecs_flat.items[off..][0..self.dim];
            }
            const off = @as(usize, id) * self.dim;
            return self.qvecs_flat.items[off..][0..self.dim];
        }

        pub fn qvecConst(self: *const Self, id: u32) []const i8 {
            if (self.slab_map) |m| {
                if (id < m.n) {
                    const off = @as(usize, id) * self.dim;
                    return m.qvecs[off..][0..self.dim];
                }
                const off = @as(usize, self.heapId(id)) * self.dim;
                return self.qvecs_flat.items[off..][0..self.dim];
            }
            const off = @as(usize, id) * self.dim;
            return self.qvecs_flat.items[off..][0..self.dim];
        }

        fn levelOf(self: *const Self, id: u32) u32 {
            if (self.slab_map) |m| {
                if (id < m.n) return m.levels[id];
                return self.levels.items[self.heapId(id)];
            }
            return self.levels.items[id];
        }

        fn qscaleOf(self: *const Self, id: u32) f32 {
            if (self.slab_map) |m| {
                if (id < m.n) return m.qscales[id];
                return self.qscales.items[self.heapId(id)];
            }
            return self.qscales.items[id];
        }

        fn setQscaleOf(self: *Self, id: u32, scale: f32) void {
            if (self.slab_map) |m| {
                if (id < m.n) {
                    m.qscales[id] = scale;
                    return;
                }
                self.qscales.items[self.heapId(id)] = scale;
                return;
            }
            self.qscales.items[id] = scale;
        }

        pub fn layerCount(self: *const Self, id: u32) u32 {
            return self.levelOf(id) + 1;
        }

        pub fn neighbors(self: *const Self, id: u32, layer: u32) []const u32 {
            const deg = self.degree(id, layer);
            return self.neighborSlotsConst(id, layer)[0..deg];
        }

        fn degPtr(self: *const Self, id: u32, layer: u32) *u16 {
            if (layer == 0) {
                if (self.slab_map) |m| {
                    if (id < m.n) return &m.l0_deg[id];
                    return @constCast(&self.l0_deg.items[self.heapId(id)]);
                }
                return @constCast(&self.l0_deg.items[id]);
            }
            if (self.slab_map) |m| {
                if (id < m.n) {
                    const slot = m.hi_start[id] + (layer - 1);
                    return &m.hi_deg[slot];
                }
                const slot = self.hi_start.items[self.heapId(id)] + (layer - 1);
                return @constCast(&self.hi_deg.items[slot]);
            }
            const slot = self.hi_start.items[id] + (layer - 1);
            return @constCast(&self.hi_deg.items[slot]);
        }

        fn degree(self: *const Self, id: u32, layer: u32) usize {
            return @atomicLoad(u16, self.degPtr(id, layer), .acquire);
        }

        fn neighborSlotsConst(self: *const Self, id: u32, layer: u32) []const u32 {
            if (layer == 0) {
                const stride = self.l0Stride();
                if (self.slab_map) |m| {
                    if (id < m.n) {
                        const off = @as(usize, id) * stride;
                        return m.l0_nbrs[off..][0..stride];
                    }
                    const off = @as(usize, self.heapId(id)) * stride;
                    return self.l0_nbrs.items[off..][0..stride];
                }
                const off = @as(usize, id) * stride;
                return self.l0_nbrs.items[off..][0..stride];
            }
            const stride = self.hiStride();
            if (self.slab_map) |m| {
                if (id < m.n) {
                    const slot = m.hi_start[id] + (layer - 1);
                    const off = slot * stride;
                    return m.hi_nbrs[off..][0..stride];
                }
                const slot = self.hi_start.items[self.heapId(id)] + (layer - 1);
                const off = slot * stride;
                return self.hi_nbrs.items[off..][0..stride];
            }
            const slot = self.hi_start.items[id] + (layer - 1);
            const off = slot * stride;
            return self.hi_nbrs.items[off..][0..stride];
        }

        fn neighborSlots(self: *Self, id: u32, layer: u32) []u32 {
            if (layer == 0) {
                const stride = self.l0Stride();
                if (self.slab_map) |m| {
                    if (id < m.n) {
                        const off = @as(usize, id) * stride;
                        return m.l0_nbrs[off..][0..stride];
                    }
                    const off = @as(usize, self.heapId(id)) * stride;
                    return self.l0_nbrs.items[off..][0..stride];
                }
                const off = @as(usize, id) * stride;
                return self.l0_nbrs.items[off..][0..stride];
            }
            const stride = self.hiStride();
            if (self.slab_map) |m| {
                if (id < m.n) {
                    const slot = m.hi_start[id] + (layer - 1);
                    const off = slot * stride;
                    return m.hi_nbrs[off..][0..stride];
                }
                const slot = self.hi_start.items[self.heapId(id)] + (layer - 1);
                const off = slot * stride;
                return self.hi_nbrs.items[off..][0..stride];
            }
            const slot = self.hi_start.items[id] + (layer - 1);
            const off = slot * stride;
            return self.hi_nbrs.items[off..][0..stride];
        }

        fn setDegree(self: *Self, id: u32, layer: u32, deg: u16) void {
            @atomicStore(u16, self.degPtr(id, layer), deg, .release);
        }

        /// Copy a stable neighbor list. Retries if a writer is mid-replace
        /// (`nbr_gen` odd or changed). `pushNeighbor` is wait-free for readers:
        /// it writes the slack slot first, then publishes degree.
        fn snapshotNeighbors(self: *const Self, id: u32, layer: u32, buf: []u32) []const u32 {
            var spins: u32 = 0;
            while (true) {
                const g1 = @atomicLoad(u32, &self.nbr_gen.items[id], .acquire);
                if ((g1 & 1) != 0) {
                    spins +%= 1;
                    if (spins > 8) std.atomic.spinLoopHint();
                    continue;
                }
                const deg = self.degree(id, layer);
                const slots = self.neighborSlotsConst(id, layer);
                const n = @min(deg, @min(buf.len, slots.len));
                if (n > 0) @memcpy(buf[0..n], slots[0..n]);
                const g2 = @atomicLoad(u32, &self.nbr_gen.items[id], .acquire);
                if (g1 == g2) return buf[0..n];
            }
        }

        fn pushNeighbor(self: *Self, id: u32, layer: u32, nid: u32) void {
            const deg = self.degree(id, layer);
            self.neighborSlots(id, layer)[deg] = nid;
            self.setDegree(id, layer, @intCast(deg + 1));
        }

        fn replaceNeighbors(self: *Self, id: u32, layer: u32, ids: []const u32) void {
            const g = @atomicLoad(u32, &self.nbr_gen.items[id], .monotonic);
            @atomicStore(u32, &self.nbr_gen.items[id], g +% 1, .release);
            const slots = self.neighborSlots(id, layer);
            @memcpy(slots[0..ids.len], ids);
            self.setDegree(id, layer, @intCast(ids.len));
            @atomicStore(u32, &self.nbr_gen.items[id], g +% 2, .release);
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
            while (true) {
                const g1 = @atomicLoad(u32, &self.vec_gen.items[id], .acquire);
                if ((g1 & 1) != 0) {
                    std.atomic.spinLoopHint();
                    continue;
                }
                const d = vecmath.cosineDistance(q, self.vectorConst(id));
                const g2 = @atomicLoad(u32, &self.vec_gen.items[id], .acquire);
                if (g1 == g2) return d;
            }
        }

        /// Pairwise distance used while splicing the graph. Exact when f32
        /// is stored; otherwise the same int8 score search already uses.
        inline fn distPair(self: *const Self, src: u32, dst: u32) f32 {
            if (self.opts.store_f32) return self.distTo(self.vectorConst(src), dst);
            return self.distQ(self.qvecConst(src), self.qscaleOf(src), dst);
        }

        inline fn distQ(self: *const Self, qq: []const i8, qs: f32, id: u32) f32 {
            // stored vectors are unit-norm; a ~= aq * scale, so
            // cos(a,b) ~= dot(aq,bq) * qs * scale_b
            while (true) {
                const g1 = @atomicLoad(u32, &self.vec_gen.items[id], .acquire);
                if ((g1 & 1) != 0) {
                    std.atomic.spinLoopHint();
                    continue;
                }
                const stored = self.qvecConst(id);
                const n = @min(qq.len, stored.len);
                const approx = @as(f64, @floatFromInt(vecmath.dotI8(qq[0..n], stored[0..n]))) *
                    qs * self.qscaleOf(id);
                const g2 = @atomicLoad(u32, &self.vec_gen.items[id], .acquire);
                if (g1 == g2) return @max(0.0, 1.0 - @as(f32, @floatCast(approx)));
            }
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
            n: usize,
            pub inline fn dist(d: @This(), id: u32) f32 {
                const n = @min(d.n, d.qq.len);
                return d.s.distQ(d.qq[0..n], d.qs, id);
            }
        };

        /// Drop the last 128 i8 dims on wide vectors (2 AVX-512 chunks).
        fn searchPrefix(self: *const Self) usize {
            if (self.dim < 512) return self.dim;
            const keep = self.dim - 128;
            return keep & ~@as(usize, 63);
        }

        fn markVisited(visited: *std.DynamicBitSetUnmanaged, alloc: std.mem.Allocator, id: u32) !void {
            if (id >= visited.capacity()) {
                try visited.resize(alloc, @max(@as(usize, id) + 256, visited.capacity() * 2), false);
            }
            visited.set(id);
        }

        fn alreadyVisited(visited: *std.DynamicBitSetUnmanaged, alloc: std.mem.Allocator, id: u32) !bool {
            if (id >= visited.capacity()) {
                try visited.resize(alloc, @max(@as(usize, id) + 256, visited.capacity() * 2), false);
                visited.set(id);
                return false;
            }
            if (visited.isSet(id)) return true;
            visited.set(id);
            return false;
        }

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
                try markVisited(visited, alloc, c.id);
                try candidates.push(alloc, c);
                try results.push(alloc, c);
            }

            while (candidates.count() > 0) {
                const c = candidates.pop().?;
                const worst = if (results.count() > 0) results.peek().?.d else c.d;
                if (c.d > worst and results.count() >= ef) break;

                if (layer >= self.layerCount(c.id)) {
                    std.debug.print("VIOLATION c.id={d} its_layers={d} layer={d} max_level={d} ep={any} ep_layers={d}\n", .{ c.id, self.layerCount(c.id), layer, self.getMaxLevel(), self.entryPoint(), if (self.entryPoint()) |e| self.layerCount(e) else 0 });
                    @panic("hnsw invariant broken");
                }
                if (candidates.count() > 0) {
                    @prefetch(self.neighborSlotsConst(candidates.peek().?.id, layer).ptr, .{
                        .rw = .read,
                        .locality = 3,
                        .cache = .data,
                    });
                }
                var nbr_buf: [64]u32 = undefined;
                const nbrs = self.snapshotNeighbors(c.id, layer, &nbr_buf);
                for (nbrs, 0..) |nid, ni| {
                    // one-ahead software prefetch: the next neighbor's int8
                    // vector starts its memory fetch while we dot this one.
                    // Skip already-visited neighbors (wasted prefetches).
                    if (ni + 1 < nbrs.len) {
                        const nxt = nbrs[ni + 1];
                        if (nxt >= visited.capacity() or !visited.isSet(nxt)) {
                            @prefetch(self.qvecConst(nxt).ptr, .{
                                .rw = .read,
                                .locality = 3,
                                .cache = .data,
                            });
                        }
                    }
                    if (try alreadyVisited(visited, alloc, nid)) continue;
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
        /// Safe under a shared lock: seqlock so in-flight dots retry.
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
            const g = @atomicLoad(u32, &self.vec_gen.items[id], .monotonic);
            @atomicStore(u32, &self.vec_gen.items[id], g +% 1, .release);
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
            self.setQscaleOf(id, scale);
            @atomicStore(u32, &self.vec_gen.items[id], g +% 2, .release);
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
                if (self.copy.len != 0) index_alloc.free(self.copy);
                if (self.q8.len != 0) index_alloc.free(self.q8);
                for (self.layers) |ids| self.scratch.free(ids);
                if (self.layers.len != 0) self.scratch.free(self.layers);
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
            const id: u32 = @intCast(self.mappedCount() + self.qscales.items.len);
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
            try self.nbr_gen.append(alloc, 0);
            try self.vec_gen.append(alloc, 0);
            if (level > 0) {
                const slots: usize = level;
                const degs = try self.hi_deg.addManyAsSlice(alloc, slots);
                @memset(degs, 0);
                const nbrs = try self.hi_nbrs.addManyAsSlice(alloc, slots * self.hiStride());
                @memset(nbrs, 0);
            }
            @atomicStore(usize, &self.published, @as(usize, id) + 1, .release);
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
            const n_layers: usize = @as(usize, @min(level, self.getMaxLevel())) + 1;
            const layers = try scratch.alloc([]u32, n_layers);
            errdefer scratch.free(layers);
            for (layers) |*slot| slot.* = &.{};

            if (self.entryPoint() == null) {
                return .{ .copy = q.copy, .q8 = q.q8, .scale = q.scale, .level = level, .layers = layers, .scratch = scratch };
            }

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(scratch, self.len() + 2048);
            defer visited.deinit(scratch);
            const qctx = QDist{ .s = self, .qq = q.q8, .qs = q.scale, .n = q.q8.len };
            const ep_id = self.entryPoint().?;
            var ep: [1]Candidate = .{.{ .id = ep_id, .d = qctx.dist(ep_id) }};
            var cur = self.getMaxLevel();
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

        /// Publish a planned node: append the payload row, write outgoing
        /// edges, and (if needed) swing `entry_point`. Does not splice
        /// back-edges. Caller must hold exclusive lock so ArrayList growth
        /// cannot race readers. Frees `plan.copy` / `plan.q8`; leaves
        /// `plan.layers` for `spliceBackEdges`.
        pub fn publishInsert(self: *Self, plan: *InsertPlan) !u32 {
            const alloc = self.allocator;
            const id = try self.appendRecord(plan.copy, plan.q8, plan.scale, plan.level);
            alloc.free(plan.copy);
            alloc.free(plan.q8);
            plan.copy = &.{};
            plan.q8 = &.{};

            const connect_n = @min(plan.layers.len, self.layerCount(id));
            var layer: u32 = 0;
            while (layer < connect_n) : (layer += 1) {
                for (plan.layers[layer]) |nid| {
                    if (nid >= self.len() or layer >= self.layerCount(nid)) continue;
                    self.pushNeighbor(id, layer, nid);
                }
            }
            return id;
        }

        /// Swing entry point after the new row is mapped in `doc_ids`.
        pub fn publishEntry(self: *Self, id: u32, level: u32) void {
            if (self.entryPoint() == null or level > self.getMaxLevel()) {
                self.setMaxLevel(level);
                self.setEntryPoint(id);
            }
        }

        /// Splice back-edges + prune. Safe under a shared lock: adjacency
        /// writes are generation-published so `snapshotNeighbors` retries
        /// instead of walking a torn list. Does not grow the packed arrays.
        /// Always frees `plan.layers`.
        pub fn spliceBackEdges(self: *Self, plan: *InsertPlan, id: u32) !void {
            defer {
                for (plan.layers) |ids| plan.scratch.free(ids);
                if (plan.layers.len != 0) plan.scratch.free(plan.layers);
                plan.layers = &.{};
            }
            const alloc = self.allocator;
            const max_m0: u32 = @intCast(self.m0());
            const connect_n = @min(plan.layers.len, self.layerCount(id));
            var layer: u32 = 0;
            while (layer < connect_n) : (layer += 1) {
                const max_m: u32 = if (layer == 0) max_m0 else self.opts.m;
                for (plan.layers[layer]) |nid| {
                    if (nid >= self.len() or layer >= self.layerCount(nid)) continue;
                    self.pushNeighbor(nid, layer, id);
                    if (self.degree(nid, layer) > max_m) {
                        var cands: std.ArrayList(Candidate) = .empty;
                        defer cands.deinit(alloc);
                        var back_buf: [64]u32 = undefined;
                        const back = self.snapshotNeighbors(nid, layer, &back_buf);
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
        }

        /// Splice a planned node into the graph (publish + back-edges).
        /// Single-threaded / exclusive-lock callers (tests, `insert`).
        pub fn commitInsert(self: *Self, plan: InsertPlan) !u32 {
            var p = plan;
            const id = try self.publishInsert(&p);
            self.publishEntry(id, p.level);
            try self.spliceBackEdges(&p, id);
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
            const ep_id = self.entryPoint() orelse return alloc.alloc(SearchResult, 0);
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
            const ctx = QDist{ .s = self, .qq = qq, .qs = qs, .n = self.searchPrefix() };

            var visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, self.len() + 2048);
            defer visited.deinit(alloc);

            var ep: [1]Candidate = .{.{ .id = ep_id, .d = ctx.dist(ep_id) }};
            var cur = self.getMaxLevel();
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
                    std.mem.writeInt(u32, b[off.*..][0..4], v, .little);
                    off.* += 4;
                }
                fn u64le(b: []u8, off: *usize, v: u64) void {
                    std.mem.writeInt(u64, b[off.*..][0..8], v, .little);
                    off.* += 8;
                }
            };
            wr.u32le(buf, &i, GRAPH_MAGIC);
            const version: u32 = if (self.opts.store_f32) GRAPH_VERSION else GRAPH_VERSION_OPT_F32;
            wr.u32le(buf, &i, version);
            wr.u64le(buf, &i, self.dim);
            wr.u64le(buf, &i, self.len());
            wr.u32le(buf, &i, self.entryPoint() orelse std.math.maxInt(u32));
            wr.u32le(buf, &i, self.getMaxLevel());
            wr.u32le(buf, &i, self.opts.m);
            wr.u32le(buf, &i, self.opts.m0_factor);
            wr.u32le(buf, &i, self.opts.ef_construction);
            var flags: u32 = FLAG_HAS_QVECS;
            if (self.opts.store_f32) flags |= FLAG_HAS_F32;
            wr.u32le(buf, &i, flags);

            var sid: u32 = 0;
            while (sid < self.len()) : (sid += 1) wr.u32le(buf, &i, self.levelOf(sid));
            sid = 0;
            while (sid < self.len()) : (sid += 1) {
                const bits: u32 = @bitCast(self.qscaleOf(sid));
                wr.u32le(buf, &i, bits);
            }
            if (self.opts.store_f32) {
                sid = 0;
                while (sid < self.len()) : (sid += 1) {
                    const row = self.vectorConst(sid);
                    const fbytes = std.mem.sliceAsBytes(row);
                    @memcpy(buf[i..][0..fbytes.len], fbytes);
                    i += fbytes.len;
                }
            }
            sid = 0;
            while (sid < self.len()) : (sid += 1) {
                const row = self.qvecConst(sid);
                const qbytes = std.mem.sliceAsBytes(row);
                @memcpy(buf[i..][0..qbytes.len], qbytes);
                i += qbytes.len;
            }
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
        /// Accepts the legacy GRAPH blob or an HMLS slab image (copy-in).
        pub fn load(self: *Self, bytes: []const u8) !void {
            if (self.len() != 0) return error.NotEmpty;
            if (bytes.len >= 4) {
                const magic = std.mem.readInt(u32, bytes[0..4], .little);
                if (magic == SLAB_MAGIC) return self.loadSlabsCopy(bytes);
            }
            if (bytes.len < 48) return error.Truncated;
            var i: usize = 0;
            const rd = struct {
                fn u32le(b: []const u8, off: *usize) !u32 {
                    if (off.* + 4 > b.len) return error.Truncated;
                    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
                    off.* += 4;
                    return v;
                }
                fn u64le(b: []const u8, off: *usize) !u64 {
                    if (off.* + 8 > b.len) return error.Truncated;
                    const v = std.mem.readInt(u64, b[off.*..][0..8], .little);
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
            self.setMaxLevel(max_level);
            if (ep_raw == std.math.maxInt(u32)) {
                @atomicStore(u32, &self.entry_id, std.math.maxInt(u32), .release);
            } else {
                self.setEntryPoint(ep_raw);
            }

            const alloc = self.allocator;
            try self.levels.ensureTotalCapacity(alloc, n);
            try self.qscales.ensureTotalCapacity(alloc, n);
            if (has_f32) try self.vectors_flat.ensureTotalCapacity(alloc, n * dim);
            try self.qvecs_flat.ensureTotalCapacity(alloc, n * dim);
            try self.l0_deg.ensureTotalCapacity(alloc, n);
            try self.l0_nbrs.ensureTotalCapacity(alloc, n * self.l0Stride());
            try self.hi_start.ensureTotalCapacity(alloc, n);
            try self.nbr_gen.ensureTotalCapacity(alloc, n);
            try self.vec_gen.ensureTotalCapacity(alloc, n);

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
                self.nbr_gen.appendAssumeCapacity(0);
                self.vec_gen.appendAssumeCapacity(0);
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
            @atomicStore(usize, &self.published, n, .release);
        }

        pub fn pageAlign(n: usize) usize {
            return std.mem.alignForward(usize, n, std.heap.page_size_min);
        }

        fn hiSlotCount(self: *const Self) usize {
            const extra = self.hi_deg.items.len;
            return if (self.slab_map) |m| m.hi_slots + extra else extra;
        }

        const SlabLayout = struct {
            n: usize,
            dim: usize,
            hi_slots: usize,
            levels_off: usize,
            qscales_off: usize,
            vectors_off: usize,
            qvecs_off: usize,
            l0_deg_off: usize,
            l0_nbrs_off: usize,
            hi_start_off: usize,
            hi_deg_off: usize,
            hi_nbrs_off: usize,
            file_len: usize,

            fn compute(n: usize, dim: usize, hi_slots: usize, l0_stride: usize, hi_stride: usize) SlabLayout {
                var off: usize = pageAlign(SLAB_HEADER_SIZE);
                const levels_off = off;
                off = pageAlign(off + n * @sizeOf(u32));
                const qscales_off = off;
                off = pageAlign(off + n * @sizeOf(f32));
                const vectors_off = off;
                off = pageAlign(off + n * dim * @sizeOf(f32));
                const qvecs_off = off;
                off = pageAlign(off + n * dim * @sizeOf(i8));
                const l0_deg_off = off;
                off = pageAlign(off + n * @sizeOf(u16));
                const l0_nbrs_off = off;
                off = pageAlign(off + n * l0_stride * @sizeOf(u32));
                const hi_start_off = off;
                off = pageAlign(off + n * @sizeOf(u32));
                const hi_deg_off = off;
                off = pageAlign(off + hi_slots * @sizeOf(u16));
                const hi_nbrs_off = off;
                off = pageAlign(off + hi_slots * hi_stride * @sizeOf(u32));
                return .{
                    .n = n,
                    .dim = dim,
                    .hi_slots = hi_slots,
                    .levels_off = levels_off,
                    .qscales_off = qscales_off,
                    .vectors_off = vectors_off,
                    .qvecs_off = qvecs_off,
                    .l0_deg_off = l0_deg_off,
                    .l0_nbrs_off = l0_nbrs_off,
                    .hi_start_off = hi_start_off,
                    .hi_deg_off = hi_deg_off,
                    .hi_nbrs_off = hi_nbrs_off,
                    .file_len = off,
                };
            }
        };

        fn writeHeaderBytes(buf: []u8, layout: SlabLayout, self: *const Self) void {
            @memset(buf[0..SLAB_HEADER_SIZE], 0);
            std.mem.writeInt(u32, buf[0..4], SLAB_MAGIC, .little);
            std.mem.writeInt(u32, buf[4..8], SLAB_VERSION, .little);
            std.mem.writeInt(u64, buf[8..16], layout.dim, .little);
            std.mem.writeInt(u64, buf[16..24], layout.n, .little);
            std.mem.writeInt(u32, buf[24..28], self.entryPoint() orelse 0, .little);
            std.mem.writeInt(u32, buf[28..32], self.getMaxLevel(), .little);
            std.mem.writeInt(u32, buf[32..36], self.opts.m, .little);
            std.mem.writeInt(u32, buf[36..40], self.opts.m0_factor, .little);
            std.mem.writeInt(u32, buf[40..44], self.opts.ef_construction, .little);
            std.mem.writeInt(u32, buf[44..48], 1, .little);
            std.mem.writeInt(u64, buf[48..56], layout.hi_slots, .little);
            std.mem.writeInt(u64, buf[56..64], layout.levels_off, .little);
            std.mem.writeInt(u64, buf[64..72], layout.qscales_off, .little);
            std.mem.writeInt(u64, buf[72..80], layout.vectors_off, .little);
            std.mem.writeInt(u64, buf[80..88], layout.qvecs_off, .little);
            std.mem.writeInt(u64, buf[88..96], layout.l0_deg_off, .little);
            std.mem.writeInt(u64, buf[96..104], layout.l0_nbrs_off, .little);
            std.mem.writeInt(u64, buf[104..112], layout.hi_start_off, .little);
            std.mem.writeInt(u64, buf[112..120], layout.hi_deg_off, .little);
            std.mem.writeInt(u64, buf[120..128], layout.hi_nbrs_off, .little);
            std.mem.writeInt(u64, buf[128..136], layout.file_len, .little);
        }

        fn parseLayout(bytes: []const u8) !struct { layout: SlabLayout, entry: u32, max_level: u32, m: u32, m0_factor: u32, efc: u32 } {
            if (bytes.len < SLAB_HEADER_SIZE) return error.Truncated;
            if (std.mem.readInt(u32, bytes[0..4], .little) != SLAB_MAGIC) return error.BadMagic;
            if (std.mem.readInt(u32, bytes[4..8], .little) != SLAB_VERSION) return error.UnsupportedVersion;
            const dim: usize = @intCast(std.mem.readInt(u64, bytes[8..16], .little));
            const n: usize = @intCast(std.mem.readInt(u64, bytes[16..24], .little));
            const entry = std.mem.readInt(u32, bytes[24..28], .little);
            const max_level = std.mem.readInt(u32, bytes[28..32], .little);
            const m = std.mem.readInt(u32, bytes[32..36], .little);
            const m0_factor = std.mem.readInt(u32, bytes[36..40], .little);
            const efc = std.mem.readInt(u32, bytes[40..44], .little);
            const hi_slots: usize = @intCast(std.mem.readInt(u64, bytes[48..56], .little));
            const layout = SlabLayout{
                .n = n,
                .dim = dim,
                .hi_slots = hi_slots,
                .levels_off = @intCast(std.mem.readInt(u64, bytes[56..64], .little)),
                .qscales_off = @intCast(std.mem.readInt(u64, bytes[64..72], .little)),
                .vectors_off = @intCast(std.mem.readInt(u64, bytes[72..80], .little)),
                .qvecs_off = @intCast(std.mem.readInt(u64, bytes[80..88], .little)),
                .l0_deg_off = @intCast(std.mem.readInt(u64, bytes[88..96], .little)),
                .l0_nbrs_off = @intCast(std.mem.readInt(u64, bytes[96..104], .little)),
                .hi_start_off = @intCast(std.mem.readInt(u64, bytes[104..112], .little)),
                .hi_deg_off = @intCast(std.mem.readInt(u64, bytes[112..120], .little)),
                .hi_nbrs_off = @intCast(std.mem.readInt(u64, bytes[120..128], .little)),
                .file_len = @intCast(std.mem.readInt(u64, bytes[128..136], .little)),
            };
            return .{ .layout = layout, .entry = entry, .max_level = max_level, .m = m, .m0_factor = m0_factor, .efc = efc };
        }

        fn sysWriteAll(fd: linux.fd_t, bytes: []const u8) !void {
            var off: usize = 0;
            while (off < bytes.len) {
                const n = linux.write(fd, bytes[off..].ptr, bytes.len - off);
                switch (posix.errno(n)) {
                    .SUCCESS => off += n,
                    .INTR => continue,
                    else => return error.WriteFailed,
                }
            }
        }

        fn sysPwriteAll(fd: linux.fd_t, bytes: []const u8, offset: u64) !void {
            var off: usize = 0;
            while (off < bytes.len) {
                const n = linux.pwrite(fd, bytes[off..].ptr, bytes.len - off, @intCast(offset + off));
                switch (posix.errno(n)) {
                    .SUCCESS => off += n,
                    .INTR => continue,
                    else => return error.WriteFailed,
                }
            }
        }

        fn sysPreadAll(fd: linux.fd_t, dest: []u8, offset: u64) !void {
            var off: usize = 0;
            while (off < dest.len) {
                const n = linux.pread(fd, dest[off..].ptr, dest.len - off, @intCast(offset + off));
                switch (posix.errno(n)) {
                    .SUCCESS => {
                        if (n == 0) return error.Truncated;
                        off += n;
                    },
                    .INTR => continue,
                    else => return error.ReadFailed,
                }
            }
        }

        fn writeConcat(fd: linux.fd_t, file_off: u64, a: []const u8, b: []const u8) !void {
            if (a.len > 0) try sysPwriteAll(fd, a, file_off);
            if (b.len > 0) try sysPwriteAll(fd, b, file_off + a.len);
        }

        fn mappedLevels(self: *const Self) []const u32 {
            return if (self.slab_map) |m| m.levels else &.{};
        }
        fn mappedQscales(self: *const Self) []const f32 {
            return if (self.slab_map) |m| m.qscales else &.{};
        }
        fn mappedVectors(self: *const Self) []const f32 {
            return if (self.slab_map) |m| m.vectors else &.{};
        }
        fn mappedQvecs(self: *const Self) []const i8 {
            return if (self.slab_map) |m| m.qvecs else &.{};
        }
        fn mappedL0Deg(self: *const Self) []const u16 {
            return if (self.slab_map) |m| m.l0_deg else &.{};
        }
        fn mappedL0Nbrs(self: *const Self) []const u32 {
            return if (self.slab_map) |m| m.l0_nbrs else &.{};
        }
        fn mappedHiStart(self: *const Self) []const u32 {
            return if (self.slab_map) |m| m.hi_start else &.{};
        }
        fn mappedHiDeg(self: *const Self) []const u16 {
            return if (self.slab_map) |m| m.hi_deg else &.{};
        }
        fn mappedHiNbrs(self: *const Self) []const u32 {
            return if (self.slab_map) |m| m.hi_nbrs else &.{};
        }

        /// Byte size of a standalone HMLS image for this index.
        pub fn slabImageSize(self: *const Self) usize {
            const n = self.len();
            return SlabLayout.compute(n, self.dim, self.hiSlotCount(), self.l0Stride(), self.hiStride()).file_len;
        }

        /// Atomically write an HMLS slab file (`path.tmp` → fsync → rename).
        /// The live snapshot is never opened writeable; a crash leaves `path` intact.
        pub fn writeSlabs(self: *const Self, path: []const u8) !void {
            var tmp_buf: [posix.PATH_MAX]u8 = undefined;
            const tmp = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{path});
            const fd = try posix.openat(posix.AT.FDCWD, tmp, .{
                .ACCMODE = .WRONLY,
                .CREAT = true,
                .TRUNC = true,
                .CLOEXEC = true,
            }, 0o644);
            var closed = false;
            defer {
                if (!closed) _ = linux.close(fd);
            }
            errdefer {
                if (posix.toPosixPath(tmp)) |z| {
                    _ = linux.unlink(&z);
                } else |_| {}
            }
            _ = try self.writeSlabsAt(fd, 0);
            switch (posix.errno(linux.fdatasync(fd))) {
                .SUCCESS => {},
                else => return error.SyncFailed,
            }
            _ = linux.close(fd);
            closed = true;
            const z_tmp = try posix.toPosixPath(tmp);
            const z_dst = try posix.toPosixPath(path);
            switch (posix.errno(linux.renameat(posix.AT.FDCWD, &z_tmp, posix.AT.FDCWD, &z_dst))) {
                .SUCCESS => {},
                else => return error.RenameFailed,
            }
        }

        /// Write an HMLS image starting at `hmls_off` in an already-open file.
        /// Used by persist to prefix a SNAP envelope. Returns image size.
        pub fn writeSlabsAt(self: *const Self, fd: linux.fd_t, hmls_off: u64) !u64 {
            const n = self.len();
            const layout = SlabLayout.compute(n, self.dim, self.hiSlotCount(), self.l0Stride(), self.hiStride());
            var hdr: [SLAB_HEADER_SIZE]u8 = undefined;
            writeHeaderBytes(&hdr, layout, self);
            try sysPwriteAll(fd, &hdr, hmls_off);

            const base = hmls_off;
            try writeConcat(fd, base + layout.levels_off, std.mem.sliceAsBytes(self.mappedLevels()), std.mem.sliceAsBytes(self.levels.items));
            try writeConcat(fd, base + layout.qscales_off, std.mem.sliceAsBytes(self.mappedQscales()), std.mem.sliceAsBytes(self.qscales.items));
            try writeConcat(fd, base + layout.vectors_off, std.mem.sliceAsBytes(self.mappedVectors()), std.mem.sliceAsBytes(self.vectors_flat.items));
            try writeConcat(fd, base + layout.qvecs_off, std.mem.sliceAsBytes(self.mappedQvecs()), std.mem.sliceAsBytes(self.qvecs_flat.items));
            try writeConcat(fd, base + layout.l0_deg_off, std.mem.sliceAsBytes(self.mappedL0Deg()), std.mem.sliceAsBytes(self.l0_deg.items));
            try writeConcat(fd, base + layout.l0_nbrs_off, std.mem.sliceAsBytes(self.mappedL0Nbrs()), std.mem.sliceAsBytes(self.l0_nbrs.items));

            const mapped_hi = if (self.slab_map) |m| m.hi_slots else 0;
            try writeConcat(fd, base + layout.hi_start_off, std.mem.sliceAsBytes(self.mappedHiStart()), &.{});
            if (self.hi_start.items.len > 0) {
                var bias_buf: std.ArrayList(u32) = .empty;
                defer bias_buf.deinit(self.allocator);
                try bias_buf.ensureTotalCapacity(self.allocator, self.hi_start.items.len);
                const bias: u32 = @intCast(mapped_hi);
                for (self.hi_start.items) |v| bias_buf.appendAssumeCapacity(v + bias);
                try sysPwriteAll(fd, std.mem.sliceAsBytes(bias_buf.items), base + layout.hi_start_off + self.mappedHiStart().len * 4);
            }
            try writeConcat(fd, base + layout.hi_deg_off, std.mem.sliceAsBytes(self.mappedHiDeg()), std.mem.sliceAsBytes(self.hi_deg.items));
            try writeConcat(fd, base + layout.hi_nbrs_off, std.mem.sliceAsBytes(self.mappedHiNbrs()), std.mem.sliceAsBytes(self.hi_nbrs.items));

            const end = hmls_off + layout.file_len;
            switch (posix.errno(linux.ftruncate(fd, @intCast(end)))) {
                .SUCCESS => {},
                else => return error.TruncateFailed,
            }
            return layout.file_len;
        }

        /// Sparse HMLS of `n` zero vectors / empty graph. Cheap RSS fixture.
        pub fn writeBlankSlabs(path: []const u8, n: usize, dim: usize, opts: Options) !void {
            var index = Self.init(std.heap.page_allocator, dim, opts);
            defer index.deinit();
            if (n == 0) {
                @atomicStore(u32, &index.entry_id, std.math.maxInt(u32), .release);
            } else {
                index.setEntryPoint(0);
            }
            const layout = SlabLayout.compute(n, dim, 0, index.l0Stride(), index.hiStride());
            var tmp_buf: [posix.PATH_MAX]u8 = undefined;
            const tmp = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{path});
            const fd = try posix.openat(posix.AT.FDCWD, tmp, .{
                .ACCMODE = .WRONLY,
                .CREAT = true,
                .TRUNC = true,
                .CLOEXEC = true,
            }, 0o644);
            var hdr: [SLAB_HEADER_SIZE]u8 = undefined;
            writeHeaderBytes(&hdr, layout, &index);
            const wr = sysPwriteAll(fd, &hdr, 0);
            const tr = linux.ftruncate(fd, @intCast(layout.file_len));
            const ds = linux.fdatasync(fd);
            _ = linux.close(fd);
            wr catch {
                const z = posix.toPosixPath(tmp) catch return error.WriteFailed;
                _ = linux.unlink(&z);
                return error.WriteFailed;
            };
            if (posix.errno(tr) != .SUCCESS or posix.errno(ds) != .SUCCESS) {
                const z = posix.toPosixPath(tmp) catch return error.WriteFailed;
                _ = linux.unlink(&z);
                return error.WriteFailed;
            }
            const z_tmp = try posix.toPosixPath(tmp);
            const z_dst = try posix.toPosixPath(path);
            switch (posix.errno(linux.renameat(posix.AT.FDCWD, &z_tmp, posix.AT.FDCWD, &z_dst))) {
                .SUCCESS => {},
                else => return error.RenameFailed,
            }
        }

        fn sliceField(comptime T: type, bytes: []u8, off: usize, count: usize) ![]T {
            const nbytes = count * @sizeOf(T);
            if (off + nbytes > bytes.len) return error.Truncated;
            if (count == 0) return @as([*]T, undefined)[0..0];
            if (off % @alignOf(T) != 0) return error.Misaligned;
            const raw = bytes[off..][0..nbytes];
            return @as([*]T, @ptrCast(@alignCast(raw.ptr)))[0..count];
        }

        fn attachMapped(self: *Self, bytes: []align(std.heap.page_size_min) u8, hmls_off: usize) !void {
            if (hmls_off + SLAB_HEADER_SIZE > bytes.len) return error.Truncated;
            const parsed = try parseLayout(bytes[hmls_off..]);
            const L = parsed.layout;
            if (self.dim == 0) self.dim = L.dim;
            if (self.dim != L.dim) return error.DimensionMismatch;
            self.opts.m = parsed.m;
            self.opts.m0_factor = parsed.m0_factor;
            self.opts.ef_construction = parsed.efc;
            self.setMaxLevel(parsed.max_level);
            if (L.n == 0) {
                @atomicStore(u32, &self.entry_id, std.math.maxInt(u32), .release);
            } else {
                self.setEntryPoint(parsed.entry);
            }
            @atomicStore(usize, &self.published, L.n, .release);

            const base = hmls_off;
            const l0_stride = self.l0Stride();
            const hi_stride = self.hiStride();
            const need = base + L.file_len;
            if (bytes.len < need) return error.Truncated;

            self.slab_map = .{
                .bytes = bytes,
                .n = L.n,
                .hi_slots = L.hi_slots,
                .levels = try sliceField(u32, bytes, base + L.levels_off, L.n),
                .qscales = try sliceField(f32, bytes, base + L.qscales_off, L.n),
                .vectors = try sliceField(f32, bytes, base + L.vectors_off, L.n * L.dim),
                .qvecs = try sliceField(i8, bytes, base + L.qvecs_off, L.n * L.dim),
                .l0_deg = try sliceField(u16, bytes, base + L.l0_deg_off, L.n),
                .l0_nbrs = try sliceField(u32, bytes, base + L.l0_nbrs_off, L.n * l0_stride),
                .hi_start = try sliceField(u32, bytes, base + L.hi_start_off, L.n),
                .hi_deg = try sliceField(u16, bytes, base + L.hi_deg_off, L.hi_slots),
                .hi_nbrs = try sliceField(u32, bytes, base + L.hi_nbrs_off, L.hi_slots * hi_stride),
            };
        }

        fn hmlsOffsetIn(bytes: []const u8) !usize {
            if (bytes.len < 4) return error.Truncated;
            const magic = std.mem.readInt(u32, bytes[0..4], .little);
            if (magic == SLAB_MAGIC) return 0;
            if (magic != SNAP_MAGIC_DUP) return error.BadMagic;
            if (bytes.len < 32) return error.Truncated;
            const version = std.mem.readInt(u32, bytes[4..8], .little);
            const n: usize = @intCast(std.mem.readInt(u64, bytes[24..32], .little));
            const raw_off = 32 + n * 8;
            // v2 persist slab snapshots page-align the HMLS image.
            if (version == 2) return pageAlign(raw_off);
            return raw_off;
        }

        /// mmap a raw HMLS file or a persist v2 envelope+HMLS file.
        /// MAP_PRIVATE: reads demand-page; writes COW and never dirty the file.
        pub fn loadMmap(self: *Self, path: []const u8) !void {
            if (self.len() != 0) return error.NotEmpty;
            const fd = try posix.openat(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            const size64 = linux.lseek(fd, 0, linux.SEEK.END);
            if (posix.errno(size64) != .SUCCESS) {
                _ = linux.close(fd);
                return error.SeekFailed;
            }
            const size: usize = @intCast(size64);
            if (size == 0) {
                _ = linux.close(fd);
                return error.Truncated;
            }
            const mapped = posix.mmap(
                null,
                size,
                .{ .READ = true, .WRITE = true },
                .{ .TYPE = .PRIVATE },
                fd,
                0,
            ) catch {
                _ = linux.close(fd);
                return error.MmapFailed;
            };
            _ = linux.close(fd);
            errdefer posix.munmap(mapped);
            const off = try hmlsOffsetIn(mapped);
            try self.attachMapped(mapped, off);
            const n = self.mappedCount();
            try self.nbr_gen.ensureTotalCapacity(self.allocator, n);
            try self.vec_gen.ensureTotalCapacity(self.allocator, n);
            var gi: usize = 0;
            while (gi < n) : (gi += 1) {
                self.nbr_gen.appendAssumeCapacity(0);
                self.vec_gen.appendAssumeCapacity(0);
            }
        }

        fn copySlabInto(comptime T: type, list: *std.ArrayList(T), alloc: std.mem.Allocator, src: []const T) !void {
            const dest = try list.addManyAsSlice(alloc, src.len);
            if (src.len > 0) @memcpy(dest, src);
        }

        /// Copy-in HMLS image (load-via-alloc). Same bytes as `loadMmap`, full RSS.
        pub fn loadSlabsCopy(self: *Self, bytes: []const u8) !void {
            if (self.len() != 0) return error.NotEmpty;
            const off = try hmlsOffsetIn(bytes);
            if (off + SLAB_HEADER_SIZE > bytes.len) return error.Truncated;
            const parsed = try parseLayout(bytes[off..]);
            const L = parsed.layout;
            if (self.dim == 0) self.dim = L.dim;
            if (self.dim != L.dim) return error.DimensionMismatch;
            self.opts.m = parsed.m;
            self.opts.m0_factor = parsed.m0_factor;
            self.opts.ef_construction = parsed.efc;
            self.setMaxLevel(parsed.max_level);
            if (L.n == 0) {
                @atomicStore(u32, &self.entry_id, std.math.maxInt(u32), .release);
            } else {
                self.setEntryPoint(parsed.entry);
            }
            @atomicStore(usize, &self.published, L.n, .release);
            if (off + L.file_len > bytes.len) return error.Truncated;

            const alloc = self.allocator;
            const l0_stride = self.l0Stride();
            const hi_stride = self.hiStride();
            const base = bytes[off..];

            const take = struct {
                fn sl(comptime T: type, b: []const u8, o: usize, c: usize) ![]const T {
                    const nbytes = c * @sizeOf(T);
                    if (o + nbytes > b.len) return error.Truncated;
                    if (c == 0) return &[_]T{};
                    return @as([*]const T, @ptrCast(@alignCast(b[o..].ptr)))[0..c];
                }
            };
            try copySlabInto(u32, &self.levels, alloc, try take.sl(u32, base, L.levels_off, L.n));
            try copySlabInto(f32, &self.qscales, alloc, try take.sl(f32, base, L.qscales_off, L.n));
            try copySlabInto(f32, &self.vectors_flat, alloc, try take.sl(f32, base, L.vectors_off, L.n * L.dim));
            try copySlabInto(i8, &self.qvecs_flat, alloc, try take.sl(i8, base, L.qvecs_off, L.n * L.dim));
            try copySlabInto(u16, &self.l0_deg, alloc, try take.sl(u16, base, L.l0_deg_off, L.n));
            try copySlabInto(u32, &self.l0_nbrs, alloc, try take.sl(u32, base, L.l0_nbrs_off, L.n * l0_stride));
            try copySlabInto(u32, &self.hi_start, alloc, try take.sl(u32, base, L.hi_start_off, L.n));
            try copySlabInto(u16, &self.hi_deg, alloc, try take.sl(u16, base, L.hi_deg_off, L.hi_slots));
            try copySlabInto(u32, &self.hi_nbrs, alloc, try take.sl(u32, base, L.hi_nbrs_off, L.hi_slots * hi_stride));
            try self.nbr_gen.ensureTotalCapacity(alloc, L.n);
            try self.vec_gen.ensureTotalCapacity(alloc, L.n);
            var gi: usize = 0;
            while (gi < L.n) : (gi += 1) {
                self.nbr_gen.appendAssumeCapacity(0);
                self.vec_gen.appendAssumeCapacity(0);
            }
        }

        /// Read an HMLS (or persist v2) file into heap ArrayLists. No mmap.
        pub fn loadSlabsFile(self: *Self, path: []const u8) !void {
            if (self.len() != 0) return error.NotEmpty;
            const fd = try posix.openat(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            defer _ = linux.close(fd);
            var peek: [4096]u8 = undefined;
            try sysPreadAll(fd, peek[0..32], 0);
            const off = try hmlsOffsetIn(peek[0..32]);
            var hdr: [SLAB_HEADER_SIZE]u8 = undefined;
            try sysPreadAll(fd, &hdr, off);
            const parsed = try parseLayout(&hdr);
            const L = parsed.layout;
            if (self.dim == 0) self.dim = L.dim;
            if (self.dim != L.dim) return error.DimensionMismatch;
            self.opts.m = parsed.m;
            self.opts.m0_factor = parsed.m0_factor;
            self.opts.ef_construction = parsed.efc;
            self.setMaxLevel(parsed.max_level);
            if (L.n == 0) {
                @atomicStore(u32, &self.entry_id, std.math.maxInt(u32), .release);
            } else {
                self.setEntryPoint(parsed.entry);
            }
            @atomicStore(usize, &self.published, L.n, .release);

            const alloc = self.allocator;
            const l0_stride = self.l0Stride();
            const hi_stride = self.hiStride();

            const readInto = struct {
                fn go(comptime T: type, list: *std.ArrayList(T), a: std.mem.Allocator, f: linux.fd_t, file_off: u64, count: usize) !void {
                    const dest = try list.addManyAsSlice(a, count);
                    if (count == 0) return;
                    try sysPreadAll(f, std.mem.sliceAsBytes(dest), file_off);
                }
            };
            try readInto.go(u32, &self.levels, alloc, fd, off + L.levels_off, L.n);
            try readInto.go(f32, &self.qscales, alloc, fd, off + L.qscales_off, L.n);
            try readInto.go(f32, &self.vectors_flat, alloc, fd, off + L.vectors_off, L.n * L.dim);
            try readInto.go(i8, &self.qvecs_flat, alloc, fd, off + L.qvecs_off, L.n * L.dim);
            try readInto.go(u16, &self.l0_deg, alloc, fd, off + L.l0_deg_off, L.n);
            try readInto.go(u32, &self.l0_nbrs, alloc, fd, off + L.l0_nbrs_off, L.n * l0_stride);
            try readInto.go(u32, &self.hi_start, alloc, fd, off + L.hi_start_off, L.n);
            try readInto.go(u16, &self.hi_deg, alloc, fd, off + L.hi_deg_off, L.hi_slots);
            try readInto.go(u32, &self.hi_nbrs, alloc, fd, off + L.hi_nbrs_off, L.hi_slots * hi_stride);
            try self.nbr_gen.ensureTotalCapacity(alloc, L.n);
            try self.vec_gen.ensureTotalCapacity(alloc, L.n);
            var gi: usize = 0;
            while (gi < L.n) : (gi += 1) {
                self.nbr_gen.appendAssumeCapacity(0);
                self.vec_gen.appendAssumeCapacity(0);
            }
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
    try queue.append(alloc, index.entryPoint().?);
    visited.set(index.entryPoint().?);
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
    try std.testing.expectEqual(index.entryPoint(), loaded.entryPoint());
    try std.testing.expectEqual(index.getMaxLevel(), loaded.getMaxLevel());
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
    try std.testing.expectEqual(index.entryPoint(), loaded.entryPoint());
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

fn rssKiBSelf() ?u64 {
    return rss.currentKiB();
}

test "hnsw slab mmap reopen + insert append buffer" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const dim = 32;
    var rng = std.Random.DefaultPrng.init(11);
    var index = Hnsw(void).init(alloc, dim, .{});
    defer index.deinit();

    var v: [dim]f32 = undefined;
    for (0..60) |_| {
        for (&v) |*x| x.* = rng.random().floatNorm(f32);
        _ = try index.insert(&v);
    }
    @memset(&v, 0);
    v[2] = 1;
    const planted = try index.insert(&v);
    const before = try index.search(&v, 5, 64, alloc);

    var path_buf: [160]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/tmp/openpuffer-mmap-roundtrip-{d}-{x}.slabs", .{
        if (builtin.os.tag == .linux) @as(u64, @intCast(std.os.linux.getpid())) else 0,
        @intFromPtr(&path_buf),
    });
    try index.writeSlabs(path);

    var loaded = Hnsw(void).init(alloc, dim, .{});
    defer loaded.deinit();
    try loaded.loadMmap(path);
    try std.testing.expect(loaded.isMmapBacked());
    try std.testing.expectEqual(index.len(), loaded.len());
    try std.testing.expectEqual(index.entryPoint(), loaded.entryPoint());
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
    for (before, after) |b, a| {
        try std.testing.expectEqual(b.id, a.id);
        try std.testing.expectApproxEqAbs(b.distance, a.distance, 1e-6);
    }

    var extra: [dim]f32 = undefined;
    @memset(&extra, 0);
    extra[5] = 1;
    const new_id = try loaded.insert(&extra);
    try std.testing.expectEqual(index.len() + 1, loaded.len());
    try std.testing.expect(new_id >= index.len());
    const found = try loaded.search(&extra, 1, 16, alloc);
    try std.testing.expectEqual(new_id, found[0].id);

    var flipped_v: [dim]f32 = undefined;
    @memset(&flipped_v, 0);
    flipped_v[7] = 1;
    try loaded.update(planted, &flipped_v);
    const flipped = try loaded.search(&flipped_v, 3, 32, alloc);
    try std.testing.expectEqual(planted, flipped[0].id);

    var copied = Hnsw(void).init(alloc, dim, .{});
    defer copied.deinit();
    try copied.loadSlabsFile(path);
    try std.testing.expect(!copied.isMmapBacked());
    try std.testing.expectEqual(index.len(), copied.len());
}

test "slab mmap vs alloc RSS (blank slabs)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const scale = builtin.mode != .debug;
    const cases = [_]struct { n: usize, dim: usize }{
        .{ .n = if (scale) 50_000 else 256, .dim = if (scale) 1536 else 64 },
        .{ .n = if (scale) 200_000 else 0, .dim = 1536 },
    };
    const gpa = std.heap.page_allocator;
    for (cases) |c| {
        if (c.n == 0) continue;
        var path_buf: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "/tmp/openpuffer-mmap-rss-{d}-{d}-{x}.slabs", .{
            c.n,
            if (builtin.os.tag == .linux) @as(u64, @intCast(std.os.linux.getpid())) else 0,
            @intFromPtr(&path_buf),
        });
        try Hnsw(void).writeBlankSlabs(path, c.n, c.dim, .{});

        const baseline = rssKiBSelf() orelse 0;

        var mapped = Hnsw(void).init(gpa, c.dim, .{});
        try mapped.loadMmap(path);
        const mmap_idle = rssKiBSelf() orelse 0;
        const _touch = mapped.vectorConst(0)[0];
        _ = _touch;
        mapped.deinit();

        var copied = Hnsw(void).init(gpa, c.dim, .{});
        try copied.loadSlabsFile(path);
        const alloc_rss = rssKiBSelf() orelse 0;
        copied.deinit();

        const mmap_delta = mmap_idle -| baseline;
        const alloc_delta = alloc_rss -| baseline;
        if (c.n >= 10_000) {
            try std.testing.expect(mmap_delta * 4 < alloc_delta);
            try std.testing.expect(alloc_delta + 1024 >= (c.n * c.dim * 5) / 1024 / 2);
        }
    }
}

test "publishInsert then spliceBackEdges matches commitInsert" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();
    const dim = 8;
    var rng = std.Random.DefaultPrng.init(3);

    var via_commit = Hnsw(void).init(alloc, dim, .{ .seed = 99 });
    defer via_commit.deinit();
    var via_split = Hnsw(void).init(alloc, dim, .{ .seed = 99 });
    defer via_split.deinit();

    var v: [dim]f32 = undefined;
    for (0..40) |_| {
        for (&v) |*x| x.* = rng.random().floatNorm(f32);
        _ = try via_commit.insert(&v);
        _ = try via_split.insert(&v);
    }
    for (&v) |*x| x.* = rng.random().floatNorm(f32);
    const level = via_split.nextLevel();
    _ = via_commit.nextLevel();
    var plan_split = try via_split.planInsert(&v, level, alloc);
    const plan_commit = try via_commit.planInsert(&v, level, alloc);
    const id = try via_split.publishInsert(&plan_split);
    try via_split.spliceBackEdges(&plan_split, id);
    _ = try via_commit.commitInsert(plan_commit);

    try std.testing.expectEqual(via_commit.len(), via_split.len());
    try std.testing.expectEqual(via_commit.entryPoint(), via_split.entryPoint());
    var nid: u32 = 0;
    while (nid < via_commit.len()) : (nid += 1) {
        var layer: u32 = 0;
        while (layer < via_commit.layerCount(nid)) : (layer += 1) {
            try std.testing.expectEqualSlices(u32, via_commit.neighbors(nid, layer), via_split.neighbors(nid, layer));
        }
    }
    const q = via_split.vectorConst(id);
    const a = try via_commit.search(q, 5, 32, alloc);
    const b = try via_split.search(q, 5, 32, alloc);
    try std.testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try std.testing.expectEqual(x.id, y.id);
        try std.testing.expectApproxEqAbs(x.distance, y.distance, 1e-6);
    }
}

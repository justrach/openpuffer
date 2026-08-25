//! openpuffer — in-memory vector search engine + dual benchmark against turbopuffer.
//!
//! Commands:
//!   selftest            run the built-in correctness tests
//!   bench-synthetic     local HNSW vs brute-force on synthetic vectors
//!   bench-live          Gemini embeddings -> load BOTH openpuffer and turbopuffer,
//!                       then time queries against both and score recall
//!                       (openpuffer ANN vs exact ground truth, tpuf vs same GT)

const std = @import("std");
const vecmath = @import("vector.zig");
const hnsw_mod = @import("hnsw.zig");
const gemini = @import("gemini.zig");
const tpuf_mod = @import("tpuf.zig");
const server_mod = @import("server.zig");
const s3_mod = @import("s3.zig");

const Hnsw = hnsw_mod.Hnsw(void);

/// Monotonic stopwatch on top of std.Io.
pub const Sw = struct {
    t0: std.Io.Timestamp,
    io: std.Io,
    fn start(io: std.Io) Sw {
        return .{ .t0 = std.Io.Timestamp.now(io, .awake), .io = io };
    }
    fn readNs(self: *const Sw) u64 {
        const d = self.t0.durationTo(std.Io.Timestamp.now(self.io, .awake));
        return @intCast(@max(0, d.nanoseconds));
    }
};

const BenchOpts = struct {
    n: usize = 512,
    q: usize = 30,
    dim: usize = 768,
    k: usize = 10,
    ef: usize = 200,
    namespace: []const u8 = "openpuffer-bench-1",
    model: []const u8 = "gemini-embedding-2",
    tpuf_key: []const u8 = "",
    google_key: []const u8 = "",
    local_endpoint: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(gpa);
    var args_it = init.minimal.args.iterate();
    defer args_it.deinit();
    _ = args_it.next(); // argv0
    while (args_it.next()) |a| try args.append(gpa, a);

    var stdout_buf: [8192]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var w = stdout_file.writer(io, &stdout_buf);
    defer w.interface.flush() catch {};

    const cmd = if (args.items.len > 0) args.items[0] else "help";

    if (std.mem.eql(u8, cmd, "selftest")) {
        _ = vecmath.cosineDistance(&.{ 1, 0 }, &.{ 1, 0 });
        try w.interface.print("selftest ok\n", .{});
    } else if (std.mem.eql(u8, cmd, "bench-synthetic")) {
        try benchSynthetic(gpa, io, &w.interface, optsFromArgs(args.items[1..], .{
            .n = 20_000, .q = 100, .dim = 1536, .k = 10, .ef = 200,
        }));
    } else if (std.mem.eql(u8, cmd, "bench-live")) {
        const tpuf_key = init.environ_map.get("TURBOPUFFER_API_KEY") orelse {
            try w.interface.print("missing TURBOPUFFER_API_KEY\n", .{});
            return error.MissingKey;
        };
        const google_key = init.environ_map.get("GEMINI_API_KEY") orelse {
            try w.interface.print("missing GEMINI_API_KEY\n", .{});
            return error.MissingKey;
        };
        try benchLive(gpa, io, &w.interface, optsFromArgs(args.items[1..], .{
            .n = 512, .q = 30, .dim = 768, .k = 10, .ef = 200,
            .tpuf_key = tpuf_key, .google_key = google_key,
        }));
    } else if (std.mem.eql(u8, cmd, "serve")) {
        var port: u16 = 8080;
        var ef: u32 = 256;
        var workers: ?usize = null;
        var s3_cfg: ?s3_mod.Config = null;
        {
            const a = args.items[1..];
            if (optOf(a, "--port")) |v| port = std.fmt.parseInt(u16, v, 10) catch 8080;
            if (optOf(a, "--ef")) |v| ef = std.fmt.parseInt(u32, v, 10) catch 256;
            if (optOf(a, "--workers")) |v| workers = std.fmt.parseInt(usize, v, 10) catch null;
            if (init.environ_map.get("OPENPUFFER_WORKERS")) |s| {
                workers = std.fmt.parseInt(usize, s, 10) catch workers;
            }
            if (optOf(a, "--s3-bucket")) |bucket| {
                const creds = s3_mod.resolveCredentials(gpa, io, init.environ_map) catch {
                    try w.interface.print("no AWS/R2 credentials found (env AWS_ACCESS_KEY_ID/SECRET or ~/.aws/credentials)\n", .{});
                    return error.MissingCredentials;
                };
                s3_cfg = .{
                    .access_key = creds.access,
                    .secret_key = creds.secret,
                    .region = optOf(a, "--s3-region") orelse creds.region,
                    .bucket = bucket,
                    .endpoint = optOf(a, "--s3-endpoint"),
                };
            }
        }
        try server_mod.serve(gpa, io, .{ .port = port, .ef = ef, .workers = workers, .s3_cfg = s3_cfg }, &w.interface);
    } else {
        try w.interface.writeAll(
            \\openpuffer — fast vector search engine + turbopuffer benchmark
            \\
            \\usage:
            \\  openpuffer selftest
            \\  openpuffer bench-synthetic [--n 20000] [--queries 100] [--dim 1536] [--k 10] [--ef 200]
            \\  openpuffer serve [--port 8080] [--ef 256] [--workers N]
            \\                  [--s3-bucket B] [--s3-region R] [--s3-endpoint URL]
            \\                  (OPENPUFFER_WORKERS is an alias for --workers)
            \\  openpuffer bench-live [--namespace openpuffer-bench-1] [--n 512] [--queries 30] [--dim 768]
            \\                  [--model gemini-embedding-2] [--k 10] [--ef 200]
            \\                  (env: TURBOPUFFER_API_KEY, GEMINI_API_KEY)
            \\
        );
    }
}

fn optOf(args: []const []const u8, name: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], name) and i + 1 < args.len) return args[i + 1];
    }
    return null;
}

fn optsFromArgs(args: []const []const u8, base: BenchOpts) BenchOpts {
    var o = base;
    if (optOf(args, "--n")) |v| o.n = std.fmt.parseInt(usize, v, 10) catch o.n;
    if (optOf(args, "--queries")) |v| o.q = std.fmt.parseInt(usize, v, 10) catch o.q;
    if (optOf(args, "--dim")) |v| o.dim = std.fmt.parseInt(usize, v, 10) catch o.dim;
    if (optOf(args, "--k")) |v| o.k = std.fmt.parseInt(usize, v, 10) catch o.k;
    if (optOf(args, "--ef")) |v| o.ef = std.fmt.parseInt(usize, v, 10) catch o.ef;
    if (optOf(args, "--namespace")) |v| o.namespace = v;
    if (optOf(args, "--model")) |v| o.model = v;
    if (optOf(args, "--local-endpoint")) |v| o.local_endpoint = v;
    return o;
}

// ---------------------------------------------------------------------------

const Lat = struct {
    samples: []u64,
    len: usize = 0,

    fn init(alloc: std.mem.Allocator, cap: usize) !Lat {
        return .{ .samples = try alloc.alloc(u64, cap) };
    }
    fn push(self: *Lat, ns: u64) void {
        if (self.len < self.samples.len) {
            self.samples[self.len] = ns;
            self.len += 1;
        }
    }
    fn pct(self: *Lat, p: f64) f64 {
        if (self.len == 0) return 0;
        std.mem.sort(u64, self.samples[0..self.len], {}, comptime std.sort.asc(u64));
        const idx = @min(self.len - 1, @as(usize, @intFromFloat(p * @as(f64, @floatFromInt(self.len)))));
        return @as(f64, @floatFromInt(self.samples[idx])) / 1e6; // ms
    }
    fn mean(self: *Lat) f64 {
        if (self.len == 0) return 0;
        var s: u64 = 0;
        for (self.samples[0..self.len]) |x| s += x;
        return @as(f64, @floatFromInt(s)) / @as(f64, @floatFromInt(self.len)) / 1e6;
    }
};

fn report(out: *std.Io.Writer, label: []const u8, lat: *Lat, qps: f64) !void {
    try out.print(
        "{s:<14} p50 {d:7.3}ms  p95 {d:7.3}ms  p99 {d:7.3}ms  mean {d:7.3}ms  {d:8.1} qps\n",
        .{ label, lat.pct(0.50), lat.pct(0.95), lat.pct(0.99), lat.mean(), qps },
    );
}

/// Exact top-k by cosine distance (ground truth / brute force baseline).
fn bruteForce(alloc: std.mem.Allocator, data: []const []const f32, q: []const f32, k: usize) ![]u32 {
    const Dist = struct { id: u32, d: f32 };
    var all: std.ArrayList(Dist) = .empty;
    defer all.deinit(alloc);
    try all.ensureTotalCapacity(alloc, data.len);
    const nq = try alloc.dupe(f32, q);
    defer alloc.free(nq);
    vecmath.normalize(nq);
    for (data, 0..) |v, i| {
        all.appendAssumeCapacity(.{ .id = @intCast(i), .d = vecmath.cosineDistance(nq, v) });
    }
    std.mem.sort(Dist, all.items, {}, struct {
        fn lt(_: void, a: Dist, b: Dist) bool {
            return a.d < b.d;
        }
    }.lt);
    const out = try alloc.alloc(u32, @min(k, all.items.len));
    for (out, 0..) |*o, i| o.* = all.items[i].id;
    return out;
}

fn recallAtK(gt: []const u32, got: []const u32) f64 {
    if (gt.len == 0) return 0;
    var hits: usize = 0;
    outer: for (got) |g| {
        for (gt) |t| {
            if (g == t) {
                hits += 1;
                continue :outer;
            }
        }
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(gt.len));
}

// ---------------------------------------------------------------------------

fn benchSynthetic(gpa: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, o: BenchOpts) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();

    try out.print("generating {d} random {d}-dim vectors...\n", .{ o.n, o.dim });
    const data: [][]f32 = try alloc.alloc([]f32, o.n);
    for (data) |*row| {
        row.* = try alloc.alloc(f32, o.dim);
        for (row.*) |*x| x.* = rand.floatNorm(f32);
        vecmath.normalize(row.*);
    }

    var index = Hnsw.init(gpa, o.dim, .{});
    defer index.deinit();

    var build_timer = Sw.start(io);
    for (data) |v| _ = try index.insert(v);
    const build_ms = @as(f64, @floatFromInt(build_timer.readNs())) / 1e6;
    try out.print("index built: {d} vectors in {d:.1}ms ({d:.0} vec/s)\n", .{
        o.n, build_ms, @as(f64, @floatFromInt(o.n)) / (build_ms / 1000.0),
    });

    const queries: [][]f32 = try alloc.alloc([]f32, o.q);
    for (queries) |*row| {
        row.* = try alloc.alloc(f32, o.dim);
        for (row.*) |*x| x.* = rand.floatNorm(f32);
        vecmath.normalize(row.*);
    }

    // local ANN
    var lat_local = try Lat.init(alloc, o.q);
    var recall_sum: f64 = 0;
    for (queries) |q| {
        var t = Sw.start(io);
        const res = try index.search(q, o.k, @intCast(o.ef), alloc);
        lat_local.push(t.readNs());
        var got: [64]u32 = undefined;
        for (res, 0..) |r, i| got[i] = r.id;
        const gt = try bruteForce(alloc, data, q, o.k);
        recall_sum += recallAtK(gt, got[0..res.len]);
    }
    const qps_local = @as(f64, @floatFromInt(o.q)) / (lat_local.mean() / 1000.0);
    try report(out, "openpuffer (ANN)", &lat_local, qps_local);
    try out.print("openpuffer recall@{d} vs exact: {d:.4}\n", .{ o.k, recall_sum / @as(f64, @floatFromInt(o.q)) });

    // brute force baseline
    var lat_bf = try Lat.init(alloc, o.q);
    for (queries) |q| {
        var t = Sw.start(io);
        _ = try bruteForce(alloc, data, q, o.k);
        lat_bf.push(t.readNs());
    }
    const qps_bf = @as(f64, @floatFromInt(o.q)) / (lat_bf.mean() / 1000.0);
    try report(out, "exact scan", &lat_bf, qps_bf);
}

// ---------------------------------------------------------------------------

const SAMPLE_SENTENCES = [_][]const u8{
    "the quick brown fox jumps over the lazy dog",
    "a vector database stores high dimensional embeddings for similarity search",
    "turbopuffer is a vector search engine built on object storage",
    "hierarchical navigable small world graphs enable fast approximate nearest neighbor search",
    "gemini embeddings map text and images into a unified semantic space",
    "cosine distance measures the angle between two vectors",
    "quantization reduces vector size at the cost of some accuracy",
    "the cat slept on the warm windowsill in the afternoon sun",
    "distributed systems trade consistency for latency",
    "rust and zig are systems programming languages with memory safety features",
    "a bloom filter probabilistically tests set membership",
    "bm25 ranks documents by term frequency and inverse document frequency",
    "the stock market rallied after the central bank cut interest rates",
    "machine learning models are trained on large datasets with gradient descent",
    "sharded indexes spread data across many nodes for scale",
    "the chef simmered the sauce for three hours until it thickened",
    "satellite imagery helps meteorologists track hurricanes",
    "a hash map provides amortized constant time lookups",
    "recurrent neural networks process sequential data one token at a time",
    "the marathon runner hydrated every five kilometers",
};

fn benchLive(gpa: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, o: BenchOpts) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    var emb = gemini.Embedder.init(gpa, io, .{
        .api_key = o.google_key,
        .model = o.model,
        .dim = o.dim,
    });
    defer emb.deinit();

    // 1) embed corpus (repeat sample sentences to reach n) + queries
    try out.print("embedding {d} docs + {d} queries with {s} (dim {d})...\n", .{ o.n, o.q, o.model, o.dim });
    var docs: std.ArrayList([]const u8) = .empty;
    var doc_vecs: std.ArrayList([]f32) = .empty;
    var i: usize = 0;
    while (i < o.n) {
        const batch = @min(100, o.n - i);
        var req: std.ArrayList([]const u8) = .empty;
        defer req.deinit(alloc);
        for (0..batch) |j| {
            const gi = i + j;
            // combine two topics so every document embedding is unique
            const text = try std.fmt.allocPrint(alloc, "{s}. {s}.", .{
                SAMPLE_SENTENCES[gi % SAMPLE_SENTENCES.len],
                SAMPLE_SENTENCES[(gi / SAMPLE_SENTENCES.len + 3) % SAMPLE_SENTENCES.len],
            });
            try req.append(alloc, text);
        }
        const vs = try emb.embedBatch(req.items, "RETRIEVAL_DOCUMENT");
        defer alloc.free(vs);
        for (vs) |v| {
            try doc_vecs.append(alloc, v);
            try docs.append(gpa, SAMPLE_SENTENCES[docs.items.len % SAMPLE_SENTENCES.len]);
        }
        i += batch;
        try out.print("  embedded {d}/{d}\n", .{ i, o.n });
    }

    var q_vecs: std.ArrayList([]f32) = .empty;
    {
        var req: std.ArrayList([]const u8) = .empty;
        defer req.deinit(alloc);
        for (0..o.q) |j| try req.append(alloc, SAMPLE_SENTENCES[(j * 7) % SAMPLE_SENTENCES.len]);
        const vs = try emb.embedBatch(req.items, "RETRIEVAL_QUERY");
        defer alloc.free(vs);
        for (vs) |v| try q_vecs.append(alloc, v);
    }

    const n = doc_vecs.items.len;
    const dim = o.dim;

    // 2) load into openpuffer (local HNSW)
    try out.print("building openpuffer index ({d} x {d})...\n", .{ n, dim });
    var index = Hnsw.init(gpa, dim, .{});
    defer index.deinit();
    var t0 = Sw.start(io);
    for (doc_vecs.items) |v| _ = try index.insert(v);
    try out.print("openpuffer build: {d:.1}ms\n", .{@as(f64, @floatFromInt(t0.readNs())) / 1e6});

    // 3) load into turbopuffer
    var tp = tpuf_mod.Client.init(gpa, io, .{ .api_key = o.tpuf_key });
    defer tp.deinit();
    try out.print("uploading {d} vectors to turbopuffer ns '{s}'...\n", .{ n, o.namespace });
    t0 = Sw.start(io);
    var off: usize = 0;
    var ids_buf: [256]u64 = undefined;
    while (off < n) {
        const batch = @min(@as(usize, 256), n - off);
        for (0..batch) |j| ids_buf[j] = @intCast(off + j);
        try tp.upsert(o.namespace, ids_buf[0..batch], doc_vecs.items[off .. off + batch]);
        off += batch;
        try out.print("  uploaded {d}/{d}\n", .{ off, n });
    }
    try out.print("tpuf upload: {d:.1}ms\n", .{@as(f64, @floatFromInt(t0.readNs())) / 1e6});
    try out.print("waiting 5s for tpuf index warmup...\n", .{});
    try io.sleep(.{ .nanoseconds = 5_000_000_000 }, .awake);

    // 4) benchmark both
    var lat_local = try Lat.init(alloc, o.q);
    var lat_tpuf = try Lat.init(alloc, o.q);
    var recall_openpuffer: f64 = 0;
    var recall_tpuf: f64 = 0;
    var hits_buf: [64]tpuf_mod.QueryHit = undefined;
    var got_buf: [64]u32 = undefined;

    for (q_vecs.items) |qv| {
        // openpuffer
        {
            var t = Sw.start(io);
            const res = try index.search(qv, o.k, @intCast(o.ef), alloc);
            lat_local.push(t.readNs());
            for (res, 0..) |r, j| got_buf[j] = r.id;
            const gt = try bruteForce(alloc, doc_vecs.items, qv, o.k);
            recall_openpuffer += recallAtK(gt, got_buf[0..res.len]);
        }
        // turbopuffer
        {
            const r = try tp.query(o.namespace, .{ .vector = qv, .top_k = o.k, .ef = o.ef }, &hits_buf);
            lat_tpuf.push(r.latency_ns);
            var got: usize = 0;
            for (hits_buf[0..r.count]) |h| {
                got_buf[got] = @intCast(h.id);
                got += 1;
            }
            const gt = try bruteForce(alloc, doc_vecs.items, qv, o.k);
            recall_tpuf += recallAtK(gt, got_buf[0..got]);
        }
    }

    const qps_tpuf = @as(f64, @floatFromInt(o.q)) / (lat_tpuf.mean() / 1000.0);
    const qps_local = @as(f64, @floatFromInt(o.q)) / (lat_local.mean() / 1000.0);
    try report(out, "openpuffer (local)", &lat_local, qps_local);
    try report(out, "turbopuffer", &lat_tpuf, qps_tpuf);
    try out.print("openpuffer recall@{d}:   {d:.4}\n", .{ o.k, recall_openpuffer / @as(f64, @floatFromInt(o.q)) });
    try out.print("tpuf recall@{d}:   {d:.4}  (vs local exact GT)\n", .{ o.k, recall_tpuf / @as(f64, @floatFromInt(o.q)) });

    if (o.local_endpoint) |ep| {
        var lc = tpuf_mod.Client.init(gpa, io, .{ .api_key = "local", .endpoint = ep });
        defer lc.deinit();
        var t_up = Sw.start(io);
        var loff: usize = 0;
        while (loff < n) {
            const batch = @min(@as(usize, 256), n - loff);
            for (0..batch) |j| ids_buf[j] = @intCast(loff + j);
            try lc.upsert(o.namespace, ids_buf[0..batch], doc_vecs.items[loff .. loff + batch]);
            loff += batch;
        }
        try out.print("local-api upload: {d:.1}ms\n", .{@as(f64, @floatFromInt(t_up.readNs())) / 1e6});

        var lat_http = try Lat.init(alloc, o.q);
        var recall_http: f64 = 0;
        for (q_vecs.items) |qv| {
            const r = try lc.query(o.namespace, .{ .vector = qv, .top_k = o.k, .ef = o.ef }, &hits_buf);
            lat_http.push(r.latency_ns);
            var got: usize = 0;
            for (hits_buf[0..r.count]) |h| {
                got_buf[got] = @intCast(h.id);
                got += 1;
            }
            const gt = try bruteForce(alloc, doc_vecs.items, qv, o.k);
            recall_http += recallAtK(gt, got_buf[0..got]);
        }
        const qps_http = @as(f64, @floatFromInt(o.q)) / (lat_http.mean() / 1000.0);
        try report(out, "openpuffer (HTTP API)", &lat_http, qps_http);
        try out.print("openpuffer HTTP recall@{d}: {d:.4}\n", .{ o.k, recall_http / @as(f64, @floatFromInt(o.q)) });
    }
}

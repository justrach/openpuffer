//! Object-storage persistence for namespaces, modeled on turbopuffer's
//! architecture: every write batch appends one WAL segment object; snapshots
//! compact consumed WAL entries. Reads stay local/in-memory.
//!
//! Layout under the bucket:
//!   openpuffer/{ns}/snapshot.bin      binary graph+vectors (preferred)
//!   openpuffer/{ns}/snapshot.json     legacy JSON vectors-only
//!   openpuffer/{ns}/wal/{seq:020}.json {"docs":[[id,[...]],...]}

const std = @import("std");
const s3 = @import("s3.zig");
const hnsw_mod = @import("hnsw.zig");
const namespace = @import("namespace.zig");

const Hnsw = hnsw_mod.Hnsw(void);

pub const SNAP_MAGIC: u32 = 0x4E53504F; // 'OPSN' le
pub const SNAP_VERSION: u32 = 1;
/// Envelope + page-aligned HMLS slab image (mmap reopen, no copy-in).
pub const SNAP_VERSION_SLAB: u32 = 2;

/// Parse a WAL object filename (prefix already stripped) into a seq number.
/// Accepts `00000000000000000001.json` and bare integers.
pub fn parseWalSeq(fname_raw: []const u8) ?u64 {
    var fname = std.mem.trim(u8, fname_raw, " \t\r\n\"");
    if (std.mem.endsWith(u8, fname, ".json")) fname = fname[0 .. fname.len - 5];
    if (std.mem.endsWith(u8, fname, ".bin")) fname = fname[0 .. fname.len - 4];
    if (fname.len == 0) return null;
    return std.fmt.parseInt(u64, fname, 10) catch null;
}

pub const SnapshotKind = enum { json, binary };

pub const Snapshot = struct {
    seq: u64,
    dim: usize,
    kind: SnapshotKind,
    /// full object body (caller frees via Store.alloc)
    bytes: []u8,
};

pub const Store = struct {
    client: *s3.Client,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, client: *s3.Client) Store {
        return .{ .client = client, .alloc = alloc };
    }

    fn keyFor(self: *Store, comptime kind: []const u8, ns: []const u8, seq: u64) ![]u8 {
        try namespace.validate(ns);
        if (comptime std.mem.eql(u8, kind, "snapshot_bin")) {
            return std.fmt.allocPrint(self.alloc, "openpuffer/{s}/snapshot.bin", .{ns});
        } else if (comptime std.mem.eql(u8, kind, "snapshot")) {
            return std.fmt.allocPrint(self.alloc, "openpuffer/{s}/snapshot.json", .{ns});
        } else {
            return std.fmt.allocPrint(self.alloc, "openpuffer/{s}/wal/{d:0>20}.json", .{ ns, seq });
        }
    }

    /// Append one write batch as a WAL segment.
    pub fn appendWal(self: *Store, ns: []const u8, seq: u64, body: []const u8) !void {
        const key = try self.keyFor("wal", ns, seq);
        defer self.alloc.free(key);
        try self.client.putObject(key, body);
    }

    pub fn putSnapshotBin(self: *Store, ns: []const u8, body: []const u8) !void {
        const key = try self.keyFor("snapshot_bin", ns, 0);
        defer self.alloc.free(key);
        try self.client.putObject(key, body);
    }

    fn getObjectOrNull(self: *Store, key: []const u8) !?[]u8 {
        return self.client.getObject(key) catch |e| switch (e) {
            error.S3RequestFailed => return null,
            else => return e,
        };
    }

    /// Fetch latest snapshot, preferring binary over legacy JSON.
    pub fn getSnapshot(self: *Store, ns: []const u8) !?Snapshot {
        return self.getSnapshotKnown(ns, true, true);
    }

    /// Like `getSnapshot` but skips keys the caller already knows are absent
    /// (avoids a ~200–400ms 404 RTT per missing object).
    pub fn getSnapshotKnown(self: *Store, ns: []const u8, has_bin: bool, has_json: bool) !?Snapshot {
        if (has_bin) {
            const bin_key = try self.keyFor("snapshot_bin", ns, 0);
            defer self.alloc.free(bin_key);
            if (try self.getObjectOrNull(bin_key)) |body| {
                if (parseBinaryHeader(body)) |hdr| {
                    return .{ .seq = hdr.seq, .dim = hdr.dim, .kind = .binary, .bytes = body };
                } else |_| {
                    self.alloc.free(body);
                }
            }
        }

        if (!has_json) return null;
        const json_key = try self.keyFor("snapshot", ns, 0);
        defer self.alloc.free(json_key);
        const body = try self.getObjectOrNull(json_key) orelse return null;
        if (parseBinaryHeader(body)) |hdr| {
            return .{ .seq = hdr.seq, .dim = hdr.dim, .kind = .binary, .bytes = body };
        } else |_| {}
        const parsed = std.json.parseFromSlice(std.json.Value, self.alloc, body, .{}) catch {
            self.alloc.free(body);
            return null;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            self.alloc.free(body);
            return null;
        }
        const obj = parsed.value.object;
        const seq: u64 = switch (obj.get("seq") orelse {
            self.alloc.free(body);
            return null;
        }) {
            .integer => |n| @intCast(n),
            else => {
                self.alloc.free(body);
                return null;
            },
        };
        const dim: usize = switch (obj.get("dim") orelse {
            self.alloc.free(body);
            return null;
        }) {
            .integer => |n| @intCast(n),
            else => {
                self.alloc.free(body);
                return null;
            },
        };
        return .{ .seq = seq, .dim = dim, .kind = .json, .bytes = body };
    }

    /// List WAL segment numbers above `after_seq`, ascending.
    pub fn listWal(self: *Store, ns: []const u8, after_seq: u64) ![]u64 {
        const prefix = try std.fmt.allocPrint(self.alloc, "openpuffer/{s}/wal/", .{ns});
        defer self.alloc.free(prefix);
        const keys = try self.client.listKeys(prefix);
        defer {
            for (keys) |k| self.alloc.free(k);
            self.alloc.free(keys);
        }
        var seqs: std.ArrayList(u64) = .empty;
        errdefer seqs.deinit(self.alloc);
        for (keys) |k| {
            const fname = if (k.len >= prefix.len) k[prefix.len..] else k;
            const seq = parseWalSeq(fname) orelse continue;
            if (seq > after_seq) try seqs.append(self.alloc, seq);
        }
        std.mem.sort(u64, seqs.items, {}, comptime std.sort.asc(u64));
        return seqs.toOwnedSlice(self.alloc);
    }

    pub fn getWalBody(self: *Store, ns: []const u8, seq: u64) ![]u8 {
        const key = try self.keyFor("wal", ns, seq);
        defer self.alloc.free(key);
        return self.client.getObject(key);
    }

    pub fn deleteWal(self: *Store, ns: []const u8, seq: u64) !void {
        const key = try self.keyFor("wal", ns, seq);
        defer self.alloc.free(key);
        try self.client.deleteObject(key);
    }

    /// Drop every WAL segment with seq <= `upto` (consumed by a snapshot).
    pub fn deleteWalUpTo(self: *Store, ns: []const u8, upto: u64) void {
        const seqs = self.listWal(ns, 0) catch return;
        defer self.alloc.free(seqs);
        for (seqs) |seq| {
            if (seq <= upto) self.deleteWal(ns, seq) catch {};
        }
    }

    /// Same as `deleteWalUpTo` but uses an already-fetched key listing (no extra LIST).
    pub fn deleteWalListed(self: *Store, ns: []const u8, keys: []const []const u8, upto: u64) void {
        const prefix = std.fmt.allocPrint(self.alloc, "openpuffer/{s}/wal/", .{ns}) catch return;
        defer self.alloc.free(prefix);
        for (keys) |k| {
            if (!std.mem.startsWith(u8, k, prefix)) continue;
            const seq = parseWalSeq(k[prefix.len..]) orelse continue;
            if (seq <= upto) self.deleteWal(ns, seq) catch {};
        }
    }

    /// Collect WAL seqs for `ns` from a prefix listing, greater than `after_seq`.
    pub fn walSeqsFromKeys(alloc: std.mem.Allocator, keys: []const []const u8, ns: []const u8, after_seq: u64) ![]u64 {
        const prefix = try std.fmt.allocPrint(alloc, "openpuffer/{s}/wal/", .{ns});
        defer alloc.free(prefix);
        var seqs: std.ArrayList(u64) = .empty;
        errdefer seqs.deinit(alloc);
        for (keys) |k| {
            if (!std.mem.startsWith(u8, k, prefix)) continue;
            const seq = parseWalSeq(k[prefix.len..]) orelse continue;
            if (seq > after_seq) try seqs.append(alloc, seq);
        }
        std.mem.sort(u64, seqs.items, {}, comptime std.sort.asc(u64));
        return seqs.toOwnedSlice(alloc);
    }

    /// Legacy JSON snapshot (vectors only). Kept so old fixtures still encode.
    pub fn buildSnapshot(
        alloc: std.mem.Allocator,
        seq: u64,
        dim: usize,
        doc_ids: []const u64,
        vectors: []const []const f32,
    ) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(alloc);
        errdefer out.deinit();
        const w = &out.writer;
        try w.print("{{\"seq\":{d},\"dim\":{d},\"docs\":[", .{ seq, dim });
        for (doc_ids, 0..) |id, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("[{d},[", .{id});
            for (vectors[i], 0..) |x, j| {
                if (j > 0) try w.writeAll(",");
                try w.print("{d}", .{x});
            }
            try w.writeAll("]]");
        }
        try w.writeAll("]}");
        return out.toOwnedSlice();
    }

    /// Binary snapshot: envelope + doc_ids + HNSW graph. Caller frees.
    pub fn buildBinarySnapshot(
        alloc: std.mem.Allocator,
        seq: u64,
        dim: usize,
        doc_ids: []const u64,
        index: *const Hnsw,
    ) ![]u8 {
        const graph = try index.serialize(alloc);
        defer alloc.free(graph);
        const n = doc_ids.len;
        const header_len: usize = 32 + n * 8;
        const buf = try alloc.alloc(u8, header_len + graph.len);
        var i: usize = 0;
        std.mem.writeInt(u32, buf[i..][0..4], SNAP_MAGIC, .little);
        i += 4;
        std.mem.writeInt(u32, buf[i..][0..4], SNAP_VERSION, .little);
        i += 4;
        std.mem.writeInt(u64, buf[i..][0..8], seq, .little);
        i += 8;
        std.mem.writeInt(u64, buf[i..][0..8], dim, .little);
        i += 8;
        std.mem.writeInt(u64, buf[i..][0..8], n, .little);
        i += 8;
        for (doc_ids) |id| {
            std.mem.writeInt(u64, buf[i..][0..8], id, .little);
            i += 8;
        }
        @memcpy(buf[i..], graph);
        return buf;
    }

    /// Local mmap-shaped snapshot: OPSN v2 envelope + page-aligned HMLS.
    /// Written atomically (`path.tmp` → fsync → rename). Reopen with
    /// `Hnsw.loadMmap(path)` — the file is never opened writeable by load.
    pub fn writeSlabSnapshotFile(
        path: []const u8,
        seq: u64,
        dim: usize,
        doc_ids: []const u64,
        index: *const Hnsw,
    ) !void {
        const linux = std.os.linux;
        const posix = std.posix;
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

        const n = doc_ids.len;
        var hdr: [32]u8 = undefined;
        std.mem.writeInt(u32, hdr[0..4], SNAP_MAGIC, .little);
        std.mem.writeInt(u32, hdr[4..8], SNAP_VERSION_SLAB, .little);
        std.mem.writeInt(u64, hdr[8..16], seq, .little);
        std.mem.writeInt(u64, hdr[16..24], dim, .little);
        std.mem.writeInt(u64, hdr[24..32], n, .little);

        var wrote = linux.pwrite(fd, &hdr, hdr.len, 0);
        if (posix.errno(wrote) != .SUCCESS or wrote != hdr.len) return error.WriteFailed;
        if (n > 0) {
            const id_bytes = std.mem.sliceAsBytes(doc_ids);
            var done: usize = 0;
            while (done < id_bytes.len) {
                wrote = linux.pwrite(fd, id_bytes[done..].ptr, id_bytes.len - done, @intCast(32 + done));
                if (posix.errno(wrote) != .SUCCESS) return error.WriteFailed;
                done += wrote;
            }
        }
        const hmls_off: u64 = Hnsw.pageAlign(32 + n * 8);
        _ = try index.writeSlabsAt(fd, hmls_off);
        if (posix.errno(linux.fdatasync(fd)) != .SUCCESS) return error.SyncFailed;
        _ = linux.close(fd);
        closed = true;
        const z_tmp = try posix.toPosixPath(tmp);
        const z_dst = try posix.toPosixPath(path);
        if (posix.errno(linux.renameat(posix.AT.FDCWD, &z_tmp, posix.AT.FDCWD, &z_dst)) != .SUCCESS)
            return error.RenameFailed;
    }
};

pub const BinaryHeader = struct {
    seq: u64,
    dim: usize,
    n: usize,
    doc_ids_off: usize,
    graph_off: usize,
    version: u32 = SNAP_VERSION,
};

pub fn parseBinaryHeader(body: []const u8) !BinaryHeader {
    if (body.len < 32) return error.Truncated;
    const magic = std.mem.readInt(u32, body[0..4], .little);
    if (magic != SNAP_MAGIC) return error.BadMagic;
    const version = std.mem.readInt(u32, body[4..8], .little);
    if (version != SNAP_VERSION and version != SNAP_VERSION_SLAB) return error.UnsupportedVersion;
    const seq = std.mem.readInt(u64, body[8..16], .little);
    const dim: usize = @intCast(std.mem.readInt(u64, body[16..24], .little));
    const n: usize = @intCast(std.mem.readInt(u64, body[24..32], .little));
    const doc_off: usize = 32;
    const raw_graph = 32 + n * 8;
    const graph_off = if (version == SNAP_VERSION_SLAB) Hnsw.pageAlign(raw_graph) else raw_graph;
    return .{ .seq = seq, .dim = dim, .n = n, .doc_ids_off = doc_off, .graph_off = graph_off, .version = version };
}

pub fn docIdsFromBinary(body: []const u8, hdr: BinaryHeader, out: []u64) void {
    var i = hdr.doc_ids_off;
    for (out) |*id| {
        id.* = std.mem.readInt(u64, body[i..][0..8], .little);
        i += 8;
    }
}

test "keyFor rejects ambiguous namespace names" {
    const alloc = std.testing.allocator;
    var dummy_client: s3.Client = undefined;
    var store = Store.init(alloc, &dummy_client);
    try std.testing.expectError(error.InvalidNamespace, store.keyFor("snapshot_bin", "codedb/nested", 0));
    try std.testing.expectError(error.InvalidNamespace, store.keyFor("wal", "", 1));
    try std.testing.expectError(error.InvalidNamespace, store.keyFor("snapshot", "..", 0));
    const key = try store.keyFor("snapshot_bin", "codedb", 0);
    defer alloc.free(key);
    try std.testing.expectEqualStrings("openpuffer/codedb/snapshot.bin", key);
}

test "walSeqsFromKeys filters by ns and seq" {
    const alloc = std.testing.allocator;
    const keys = [_][]const u8{
        "openpuffer/a/snapshot.bin",
        "openpuffer/a/wal/00000000000000000001.json",
        "openpuffer/a/wal/00000000000000000003.json",
        "openpuffer/b/wal/00000000000000000002.json",
    };
    const seqs = try Store.walSeqsFromKeys(alloc, &keys, "a", 1);
    defer alloc.free(seqs);
    try std.testing.expectEqualSlices(u64, &[_]u64{3}, seqs);
}

test "parseWalSeq strips json suffix" {
    try std.testing.expectEqual(@as(?u64, 1), parseWalSeq("00000000000000000001.json"));
    try std.testing.expectEqual(@as(?u64, 37), parseWalSeq("00000000000000000037.json"));
    try std.testing.expectEqual(@as(?u64, 8), parseWalSeq("8"));
    try std.testing.expectEqual(@as(?u64, null), parseWalSeq("snapshot.json"));
    try std.testing.expectEqual(@as(?u64, 2), parseWalSeq("\"00000000000000000002.json\""));
}

test "binary snapshot header roundtrip" {
    const alloc = std.testing.allocator;
    var index = Hnsw.init(alloc, 4, .{});
    defer index.deinit();
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    _ = try index.insert(&a);
    _ = try index.insert(&b);
    const ids = [_]u64{ 10, 20 };
    const body = try Store.buildBinarySnapshot(alloc, 7, 4, &ids, &index);
    defer alloc.free(body);
    const hdr = try parseBinaryHeader(body);
    try std.testing.expectEqual(@as(u64, 7), hdr.seq);
    try std.testing.expectEqual(@as(usize, 4), hdr.dim);
    try std.testing.expectEqual(@as(usize, 2), hdr.n);
    var out_ids: [2]u64 = undefined;
    docIdsFromBinary(body, hdr, &out_ids);
    try std.testing.expectEqualSlices(u64, &ids, &out_ids);

    var loaded = Hnsw.init(alloc, 4, .{});
    defer loaded.deinit();
    try loaded.load(body[hdr.graph_off..]);
    try std.testing.expectEqual(index.len(), loaded.len());
    try std.testing.expectEqual(index.entryPoint(), loaded.entryPoint());
}

test "slab snapshot file mmap reopen" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    var index = Hnsw.init(alloc, 4, .{});
    defer index.deinit();
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    _ = try index.insert(&a);
    _ = try index.insert(&b);
    const ids = [_]u64{ 10, 20 };
    const path = "/tmp/openpuffer-persist-mmap.snap";
    try Store.writeSlabSnapshotFile(path, 9, 4, &ids, &index);

    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
    var peek: [32]u8 = undefined;
    const nread = std.os.linux.pread(fd, &peek, peek.len, 0);
    _ = std.os.linux.close(fd);
    try std.testing.expectEqual(std.posix.errno(nread), .SUCCESS);
    const hdr = try parseBinaryHeader(&peek);
    try std.testing.expectEqual(@as(u32, SNAP_VERSION_SLAB), hdr.version);
    try std.testing.expectEqual(@as(u64, 9), hdr.seq);
    try std.testing.expectEqual(@as(usize, 2), hdr.n);
    try std.testing.expectEqual(Hnsw.pageAlign(32 + 16), hdr.graph_off);

    var loaded = Hnsw.init(alloc, 4, .{});
    defer loaded.deinit();
    try loaded.loadMmap(path);
    try std.testing.expect(loaded.isMmapBacked());
    try std.testing.expectEqual(index.len(), loaded.len());
    try std.testing.expectEqual(index.entryPoint(), loaded.entryPoint());
}

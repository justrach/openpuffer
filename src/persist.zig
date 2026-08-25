//! Object-storage persistence for namespaces, modeled on turbopuffer's
//! architecture: every write batch appends one WAL segment object; snapshots
//! compact consumed WAL entries. Reads stay local/in-memory.
//!
//! Layout under the bucket:
//!   openpuffer/{ns}/snapshot.json     {"seq":N,"dim":D,"docs":[[id,[...]],...]}
//!   openpuffer/{ns}/wal/{seq:020}.json {"docs":[[id,[...]],...]}

const std = @import("std");
const s3 = @import("s3.zig");

pub const Store = struct {
    client: *s3.Client,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, client: *s3.Client) Store {
        return .{ .client = client, .alloc = alloc };
    }

    fn keyFor(self: *Store, comptime kind: []const u8, ns: []const u8, seq: u64) ![]u8 {
        if (comptime std.mem.eql(u8, kind, "snapshot")) {
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

    pub const Snapshot = struct {
        seq: u64,
        dim: usize,
        /// rows of [doc_id, vector]
        docs_json: []u8,
    };

    /// Fetch latest snapshot, or null when the namespace was never snapshotted.
    pub fn getSnapshot(self: *Store, ns: []const u8) !?Snapshot {
        const key = try self.keyFor("snapshot", ns, 0);
        defer self.alloc.free(key);
        const body = self.client.getObject(key) catch |e| switch (e) {
            error.S3RequestFailed => return null,
            else => return e,
        };
        // parse just seq/dim at top level; docs stay raw for the caller
        const parsed = try std.json.parseFromSlice(std.json.Value, self.alloc, body, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const obj = parsed.value.object;
        const seq: u64 = switch (obj.get("seq") orelse return null) {
            .integer => |n| @intCast(n),
            else => return null,
        };
        const dim: usize = switch (obj.get("dim") orelse return null) {
            .integer => |n| @intCast(n),
            else => return null,
        };
        return .{ .seq = seq, .dim = dim, .docs_json = body };
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
            const fname = std.mem.trim(u8, k[prefix.len..], "\"");
            const seq = std.fmt.parseInt(u64, fname, 10) catch continue;
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

    /// Serialize current namespace state into a snapshot body (caller frees).
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
};

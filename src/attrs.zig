//! Per-namespace document attributes and turbopuffer-style filters.
//!
//! Attributes are stored column-wise, one `Value` per local row id, so a
//! filter is a linear scan of one column per condition into a bitset over
//! row ids. The bitset then drives the vector search (see server.zig): few
//! matching rows are scanned exactly, many are found by a graph walk that
//! only accepts matching rows.
//!
//! Filter grammar (a subset of turbopuffer's):
//!   [attr, "Eq" | "NotEq" | "Lt" | "Lte" | "Gt" | "Gte", value]
//!   [attr, "In" | "NotIn", [value, ...]]
//!   ["And", [filter, ...]]   ["Or", [filter, ...]]   ["Not", filter]
//! `attr` "id" refers to the document id. Values are strings, integers,
//! floats, booleans or null; a missing attribute is null. Numbers compare
//! numerically across int/float, strings lexicographically, and an ordering
//! comparison never matches null.

const std = @import("std");

pub const Value = union(enum) {
    null,
    bool: bool,
    int: i64,
    float: f64,
    /// interned string id (see `Store.strings`)
    str: u32,
};

pub const Error = error{ BadAttribute, BadFilter, OutOfMemory };

/// Every column holds exactly `rows` values (null-filled), so writing a row
/// below `rows` never allocates. The server reserves rows alongside the
/// vector slabs and writes a new document's attributes before the row is
/// published to queries.
pub const Store = struct {
    alloc: std.mem.Allocator,
    rows: usize = 0,
    names: std.ArrayList([]u8) = .empty,
    cols: std.ArrayList(std.ArrayList(Value)) = .empty,
    strings: std.ArrayList([]u8) = .empty,
    string_ids: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn init(alloc: std.mem.Allocator) Store {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Store) void {
        for (self.names.items) |n| self.alloc.free(n);
        self.names.deinit(self.alloc);
        for (self.cols.items) |*c| c.deinit(self.alloc);
        self.cols.deinit(self.alloc);
        for (self.strings.items) |str| self.alloc.free(str);
        self.strings.deinit(self.alloc);
        self.string_ids.deinit(self.alloc);
    }

    pub fn isEmpty(self: *const Store) bool {
        return self.names.items.len == 0;
    }

    pub fn column(self: *const Store, name: []const u8) ?usize {
        for (self.names.items, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return i;
        }
        return null;
    }

    pub fn ensureColumn(self: *Store, name: []const u8) !usize {
        if (self.column(name)) |i| return i;
        const owned = try self.alloc.dupe(u8, name);
        errdefer self.alloc.free(owned);
        var col: std.ArrayList(Value) = .empty;
        errdefer col.deinit(self.alloc);
        try col.appendNTimes(self.alloc, .null, self.rows);
        try self.cols.append(self.alloc, col);
        errdefer _ = self.cols.pop();
        try self.names.append(self.alloc, owned);
        return self.names.items.len - 1;
    }

    /// Make every column (current and future) hold at least `n` rows.
    pub fn reserveRows(self: *Store, n: usize) !void {
        if (n <= self.rows) return;
        for (self.cols.items) |*c| try c.appendNTimes(self.alloc, .null, n - c.items.len);
        self.rows = n;
    }

    pub fn intern(self: *Store, s: []const u8) !u32 {
        if (self.string_ids.get(s)) |id| return id;
        const owned = try self.alloc.dupe(u8, s);
        errdefer self.alloc.free(owned);
        const id: u32 = @intCast(self.strings.items.len);
        try self.strings.append(self.alloc, owned);
        errdefer _ = self.strings.pop();
        try self.string_ids.put(self.alloc, owned, id);
        return id;
    }

    pub fn get(self: *const Store, col: usize, row: u32) Value {
        const c = self.cols.items[col].items;
        return if (row < c.len) c[row] else .null;
    }

    /// Set one value, growing every column if `row` is past `rows`.
    pub fn set(self: *Store, col: usize, row: u32, v: Value) !void {
        if (row >= self.rows) {
            if (v == .null) return;
            try self.reserveRows(@max(@as(usize, row) + 1, self.rows * 2));
        }
        self.setReserved(col, row, v);
    }

    /// Set one value in a row below `rows` (never allocates).
    pub fn setReserved(self: *Store, col: usize, row: u32, v: Value) void {
        std.debug.assert(row < self.rows);
        self.cols.items[col].items[row] = v;
    }

    /// Null every attribute of `row` (an upsert replaces the whole document).
    pub fn clearRow(self: *Store, row: u32) void {
        for (self.cols.items) |*c| {
            if (row < c.items.len) c.items[row] = .null;
        }
    }

    /// Convert a JSON attribute value. Arrays and objects are not supported.
    pub fn fromJson(self: *Store, v: std.json.Value) Error!Value {
        return switch (v) {
            .null => .null,
            .bool => |b| .{ .bool = b },
            .integer => |n| .{ .int = n },
            .float => |f| .{ .float = f },
            .number_string => |s| .{ .float = std.fmt.parseFloat(f64, s) catch return error.BadAttribute },
            .string => |s| .{ .str = try self.intern(s) },
            .array, .object => error.BadAttribute,
        };
    }

    /// Like `fromJson` but never interns: a string the store has never seen
    /// cannot equal any stored value, so it maps to null-match semantics via
    /// `Literal.missing_str`.
    fn literal(self: *const Store, v: std.json.Value) Error!Literal {
        return switch (v) {
            .null => .{ .val = .null },
            .bool => |b| .{ .val = .{ .bool = b } },
            .integer => |n| .{ .val = .{ .int = n } },
            .float => |f| .{ .val = .{ .float = f } },
            .number_string => |s| .{ .val = .{ .float = std.fmt.parseFloat(f64, s) catch return error.BadFilter } },
            .string => |s| if (self.string_ids.get(s)) |id| .{ .val = .{ .str = id }, .text = s } else .{ .val = .null, .text = s, .missing_str = true },
            .array, .object => error.BadFilter,
        };
    }

    pub fn writeJson(self: *const Store, w: *std.Io.Writer, v: Value) !void {
        switch (v) {
            .null => try w.writeAll("null"),
            .bool => |b| try w.writeAll(if (b) "true" else "false"),
            .int => |n| try w.print("{d}", .{n}),
            .float => |f| try w.print("{d}", .{f}),
            .str => |id| try writeJsonString(w, self.strings.items[id]),
        }
    }

    /// `,"name":value` for each non-null attribute of `row` (appended to a
    /// row object that already has id and distance). `only`, when set,
    /// limits it to those names.
    pub fn writeRowFields(self: *const Store, w: *std.Io.Writer, row: u32, only: ?[]const []const u8) !void {
        for (self.names.items, 0..) |name, col| {
            if (only) |names| {
                var want = false;
                for (names) |n| want = want or std.mem.eql(u8, n, name);
                if (!want) continue;
            }
            const v = self.get(col, row);
            if (v == .null) continue;
            try w.writeAll(",");
            try writeJsonString(w, name);
            try w.writeAll(":");
            try self.writeJson(w, v);
        }
    }

    /// Evaluate `filter` over rows `0..n` into a fresh bitset. `doc_ids`
    /// backs the special attribute "id". Deleted rows are the caller's job.
    pub fn eval(self: *const Store, filter: std.json.Value, doc_ids: []const u64, n: usize, alloc: std.mem.Allocator) Error!std.DynamicBitSetUnmanaged {
        var out = try std.DynamicBitSetUnmanaged.initEmpty(alloc, n);
        errdefer out.deinit(alloc);
        try self.evalInto(filter, doc_ids, &out, alloc);
        return out;
    }

    fn evalInto(self: *const Store, filter: std.json.Value, doc_ids: []const u64, out: *std.DynamicBitSetUnmanaged, alloc: std.mem.Allocator) Error!void {
        if (filter != .array) return error.BadFilter;
        const parts = filter.array.items;
        if (parts.len < 2 or parts[0] != .string) return error.BadFilter;
        const head = parts[0].string;
        if (parts.len == 2 and (std.mem.eql(u8, head, "And") or std.mem.eql(u8, head, "Or"))) {
            if (parts[1] != .array) return error.BadFilter;
            const is_and = std.mem.eql(u8, head, "And");
            if (is_and) out.setRangeValue(.{ .start = 0, .end = out.bit_length }, true);
            var tmp = try std.DynamicBitSetUnmanaged.initEmpty(alloc, out.bit_length);
            defer tmp.deinit(alloc);
            for (parts[1].array.items) |sub| {
                tmp.unsetAll();
                try self.evalInto(sub, doc_ids, &tmp, alloc);
                if (is_and) out.setIntersection(tmp) else out.setUnion(tmp);
            }
            return;
        }
        if (parts.len == 2 and std.mem.eql(u8, head, "Not")) {
            try self.evalInto(parts[1], doc_ids, out, alloc);
            out.toggleAll();
            return;
        }
        if (parts.len != 3 or parts[1] != .string) return error.BadFilter;
        const op = std.meta.stringToEnum(Op, parts[1].string) orelse return error.BadFilter;
        var lits: std.ArrayList(Literal) = .empty;
        defer lits.deinit(alloc);
        switch (op) {
            .In, .NotIn => {
                if (parts[2] != .array) return error.BadFilter;
                for (parts[2].array.items) |v| try lits.append(alloc, try self.literal(v));
            },
            else => try lits.append(alloc, try self.literal(parts[2])),
        }
        const n = out.bit_length;
        if (std.mem.eql(u8, head, "id")) {
            for (0..@min(n, doc_ids.len)) |row| {
                if (op.matches(self, .{ .int = @bitCast(doc_ids[row]) }, lits.items)) out.set(row);
            }
            return;
        }
        const col = self.column(head) orelse {
            // unknown attribute: every row is null
            if (op.matches(self, .null, lits.items)) out.setRangeValue(.{ .start = 0, .end = n }, true);
            return;
        };
        const vals = self.cols.items[col].items;
        const stored = vals[0..@min(n, vals.len)];
        // Fast path for the common case, equality with one string or integer.
        if (op == .Eq and !lits.items[0].missing_str) switch (lits.items[0].val) {
            .str => |want| {
                for (stored, 0..) |v, row| {
                    if (v == .str and v.str == want) out.set(row);
                }
                return;
            },
            .int => |want| {
                const wf: f64 = @floatFromInt(want);
                for (stored, 0..) |v, row| {
                    const hit = switch (v) {
                        .int => |x| x == want,
                        .float => |x| x == wf,
                        else => false,
                    };
                    if (hit) out.set(row);
                }
                return;
            },
            else => {},
        };
        for (stored, 0..) |v, row| {
            if (op.matches(self, v, lits.items)) out.set(row);
        }
        // rows past the stored values are null
        if (stored.len < n and op.matches(self, .null, lits.items)) out.setRangeValue(.{ .start = stored.len, .end = n }, true);
    }
};

const Literal = struct {
    val: Value,
    /// original text for string literals (ordering comparisons)
    text: []const u8 = "",
    /// a string no row holds: equal to nothing
    missing_str: bool = false,
};

const Op = enum {
    Eq,
    NotEq,
    In,
    NotIn,
    Lt,
    Lte,
    Gt,
    Gte,

    fn matches(op: Op, store: *const Store, v: Value, lits: []const Literal) bool {
        return switch (op) {
            .Eq => equal(v, lits[0]),
            .NotEq => !equal(v, lits[0]),
            .In => for (lits) |l| {
                if (equal(v, l)) break true;
            } else false,
            .NotIn => for (lits) |l| {
                if (equal(v, l)) break false;
            } else true,
            .Lt, .Lte, .Gt, .Gte => blk: {
                const ord = order(store, v, lits[0]) orelse break :blk false;
                break :blk switch (op) {
                    .Lt => ord == .lt,
                    .Lte => ord != .gt,
                    .Gt => ord == .gt,
                    .Gte => ord != .lt,
                    else => unreachable,
                };
            },
        };
    }
};

fn num(v: Value) ?f64 {
    return switch (v) {
        .int => |n| @floatFromInt(n),
        .float => |f| f,
        else => null,
    };
}

fn equal(v: Value, l: Literal) bool {
    if (l.missing_str) return false;
    if (num(v)) |a| {
        if (v == .int and l.val == .int) return v.int == l.val.int;
        const b = num(l.val) orelse return false;
        return a == b;
    }
    return switch (v) {
        .null => l.val == .null,
        .bool => |b| l.val == .bool and l.val.bool == b,
        .str => |id| l.val == .str and l.val.str == id,
        else => false,
    };
}

fn order(store: *const Store, v: Value, l: Literal) ?std.math.Order {
    if (num(v)) |a| {
        if (v == .int and l.val == .int) return std.math.order(v.int, l.val.int);
        const b = num(l.val) orelse return null;
        return std.math.order(a, b);
    }
    if (v == .str and (l.val == .str or l.missing_str)) {
        return std.mem.order(u8, store.strings.items[v.str], l.text);
    }
    return null;
}

pub fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeAll("\"");
    for (s) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => if (c < 0x20) try w.print("\\u{x:0>4}", .{c}) else try w.writeByte(c),
        }
    }
    try w.writeAll("\"");
}

/// Byte span of the JSON value that follows `"key":` in `body`, without
/// parsing the rest (query bodies carry a large vector). Null when absent.
pub fn valueSpan(body: []const u8, key: []const u8) ?[]const u8 {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, body, from, key)) |p| {
        from = p + key.len;
        var j = from;
        while (j < body.len and (body[j] == ' ' or body[j] == '\t' or body[j] == '\n' or body[j] == '\r')) : (j += 1) {}
        if (j >= body.len or body[j] != ':') continue;
        j += 1;
        while (j < body.len and (body[j] == ' ' or body[j] == '\t' or body[j] == '\n' or body[j] == '\r')) : (j += 1) {}
        const start = j;
        var depth: usize = 0;
        var in_str = false;
        while (j < body.len) : (j += 1) {
            const c = body[j];
            if (in_str) {
                if (c == '\\') j += 1 else if (c == '"') {
                    in_str = false;
                    if (depth == 0) return body[start .. j + 1];
                }
                continue;
            }
            switch (c) {
                '"' => in_str = true,
                '[', '{' => depth += 1,
                ']', '}' => {
                    if (depth == 0) return body[start..j];
                    depth -= 1;
                    if (depth == 0) return body[start .. j + 1];
                },
                ',' => if (depth == 0) return body[start..j],
                else => {},
            }
        }
        return body[start..j];
    }
    return null;
}

test "filters over string, int and float columns" {
    const alloc = std.testing.allocator;
    var st = Store.init(alloc);
    defer st.deinit();
    const kind = try st.ensureColumn("kind");
    const pri = try st.ensureColumn("priority");
    try st.set(kind, 0, .{ .str = try st.intern("fact") });
    try st.set(kind, 1, .{ .str = try st.intern("todo") });
    try st.set(kind, 3, .{ .str = try st.intern("fact") });
    try st.set(pri, 0, .{ .int = 1 });
    try st.set(pri, 1, .{ .float = 2.5 });
    try st.set(pri, 2, .{ .int = 5 });
    const ids = [_]u64{ 10, 11, 12, 13 };

    const cases = [_]struct { f: []const u8, want: []const usize }{
        .{ .f = "[\"kind\",\"Eq\",\"fact\"]", .want = &.{ 0, 3 } },
        .{ .f = "[\"kind\",\"NotEq\",\"fact\"]", .want = &.{ 1, 2 } },
        .{ .f = "[\"kind\",\"Eq\",\"never-stored\"]", .want = &.{} },
        .{ .f = "[\"kind\",\"Eq\",null]", .want = &.{2} },
        .{ .f = "[\"priority\",\"Gte\",2]", .want = &.{ 1, 2 } },
        .{ .f = "[\"priority\",\"Lt\",2.5]", .want = &.{0} },
        .{ .f = "[\"priority\",\"In\",[1,5]]", .want = &.{ 0, 2 } },
        .{ .f = "[\"id\",\"In\",[11,13]]", .want = &.{ 1, 3 } },
        .{ .f = "[\"And\",[[\"kind\",\"Eq\",\"fact\"],[\"priority\",\"Eq\",1]]]", .want = &.{0} },
        .{ .f = "[\"Or\",[[\"kind\",\"Eq\",\"todo\"],[\"priority\",\"Gt\",4]]]", .want = &.{ 1, 2 } },
        .{ .f = "[\"Not\",[\"kind\",\"In\",[\"fact\",\"todo\"]]]", .want = &.{2} },
        .{ .f = "[\"missing\",\"Eq\",null]", .want = &.{ 0, 1, 2, 3 } },
        .{ .f = "[\"kind\",\"Gt\",\"g\"]", .want = &.{1} },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, c.f, .{});
        defer parsed.deinit();
        var bits = try st.eval(parsed.value, &ids, 4, alloc);
        defer bits.deinit(alloc);
        var got: std.ArrayList(usize) = .empty;
        defer got.deinit(alloc);
        var it = bits.iterator(.{});
        while (it.next()) |i| try got.append(alloc, i);
        std.testing.expectEqualSlices(usize, c.want, got.items) catch |e| {
            std.debug.print("filter {s}\n", .{c.f});
            return e;
        };
    }

    const bad = [_][]const u8{ "{}", "[\"kind\"]", "[\"kind\",\"Like\",\"x\"]", "[\"And\",{}]", "[\"kind\",\"In\",\"fact\"]" };
    for (bad) |f| {
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, f, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.BadFilter, st.eval(parsed.value, &ids, 4, alloc));
    }
}

test "row json skips nulls and escapes strings" {
    const alloc = std.testing.allocator;
    var st = Store.init(alloc);
    defer st.deinit();
    const a = try st.ensureColumn("note");
    const b = try st.ensureColumn("n");
    try st.set(a, 0, .{ .str = try st.intern("say \"hi\"\n") });
    try st.set(b, 1, .{ .int = 3 });
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try st.writeRowFields(&out.writer, 0, null);
    try std.testing.expectEqualStrings(",\"note\":\"say \\\"hi\\\"\\n\"", out.written());
    out.clearRetainingCapacity();
    try st.writeRowFields(&out.writer, 1, &.{"note"});
    try std.testing.expectEqualStrings("", out.written());
    st.clearRow(1);
    out.clearRetainingCapacity();
    try st.writeRowFields(&out.writer, 1, null);
    try std.testing.expectEqualStrings("", out.written());
}

test "valueSpan finds values without parsing the vector" {
    const body = "{\"rank_by\":[\"vector\",\"ANN\",[0.1,0.2]],\"filters\": [\"And\",[[\"k\",\"Eq\",\"a]\"],[\"n\",\"Gt\",1]]],\"include_attributes\":true}";
    try std.testing.expectEqualStrings("[\"And\",[[\"k\",\"Eq\",\"a]\"],[\"n\",\"Gt\",1]]]", valueSpan(body, "\"filters\"").?);
    try std.testing.expectEqualStrings("true", valueSpan(body, "\"include_attributes\"").?);
    try std.testing.expect(valueSpan(body, "\"missing\"") == null);
    try std.testing.expectEqualStrings("\"x\"", valueSpan("{\"a\":\"x\"}", "\"a\"").?);
}

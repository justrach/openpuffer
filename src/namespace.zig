//! Canonical namespace names for HTTP routes and object-store keys.
//!
//! A name is one URL path segment: non-empty, ≤128 bytes, charset
//! `[A-Za-z0-9._-]`, not a dot-segment, and not a reserved storage
//! component (`wal`, `snapshot`, `snapshot.bin`, `snapshot.json`).
//! Slashes, backslashes, controls, and spaces are rejected so
//! `openpuffer/{ns}/...` keys always round-trip.

const std = @import("std");

pub const max_len: usize = 128;

pub const reserved = [_][]const u8{
    ".",
    "..",
    "wal",
    "snapshot",
    "snapshot.bin",
    "snapshot.json",
};

pub const Error = error{InvalidNamespace};

pub fn validate(name: []const u8) Error!void {
    if (name.len == 0 or name.len > max_len) return error.InvalidNamespace;
    for (reserved) |r| {
        if (std.mem.eql(u8, name, r)) return error.InvalidNamespace;
    }
    for (name) |c| {
        const ok = (c >= 'A' and c <= 'Z') or
            (c >= 'a' and c <= 'z') or
            (c >= '0' and c <= '9') or
            c == '_' or c == '-' or c == '.';
        if (!ok) return error.InvalidNamespace;
    }
}

pub fn isValid(name: []const u8) bool {
    validate(name) catch return false;
    return true;
}

/// JSON-escape a namespace (or any short token) into `buf`. After
/// `validate`, the charset is already JSON-safe; this still escapes
/// controls and quotes so invalid names can appear in 400 bodies.
pub fn jsonEscape(name: []const u8, buf: []u8) []const u8 {
    var i: usize = 0;
    for (name) |c| {
        if (i + 6 > buf.len) break;
        switch (c) {
            '"' => {
                buf[i] = '\\';
                buf[i + 1] = '"';
                i += 2;
            },
            '\\' => {
                buf[i] = '\\';
                buf[i + 1] = '\\';
                i += 2;
            },
            0x08 => {
                buf[i] = '\\';
                buf[i + 1] = 'b';
                i += 2;
            },
            0x09 => {
                buf[i] = '\\';
                buf[i + 1] = 't';
                i += 2;
            },
            0x0a => {
                buf[i] = '\\';
                buf[i + 1] = 'n';
                i += 2;
            },
            0x0d => {
                buf[i] = '\\';
                buf[i + 1] = 'r';
                i += 2;
            },
            else => if (c < 0x20) {
                const hex = "0123456789abcdef";
                buf[i] = '\\';
                buf[i + 1] = 'u';
                buf[i + 2] = '0';
                buf[i + 3] = '0';
                buf[i + 4] = hex[c >> 4];
                buf[i + 5] = hex[c & 0xf];
                i += 6;
            } else {
                buf[i] = c;
                i += 1;
            },
        }
    }
    return buf[0..i];
}

test "ordinary codedb-style ids" {
    try validate("codedb");
    try validate("openpuffer-bench-1");
    try validate("ns_01");
    try validate("a.b");
    try validate("A");
}

test "nested paths and slashes" {
    try std.testing.expectError(error.InvalidNamespace, validate("codedb/nested"));
    try std.testing.expectError(error.InvalidNamespace, validate("a\\b"));
    try std.testing.expectError(error.InvalidNamespace, validate("a/b/c"));
}

test "empty names" {
    try std.testing.expectError(error.InvalidNamespace, validate(""));
}

test "dot-segments" {
    try std.testing.expectError(error.InvalidNamespace, validate("."));
    try std.testing.expectError(error.InvalidNamespace, validate(".."));
}

test "control characters" {
    try std.testing.expectError(error.InvalidNamespace, validate("a\nb"));
    try std.testing.expectError(error.InvalidNamespace, validate("a\x00b"));
    try std.testing.expectError(error.InvalidNamespace, validate("a b"));
}

test "maximum length" {
    var ok: [max_len]u8 = undefined;
    @memset(&ok, 'n');
    try validate(&ok);
    var too: [max_len + 1]u8 = undefined;
    @memset(&too, 'n');
    try std.testing.expectError(error.InvalidNamespace, validate(&too));
}

test "reserved storage components" {
    try std.testing.expectError(error.InvalidNamespace, validate("wal"));
    try std.testing.expectError(error.InvalidNamespace, validate("snapshot"));
    try std.testing.expectError(error.InvalidNamespace, validate("snapshot.bin"));
    try std.testing.expectError(error.InvalidNamespace, validate("snapshot.json"));
}

test "jsonEscape quotes and controls" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("codedb", jsonEscape("codedb", &buf));
    try std.testing.expectEqualStrings("a\\\"b", jsonEscape("a\"b", &buf));
    try std.testing.expectEqualStrings("a\\u0000b", jsonEscape("a\x00b", &buf));
}

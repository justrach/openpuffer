//! Linux io_uring helpers for the HTTP serve path.
//!
//! The in-memory HNSW loop does no I/O — io_uring cannot make dots faster.
//! It *can* cut the per-query HTTP cost (accept + recv + send + Nagle) that
//! qa_bench / urllib pay by opening a new TCP connection every time.
//! Concurrent queries are handled by a fixed OS-thread worker pool; each
//! worker uses blocking posix read/write so keep-alive and parallel ANN
//! searches do not serialize on the accept ring.

const std = @import("std");
const builtin = @import("builtin");

pub const is_linux = builtin.os.tag == .linux;

pub fn enableTcpNoDelay(fd: std.posix.fd_t) void {
    if (builtin.os.tag != .linux) return;
    const yes: c_int = 1;
    std.posix.setsockopt(
        fd,
        std.os.linux.IPPROTO.TCP,
        std.os.linux.TCP.NODELAY,
        std.mem.asBytes(&yes),
    ) catch {};
}

pub const Ring = if (is_linux) RingLinux else RingStub;

const RingStub = struct {
    pub fn init() error{Unsupported}!RingStub {
        return error.Unsupported;
    }
    pub fn deinit(_: *RingStub) void {}
};

const RingLinux = struct {
    ring: std.os.linux.IoUring,

    pub fn init() !RingLinux {
        return .{ .ring = try std.os.linux.IoUring.init(64, 0) };
    }

    pub fn deinit(self: *RingLinux) void {
        self.ring.deinit();
    }

    fn wait1(self: *RingLinux) !i32 {
        _ = try self.ring.submit_and_wait(1);
        const cqe = try self.ring.copy_cqe();
        return cqe.res;
    }

    pub fn accept(self: *RingLinux, listen_fd: std.posix.fd_t) !std.posix.fd_t {
        _ = try self.ring.accept(1, listen_fd, null, null, std.os.linux.SOCK.CLOEXEC);
        const res = try self.wait1();
        if (res < 0) return error.AcceptFailed;
        return @intCast(res);
    }

    pub fn recv(self: *RingLinux, fd: std.posix.fd_t, buf: []u8) !usize {
        _ = try self.ring.recv(2, fd, .{ .buffer = buf }, 0);
        const res = try self.wait1();
        if (res < 0) return error.RecvFailed;
        return @intCast(res);
    }

    pub fn sendAll(self: *RingLinux, fd: std.posix.fd_t, data: []const u8) !void {
        var off: usize = 0;
        while (off < data.len) {
            _ = try self.ring.send(3, fd, data[off..], 0);
            const res = try self.wait1();
            if (res <= 0) return error.SendFailed;
            off += @as(usize, @intCast(res));
        }
    }
};

/// Blocking read/write for worker threads when a second io_uring cannot be created.
pub fn posixRecv(fd: std.posix.fd_t, buf: []u8) !usize {
    if (!is_linux) return error.RecvFailed;
    const res = std.os.linux.read(fd, buf.ptr, buf.len);
    const e = std.os.linux.errno(res);
    if (e != .SUCCESS) return error.RecvFailed;
    return res;
}

pub fn posixSendAll(fd: std.posix.fd_t, data: []const u8) !void {
    if (!is_linux) return error.SendFailed;
    var off: usize = 0;
    while (off < data.len) {
        const res = std.os.linux.write(fd, data[off..].ptr, data.len - off);
        const e = std.os.linux.errno(res);
        if (e != .SUCCESS or res == 0) return error.SendFailed;
        off += res;
    }
}

/// How many bytes of `buf` form a complete HTTP/1.1 request, or 0 if more data needed.
pub fn completeHttpRequest(buf: []const u8) usize {
    const sep = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return 0;
    const header_end = sep + 4;
    const headers = buf[0..sep];
    var content_len: usize = 0;
    var it = std.mem.splitSequence(u8, headers, "\r\n");
    _ = it.next(); // request line
    while (it.next()) |line| {
        if (line.len >= 15 and std.ascii.eqlIgnoreCase(line[0..15], "content-length:")) {
            const n = std.mem.trim(u8, line[15..], " \t");
            content_len = std.fmt.parseInt(usize, n, 10) catch 0;
        }
    }
    const need = header_end + content_len;
    if (buf.len < need) return 0;
    return need;
}

pub const HttpReq = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
};

pub fn parseHttpRequest(buf: []const u8) !HttpReq {
    const header_end = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return error.Incomplete;
    const head = buf[0..header_end];
    const nl = std.mem.indexOf(u8, head, "\r\n") orelse return error.BadRequest;
    const line = head[0..nl];
    const sp1 = std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadRequest;
    const rest = line[sp1 + 1 ..];
    const sp2 = std.mem.indexOfScalar(u8, rest, ' ') orelse return error.BadRequest;
    return .{
        .method = line[0..sp1],
        .path = rest[0..sp2],
        .body = buf[header_end + 4 ..],
    };
}

pub fn wantsClose(buf: []const u8) bool {
    const sep = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return true;
    const head = buf[0..sep];
    const nl = std.mem.indexOf(u8, head, "\r\n") orelse return true;
    const line = head[0..nl];
    if (std.mem.endsWith(u8, line, "HTTP/1.0")) return true;
    var it = std.mem.splitSequence(u8, head, "\r\n");
    _ = it.next();
    while (it.next()) |hdr| {
        if (hdr.len >= 11 and std.ascii.eqlIgnoreCase(hdr[0..11], "connection:")) {
            const v = std.mem.trim(u8, hdr[11..], " \t");
            if (std.ascii.eqlIgnoreCase(v, "close")) return true;
        }
    }
    return false;
}

pub fn formatResponse(alloc: std.mem.Allocator, status: u16, reason: []const u8, body: []const u8, keepalive: bool) ![]u8 {
    const conn = if (keepalive) "keep-alive" else "close";
    return std.fmt.allocPrint(alloc, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: {s}\r\n\r\n{s}", .{
        status, reason, body.len, conn, body,
    });
}

test "completeHttpRequest waits for body" {
    const partial = "POST /v2/namespaces/x/query HTTP/1.1\r\nContent-Length: 5\r\n\r\nab";
    try std.testing.expectEqual(@as(usize, 0), completeHttpRequest(partial));
    const full = "POST /v2/namespaces/x/query HTTP/1.1\r\nContent-Length: 5\r\n\r\nabcde";
    try std.testing.expectEqual(full.len, completeHttpRequest(full));
}

test "wantsClose honors Connection and HTTP/1.0" {
    const c = "POST /q HTTP/1.1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
    try std.testing.expect(wantsClose(c));
    const k = "POST /q HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
    try std.testing.expect(!wantsClose(k));
    const old = "POST /q HTTP/1.0\r\nContent-Length: 0\r\n\r\n";
    try std.testing.expect(wantsClose(old));
}

test "parseHttpRequest extracts method path body" {
    const raw = "POST /v2/namespaces/ns/query HTTP/1.1\r\nContent-Length: 4\r\n\r\nABCD";
    const r = try parseHttpRequest(raw);
    try std.testing.expectEqualStrings("POST", r.method);
    try std.testing.expectEqualStrings("/v2/namespaces/ns/query", r.path);
    try std.testing.expectEqualStrings("ABCD", r.body);
}

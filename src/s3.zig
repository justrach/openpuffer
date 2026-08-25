//! Minimal S3-compatible object storage client (works with AWS S3 and
//! Cloudflare R2, which speaks the same SigV4 API).
//!
//! Auth: env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION, or an
//! entry in ~/.aws/credentials selected via AWS_PROFILE.
//! R2 usage: endpoint = https://<accountid>.r2.cloudflarestorage.com,
//! region = "auto", access/secret from an R2 API token.

const std = @import("std");

pub const Config = struct {
    access_key: []const u8,
    secret_key: []const u8,
    region: []const u8 = "us-east-1",
    bucket: []const u8,
    /// null => virtual-hosted style https://<bucket>.s3.<region>.amazonaws.com
    endpoint: ?[]const u8 = null, // e.g. https://abc123.r2.cloudflarestorage.com or MinIO
    debug_sig: bool = false,
};

pub const Client = struct {
    cfg: Config,
    client: std.http.Client,
    io: std.Io,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, cfg: Config) Client {
        return .{
            .cfg = cfg,
            .client = .{ .allocator = alloc, .io = io },
            .io = io,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Client) void {
        self.client.deinit();
    }

    // ---- SigV4 -------------------------------------------------------------

    fn hmacSha256(key: []const u8, data: []const u8) [32]u8 {
        var out: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&out, data, key);
        return out;
    }

    fn sha256Hex(data: []const u8) [64]u8 {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data, &hash, .{});
        var out: [64]u8 = undefined;
        const digits = "0123456789abcdef";
        for (hash, 0..) |b, i| {
            out[i * 2] = digits[b >> 4];
            out[i * 2 + 1] = digits[b & 15];
        }
        return out;
    }

    const SignedRequest = struct {
        amz_date: [16]u8, // YYYYMMDDTHHMMSSZ
        authorization: [512]u8,
        auth_len: usize,
    };

    fn signRequest(
        self: *Client,
        method: []const u8,
        uri_path: []const u8, // /bucket/key (already URI-encoded)
        query: []const u8, // "" or "list-type=2&..." sorted
        payload_hash: *const [64]u8,
    ) !SignedRequest {
        // derive date strings from the real-time clock
        const now = std.Io.Timestamp.now(self.io, .real);
        const secs: u64 = @intCast(@divFloor(now.nanoseconds, std.time.ns_per_s));
        const epoch = std.time.epoch.EpochSeconds{ .secs = secs };
        const day = epoch.getEpochDay();
        const year_day = day.calculateYearDay();
        const month_day = year_day.calculateMonthDay();
        const dsecs = epoch.getDaySeconds();
        var amz_date: [16]u8 = undefined;
        _ = try std.fmt.bufPrint(&amz_date, "{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z", .{
            year_day.year,
            month_day.month.numeric(),
            month_day.day_index + 1,
            dsecs.getHoursIntoDay(),
            dsecs.getMinutesIntoHour(),
            dsecs.getSecondsIntoMinute(),
        });
        const short_date = amz_date[0..8];

        const host = try self.hostHeader();
        var chost: [256]u8 = undefined;
        const h = try std.fmt.bufPrint(&chost, "{s}", .{host});

        // canonical headers: host + x-amz-content-sha256 + x-amz-date
        var canonical_headers: [1024]u8 = undefined;
        const ch = try std.fmt.bufPrint(&canonical_headers, "host:{s}\nx-amz-content-sha256:{s}\nx-amz-date:{s}\n", .{ h, payload_hash.*, amz_date });
        const signed_headers = "host;x-amz-content-sha256;x-amz-date";

        var canonical_request: [2048]u8 = undefined;
        const cr = try std.fmt.bufPrint(&canonical_request, "{s}\n{s}\n{s}\n{s}\n{s}\n{s}", .{
            method, uri_path, query, ch, signed_headers, payload_hash.*,
        });
        const cr_hash = sha256Hex(cr);

        var scope: [64]u8 = undefined;
        const sc = try std.fmt.bufPrint(&scope, "{s}/{s}/s3/aws4_request", .{ short_date, self.cfg.region });

        var string_to_sign: [256]u8 = undefined;
        const sts = try std.fmt.bufPrint(&string_to_sign, "AWS4-HMAC-SHA256\n{s}\n{s}\n{s}", .{ amz_date, sc, cr_hash });

        var k_secret: [128]u8 = undefined;
        const ks = try std.fmt.bufPrint(&k_secret, "AWS4{s}", .{self.cfg.secret_key});
        var k_date = hmacSha256(ks[0..ks.len], short_date);
        var k_region = hmacSha256(&k_date, self.cfg.region);
        var k_service = hmacSha256(&k_region, "s3");
        var k_signing = hmacSha256(&k_service, "aws4_request");
        const signature_raw = hmacSha256(&k_signing, sts);
        var signature: [64]u8 = undefined;
        {
            const digits = "0123456789abcdef";
            for (signature_raw, 0..) |b, i| {
                signature[i * 2] = digits[b >> 4];
                signature[i * 2 + 1] = digits[b & 15];
            }
        }

        if (self.cfg.debug_sig) {
            std.debug.print("CANONICAL:\n{s}\n---\nSTS:\n{s}\n---\n", .{ cr, sts });
        }
        var out = SignedRequest{ .amz_date = amz_date, .authorization = undefined, .auth_len = 0 };
        const auth = try std.fmt.bufPrint(&out.authorization, "AWS4-HMAC-SHA256 Credential={s}/{s}, SignedHeaders={s}, Signature={s}", .{
            self.cfg.access_key, sc, signed_headers, signature,
        });
        out.auth_len = auth.len;
        return out;
    }

    fn hostHeader(self: *Client) ![]const u8 {
        if (self.cfg.endpoint) |ep| {
            // strip scheme
            const stripped = if (std.mem.startsWith(u8, ep, "https://")) ep["https://".len..] else if (std.mem.startsWith(u8, ep, "http://")) ep["http://".len..] else ep;
            return stripped;
        }
        return std.fmt.allocPrint(self.alloc, "{s}.s3.{s}.amazonaws.com", .{ self.cfg.bucket, self.cfg.region });
    }

    /// RFC 3986 strict encoding for query values ('/' must be escaped).
    fn urlEncodeQueryValue(self: *Client, s: []const u8) ![]u8 {
        var buf: std.Io.Writer.Allocating = .init(self.alloc);
        errdefer buf.deinit();
        const w = &buf.writer;
        for (s) |c| {
            if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '.' or c == '-' or c == '_' or c == '~') {
                try w.writeAll(&.{c});
            } else {
                try w.print("%{X:0>2}", .{c});
            }
        }
        return buf.toOwnedSlice();
    }

    fn urlEncodePath(self: *Client, key: []const u8) ![]u8 {
        // encode each path segment but keep '/'
        var buf: std.Io.Writer.Allocating = .init(self.alloc);
        errdefer buf.deinit();
        const w = &buf.writer;
        for (key) |c| {
            if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '/' or c == '.' or c == '-' or c == '_') {
                try w.writeAll(&.{c});
            } else {
                try w.print("%{X:0>2}", .{c});
            }
        }
        return buf.toOwnedSlice();
    }

    fn request(
        self: *Client,
        method: std.http.Method,
        key: []const u8,
        query: []const u8,
        payload: ?[]const u8,
    ) ![]u8 {
        const alloc = self.alloc;
        const encoded_key = try self.urlEncodePath(key);
        defer alloc.free(encoded_key);

        var uri_path_buf: [1024]u8 = undefined;
        var uri_path: []const u8 = undefined;
        const root_key = std.mem.eql(u8, key, "/") or key.len == 0;
        if (self.cfg.endpoint != null) {
            // path-style: /bucket/key
            uri_path = if (root_key)
                try std.fmt.bufPrint(&uri_path_buf, "/{s}/", .{self.cfg.bucket})
            else
                try std.fmt.bufPrint(&uri_path_buf, "/{s}/{s}", .{ self.cfg.bucket, encoded_key });
        } else {
            uri_path = try std.fmt.bufPrint(&uri_path_buf, "/{s}", .{encoded_key});
        }

        const payload_bytes = payload orelse "";
        const phash = sha256Hex(payload_bytes);
        var signed = try self.signRequest(@tagName(method), uri_path, query, &phash);

        var url_buf: [1200]u8 = undefined;
        const url = blk: {
            // `query` arrives without its leading '?' (it was stripped for
            // SigV4 canonicalization); re-add it for the actual URL.
            const qsep: []const u8 = if (query.len > 0) "?" else "";
            if (self.cfg.endpoint) |ep| {
                break :blk try std.fmt.bufPrint(&url_buf, "{s}{s}{s}{s}", .{ ep, uri_path, qsep, query });
            }
            break :blk try std.fmt.bufPrint(&url_buf, "https://{s}.s3.{s}.amazonaws.com/{s}?{s}", .{
                self.cfg.bucket, self.cfg.region, encoded_key, query,
            });
        };

        var auth_header_buf: [600]u8 = undefined;
        const auth_value = try std.fmt.bufPrint(&auth_header_buf, "{s}", .{signed.authorization[0..signed.auth_len]});
        var date_header_buf: [40]u8 = undefined;
        const date_value = try std.fmt.bufPrint(&date_header_buf, "{s}", .{signed.amz_date});
        var sha_header_buf: [80]u8 = undefined;
        const sha_value = try std.fmt.bufPrint(&sha_header_buf, "{s}", .{phash});

        var resp: std.Io.Writer.Allocating = .init(alloc);
        defer resp.deinit();
        const result = try self.client.fetch(.{
            .location = .{ .url = url },
            .method = method,
            .payload = if (payload) |p| p else null,
            // NOTE: this std version never emits privileged_headers, so the
            // signature goes out via extra_headers.
            .extra_headers = &.{
                .{ .name = "x-amz-date", .value = date_value },
                .{ .name = "x-amz-content-sha256", .value = sha_value },
                .{ .name = "authorization", .value = auth_value },
            },
            .response_writer = &resp.writer,
            .keep_alive = false,
        });
        if (result.status.class() != .success) {
            if (result.status != .not_found) {
                std.debug.print("s3 {s} {s} -> {d}: {s}\n", .{ @tagName(method), key, @intFromEnum(result.status), resp.written()[0..@min(resp.written().len, 300)] });
            }
            return error.S3RequestFailed;
        }
        return alloc.dupe(u8, resp.written());
    }

    /// Store an object. Overwrites existing key.
    pub fn putObject(self: *Client, key: []const u8, data: []const u8) !void {
        const body = try self.request(.PUT, key, "", data);
        defer self.alloc.free(body);
    }

    /// Fetch an object; caller owns returned bytes.
    pub fn getObject(self: *Client, key: []const u8) ![]u8 {
        return self.request(.GET, key, "", null);
    }

    pub fn deleteObject(self: *Client, key: []const u8) !void {
        const body = try self.request(.DELETE, key, "", "");
        defer self.alloc.free(body);
    }

    pub const ListedKey = struct { key: []u8 };

    /// List keys under a prefix (handles up to 1 page of 1000 — plenty here).
    pub fn listKeys(self: *Client, prefix: []const u8) ![][]u8 {
        const alloc = self.alloc;
        const enc_prefix = try self.urlEncodeQueryValue(prefix);
        defer alloc.free(enc_prefix);
        var qbuf: [600]u8 = undefined;
        const query = try std.fmt.bufPrint(&qbuf, "?list-type=2&prefix={s}", .{enc_prefix});

        // List uses different canonical query handling; simplest correct path:
        // sign with the same query string (without '?').
        const body = try self.request(.GET, "/", query[1..], null);
        defer alloc.free(body);

        // parse <Key>...</Key> entries
        var keys: std.ArrayList([]u8) = .empty;
        var rest: []const u8 = body;
        while (std.mem.indexOf(u8, rest, "<Key>")) |start| {
            const after = rest[start + 5 ..];
            const end = std.mem.indexOf(u8, after, "</Key>") orelse break;
            const raw = after[0..end];
            // XML-unescape the minimal set we may emit
            const decoded = if (std.mem.indexOfScalar(u8, raw, '&') != null)
                try xmlUnescape(alloc, raw)
            else
                try alloc.dupe(u8, raw);
            try keys.append(alloc, decoded);
            rest = after[end + 6 ..];
        }
        return keys.toOwnedSlice(alloc);
    }

    fn xmlUnescape(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(alloc);
        errdefer out.deinit();
        var i: usize = 0;
        while (i < s.len) {
            if (std.mem.startsWith(u8, s[i..], "&amp;")) {
                try out.writer.writeAll("&");
                i += 5;
            } else if (std.mem.startsWith(u8, s[i..], "&lt;")) {
                try out.writer.writeAll("<");
                i += 4;
            } else if (std.mem.startsWith(u8, s[i..], "&gt;")) {
                try out.writer.writeAll(">");
                i += 4;
            } else if (std.mem.startsWith(u8, s[i..], "&quot;")) {
                try out.writer.writeAll("\"");
                i += 6;
            } else {
                try out.writer.writeAll(s[i .. i + 1]);
                i += 1;
            }
        }
        return out.toOwnedSlice();
    }
};

/// Resolve credentials: env first, then ~/.aws/credentials profile.
pub fn resolveCredentials(alloc: std.mem.Allocator, io: std.Io, env_map: *std.process.Environ.Map) !Credentials {
    const C = struct { access: []u8, secret: []u8, region: []u8 };
    _ = C;
    return resolveCredentialsInner(alloc, io, env_map);
}

const Credentials = struct { access: []u8, secret: []u8, region: []u8 };

fn resolveCredentialsInner(alloc: std.mem.Allocator, io: std.Io, env_map: *std.process.Environ.Map) !Credentials {
    if (env_map.get("AWS_ACCESS_KEY_ID")) |ak| {
        if (env_map.get("AWS_SECRET_ACCESS_KEY")) |sk| {
            return .{
                .access = try alloc.dupe(u8, ak),
                .secret = try alloc.dupe(u8, sk),
                .region = try alloc.dupe(u8, env_map.get("AWS_REGION") orelse env_map.get("AWS_DEFAULT_REGION") orelse "us-east-1"),
            };
        }
    }
    // fall back to ~/.aws/credentials
    const home = env_map.get("HOME") orelse return error.NoCredentials;
    var pbuf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/.aws/credentials", .{home});
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var fr = file.reader(io, &.{});
    const contents = try fr.interface.allocRemaining(alloc, .limited(65536));
    const profile: []const u8 = env_map.get("AWS_PROFILE") orelse "default";

    var section: ?[]const u8 = null;
    var access: ?[]u8 = null;
    var secret: ?[]u8 = null;
    var region: ?[]u8 = null;
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or line[0] == ';') continue;
        if (line[0] == '[' and line[line.len - 1] == ']') {
            section = line[1 .. line.len - 1];
            continue;
        }
        const sec = section orelse continue;
        if (!std.mem.eql(u8, sec, profile)) continue;
        if (std.mem.indexOfScalar(u8, line, '=')) |eq| {
            const k = std.mem.trim(u8, line[0..eq], " \t");
            const v = std.mem.trim(u8, line[eq + 1 ..], " \t");
            if (std.mem.eql(u8, k, "aws_access_key_id")) access = try alloc.dupe(u8, v);
            if (std.mem.eql(u8, k, "aws_secret_access_key")) secret = try alloc.dupe(u8, v);
            if (std.mem.eql(u8, k, "region")) region = try alloc.dupe(u8, v);
        }
    }
    if (access == null or secret == null) return error.NoCredentials;
    return .{
        .access = access.?,
        .secret = secret.?,
        .region = region orelse try alloc.dupe(u8, "us-east-1"),
    };
}

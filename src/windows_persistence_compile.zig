//! Compile-only regression for the Windows public persistence surface.
//! This object is never run; compiling it proves the generic methods resolve
//! to explicit UnsupportedPlatform paths without analyzing POSIX mmap code.

const std = @import("std");
const openpuffer = @import("openpuffer");

export fn openpufferWindowsPersistenceGuard() void {
    const Index = openpuffer.Hnsw(void);
    var index = Index.init(std.heap.page_allocator, 2, .{});

    index.writeSlabs("unused.slabs") catch |err| switch (err) {
        error.UnsupportedPlatform => {},
        else => unreachable,
    };
    index.loadMmap("unused.slabs") catch |err| switch (err) {
        error.UnsupportedPlatform => {},
        else => unreachable,
    };
}

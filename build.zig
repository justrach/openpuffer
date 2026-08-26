const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Public in-process engine. Dependents `@import("openpuffer")` get Hnsw /
    // Options / SearchResult without the HTTP server or cloud clients.
    _ = b.addModule("openpuffer", .{
        .root_source_file = b.path("src/hnsw.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "openpuffer",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run openpuffer");
    run_step.dependOn(&run_cmd.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Official `zig build test` used to only compile src/main.zig, which has
    // no tests. The engine tests live in src/hnsw.zig (imported modules are
    // not executed). Run them here so the loop gate is real.
    const hnsw_mod = b.createModule(.{
        .root_source_file = b.path("src/hnsw.zig"),
        .target = target,
        .optimize = optimize,
    });
    const hnsw_tests = b.addTest(.{ .root_module = hnsw_mod });
    const run_hnsw_tests = b.addRunArtifact(hnsw_tests);

    const shard_mod = b.createModule(.{
        .root_source_file = b.path("src/shard_router.zig"),
        .target = target,
        .optimize = optimize,
    });
    const shard_tests = b.addTest(.{ .root_module = shard_mod });
    const run_shard_tests = b.addRunArtifact(shard_tests);

    const iouring_mod = b.createModule(.{
        .root_source_file = b.path("src/iouring_sock.zig"),
        .target = target,
        .optimize = optimize,
    });
    const iouring_tests = b.addTest(.{ .root_module = iouring_mod });
    const run_iouring_tests = b.addRunArtifact(iouring_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_hnsw_tests.step);
    test_step.dependOn(&run_shard_tests.step);
    test_step.dependOn(&run_iouring_tests.step);
}

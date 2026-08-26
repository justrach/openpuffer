const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Public in-process engine. Dependents `b.dependency("openpuffer", …)`
    // then `addImport("openpuffer", dep.module("openpuffer"))`. HTTP serve
    // is not in this root — it stays on the `openpuffer` binary.
    const lib_mod = b.addModule("openpuffer", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "openpuffer",
        .root_module = lib_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

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

    const lib_tests = b.addTest(.{ .root_module = lib_mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const consumer_mod = b.createModule(.{
        .root_source_file = b.path("src/lib_consumer_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "openpuffer", .module = lib_mod },
        },
    });
    const consumer_tests = b.addTest(.{ .root_module = consumer_mod });
    const run_consumer_tests = b.addRunArtifact(consumer_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_hnsw_tests.step);
    test_step.dependOn(&run_shard_tests.step);
    test_step.dependOn(&run_iouring_tests.step);
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_consumer_tests.step);
}

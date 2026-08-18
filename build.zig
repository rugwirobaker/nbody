const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ---- nbody: the core library. Renderer-free, I/O-free. ----
    const nbody = b.addModule("nbody", .{
        .root_source_file = b.path("src/nbody.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ---- nbody-bench: the measurement harness. ----
    const bench = b.addExecutable(.{
        .name = "nbody-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "nbody", .module = nbody }},
        }),
    });
    b.installArtifact(bench);

    const bench_run = b.addRunArtifact(bench);
    bench_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench_run.addArgs(args);
    // RFC §2.5 rule 1: builds are compared only under ReleaseFast. Warn loudly
    // if someone benchmarks a Debug build (which measures register spills).
    if (optimize != .ReleaseFast) {
        bench_run.step.name = "run nbody-bench (WARNING: not ReleaseFast, see RFC §2.5)";
    }
    b.step("bench", "Run the benchmark harness").dependOn(&bench_run.step);

    // ---- tests ----
    const lib_tests = b.addTest(.{ .root_module = nbody });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    b.step("test", "Run the library tests").dependOn(&run_lib_tests.step);
}

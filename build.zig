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
    const test_step = b.step("test", "Run the library tests");
    test_step.dependOn(&run_lib_tests.step);

    // ---- nbody-viz: the demo client. WebGL2 over a wasm build of `nbody`. ----
    //
    // The renderer is hand-written JavaScript against WebGL2; the browser
    // supplies the window and the swapchain, and this wasm module supplies the
    // physics. Nothing here is fetched: `nbody` is renderer-free and I/O-free,
    // so it targets `wasm32-freestanding` directly with no emscripten and no
    // package dependencies.
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        // Without simd128, `std.simd.suggestVectorLength(f32)` is null and the
        // SIMD kernel would fall back to an 8-wide vector that wasm has to
        // emulate. With it the target reports 4, the same width as NEON, and
        // `simd.zig` gets real v128 registers.
        .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}),
    });

    // Deliberately not `optimize`: the page reports Phase-A ns/tick, and a
    // Debug build would report register spills instead (RFC §2.5 rule 1).
    const wasm_optimize: std.builtin.OptimizeMode = .ReleaseFast;

    // DWARF is most of an unstripped wasm and nothing in the browser reads it
    // here; stripping took the module from 742 KB to 25 KB.
    const nbody_wasm = b.createModule(.{
        .root_source_file = b.path("src/nbody.zig"),
        .target = wasm_target,
        .optimize = wasm_optimize,
        .strip = true,
    });

    const viz = b.addExecutable(.{
        .name = "nbody",
        .root_module = b.createModule(.{
            .root_source_file = b.path("viz/main.zig"),
            .target = wasm_target,
            .optimize = wasm_optimize,
            .strip = true,
            .imports = &.{.{ .name = "nbody", .module = nbody_wasm }},
        }),
    });
    // A wasm library, not a program: no `_start`, and the exports have to
    // survive the linker's garbage collection so JavaScript can call them.
    viz.entry = .disabled;
    viz.rdynamic = true;

    const web_dir: std.Build.InstallDir = .{ .custom = "web" };
    const install_wasm = b.addInstallArtifact(viz, .{
        .dest_dir = .{ .override = web_dir },
    });
    const install_static = b.addInstallDirectory(.{
        .source_dir = b.path("viz"),
        .install_dir = web_dir,
        .install_subdir = "",
        .include_extensions = &.{ ".html", ".js", ".css" },
    });

    const viz_step = b.step("viz", "Build the WebGL2 demo into zig-out/web");
    viz_step.dependOn(&install_wasm.step);
    viz_step.dependOn(&install_static.step);

    // The fixed-timestep clamp (RFC §2.4) is target-independent, so it is
    // tested on the host rather than only exercised in a browser.
    const timestep_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("viz/timestep.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(timestep_tests).step);
}

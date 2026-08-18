//! nbody-bench — the measurement harness (RFC §2.5).
//!
//! Measurement rules, restated because they are easy to violate by accident:
//!   1. Builds are compared only under `-Doptimize=ReleaseFast`.
//!   2. The metric is Phase-A wall time per tick in nanoseconds — never FPS.
//!   3. Measurements are only meaningful at fixed n, so `merging = false`.
//!   4. All randomness flows from the config seed, recorded with every number.

const std = @import("std");
const builtin = @import("builtin");
const nbody = @import("nbody");

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_file.interface;

    const cfg = nbody.Config{};

    try out.print("nbody-bench\n", .{});
    try out.print("  target      {s}-{s}\n", .{ @tagName(builtin.cpu.arch), @tagName(builtin.os.tag) });
    try out.print("  optimize    {s}{s}\n", .{
        @tagName(builtin.mode),
        if (builtin.mode == .ReleaseFast) "" else "  (RFC §2.5: measurements require ReleaseFast)",
    });
    try out.print("  lane count  {d} (f32)\n", .{nbody.lane_count});
    try out.print("  seed        0x{X}\n", .{cfg.seed});
    try out.print("  n           {d}\n", .{cfg.n});
    try out.print("  g/dt/eps2   {d} / {d} / {d}\n", .{ cfg.g, cfg.dt, cfg.eps2 });
    try out.print("\n  no kernels yet — Part 2 lands the scalar baseline.\n", .{});

    try out.flush();
}

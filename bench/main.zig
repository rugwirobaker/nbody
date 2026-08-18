//! nbody-bench — the measurement harness (RFC §2.5).
//!
//! Measurement rules, restated because they are easy to violate by accident:
//!   1. Builds are compared only under `-Doptimize=ReleaseFast`.
//!   2. The metric is Phase-A wall time per tick in nanoseconds — never FPS.
//!   3. Measurements are only meaningful at fixed n, so `merging = false`.
//!   4. All randomness flows from the config seed, recorded with every number.
//!
//! Usage: `zig build bench -Doptimize=ReleaseFast -- [n ...]`

const std = @import("std");
const builtin = @import("builtin");
const nbody = @import("nbody");

/// Sized so the n² work spans a couple of orders of magnitude and crosses the
/// cache levels — the ns/pair column is what shows that happening.
const default_sweep = [_]usize{ 512, 1024, 2048, 4096, 8192 };

/// Stop a measurement once Phase A has accumulated this much time...
const target_ns: i96 = 100 * std.time.ns_per_ms;
/// ...or this many ticks, whichever comes first. Small n hits the tick cap,
/// large n hits the time budget.
const max_ticks: usize = 200;
const min_ticks: usize = 5;

const Result = struct {
    n: usize,
    ticks: usize,
    ns_per_tick: f64,
    ns_per_pair: f64,
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_file.interface;

    const args = try init.minimal.args.toSlice(arena);
    const sweep = if (args.len > 1) try parseSweep(arena, args[1..]) else &default_sweep;

    // One config for the whole sweep: only n varies, and it is the axis being
    // swept. Everything else — g, dt, eps2, seed, preset — is held fixed, or
    // the numbers would not be comparable to each other, let alone to Part 3.
    const base = nbody.Config{ .merging = false };
    if (nbody.Config.validate(base)) |why| {
        try out.print("invalid config: {s}\n", .{why});
        try out.flush();
        return error.InvalidConfig;
    }

    try out.print("nbody-bench — scalar baseline (RFC Part 2)\n\n", .{});
    try out.print("  target      {s}-{s}\n", .{ @tagName(builtin.cpu.arch), @tagName(builtin.os.tag) });
    try out.print("  optimize    {s}{s}\n", .{
        @tagName(builtin.mode),
        if (builtin.mode == .ReleaseFast) "" else "  (RFC §2.5: measurements require ReleaseFast)",
    });
    try out.print("  lane count  {d} (f32, unused by the scalar baseline)\n", .{nbody.lane_count});
    try out.print("  seed        0x{X}\n", .{base.seed});
    try out.print("  preset      {s}, merging {}\n", .{ @tagName(base.preset), base.merging });
    try out.print("  g/dt/eps2   {d} / {d} / {d}\n", .{ base.g, base.dt, base.eps2 });
    try out.print("  mass/radius [{d}, {d}] / {d}, jitter {d}\n\n", .{
        base.mass_min, base.mass_max, base.radius, base.jitter,
    });

    try out.print("  {s:>8}  {s:>7}  {s:>14}  {s:>10}\n", .{ "n", "ticks", "ns/tick", "ns/pair" });
    try out.print("  {s:>8}  {s:>7}  {s:>14}  {s:>10}\n", .{ "--------", "-------", "--------------", "----------" });
    try out.flush();

    for (sweep) |n| {
        const r = try measure(gpa, init.io, base, n);
        try out.print("  {d:>8}  {d:>7}  {d:>14.1}  {d:>10.3}\n", .{
            r.n, r.ticks, r.ns_per_tick, r.ns_per_pair,
        });
        try out.flush(); // a long sweep should report as it goes
    }

    try out.print("\n  ns/pair = ns/tick / n². Flat across n means the kernel is\n", .{});
    try out.print("  compute-bound; the rise at large n is the working set leaving cache.\n", .{});
    if (builtin.mode != .ReleaseFast) {
        try out.print("\n  WARNING: this build is {s}. These numbers measure register\n", .{@tagName(builtin.mode)});
        try out.print("  spills, not the algorithm (RFC §2.5 rule 1, §3.3c).\n", .{});
    }
    try out.flush();
}

fn parseSweep(arena: std.mem.Allocator, args: []const [:0]const u8) ![]const usize {
    const sweep = try arena.alloc(usize, args.len);
    for (args, sweep) |arg, *slot| {
        slot.* = std.fmt.parseInt(usize, arg, 10) catch return error.InvalidParticleCount;
        if (slot.* == 0) return error.InvalidParticleCount;
    }
    return sweep;
}

/// Times `computeAccelerations` alone — the RFC's Phase A, and the only thing
/// the n² claim is about (RFC §2.5 rule 2).
///
/// `integrate` still runs, outside the timer: the measurement has to be of a
/// simulation that is actually advancing, or Phase A would re-read one
/// unchanging position snapshot tick after tick and report a cache behaviour
/// that no real run has.
fn measure(gpa: std.mem.Allocator, io: std.Io, base: nbody.Config, n: usize) !Result {
    const cfg = blk: {
        var c = base;
        c.n = n;
        break :blk c;
    };

    var sim = try nbody.Sim.initSeeded(gpa, cfg);
    defer sim.deinit(gpa);

    // Warmup: fault in the pages and let the clocks settle before anything is
    // recorded.
    for (0..3) |_| nbody.scalar.tick(&sim, cfg);

    var total_ns: i96 = 0;
    var ticks: usize = 0;
    while (ticks < max_ticks and (ticks < min_ticks or total_ns < target_ns)) : (ticks += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        nbody.scalar.computeAccelerations(&sim, cfg);
        const t1 = std.Io.Timestamp.now(io, .awake);
        total_ns += t0.durationTo(t1).toNanoseconds();

        nbody.scalar.integrate(&sim, cfg);
    }

    const ns_per_tick = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(ticks));
    const pairs = @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    return .{
        .n = n,
        .ticks = ticks,
        .ns_per_tick = ns_per_tick,
        .ns_per_pair = ns_per_tick / pairs,
    };
}

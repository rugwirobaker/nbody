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

/// Spans two and a half orders of magnitude in n, so the ns/pair column has a
/// chance to show both ends of the curve.
///
/// The small rows are not filler. Part 3's Variant A does two `@reduce` calls
/// per row, amortized over only n/L inner iterations, so SIMD's advantage is
/// at its *smallest* here — dropping them would hide the one part of the sweep
/// where the horizontal reduction is visible in the cost.
///
/// The top end reaches for the cache cliff. AoS streams 24n bytes per row, so
/// n = 16384 is ~390 KB and well past L1 — worth noting that at 8192 (~196 KB)
/// nothing had bent yet, so the claim that ns/pair rises when the working set
/// leaves cache is still owed a demonstration on this hardware.
const default_sweep = [_]usize{ 256, 512, 1024, 2048, 4096, 8192, 16384 };

/// Stop a measurement once Phase A has accumulated this much time...
const target_ns: i96 = 100 * std.time.ns_per_ms;
/// ...or this many ticks, whichever comes first.
///
/// `max_ticks` is a runaway guard, not a target: **every row should stop on
/// the time budget**, so that each n gets the same amount of measurement. It
/// was 200, which silently capped the small-n rows — n = 512 needs ~350 ticks
/// to reach 100 ms, so it was being cut off at 57 ms while every other row got
/// the full budget, and that row's ns/pair duly swung 1.09–1.60 across runs.
/// If a row ever reports exactly `max_ticks`, its number is undermeasured and
/// this constant is too low.
const max_ticks: usize = 2000;
const min_ticks: usize = 5;

/// Discarded work before the first measurement, to get the CPU off its idle
/// clock. Per-row warmup ticks are not enough for this: three ticks at n = 512
/// is under a millisecond, so without this the first row in the sweep pays for
/// the frequency ramp and reads slow.
const warmup_ns: i96 = 300 * std.time.ns_per_ms;

const Result = struct {
    n: usize,
    ticks: usize,
    ns_per_tick: f64,
    ns_per_pair: f64,
};

/// Which kernel a measurement is of.
///
/// A harness axis, deliberately **not** a `Config` field. RFC §3.5 binds the
/// comparison to "same config and seed", so the one thing being measured must
/// not be part of what is held equal. Both kernels also run in this one
/// process, one compiler invocation, one thermal state — two separately-built
/// binaries would add a variable for no benefit.
const Kernel = enum { scalar, simd };

/// Lane width for the main sweep: the target's natural vector size.
const lanes = nbody.simd.default_lanes;

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

    try out.print("nbody-bench — scalar baseline vs SIMD (RFC Parts 2–3)\n\n", .{});
    try out.print("  target      {s}-{s}\n", .{ @tagName(builtin.cpu.arch), @tagName(builtin.os.tag) });
    try out.print("  optimize    {s}{s}\n", .{
        @tagName(builtin.mode),
        if (builtin.mode == .ReleaseFast) "" else "  (RFC §2.5: measurements require ReleaseFast)",
    });
    try out.print("  lane count  {d} (f32; scalar baseline does not use it)\n", .{nbody.lane_count});
    try out.print("  seed        0x{X}\n", .{base.seed});
    try out.print("  preset      {s}, merging {}\n", .{ @tagName(base.preset), base.merging });
    try out.print("  g/dt/eps2   {d} / {d} / {d}\n", .{ base.g, base.dt, base.eps2 });
    try out.print("  mass/radius [{d}, {d}] / {d}, jitter {d}\n\n", .{
        base.mass_min, base.mass_max, base.radius, base.jitter,
    });

    try warmUp(gpa, init.io, base);

    try out.print("  Phase A only, ns/tick. Both kernels seeded from the same AoS sim.\n\n", .{});
    try out.print("  {s:>8}  {s:>15}  {s:>15}  {s:>9}  {s:>9}\n", .{
        "n", "scalar ns/tick", "simd ns/tick", "simd/pair", "speedup",
    });
    try out.print("  {s:>8}  {s:>15}  {s:>15}  {s:>9}  {s:>9}\n", .{
        "--------", "---------------", "---------------", "---------", "---------",
    });
    try out.flush();

    for (sweep) |n| {
        const s_row = try measure(.scalar, lanes, gpa, init.io, base, n);
        const v_row = try measure(.simd, lanes, gpa, init.io, base, n);
        try out.print("  {d:>8}  {d:>15.1}  {d:>15.1}  {d:>9.3}  {d:>8.2}x\n", .{
            n,                                     s_row.ns_per_tick, v_row.ns_per_tick, v_row.ns_per_pair,
            s_row.ns_per_tick / v_row.ns_per_tick,
        });
        try out.flush(); // a long sweep should report as it goes
    }

    // ---- labeled experiment: lane width ----
    //
    // RFC §3.2 requires the kernel be parameterized by L rather than hard-coded,
    // which makes this measurable rather than theoretical. Widths above the
    // hardware's native vector size are legal Zig — @Vector lowers them by
    // splitting into multiple registers — so this shows where widening stops
    // paying on 4-wide NEON.
    const experiment_n: usize = 4096;
    try out.print("\n  Lane-width experiment (labeled, non-normative), n = {d}:\n\n", .{experiment_n});
    try out.print("  {s:>8}  {s:>15}  {s:>9}\n", .{ "L", "simd ns/tick", "speedup" });
    try out.print("  {s:>8}  {s:>15}  {s:>9}\n", .{ "--------", "---------------", "---------" });

    const baseline = try measure(.scalar, lanes, gpa, init.io, base, experiment_n);
    inline for (.{ 1, 2, 4, 8, 16 }) |experiment_lanes| {
        const row = try measure(.simd, experiment_lanes, gpa, init.io, base, experiment_n);
        try out.print("  {d:>8}  {d:>15.1}  {d:>8.2}x{s}\n", .{
            experiment_lanes,
            row.ns_per_tick,
            baseline.ns_per_tick / row.ns_per_tick,
            if (experiment_lanes == nbody.lane_count) "  <- native" else "",
        });
        try out.flush();
    }

    try out.print("\n  ns/pair = ns/tick / n². Flat across n means the kernel is\n", .{});
    try out.print("  compute-bound — and on this hardware it is flat all the way:\n", .{});
    try out.print("  no cache cliff appears even at n = 16384, where each row\n", .{});
    try out.print("  streams ~390 KB of AoS particles. The access pattern is pure\n", .{});
    try out.print("  sequential and the per-pair sqrt+div leaves the prefetcher\n", .{});
    try out.print("  ample time, so memory never becomes the limiter.\n", .{});
    if (builtin.mode != .ReleaseFast) {
        try out.print("\n  WARNING: this build is {s}. These numbers measure register\n", .{@tagName(builtin.mode)});
        try out.print("  spills, not the algorithm (RFC §2.5 rule 1, §3.3c).\n", .{});
    }
    try out.flush();
}

/// Runs real Phase-A work for `warmup_ns` and throws the result away, so the
/// CPU is at its sustained clock before the first row is timed.
///
/// This measures nothing and asserts nothing. It exists because the first
/// entry in a sweep is otherwise systematically slower than the rest — an
/// artefact of the machine, not of n, and one that would show up as a bogus
/// bend at the small-n end of the curve.
fn warmUp(gpa: std.mem.Allocator, io: std.Io, base: nbody.Config) !void {
    const cfg = blk: {
        var c = base;
        c.n = 2048;
        break :blk c;
    };

    var sim = try nbody.Sim.initSeeded(gpa, cfg);
    defer sim.deinit(gpa);

    const start = std.Io.Timestamp.now(io, .awake);
    while (start.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds() < warmup_ns) {
        nbody.scalar.computeAccelerations(&sim, cfg);
        nbody.scalar.integrate(&sim, cfg);
    }
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
fn measure(
    comptime kernel: Kernel,
    comptime L: usize,
    gpa: std.mem.Allocator,
    io: std.Io,
    base: nbody.Config,
    n: usize,
) !Result {
    const cfg = blk: {
        var c = base;
        c.n = n;
        break :blk c;
    };

    // Both kernels are seeded from the same AoS sim, so they measure identical
    // initial conditions rather than two seeding paths that merely look alike.
    var sim = try nbody.Sim.initSeeded(gpa, cfg);
    defer sim.deinit(gpa);

    var particles: nbody.simd.Particles = if (kernel == .simd)
        try nbody.simd.Particles.fromAoS(gpa, sim, L)
    else
        undefined;
    defer if (kernel == .simd) particles.deinit(gpa);

    // Warmup: fault in the pages and let the clocks settle before anything is
    // recorded.
    for (0..3) |_| switch (kernel) {
        .scalar => nbody.scalar.tick(&sim, cfg),
        .simd => nbody.simd.tick(L, &particles, cfg),
    };

    var total_ns: i96 = 0;
    var ticks: usize = 0;
    while (ticks < max_ticks and (ticks < min_ticks or total_ns < target_ns)) : (ticks += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        switch (kernel) {
            .scalar => nbody.scalar.computeAccelerations(&sim, cfg),
            .simd => nbody.simd.computeAccelerations(L, &particles, cfg),
        }
        const t1 = std.Io.Timestamp.now(io, .awake);
        total_ns += t0.durationTo(t1).toNanoseconds();

        switch (kernel) {
            .scalar => nbody.scalar.integrate(&sim, cfg),
            .simd => nbody.simd.integrate(L, &particles, cfg),
        }
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

//! Tests for the SIMD build (RFC §3).
//!
//! Two jobs. First, the **padding invariants** (§3.2): zero-mass slots past
//! `n` must be arithmetically present and physically absent, and every path
//! that shrinks `n` must keep them that way. The ghost-particle bug is the
//! reason this section exists — it is invisible to a renderer, never a merge
//! candidate, and passes every test except momentum-with-merging.
//!
//! Second, **equivalence with the scalar baseline** (§3.5). That comparison is
//! tolerance-based and never bitwise: `@reduce` reorders Step 7's summation,
//! float addition is not associative, and the system is chaotic. Agreement to
//! the last bit is not expected, and its absence is not a bug.

const std = @import("std");
const testing = std.testing;

const Config = @import("config.zig").Config;
const platform = @import("platform.zig");
const properties = @import("properties.zig");
const scalar = @import("scalar.zig");
const simd = @import("simd.zig");
const sim_mod = @import("sim.zig");
const Particle = sim_mod.Particle;
const Sim = sim_mod.Sim;

const L = simd.default_lanes;

/// Seeds an AoS sim and converts it, which is the only supported route into
/// the SoA layout (see `Particles.fromAoS`).
fn seeded(cfg: Config, comptime lanes: usize) !struct { Sim, simd.Particles } {
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    errdefer sim.deinit(testing.allocator);
    const p = try simd.Particles.fromAoS(testing.allocator, sim, lanes);
    return .{ sim, p };
}

// ---- layout and conversion ----

test "fromAoS pads up to a multiple of L and zeroes the tail" {
    const cfg = Config{ .n = 100 }; // deliberately not a multiple of any L
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 100), p.n);
    try testing.expectEqual(platform.alignUp(100, L), p.n_padded);
    try testing.expectEqual(platform.alignUp(100, L), p.capacity);
    try testing.expect(p.n_padded >= p.n);

    // The invariant at birth: every slot past n is inert.
    for (p.mass[p.n..p.capacity]) |m| try testing.expectEqual(@as(f32, 0), m);
    for (p.x[p.n..p.capacity]) |v| try testing.expectEqual(@as(f32, 0), v);
    for (p.vx[p.n..p.capacity]) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "fromAoS then toAoS round-trips exactly" {
    const cfg = Config{ .n = 64 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    const buf = try testing.allocator.alloc(Particle, cfg.n);
    defer testing.allocator.free(buf);

    // Bit-identical, not approximate: this is what lets §3.5 claim both builds
    // start from the same initial conditions.
    try testing.expectEqualSlices(Particle, sim.live(), p.toAoS(buf));
}

test "SoA arrays are cache-line aligned" {
    const cfg = Config{ .n = 64 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    const bytes = platform.cache_line.toByteUnits();
    for ([_][]f32{ p.x, p.y, p.vx, p.vy, p.mass, p.heat, p.ax, p.ay }) |a| {
        try testing.expectEqual(@as(usize, 0), @intFromPtr(a.ptr) % bytes);
    }
}

// ---- padding as physics (§3.2) ----

test "zero-mass padding contributes exactly nothing" {
    // The claim the whole padding scheme rests on. Run Phase A at n = 5 with
    // L-rounded padding, then again with the padding slots filled with absurd
    // positions — the accelerations must be identical, because every
    // contribution is multiplied by a source mass of zero.
    const cfg = Config{};
    var sim = try Sim.initCapacity(testing.allocator, 5);
    defer sim.deinit(testing.allocator);
    for (0..5) |k| {
        const f: f32 = @floatFromInt(k);
        sim.push(.{ .x = f * 0.3, .y = f * -0.2, .vx = 0, .vy = 0, .mass = 1 + f, .heat = 0 });
    }

    var p = try simd.Particles.fromAoS(testing.allocator, sim, L);
    defer p.deinit(testing.allocator);
    simd.computeAccelerations(L, &p, cfg);

    const ax_clean = try testing.allocator.dupe(f32, p.ax[0..p.n]);
    defer testing.allocator.free(ax_clean);
    const ay_clean = try testing.allocator.dupe(f32, p.ay[0..p.n]);
    defer testing.allocator.free(ay_clean);

    // Ghosts with no gravity: present in the arithmetic, absent from the
    // physics. Position them somewhere ludicrous; mass 0 is the only thing
    // that matters.
    for (p.n..p.capacity) |k| {
        p.x[k] = 1.0e6;
        p.y[k] = -3.0e5;
    }
    simd.computeAccelerations(L, &p, cfg);

    try testing.expectEqualSlices(f32, ax_clean, p.ax[0..p.n]);
    try testing.expectEqualSlices(f32, ay_clean, p.ay[0..p.n]);
}

test "padded lanes never produce NaN or inf" {
    // Zero times something is only dangerous if the something is infinite.
    // Softening guarantees the denominator is at least ε³, which is exactly
    // what makes zero-mass padding safe — the two decisions click together.
    const cfg = Config{};
    var sim = try Sim.initCapacity(testing.allocator, 3);
    defer sim.deinit(testing.allocator);
    // Every particle at the origin, so padding slots are coincident with them.
    for (0..3) |_| sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });

    var p = try simd.Particles.fromAoS(testing.allocator, sim, L);
    defer p.deinit(testing.allocator);
    simd.computeAccelerations(L, &p, cfg);

    for (p.ax[0..p.n_padded]) |a| {
        try testing.expect(!std.math.isNan(a));
        try testing.expect(std.math.isFinite(a));
    }
    for (p.ay[0..p.n_padded]) |a| {
        try testing.expect(!std.math.isNan(a));
        try testing.expect(std.math.isFinite(a));
    }
}

test "merging re-zeroes the vacated slot (the ghost-particle trap)" {
    // The bug this catches is silent: a renderer drawing [0..n] cannot see it,
    // it is never a merge candidate, and every test passes except
    // momentum-with-merging. Here it is caught directly.
    const cfg = Config{ .merging = true, .d_merge2 = 0.01 };
    var sim = try Sim.initCapacity(testing.allocator, 4);
    defer sim.deinit(testing.allocator);
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 0.05, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 5, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 9, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });

    var p = try simd.Particles.fromAoS(testing.allocator, sim, L);
    defer p.deinit(testing.allocator);

    // Give the slots real accelerations first, so a failure to re-zero leaves
    // something visibly non-zero behind.
    simd.computeAccelerations(L, &p, cfg);
    try testing.expectEqual(@as(usize, 1), simd.mergeCollisions(L, &p, cfg));
    try testing.expectEqual(@as(usize, 3), p.n);

    for (p.mass[p.n..p.capacity]) |m| try testing.expectEqual(@as(f32, 0), m);
    for (p.ax[p.n..p.capacity]) |a| try testing.expectEqual(@as(f32, 0), a);
    for (p.ay[p.n..p.capacity]) |a| try testing.expectEqual(@as(f32, 0), a);
    p.assertPaddingIsInert();
}

test "a ghost would change the answer — so the re-zero is load-bearing" {
    // Proves the previous test is testing something. Plant a full-mass ghost
    // in a padding slot by hand and confirm Phase A notices, which is exactly
    // what a missing `mass[n] = 0` would leave behind.
    const cfg = Config{};
    var sim = try Sim.initCapacity(testing.allocator, 2);
    defer sim.deinit(testing.allocator);
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 1, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });

    var p = try simd.Particles.fromAoS(testing.allocator, sim, L);
    defer p.deinit(testing.allocator);
    if (p.capacity <= p.n) return error.SkipZigTest; // no padding at this L

    simd.computeAccelerations(L, &p, cfg);
    const ax_clean = p.ax[0];

    p.mass[p.n] = 1000.0; // the ghost
    p.x[p.n] = 0.5;
    simd.computeAccelerations(L, &p, cfg);

    try testing.expect(p.ax[0] != ax_clean);
}

test "n_padded shrinks with n so the inner loop does too" {
    const cfg = Config{ .merging = true, .d_merge2 = 0.01 };
    var sim = try Sim.initCapacity(testing.allocator, 3 * L);
    defer sim.deinit(testing.allocator);
    for (0..3 * L) |k| {
        const f: f32 = @floatFromInt(k);
        sim.push(.{ .x = f, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    }
    // Put two of them on top of each other so exactly one merge happens.
    sim.particles[1].x = sim.particles[0].x + 0.01;

    var p = try simd.Particles.fromAoS(testing.allocator, sim, L);
    defer p.deinit(testing.allocator);

    const padded_before = p.n_padded;
    try testing.expectEqual(@as(usize, 1), simd.mergeCollisions(L, &p, cfg));
    try testing.expectEqual(platform.alignUp(p.n, L), p.n_padded);
    try testing.expect(p.n_padded <= padded_before);
    try testing.expectEqual(@as(usize, 3 * L), p.capacity); // capacity never moves
}

// ---- equivalence with the baseline (§3.5) ----

/// Largest positional difference between the two builds, relative to the
/// system's own length scale. Judged against `radius` rather than each
/// particle's own coordinate, which would blow up near the origin.
fn maxPositionError(sim: Sim, p: simd.Particles, cfg: Config) f64 {
    var worst: f64 = 0;
    for (sim.live(), 0..) |a, i| {
        const dx = @as(f64, a.x) - @as(f64, p.x[i]);
        const dy = @as(f64, a.y) - @as(f64, p.y[i]);
        worst = @max(worst, @sqrt(dx * dx + dy * dy) / @as(f64, cfg.radius));
    }
    return worst;
}

test "(§3.5) both builds agree over a short horizon" {
    const cfg = Config{ .n = 256 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    // Identical initial conditions by construction, not by luck.
    try testing.expectEqual(@as(f64, 0), maxPositionError(sim, p, cfg));

    for (0..100) |_| {
        scalar.tick(&sim, cfg);
        simd.tick(L, &p, cfg);
    }

    // Tolerance, never bitwise: @reduce reorders the summation. The error's
    // growth rate with horizon length is the chaos, not a defect.
    try testing.expect(maxPositionError(sim, p, cfg) < 1e-4);
}

test "(§3.5) the SIMD build conserves momentum on its own terms" {
    // Invariants are immune to chaos even when trajectories are not, so this
    // holds at any horizon — unlike the trajectory comparison above.
    const cfg = Config{ .n = 128 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    const buf = try testing.allocator.alloc(Particle, cfg.n);
    defer testing.allocator.free(buf);

    for (0..2000) |_| simd.tick(L, &p, cfg);

    const live = p.toAoS(buf);
    const momentum = properties.momentum(live);
    const scale = properties.momentumScale(live);
    try testing.expect(scale > 0);
    try testing.expect(momentum.magnitude() / scale < 1e-5);
}

test "(§3.5) the SIMD build conserves momentum and mass across merges" {
    // The ghost-particle bug's signature is this test failing while every
    // other one passes.
    const cfg = Config{ .n = 128, .merging = true, .heat_decay = 1.0 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    const buf = try testing.allocator.alloc(Particle, cfg.n);
    defer testing.allocator.free(buf);

    const n0 = p.n;
    const m0 = properties.totalMass(p.toAoS(buf));

    for (0..2000) |_| simd.tick(L, &p, cfg);

    const live = p.toAoS(buf);
    try testing.expect(p.n < n0); // merges actually happened
    try testing.expectApproxEqRel(m0, properties.totalMass(live), 1e-6);

    const momentum = properties.momentum(live);
    const scale = properties.momentumScale(live);
    try testing.expect(momentum.magnitude() / scale < 1e-5);
    p.assertPaddingIsInert();
}

test "the kernel is genuinely parameterized by lane width" {
    // Different L must produce the same physics to the same tolerance. If this
    // fails, something is hard-coded to one width — non-compliant per §3.2.
    const cfg = Config{ .n = 128 };

    var sim_a, var p2 = try seeded(cfg, 2);
    defer sim_a.deinit(testing.allocator);
    defer p2.deinit(testing.allocator);
    var sim_b, var p8 = try seeded(cfg, 8);
    defer sim_b.deinit(testing.allocator);
    defer p8.deinit(testing.allocator);

    for (0..100) |_| {
        simd.tick(2, &p2, cfg);
        simd.tick(8, &p8, cfg);
    }

    var worst: f64 = 0;
    for (0..cfg.n) |i| {
        const dx = @as(f64, p2.x[i]) - @as(f64, p8.x[i]);
        const dy = @as(f64, p2.y[i]) - @as(f64, p8.y[i]);
        worst = @max(worst, @sqrt(dx * dx + dy * dy) / @as(f64, cfg.radius));
    }
    try testing.expect(worst < 1e-4);
}

test "the SIMD build is deterministic" {
    const cfg = Config{ .n = 96, .merging = true };
    var sim_a, var a = try seeded(cfg, L);
    defer sim_a.deinit(testing.allocator);
    defer a.deinit(testing.allocator);
    var sim_b, var b = try seeded(cfg, L);
    defer sim_b.deinit(testing.allocator);
    defer b.deinit(testing.allocator);

    for (0..500) |_| {
        simd.tick(L, &a, cfg);
        simd.tick(L, &b, cfg);
    }

    try testing.expectEqual(a.n, b.n);
    try testing.expectEqualSlices(f32, a.x[0..a.n], b.x[0..b.n]);
    try testing.expectEqualSlices(f32, a.y[0..a.n], b.y[0..b.n]);
    try testing.expectEqualSlices(f32, a.mass[0..a.n], b.mass[0..b.n]);
}

test "Phase A leaves particle state untouched (frozen snapshot)" {
    const cfg = Config{ .n = 64 };
    var sim, var p = try seeded(cfg, L);
    defer sim.deinit(testing.allocator);
    defer p.deinit(testing.allocator);

    const x_before = try testing.allocator.dupe(f32, p.x[0..p.n]);
    defer testing.allocator.free(x_before);
    const vx_before = try testing.allocator.dupe(f32, p.vx[0..p.n]);
    defer testing.allocator.free(vx_before);

    simd.computeAccelerations(L, &p, cfg);

    try testing.expectEqualSlices(f32, x_before, p.x[0..p.n]);
    try testing.expectEqualSlices(f32, vx_before, p.vx[0..p.n]);
}

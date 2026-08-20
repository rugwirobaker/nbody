//! The RFC §2.5 acceptance tests.
//!
//! Each one verifies a *named Part 1 result* rather than a trajectory — the
//! system is chaotic, so "the particles end up here" is not a testable claim,
//! while "momentum is conserved" is, forever, at any horizon.
//!
//! Each test appears twice where merging changes what can be claimed: once
//! with `merging = false`, isolating the integrator and the force law, and
//! once with merging on, which is the stronger form — test (b) across merge
//! events is what catches an un-weighted average in `mergePair`.
//!
//! **One caveat on energy.** RFC §2.5 test (d) says `KE + PE + Σheat` holds
//! constant across merges. It doesn't, quite: Step 10 banks the destroyed
//! *kinetic* energy into `heat`, but the merged pair's mutual potential simply
//! vanishes with the pair, stepping total energy up at each merge. So the
//! energy claims here are split — proved exactly where they can be (the
//! two-body test below, where the vanished term is computable), and asserted
//! only on merge-free ticks at system scale. See `src/merge.zig`.

const std = @import("std");
const testing = std.testing;

const Config = @import("config.zig").Config;
const merge = @import("merge.zig");
const properties = @import("properties.zig");
const scalar = @import("scalar.zig");
const sim_mod = @import("sim.zig");
const Particle = sim_mod.Particle;
const Sim = sim_mod.Sim;

/// A two-body setup whose centre of mass is stationary and whose orbit is
/// circular *under the softened force law the kernel actually integrates*.
///
/// The relative orbit obeys r̈ = −G(M+m)·r/(r²+ε²)^1.5, so a circular relative
/// orbit needs v_rel = d·√(G(M+m)/(d²+ε²)^1.5); splitting v_rel between the
/// bodies in inverse proportion to their masses puts P at zero. Deriving it
/// this way rather than with the textbook √(GM/d) means the test measures the
/// kernel instead of the approximation in the test's own setup.
fn twoBody(cfg: Config, heavy: f32, light: f32, d: f32) !Sim {
    var sim = try Sim.initCapacity(testing.allocator, 2);
    errdefer sim.deinit(testing.allocator);

    const total = heavy + light;
    const soft = @as(f64, d) * d + @as(f64, cfg.eps2);
    const v_rel: f64 = @as(f64, d) * @sqrt(@as(f64, cfg.g) * total / (soft * @sqrt(soft)));

    sim.push(.{
        .x = 0,
        .y = 0,
        .vx = 0,
        .vy = @floatCast(-v_rel * light / total),
        .mass = heavy,
        .heat = 0,
    });
    sim.push(.{
        .x = d,
        .y = 0,
        .vx = 0,
        .vy = @floatCast(v_rel * heavy / total),
        .mass = light,
        .heat = 0,
    });
    return sim;
}

fn separation(sim: Sim) f64 {
    const a = sim.live()[0];
    const b = sim.live()[1];
    const dx = @as(f64, b.x) - @as(f64, a.x);
    const dy = @as(f64, b.y) - @as(f64, a.y);
    return @sqrt(dx * dx + dy * dy);
}

test "(a) two-body: the light particle orbits, the heavy one barely wobbles" {
    // RFC Step 6: each particle's acceleration carries the *other's* mass, so
    // a 1000:1 mass ratio is a 1000:1 motion ratio.
    const cfg = Config{ .heat_decay = 1.0 };
    const d0: f32 = 0.5;
    var sim = try twoBody(cfg, 1000.0, 1.0, d0);
    defer sim.deinit(testing.allocator);

    // One orbital period: T = 2πd/v_rel ≈ 3.14 s at these numbers, and
    // dt = 1e-3, so ~3142 ticks.
    const ticks = 3142;

    var min_sep: f64 = std.math.inf(f64);
    var max_sep: f64 = 0;
    var heavy_excursion: f64 = 0;
    var light_excursion: f64 = 0;

    for (0..ticks) |_| {
        scalar.tick(&sim, cfg);

        const s = separation(sim);
        min_sep = @min(min_sep, s);
        max_sep = @max(max_sep, s);

        const h = sim.live()[0];
        const l = sim.live()[1];
        heavy_excursion = @max(heavy_excursion, @sqrt(@as(f64, h.x) * h.x + @as(f64, h.y) * h.y));
        const lx = @as(f64, l.x) - d0;
        light_excursion = @max(light_excursion, @sqrt(lx * lx + @as(f64, l.y) * l.y));
    }

    // The orbit stays an orbit: separation holds to within a couple of percent
    // over a full revolution (first-order integrator, dt = 1e-3).
    try testing.expectApproxEqRel(@as(f64, d0), min_sep, 0.02);
    try testing.expectApproxEqRel(@as(f64, d0), max_sep, 0.02);

    // The asymmetry itself: the light particle sweeps a full diameter, the
    // heavy one traces a circle 1000× smaller.
    try testing.expect(light_excursion > 0.9 * d0);
    try testing.expect(light_excursion / heavy_excursion > 500.0);
}

test "(b) momentum is conserved" {
    // RFC Step 9. The cancellation is exact in the *discrete algorithm* —
    // provided Phase A used one frozen snapshot — so this is a real test of
    // Step 8, not a loose ideal. What is left is f32 rounding in Phase A's
    // accumulators, which random-walks rather than drifts.
    const cfg = Config{ .n = 128, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    // Seeding zeroes the net momentum, so the target is a clean zero.
    const scale0 = properties.momentumScale(sim.live());
    try testing.expect(scale0 > 0);
    try testing.expect(properties.momentum(sim.live()).magnitude() / scale0 < 1e-6);

    for (0..2000) |_| scalar.tick(&sim, cfg);

    const p = properties.momentum(sim.live());
    const scale = properties.momentumScale(sim.live());
    try testing.expect(p.magnitude() / scale < 1e-5);
}

test "(c) the centre of mass drifts in a straight line at constant speed" {
    // RFC Step 9 corollary. Boost the whole system so the claim has something
    // to say: with P = 0 the CoM merely sitting still would pass vacuously.
    const cfg = Config{ .n = 128, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    const boost_x: f32 = 0.3;
    const boost_y: f32 = -0.2;
    for (sim.live()) |*p| {
        p.vx += boost_x;
        p.vy += boost_y;
    }

    const c0 = properties.centerOfMass(sim.live());
    const v0 = properties.centerOfMassVelocity(sim.live());
    try testing.expectApproxEqAbs(@as(f64, boost_x), v0.x, 1e-4);
    try testing.expectApproxEqAbs(@as(f64, boost_y), v0.y, 1e-4);

    const ticks = 2000;
    var max_deviation: f64 = 0;
    for (1..ticks + 1) |k| {
        scalar.tick(&sim, cfg);

        const t = @as(f64, @floatFromInt(k)) * @as(f64, cfg.dt);
        const c = properties.centerOfMass(sim.live());
        const predicted = properties.Vec2{ .x = c0.x + v0.x * t, .y = c0.y + v0.y * t };
        max_deviation = @max(max_deviation, c.sub(predicted).magnitude());
    }

    // Judged against the distance travelled, not an absolute epsilon.
    const travelled = v0.magnitude() * @as(f64, ticks) * @as(f64, cfg.dt);
    try testing.expect(travelled > 0.5);
    try testing.expect(max_deviation / travelled < 1e-3);
}

test "(d) energy: a symplectic orbit does not spiral outward" {
    // RFC Step 5's ordering is the whole subject of this test. Explicit Euler
    // — position updated with the *old* velocity — costs the same and pumps
    // energy in; the orbit would grow monotonically here.
    const cfg = Config{ .heat_decay = 1.0 };
    const d0: f32 = 0.5;
    var sim = try twoBody(cfg, 1000.0, 1.0, d0);
    defer sim.deinit(testing.allocator);

    const ticks = 10_000; // ~3.2 orbits
    const window = 1000;

    var early_sum: f64 = 0;
    var late_sum: f64 = 0;
    const e0 = properties.totalEnergy(sim.live(), cfg);
    var max_energy_error: f64 = 0;

    for (0..ticks) |k| {
        scalar.tick(&sim, cfg);

        const s = separation(sim);
        if (k < window) early_sum += s;
        if (k >= ticks - window) late_sum += s;

        const e = properties.totalEnergy(sim.live(), cfg);
        max_energy_error = @max(max_energy_error, @abs((e - e0) / e0));
    }

    const early = early_sum / @as(f64, window);
    const late = late_sum / @as(f64, window);

    // No systematic growth: the mean radius at the end matches the beginning.
    try testing.expectApproxEqRel(early, late, 0.01);

    // Total energy oscillates around its initial value rather than climbing —
    // the defining property of a symplectic integrator.
    try testing.expect(max_energy_error < 0.01);

    // No heat without merging, and heat_decay = 1.0 keeps it that way.
    try testing.expectEqual(@as(f64, 0), properties.totalHeat(sim.live()));
}

test "(e) determinism: same seed and config, bitwise-identical state" {
    // RFC Step 8 and §2.6 ordering. This is what makes the scalar and SIMD
    // builds comparable to each other later, not merely benchmarkable.
    const cfg = Config{ .n = 64 };
    var a = try Sim.initSeeded(testing.allocator, cfg);
    defer a.deinit(testing.allocator);
    var b = try Sim.initSeeded(testing.allocator, cfg);
    defer b.deinit(testing.allocator);

    for (0..500) |_| {
        scalar.tick(&a, cfg);
        scalar.tick(&b, cfg);
    }

    try testing.expectEqualSlices(Particle, a.live(), b.live());
    try testing.expect(std.mem.eql(
        u8,
        std.mem.sliceAsBytes(a.live()),
        std.mem.sliceAsBytes(b.live()),
    ));
}

/// Runs one tick as its three separate phases, returning how many merges
/// Phase C performed. Equivalent to `scalar.tick`, but the count is what lets
/// the energy test below tell merge ticks apart from merge-free ones.
fn tickCountingMerges(sim: *Sim, cfg: Config) usize {
    scalar.computeAccelerations(sim, cfg);
    scalar.integrate(sim, cfg);
    return if (cfg.merging) merge.mergeCollisions(sim, cfg) else 0;
}

test "(b) momentum is conserved across merge events" {
    // RFC Step 9's argument uses only the pairwise equal-and-opposite
    // structure, never the form of the force — so it must survive a collision
    // rule, which is just a very strong internal interaction. This is the test
    // an un-weighted average in mergePair fails on the first merge.
    const cfg = Config{ .n = 128, .merging = true, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    const n0 = sim.n;
    for (0..2000) |_| _ = tickCountingMerges(&sim, cfg);

    // Without this the test passes vacuously if nothing ever merged.
    try testing.expect(sim.n < n0);

    const p = properties.momentum(sim.live());
    const scale = properties.momentumScale(sim.live());
    try testing.expect(scale > 0);
    try testing.expect(p.magnitude() / scale < 1e-5);
}

test "(b) total mass is conserved while n falls" {
    // Merging is the only thing that changes n, and Step 10 makes the merged
    // mass additive — so M must not move even as particles disappear.
    const cfg = Config{ .n = 128, .merging = true, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    const n0 = sim.n;
    const m0 = properties.totalMass(sim.live());

    for (0..2000) |_| _ = tickCountingMerges(&sim, cfg);

    try testing.expect(sim.n < n0);
    try testing.expectApproxEqRel(m0, properties.totalMass(sim.live()), 1e-6);
}

test "(c) the centre of mass drifts straight through merge events" {
    // Step 10's position rule is what protects this: placing the product
    // anywhere but the pair's centre of mass makes the system's CoM jump.
    const cfg = Config{ .n = 128, .merging = true, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    const boost_x: f32 = 0.3;
    const boost_y: f32 = -0.2;
    for (sim.live()) |*p| {
        p.vx += boost_x;
        p.vy += boost_y;
    }

    const n0 = sim.n;
    const c0 = properties.centerOfMass(sim.live());
    const v0 = properties.centerOfMassVelocity(sim.live());

    const ticks = 2000;
    var max_deviation: f64 = 0;
    for (1..ticks + 1) |k| {
        _ = tickCountingMerges(&sim, cfg);

        const t = @as(f64, @floatFromInt(k)) * @as(f64, cfg.dt);
        const c = properties.centerOfMass(sim.live());
        const predicted = properties.Vec2{ .x = c0.x + v0.x * t, .y = c0.y + v0.y * t };
        max_deviation = @max(max_deviation, c.sub(predicted).magnitude());
    }

    try testing.expect(sim.n < n0);
    const travelled = v0.magnitude() * @as(f64, ticks) * @as(f64, cfg.dt);
    try testing.expect(max_deviation / travelled < 1e-3);
}

test "(d) merging: the energy books close exactly for a single merge" {
    // The sharp version of test (d)'s merging clause. Two particles and
    // nothing else, so after the merge there are no pairs left and PE is
    // exactly zero — which makes the vanished mutual potential exactly the
    // pre-merge PE, and every term in the ledger computable.
    const cfg = Config{ .heat_decay = 1.0, .merging = true, .merge_radius_scale = 0.05 };
    var sim = try Sim.initCapacity(testing.allocator, 2);
    defer sim.deinit(testing.allocator);
    sim.push(.{ .x = 0, .y = 0, .vx = 1.5, .vy = -0.5, .mass = 3, .heat = 0 });
    sim.push(.{ .x = 0.05, .y = 0, .vx = -2.0, .vy = 1.0, .mass = 1, .heat = 0 });

    const ke0 = properties.kineticEnergy(sim.live());
    const pe0 = properties.potentialEnergy(sim.live(), cfg);
    const p0 = properties.momentum(sim.live());
    const m0 = properties.totalMass(sim.live());

    try testing.expectEqual(@as(usize, 1), merge.mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 1), sim.n);

    const ke1 = properties.kineticEnergy(sim.live());
    const pe1 = properties.potentialEnergy(sim.live(), cfg);
    const heat1 = properties.totalHeat(sim.live());
    const p1 = properties.momentum(sim.live());

    // Exact across the merge: momentum and mass.
    try testing.expectApproxEqRel(p0.x, p1.x, 1e-6);
    try testing.expectApproxEqRel(p0.y, p1.y, 1e-6);
    try testing.expectApproxEqRel(m0, properties.totalMass(sim.live()), 1e-6);

    // A single particle has no pairs, so PE is exactly zero.
    try testing.expectEqual(@as(f64, 0), pe1);

    // The destroyed kinetic energy went into heat, all of it: ΔKE = ½μv_rel².
    const mu = (3.0 * 1.0) / 4.0;
    const rvx = 1.5 - -2.0;
    const rvy = -0.5 - 1.0;
    const expected_dke = 0.5 * mu * (rvx * rvx + rvy * rvy);
    try testing.expectApproxEqRel(expected_dke, ke0 - ke1, 1e-5);
    try testing.expectApproxEqRel(expected_dke, heat1, 1e-5);

    // And the whole ledger: total energy steps up by exactly the mutual
    // potential that vanished with the pair. This is the term RFC test (d)
    // does not account for.
    const e0 = ke0 + pe0;
    const e1 = ke1 + pe1 + heat1;
    try testing.expectApproxEqRel(-pe0, e1 - e0, 1e-5);
    try testing.expect(e1 > e0); // upward, because the vanished term is negative
}

test "(d) merging: energy is flat on every merge-free tick" {
    // At system scale the merge jumps are real, so the claim is made only
    // where it holds: on a tick that merged nothing, this is the same
    // symplectic system as the merging-off test and must behave identically.
    const cfg = Config{ .n = 96, .merging = true, .heat_decay = 1.0 };
    var sim = try Sim.initSeeded(testing.allocator, cfg);
    defer sim.deinit(testing.allocator);

    var quiet_ticks: usize = 0;
    var merge_ticks: usize = 0;
    var max_quiet_step: f64 = 0;

    for (0..2000) |_| {
        const before = properties.totalEnergy(sim.live(), cfg);
        const merges = tickCountingMerges(&sim, cfg);
        const after = properties.totalEnergy(sim.live(), cfg);

        if (merges == 0) {
            quiet_ticks += 1;
            max_quiet_step = @max(max_quiet_step, @abs((after - before) / before));
        } else {
            merge_ticks += 1;
        }
    }

    // Both kinds of tick must actually have occurred for this to mean anything.
    try testing.expect(quiet_ticks > 100);
    try testing.expect(merge_ticks > 0);
    try testing.expect(max_quiet_step < 1e-3);
}

test "(e) determinism holds across merge events" {
    // Greedy merging in a fixed scan order is the whole reason this survives:
    // same seed ⇒ same merges ⇒ same trajectory, bit for bit.
    const cfg = Config{ .n = 96, .merging = true };
    var a = try Sim.initSeeded(testing.allocator, cfg);
    defer a.deinit(testing.allocator);
    var b = try Sim.initSeeded(testing.allocator, cfg);
    defer b.deinit(testing.allocator);

    const n0 = a.n;
    for (0..1000) |_| {
        scalar.tick(&a, cfg);
        scalar.tick(&b, cfg);
    }

    try testing.expect(a.n < n0);
    try testing.expectEqual(a.n, b.n);
    try testing.expectEqualSlices(Particle, a.live(), b.live());
}

test "a lone particle never moves" {
    // The trivial invariant that catches a sign error or a stray self-term:
    // with nothing to pull it, the self-interaction must be exactly zero.
    const cfg = Config{ .heat_decay = 1.0 };
    var sim = try Sim.initCapacity(testing.allocator, 1);
    defer sim.deinit(testing.allocator);
    sim.push(.{ .x = 2, .y = -3, .vx = 0, .vy = 0, .mass = 5, .heat = 0 });

    for (0..1000) |_| scalar.tick(&sim, cfg);

    try testing.expectEqual(@as(f32, 2), sim.live()[0].x);
    try testing.expectEqual(@as(f32, -3), sim.live()[0].y);
}

test "a merging config is valid" {
    try testing.expect(Config.validate(.{ .merging = true }) == null);
}

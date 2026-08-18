//! Initial conditions: the spinning disk (RFC §2.7).
//!
//! Seeding answers three per-particle questions — where, how heavy, how fast —
//! and one global one: why doesn't the cloud fall into its own centre? The
//! answer is tangential velocity at the circular-orbit speed, which is why the
//! total mass has to be known before any velocity can be assigned (hence two
//! passes below).
//!
//! **Determinism contract.** Every random value comes from `cfg.seed` via a
//! single PRNG, drawn in the fixed order documented at each call. Same seed +
//! same config ⇒ same run, bitwise (RFC §2.5 test (e)). Reordering or adding a
//! draw changes every run that follows it, so treat the draw order as part of
//! the observable behaviour.

const std = @import("std");

const Config = @import("config.zig").Config;
const sim_mod = @import("sim.zig");
const Particle = sim_mod.Particle;
const Sim = sim_mod.Sim;

/// Fills `sim` with `cfg.n` particles according to `cfg.preset`.
///
/// The sim must be empty and have capacity for `cfg.n`.
pub fn populate(sim: *Sim, cfg: Config) void {
    std.debug.assert(sim.n == 0);
    std.debug.assert(sim.capacity() >= cfg.n);
    if (cfg.n == 0) return;

    var prng = std.Random.DefaultPrng.init(cfg.seed);
    const r = prng.random();

    // ---- Pass 1: positions and masses (velocities need the total mass) ----

    // `keplerian` puts one heavy body at the origin, at rest, at index 0. Its
    // mass is a multiple of the *nominal* mean particle mass rather than a
    // sample statistic, so it does not depend on how many particles follow.
    const central_mass: f32 = if (cfg.preset == .keplerian)
        cfg.central_mass_factor * 0.5 * (cfg.mass_min + cfg.mass_max)
    else
        0.0;

    var first_disk_index: usize = 0;
    if (cfg.preset == .keplerian) {
        sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = central_mass, .heat = 0 });
        first_disk_index = 1;
    }

    while (sim.n < cfg.n) {
        // Draw order, fixed: theta, then radius, then mass.
        const theta = r.float(f32) * 2.0 * std.math.pi;
        // Radius must NOT be uniform in [0, R]: a ring at radius rr has area
        // ∝ rr, so uniform-in-rr over-crowds the centre. Uniform *area*
        // density needs rr = R·√u (RFC §2.7).
        const rr = cfg.radius * @sqrt(r.float(f32));
        const mass = cfg.mass_min + r.float(f32) * (cfg.mass_max - cfg.mass_min);

        sim.push(.{
            .x = rr * @cos(theta),
            .y = rr * @sin(theta),
            .vx = 0,
            .vy = 0,
            .mass = mass,
            .heat = 0,
        });
    }

    // ---- Pass 2: circular velocities ----

    const ps = sim.live();

    // Total mass of the disk population, excluding any central body.
    var disk_mass: f64 = 0;
    for (ps[first_disk_index..]) |p| disk_mass += p.mass;

    const inv_r2 = 1.0 / (@as(f64, cfg.radius) * @as(f64, cfg.radius));
    for (ps[first_disk_index..]) |*p| {
        const rr = @sqrt(@as(f64, p.x) * @as(f64, p.x) + @as(f64, p.y) * @as(f64, p.y));

        // Draw jitter unconditionally so the draw sequence does not depend on
        // where a particle happened to land (the rr == 0 branch below).
        const jx = 2.0 * r.float(f32) - 1.0;
        const jy = 2.0 * r.float(f32) - 1.0;

        if (rr == 0) continue; // a particle exactly at the origin has no tangent

        // Enclosed mass: the part that effectively pulls inward. For a
        // uniform disk M_enc(r) = M·(r/R)², giving v_circ ∝ √r. The RFC's
        // keplerian sketch approximates M_enc ≈ M_central; keeping the disk
        // term as well is strictly more accurate, still 1/√r-dominated, and
        // costs nothing here.
        const m_enc = @as(f64, central_mass) + disk_mass * (rr * rr * inv_r2);
        const v_circ: f32 = @floatCast(@sqrt(@as(f64, cfg.g) * m_enc / rr));

        // Tangential direction: rotate the outward unit vector (x/r, y/r) a
        // quarter turn counterclockwise, (a, b) ↦ (−b, a).
        const ux: f32 = @floatCast(@as(f64, p.x) / rr);
        const uy: f32 = @floatCast(@as(f64, p.y) / rr);

        // A perfectly cold disk is glassy; jitter seeds the encounters that
        // will drive merging once Phase C lands.
        p.vx = -v_circ * uy + cfg.jitter * v_circ * jx;
        p.vy = v_circ * ux + cfg.jitter * v_circ * jy;
    }

    zeroNetMomentum(ps);
}

/// Subtracts the mass-weighted mean velocity from every particle, so total
/// momentum starts at exactly zero (RFC §2.7).
///
/// Without this, jitter leaves a small net drift and the whole disk slides off
/// screen over a long run. It also makes RFC §2.5 test (b)'s target a clean
/// zero rather than "whatever it started as". Accumulated in f64: this runs
/// once at startup and is not the measured kernel.
fn zeroNetMomentum(ps: []Particle) void {
    var px: f64 = 0;
    var py: f64 = 0;
    var m_total: f64 = 0;
    for (ps) |p| {
        px += @as(f64, p.mass) * @as(f64, p.vx);
        py += @as(f64, p.mass) * @as(f64, p.vy);
        m_total += p.mass;
    }
    if (m_total == 0) return;

    const mean_vx: f32 = @floatCast(px / m_total);
    const mean_vy: f32 = @floatCast(py / m_total);
    for (ps) |*p| {
        p.vx -= mean_vx;
        p.vy -= mean_vy;
    }
}

// ---- tests ----

const testing = std.testing;

fn seeded(cfg: Config) !Sim {
    return Sim.initSeeded(testing.allocator, cfg);
}

test "every particle lands inside the disk radius" {
    const cfg = Config{ .n = 500 };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(cfg.n, sim.n);
    for (sim.live()) |p| {
        const rr = @sqrt(p.x * p.x + p.y * p.y);
        try testing.expect(rr <= cfg.radius * (1.0 + 1e-6));
    }
}

test "masses stay inside the configured band" {
    const cfg = Config{ .n = 500 };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    for (sim.live()) |p| {
        try testing.expect(p.mass >= cfg.mass_min);
        try testing.expect(p.mass <= cfg.mass_max);
    }
}

test "radial distribution is uniform in area, not in radius" {
    // The bug this catches: drawing rr uniformly in [0, R] would put ~50% of
    // the particles inside R/2 instead of the correct 25% (area ratio).
    const cfg = Config{ .n = 20_000 };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    var inside: usize = 0;
    for (sim.live()) |p| {
        if (p.x * p.x + p.y * p.y < 0.25 * cfg.radius * cfg.radius) inside += 1;
    }

    const fraction = @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(cfg.n));
    try testing.expectApproxEqAbs(@as(f64, 0.25), fraction, 0.02);
}

test "net momentum is zero after seeding" {
    const cfg = Config{ .n = 1000 };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    var px: f64 = 0;
    var py: f64 = 0;
    var scale: f64 = 0; // Σ m|v|, the natural yardstick for "small"
    for (sim.live()) |p| {
        px += @as(f64, p.mass) * @as(f64, p.vx);
        py += @as(f64, p.mass) * @as(f64, p.vy);
        scale += @as(f64, p.mass) * @sqrt(@as(f64, p.vx) * p.vx + @as(f64, p.vy) * p.vy);
    }

    try testing.expect(scale > 0); // the disk is actually moving
    try testing.expect(@abs(px) / scale < 1e-6);
    try testing.expect(@abs(py) / scale < 1e-6);
}

test "the disk rotates coherently: velocity is tangential, one sense" {
    // Stated as angular momentum rather than per-particle radial velocity,
    // because momentum zeroing subtracts one constant velocity from everyone
    // and so leaves each particle a small radial component (the sample mean of
    // n random tangential velocities is O(v/√n), not 0). The coherence ratio
    // below survives that — it is ~1 for a rotating disk, ~0 for random
    // directions, and negative if the quarter-turn (a,b) ↦ (−b,a) got flipped.
    const cfg = Config{ .n = 2000, .jitter = 0.0 };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    var angular: f64 = 0; // Σ m(x·vy − y·vx)
    var scale: f64 = 0; // Σ m·r·|v|, the value angular takes if all motion is tangential
    for (sim.live()) |p| {
        angular += @as(f64, p.mass) * (@as(f64, p.x) * p.vy - @as(f64, p.y) * p.vx);
        const rr = @sqrt(@as(f64, p.x) * p.x + @as(f64, p.y) * p.y);
        const speed = @sqrt(@as(f64, p.vx) * p.vx + @as(f64, p.vy) * p.vy);
        scale += @as(f64, p.mass) * rr * speed;
    }

    try testing.expect(scale > 0);
    try testing.expect(angular / scale > 0.99); // counterclockwise, near-perfectly tangential
}

test "keplerian puts the heavy body at index 0, at rest at the origin" {
    const cfg = Config{ .n = 100, .preset = .keplerian };
    var sim = try seeded(cfg);
    defer sim.deinit(testing.allocator);

    const central = sim.live()[0];
    try testing.expectEqual(@as(f32, 0), central.x);
    try testing.expectEqual(@as(f32, 0), central.y);
    try testing.expect(central.mass > cfg.mass_max * 10);

    // Momentum zeroing nudges even the central body, by P/M. Judged against
    // a typical disk speed rather than an absolute epsilon: the claim is that
    // the heavy body barely moves compared to what orbits it.
    var disk_speed: f64 = 0;
    for (sim.live()[1..]) |p| {
        disk_speed += @sqrt(@as(f64, p.vx) * p.vx + @as(f64, p.vy) * p.vy);
    }
    disk_speed /= @floatFromInt(sim.n - 1);

    const central_speed = @sqrt(@as(f64, central.vx) * central.vx + @as(f64, central.vy) * central.vy);
    try testing.expect(central_speed < 0.2 * disk_speed);

    for (sim.live()[1..]) |p| try testing.expect(p.mass <= cfg.mass_max);
}

test "seeding is a pure function of the seed" {
    const cfg = Config{ .n = 200 };
    var a = try seeded(cfg);
    defer a.deinit(testing.allocator);
    var b = try seeded(cfg);
    defer b.deinit(testing.allocator);
    var c = try seeded(Config{ .n = 200, .seed = cfg.seed + 1 });
    defer c.deinit(testing.allocator);

    try testing.expectEqualSlices(Particle, a.live(), b.live());
    // Bytes, not std.mem.eql: "same seed ⇒ same run" is a bitwise claim.
    try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(a.live()), std.mem.sliceAsBytes(c.live())));
}

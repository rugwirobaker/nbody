//! Whole-system properties (RFC §2.5).
//!
//! Correctness here is defined by invariants, not by eyeballing trajectories:
//! momentum constant, centre of mass drifting in a straight line, orbits that
//! do not systematically grow. These functions are what the acceptance tests
//! measure.
//!
//! **Everything accumulates in f64.** None of this runs in the measured
//! kernel, so the wider accumulator costs nothing that matters and keeps a
//! test measuring the algorithm's conservation instead of its own summation
//! error. A momentum sum over 10⁴ f32 particles loses more precision to the
//! sum than the simulation loses to a tick.

const std = @import("std");

const Config = @import("config.zig").Config;
const Particle = @import("sim.zig").Particle;

/// A 2D quantity in f64 — the return type for the vector-valued measurements
/// below (momentum, centre of mass, centre-of-mass velocity).
///
/// **Not a simulation type, and deliberately not reusable as one.** The
/// simulation never stores a vector: `Particle` holds bare `x`/`y`/`vx`/`vy`
/// f32 fields, and Part 3's SoA layout splits them into separate arrays
/// entirely (RFC §3.2) — a `Vec2` field anywhere in either layout would fight
/// the memory layout the whole project is about. This type exists only so a
/// diagnostic can hand back both components at once, and it is f64 for the
/// same reason everything here accumulates in f64.
pub const Vec2 = struct {
    x: f64,
    y: f64,

    pub fn magnitude(v: Vec2) f64 {
        return @sqrt(v.x * v.x + v.y * v.y);
    }

    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }
};

/// M = Σ mᵢ over the live particles.
///
/// The denominator of `centerOfMass` and `centerOfMassVelocity` — which is
/// most of why it exists — but also an invariant in its own right: mass is
/// conserved, merges included, because RFC Step 10 makes the merged mass
/// additive (`m_new = m_i + m_j`). That makes it the cheapest possible check
/// that Phase C neither lost nor duplicated a particle once merging lands: `n`
/// falls, M must not move.
pub fn totalMass(ps: []const Particle) f64 {
    var m: f64 = 0;
    for (ps) |p| m += p.mass;
    return m;
}

/// Total momentum P = Σ mᵢvᵢ (RFC Step 9). Conserved exactly, to rounding,
/// by the specified algorithm — including across merges.
pub fn momentum(ps: []const Particle) Vec2 {
    var px: f64 = 0;
    var py: f64 = 0;
    for (ps) |p| {
        px += @as(f64, p.mass) * @as(f64, p.vx);
        py += @as(f64, p.mass) * @as(f64, p.vy);
    }
    return .{ .x = px, .y = py };
}

/// Σ mᵢ|vᵢ| — the natural scale against which a momentum residual is "small".
/// A bare |P| threshold would silently pass a frozen simulation.
pub fn momentumScale(ps: []const Particle) f64 {
    var s: f64 = 0;
    for (ps) |p| {
        s += @as(f64, p.mass) * @sqrt(@as(f64, p.vx) * p.vx + @as(f64, p.vy) * p.vy);
    }
    return s;
}

/// Centre of mass X = (1/M) Σ mᵢxᵢ. Drifts in a straight line at constant
/// speed forever, whatever the particles do individually (RFC Step 9 corollary).
pub fn centerOfMass(ps: []const Particle) Vec2 {
    var cx: f64 = 0;
    var cy: f64 = 0;
    var m: f64 = 0;
    for (ps) |p| {
        cx += @as(f64, p.mass) * @as(f64, p.x);
        cy += @as(f64, p.mass) * @as(f64, p.y);
        m += p.mass;
    }
    if (m == 0) return .{ .x = 0, .y = 0 };
    return .{ .x = cx / m, .y = cy / m };
}

/// Centre-of-mass velocity, P/M — the constant speed of that straight line.
pub fn centerOfMassVelocity(ps: []const Particle) Vec2 {
    const m = totalMass(ps);
    if (m == 0) return .{ .x = 0, .y = 0 };
    const p = momentum(ps);
    return .{ .x = p.x / m, .y = p.y / m };
}

/// KE = Σ ½mv² (RFC Step 10). Scalar, direction-blind, and *not* conserved
/// by merging.
pub fn kineticEnergy(ps: []const Particle) f64 {
    var ke: f64 = 0;
    for (ps) |p| {
        const v2 = @as(f64, p.vx) * @as(f64, p.vx) + @as(f64, p.vy) * @as(f64, p.vy);
        ke += 0.5 * @as(f64, p.mass) * v2;
    }
    return ke;
}

/// PE = −G Σ_{i<j} mᵢmⱼ / √(d² + ε²).
///
/// This is the **softened (Plummer) potential**, i.e. the exact potential
/// whose gradient is the RFC's softened force law. Using the textbook
/// −Gmᵢmⱼ/d here instead would make the energy test drift for a reason that
/// has nothing to do with the integrator — the diagnostic must agree with the
/// force law actually being integrated.
pub fn potentialEnergy(ps: []const Particle, cfg: Config) f64 {
    var pe: f64 = 0;
    for (ps, 0..) |pi, i| {
        for (ps[i + 1 ..]) |pj| {
            const dx = @as(f64, pj.x) - @as(f64, pi.x);
            const dy = @as(f64, pj.y) - @as(f64, pi.y);
            const d = @sqrt(dx * dx + dy * dy + @as(f64, cfg.eps2));
            pe -= @as(f64, cfg.g) * @as(f64, pi.mass) * @as(f64, pj.mass) / d;
        }
    }
    return pe;
}

/// Σ heat — the kinetic energy destroyed by merges (RFC Step 10).
/// Zero until Phase C lands, and zero forever with `heat_decay` at 1.0 and
/// merging off.
pub fn totalHeat(ps: []const Particle) f64 {
    var h: f64 = 0;
    for (ps) |p| h += p.heat;
    return h;
}

/// KE + PE + Σheat — the quantity that stays constant up to integration error
/// (RFC §2.5 test (d)).
pub fn totalEnergy(ps: []const Particle, cfg: Config) f64 {
    return kineticEnergy(ps) + potentialEnergy(ps, cfg) + totalHeat(ps);
}

// ---- tests ----

const testing = std.testing;

test "momentum of a symmetric pair cancels" {
    const ps = [_]Particle{
        .{ .x = 0, .y = 0, .vx = 2, .vy = -3, .mass = 1, .heat = 0 },
        .{ .x = 1, .y = 0, .vx = -1, .vy = 1.5, .mass = 2, .heat = 0 },
    };
    const p = momentum(&ps);
    try testing.expectApproxEqAbs(@as(f64, 0), p.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), p.y, 1e-12);
}

test "centre of mass is mass-weighted, not the plain average" {
    const ps = [_]Particle{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 3, .heat = 0 },
        .{ .x = 4, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    };
    const c = centerOfMass(&ps);
    try testing.expectApproxEqAbs(@as(f64, 1.0), c.x, 1e-12); // not 2.0
    try testing.expectApproxEqAbs(@as(f64, 0.0), c.y, 1e-12);
}

test "kinetic energy is ½mv²" {
    const ps = [_]Particle{
        .{ .x = 0, .y = 0, .vx = 3, .vy = 4, .mass = 2, .heat = 0 },
    };
    try testing.expectApproxEqRel(@as(f64, 25.0), kineticEnergy(&ps), 1e-12);
}

test "potential energy is negative and softened" {
    const cfg = Config{};
    const ps = [_]Particle{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 1, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    };
    const pe = potentialEnergy(&ps, cfg);
    const expected = -@as(f64, cfg.g) / @sqrt(1.0 + @as(f64, cfg.eps2));
    try testing.expectApproxEqRel(expected, pe, 1e-12);
    try testing.expect(pe < 0);
}

test "coincident particles have finite potential energy" {
    // Without softening this would be −inf. The diagnostic has to agree with
    // the bounded force law the kernel integrates.
    const ps = [_]Particle{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    };
    try testing.expect(std.math.isFinite(potentialEnergy(&ps, .{})));
}

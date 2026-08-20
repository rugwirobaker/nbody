//! Phase C: merging, the perfectly inelastic collision (RFC §2.6, Step 10).
//!
//! Every field of a merged particle is **forced** by a conservation law, not
//! chosen: mass is additive, and position, velocity and heat are all the same
//! mass-weighted-average shape. Un-weighted averages fail the momentum test on
//! the first merge, which is what that test is for.
//!
//! The baseline-honesty rules in `scalar.zig` do not apply here. This phase is
//! demo-only and never benchmarked (RFC §2.5 rule 3 runs every measurement at
//! `merging = false`), so it is written for clarity. RFC §3.3 keeps Phase C
//! scalar even in the SIMD build; see `mergePair` for the one thing the SoA
//! layout has to do differently.
//!
//! **On energy.** Step 10 banks the destroyed kinetic energy into `heat`,
//! exactly. It leaves out the merged pair's mutual potential, which vanishes
//! from the system along with the pair, so RFC §2.5 test (d)'s
//! `KE + PE + Σheat` steps upward slightly at each merge. The spec's wording
//! overstates the invariant. The tests assert the exact claims exactly and
//! make no approximate one.

const std = @import("std");

const Config = @import("config.zig").Config;
const sim_mod = @import("sim.zig");
const Particle = sim_mod.Particle;
const Sim = sim_mod.Sim;

/// Merges every pair closer than `d_merge`, greedily, until none remain.
/// Returns the number of merges performed.
///
/// Greedy with restart (RFC §2.6): merge the first pair found in fixed scan
/// order, then rescan from scratch, because a merge invalidates the indices
/// behind it and may create a product that merges again. Each merge strictly
/// decreases `n`, so this terminates; the fixed order is what makes it
/// deterministic, and determinism is what makes runs reproducible from a seed.
///
/// Cost is O(n²) per scan, so a tick with k merges costs O(n²·k). Accepted
/// deliberately — §2.6's "correctness beats cleverness" — because merging is
/// off for every benchmark and chains are rare at sane `dt`.
pub fn mergeCollisions(sim: *Sim, cfg: Config) usize {
    var merges: usize = 0;

    var restart = true;
    while (restart) {
        restart = false;

        // Radii once per pass, not once per pair (RFC-002 §5.1). There are
        // thousands of particles and millions of pairs, which is the whole of
        // why this is hoisted: computing `radius` inside the scan costs 2.08×
        // the fixed-threshold scan, reading it from here costs 1.28×.
        for (sim.particles[0..sim.n], sim.radii[0..sim.n]) |p, *r| r.* = p.radius(cfg);

        scan: for (0..sim.n) |i| {
            for (i + 1..sim.n) |j| {
                const dx = sim.particles[j].x - sim.particles[i].x;
                const dy = sim.particles[j].y - sim.particles[i].y;
                // Squared compare, so no sqrt. No `eps2` either: this is a
                // proximity test between two points, not a force evaluation.
                const d2 = dx * dx + dy * dy;
                // Discs touching (RFC-002 §1.2), so what the renderer draws is
                // what merges.
                const contact = sim.radii[i] + sim.radii[j];
                if (d2 < contact * contact) {
                    mergePair(sim, i, j);
                    merges += 1;
                    restart = true;
                    break :scan;
                }
            }
        }
    }

    return merges;
}

/// Merges `j` into `i` by RFC Step 10's formulas, then swap-removes `j`.
///
/// `i` survives and keeps its slot; `j`'s slot is filled by the last live
/// particle. Swap-remove is O(1) and legal because nothing in the algorithm
/// depends on particle order.
///
/// **Note for the SoA build (RFC §3.2).** The scalar build never reads past
/// `n`, so stale data in the vacated slot is harmless here. The SIMD build
/// reads to `n_padded` by design, so its version of this function has to
/// re-zero the vacated slot's mass. Skipping that leaves a ghost particle:
/// invisible to the renderer, unmergeable, and still pulling on everything.
pub fn mergePair(sim: *Sim, i: usize, j: usize) void {
    std.debug.assert(i < j);
    std.debug.assert(j < sim.n);

    const a = sim.particles[i];
    const b = sim.particles[j];
    const m = a.mass + b.mass;

    // ΔKE = ½·μ·v_rel², computed BEFORE the overwrite below because it needs
    // both original velocities. Moving it down silently reads a merged one.
    const rvx = a.vx - b.vx;
    const rvy = a.vy - b.vy;
    const mu = (a.mass * b.mass) / m; // reduced mass
    const dke = 0.5 * mu * (rvx * rvx + rvy * rvy);

    sim.particles[i] = .{
        .x = (a.mass * a.x + b.mass * b.x) / m, // centre of mass  (Step 10)
        .y = (a.mass * a.y + b.mass * b.y) / m,
        .vx = (a.mass * a.vx + b.mass * b.vx) / m, // momentum     (Step 9)
        .vy = (a.mass * a.vy + b.mass * b.vy) / m,
        .mass = m, // additive
        .heat = a.heat + b.heat + dke, // energy bookkeeping (Step 10)
    };

    // Swap-remove j. When j was the last live particle this is a harmless
    // self-assignment.
    sim.n -= 1;
    sim.particles[j] = sim.particles[sim.n];
}

// ---- tests ----

const testing = std.testing;

fn simOf(ps: []const Particle) !Sim {
    var sim = try Sim.initCapacity(testing.allocator, ps.len);
    for (ps) |p| sim.push(p);
    return sim;
}

test "Step 10's worked example: equal masses, head-on, merge to rest" {
    // m and m closing at ±v: momentum forces v_new = 0, and the entire kinetic
    // budget is destroyed — ΔKE = ½·(m/2)·(2v)² = m·v².
    const m: f32 = 2.0;
    const v: f32 = 3.0;
    var sim = try simOf(&.{
        .{ .x = -1, .y = 0, .vx = v, .vy = 0, .mass = m, .heat = 0 },
        .{ .x = 1, .y = 0, .vx = -v, .vy = 0, .mass = m, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    mergePair(&sim, 0, 1);

    const p = sim.live()[0];
    try testing.expectEqual(@as(usize, 1), sim.n);
    try testing.expectEqual(@as(f32, 2 * m), p.mass);
    try testing.expectEqual(@as(f32, 0), p.vx); // forced by momentum
    try testing.expectEqual(@as(f32, 0), p.vy);
    try testing.expectEqual(@as(f32, 0), p.x); // midpoint of equal masses
    try testing.expectApproxEqRel(m * v * v, p.heat, 1e-6);
}

test "position and velocity are mass-weighted, not plain averages" {
    // 3:1 mass ratio. A plain average would put the product at x = 2 and give
    // it vx = 2; the mass-weighted answer is x = 1, vx = 1.
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 3, .heat = 0 },
        .{ .x = 4, .y = 0, .vx = 4, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    mergePair(&sim, 0, 1);

    const p = sim.live()[0];
    try testing.expectEqual(@as(f32, 4), p.mass);
    try testing.expectApproxEqRel(@as(f32, 1.0), p.x, 1e-6);
    try testing.expectApproxEqRel(@as(f32, 1.0), p.vx, 1e-6);
}

test "merging conserves momentum exactly" {
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 1.5, .vy = -2.5, .mass = 3, .heat = 0 },
        .{ .x = 1, .y = 2, .vx = -0.5, .vy = 4.0, .mass = 7, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    const px_before = 3 * 1.5 + 7 * -0.5;
    const py_before = 3 * -2.5 + 7 * 4.0;

    mergePair(&sim, 0, 1);

    const p = sim.live()[0];
    try testing.expectApproxEqRel(px_before, p.mass * p.vx, 1e-6);
    try testing.expectApproxEqRel(py_before, p.mass * p.vy, 1e-6);
}

test "dke is computed before the overwrite, not after" {
    // Two particles with distinct velocities: if dke were computed after
    // sim.particles[i] was overwritten, the relative velocity would be taken
    // against the merged velocity and come out wrong (here, far too small).
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 10, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    mergePair(&sim, 0, 1);

    // μ = 0.5, v_rel = 10 ⇒ ΔKE = ½·0.5·100 = 25.
    try testing.expectApproxEqRel(@as(f32, 25.0), sim.live()[0].heat, 1e-6);
}

test "heat pools both bodies' budgets plus the splat" {
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 4 },
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 6 },
    });
    defer sim.deinit(testing.allocator);

    // Lockstep particles merge for free: v_rel = 0 ⇒ ΔKE = 0.
    mergePair(&sim, 0, 1);
    try testing.expectEqual(@as(f32, 10), sim.live()[0].heat);
}

test "swap-remove moves the last live particle into the vacated slot" {
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 99, .y = 99, .vx = 0, .vy = 0, .mass = 5, .heat = 0 }, // the survivor
    });
    defer sim.deinit(testing.allocator);

    mergePair(&sim, 0, 1);

    try testing.expectEqual(@as(usize, 2), sim.n);
    try testing.expectEqual(@as(f32, 99), sim.live()[1].x); // moved into slot 1
    try testing.expectEqual(@as(f32, 5), sim.live()[1].mass);
}

test "swap-remove handles j being the last live particle" {
    // Self-assignment case: particles[j] = particles[n] where j == n.
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 2, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    mergePair(&sim, 0, 1);

    try testing.expectEqual(@as(usize, 1), sim.n);
    try testing.expectEqual(@as(f32, 2), sim.live()[0].mass);
    try testing.expectApproxEqRel(@as(f32, 1.0), sim.live()[0].x, 1e-6);
}

test "mergeCollisions leaves pairs outside the threshold alone" {
    const cfg = Config{ .merging = true, .merge_radius_scale = 0.05 }; // two mass-1 bodies touch at 0.1
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.2, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 2), sim.n);
}

test "mergeCollisions merges pairs inside the threshold" {
    const cfg = Config{ .merging = true, .merge_radius_scale = 0.05 };
    var sim = try simOf(&.{
        .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.05, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 1), sim.n);
}

test "unequal masses merge when their discs touch, and not before" {
    // The property RFC-002 exists to establish, and the one a fixed threshold
    // cannot express: the trigger distance depends on both masses.
    //
    // Radii are k·√m, so mass 4 gives 2k and mass 1 gives k. They touch at 3k.
    // A fixed threshold would fire at the same distance for these two as for a
    // pair of specks.
    const k: f32 = 0.05;
    const cfg = Config{ .merging = true, .merge_radius_scale = k };
    const contact = k * (@sqrt(@as(f32, 4)) + 1.0); // 0.15

    inline for (.{ .{ contact * 1.02, 0, 2 }, .{ contact * 0.98, 1, 1 } }) |case| {
        var sim = try simOf(&.{
            .{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 4, .heat = 0 },
            .{ .x = case[0], .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        });
        defer sim.deinit(testing.allocator);

        try testing.expectEqual(@as(usize, case[1]), mergeCollisions(&sim, cfg));
        try testing.expectEqual(@as(usize, case[2]), sim.n);
    }
}

test "the same separation merges one pair and not another, by mass alone" {
    // Two pairs the same distance apart. The heavy pair's discs overlap at that
    // separation and the light pair's do not, so one merges and one does not —
    // from mass, with no threshold in sight.
    const cfg = Config{ .merging = true, .merge_radius_scale = 0.05 };
    var sim = try simOf(&.{
        .{ .x = 0.00, .y = 0, .vx = 0, .vy = 0, .mass = 9, .heat = 0 },
        .{ .x = 0.18, .y = 0, .vx = 0, .vy = 0, .mass = 9, .heat = 0 }, // touch at 0.30
        .{ .x = 10.00, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 10.18, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 }, // touch at 0.10
    });
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 3), sim.n);
    // The survivor of the heavy pair carries both their masses.
    try testing.expectEqual(@as(f32, 18), sim.live()[0].mass);
}

test "chains collapse in one call (greedy with restart)" {
    // a–b and b–c both cross the threshold; a–c does not. Greedy merges the
    // first pair found, then the product merges again on the rescan.
    //
    // The positions need care, and the reason is worth knowing: the product
    // lands at the *centre of mass* of the pair, which can be farther from a
    // third particle than either original was. At x = 0, 0.08, 0.17 the a–b
    // product sits at 0.04, putting it 0.13 from c — out of reach even after
    // the product's radius grows. Chaining is not transitive; the test below
    // asserts that case directly.
    const cfg = Config{ .merging = true, .merge_radius_scale = 0.05 }; // two mass-1 bodies touch at 0.1
    var sim = try simOf(&.{
        .{ .x = 0.00, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.08, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.12, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    // a–b merge to mass 2 at x = 0.04; that product is 0.08 from c, so the
    // rescan merges again.
    try testing.expectEqual(@as(usize, 2), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 1), sim.n);
    try testing.expectEqual(@as(f32, 3), sim.live()[0].mass);
    try testing.expectApproxEqRel(@as(f32, 0.0666667), sim.live()[0].x, 1e-5);
}

test "a merged product can land out of reach of a neighbour" {
    // The other half of the lesson above, asserted directly: b is inside the
    // threshold of both a and c, but merging a–b moves the product away from
    // c and the chain stops at one merge.
    //
    // Under RFC-002 the product also *grows*: mass 2 gives it radius
    // 0.05·√2 = 0.0707, so it reaches 0.1207 towards c rather than 0.1. c has
    // to sit past that to be out of reach, which is why it is at 0.17 and not
    // the 0.16 a fixed threshold would have needed.
    const cfg = Config{ .merging = true, .merge_radius_scale = 0.05 }; // two mass-1 bodies touch at 0.1
    var sim = try simOf(&.{
        .{ .x = 0.00, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.08, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
        .{ .x = 0.17, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 },
    });
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 2), sim.n);
}

test "mergeCollisions is a no-op on an empty or single-particle sim" {
    const cfg = Config{ .merging = true };
    var sim = try Sim.initCapacity(testing.allocator, 4);
    defer sim.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), mergeCollisions(&sim, cfg));

    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    try testing.expectEqual(@as(usize, 0), mergeCollisions(&sim, cfg));
    try testing.expectEqual(@as(usize, 1), sim.n);
}

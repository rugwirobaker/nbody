//! The scalar baseline kernel (RFC §2.3) — the experimental control.
//!
//! This file is *deliberately naive*. Part 3's claim ("same algorithm,
//! different memory layout and instruction width, ~L× faster") is only
//! measurable against the honest implementation a reasonable engineer writes
//! first, so the rules below outrank tidiness, cleverness, and performance:
//!
//!   - **No fast math.** No `@setFloatMode(.optimized)`, no hand-reordering of
//!     the float reductions. Strict FP semantics are what stop LLVM from
//!     auto-vectorizing this loop; a secretly-vectorized "scalar" baseline
//!     makes the whole comparison a lie.
//!   - **Ordered pairs, full n².** RFC Step 6's pairwise-symmetry halving is a
//!     real optimization and is explicitly rejected here for comparison
//!     fairness (RFC Step 6, "Shared work").
//!   - **No `if (i != j)`.** Softening makes the self term exactly zero; the
//!     branch's absence is intentional (RFC "Softening", benefit 1).
//!   - **Phases never fuse.** See `computeAccelerations`.

const std = @import("std");

const Config = @import("config.zig").Config;
const merge = @import("merge.zig");
const Sim = @import("sim.zig").Sim;

/// Computes the net acceleration on every live particle — the RFC's **Phase A**
/// (Steps 7–8).
///
/// Computes, deliberately: it reads positions and masses and writes only
/// `sim.ax` / `sim.ay`, leaving every particle untouched. Every acceleration in
/// a tick comes from one frozen snapshot of positions, which is what makes
/// momentum conservation exact and runs reproducible under reordering
/// (RFC Step 8). Applying them is `integrate`'s job, one phase later.
pub fn computeAccelerations(sim: *Sim, cfg: Config) void {
    const ps = sim.live();

    for (ps, 0..) |pi, i| {
        var ax: f32 = 0.0;
        var ay: f32 = 0.0;
        for (ps) |pj| { // j == i term is exactly zero via softening
            const dx = pj.x - pi.x; // source minus self (RFC Part 1 convention)
            const dy = pj.y - pi.y;
            const d2 = dx * dx + dy * dy + cfg.eps2;
            const inv_d3 = 1.0 / (d2 * @sqrt(d2)); // @sqrt: the instruction, not pow()
            const s = cfg.g * pj.mass * inv_d3; // only the *source's* mass (Step 2)
            ax += s * dx;
            ay += s * dy;
        }
        sim.ax[i] = ax;
        sim.ay[i] = ay;
    }
}

/// Advances velocity and position by one `dt` — the RFC's **Phase B**
/// (Steps 4–5), semi-implicit Euler.
///
/// Velocity first, then position using the **new** velocity. The reversed
/// order costs exactly the same and systematically pumps energy in — orbits
/// spiral outward, clusters unbind (RFC Step 5, normative).
pub fn integrate(sim: *Sim, cfg: Config) void {
    for (sim.live(), 0..) |*p, i| {
        p.vx += sim.ax[i] * cfg.dt;
        p.vy += sim.ay[i] * cfg.dt;
        p.x += p.vx * cfg.dt;
        p.y += p.vy * cfg.dt;
        p.heat *= cfg.heat_decay; // presentation, not physics
    }
}

/// One tick: advance simulated time by `cfg.dt` (RFC Step 8).
pub fn tick(sim: *Sim, cfg: Config) void {
    computeAccelerations(sim, cfg);
    integrate(sim, cfg);

    // Phase C is a distinct pass *after* integration, never inside Phase A —
    // accelerations must be computed over a stable particle set (RFC §2.6).
    // Callers who need the merge count call `mergeCollisions` directly.
    if (cfg.merging) _ = merge.mergeCollisions(sim, cfg);
}

test "computeAccelerations leaves particle state untouched (frozen snapshot, Step 8)" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 3);
    defer sim.deinit(testing.allocator);

    sim.push(.{ .x = 0, .y = 0, .vx = 0.5, .vy = -0.5, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 1, .y = 0, .vx = 0, .vy = 0, .mass = 2, .heat = 0 });
    sim.push(.{ .x = 0, .y = 1, .vx = 0, .vy = 0, .mass = 3, .heat = 0 });

    const before = try testing.allocator.dupe(@TypeOf(sim.particles[0]), sim.live());
    defer testing.allocator.free(before);

    computeAccelerations(&sim, .{});

    try testing.expectEqualSlices(@TypeOf(sim.particles[0]), before, sim.live());
}

test "self-interaction contributes exactly zero (no i != j branch)" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 1);
    defer sim.deinit(testing.allocator);

    // A lone particle feels only itself. Softening makes that term 0/ε³ = 0.
    sim.push(.{ .x = 3, .y = -7, .vx = 0, .vy = 0, .mass = 1000, .heat = 0 });
    computeAccelerations(&sim, .{});

    try testing.expectEqual(@as(f32, 0), sim.ax[0]);
    try testing.expectEqual(@as(f32, 0), sim.ay[0]);
}

test "acceleration points from the accelerated particle toward the source" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 2);
    defer sim.deinit(testing.allocator);

    // Source sits to the +x side of the target: attraction must be +x.
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 1, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    computeAccelerations(&sim, .{});

    try testing.expect(sim.ax[0] > 0);
    try testing.expect(sim.ax[1] < 0);
    try testing.expectEqual(@as(f32, 0), sim.ay[0]);
    try testing.expectEqual(@as(f32, 0), sim.ay[1]);

    // Equal masses ⇒ equal and opposite (Step 6).
    try testing.expectApproxEqAbs(sim.ax[0], -sim.ax[1], 1e-12);
}

test "acceleration carries the source's mass, not the target's (Step 2)" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 2);
    defer sim.deinit(testing.allocator);

    // m=1 pulled by m=1000 accelerates 1000× harder than the reverse.
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 1, .y = 0, .vx = 0, .vy = 0, .mass = 1000, .heat = 0 });
    computeAccelerations(&sim, .{});

    try testing.expectApproxEqRel(@as(f32, 1000), sim.ax[0] / -sim.ax[1], 1e-4);
}

test "softening bounds the force at zero separation (no NaN, no inf)" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 2);
    defer sim.deinit(testing.allocator);

    // Two particles exactly on top of each other: the case that would divide
    // by zero without ε².
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    sim.push(.{ .x = 0, .y = 0, .vx = 0, .vy = 0, .mass = 1, .heat = 0 });
    computeAccelerations(&sim, .{});

    for ([_]f32{ sim.ax[0], sim.ay[0], sim.ax[1], sim.ay[1] }) |a| {
        try testing.expect(!std.math.isNan(a));
        try testing.expect(std.math.isFinite(a));
        try testing.expectEqual(@as(f32, 0), a); // dx = dy = 0 ⇒ numerator 0
    }
}

test "integrate updates position with the new velocity (Step 5 ordering)" {
    const testing = std.testing;
    var sim = try Sim.initCapacity(testing.allocator, 1);
    defer sim.deinit(testing.allocator);

    sim.push(.{ .x = 0, .y = 0, .vx = 1, .vy = 0, .mass = 1, .heat = 0 });
    sim.ax[0] = 10;
    sim.ay[0] = 0;

    const cfg = Config{ .dt = 0.1 };
    integrate(&sim, cfg);

    // Semi-implicit: v becomes 1 + 10·0.1 = 2, then x += 2·0.1 = 0.2.
    // Explicit Euler would have given x = 0.1 — the whole difference.
    const p = sim.live()[0];
    try testing.expectApproxEqRel(@as(f32, 2.0), p.vx, 1e-6);
    try testing.expectApproxEqRel(@as(f32, 0.2), p.x, 1e-6);
}

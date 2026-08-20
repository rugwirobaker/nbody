//! The scalar baseline's data layout (RFC §2.1).
//!
//! Array of structures, allocated once at startup to the initial capacity.
//! This is the experimental control — the layout a reasonable engineer reaches
//! for first — so it stays deliberately plain. Part 3's SoA transform
//! (RFC §3.2) is a separate type, not an improvement to this one.

const std = @import("std");

const Config = @import("config.zig").Config;
const seed = @import("seed.zig");

pub const Particle = struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32,
    /// RFC Step 10: accumulated ΔKE from merges. Visual/diagnostic channel;
    /// nothing in the force law reads it.
    heat: f32,

    /// This body's radius (RFC-002 §1.1): `k·√m`, from mass at constant
    /// density, since mass is area in 2D.
    ///
    /// Derived rather than stored. A stored radius would be a second copy of
    /// what `mass` already says, and every write to `mass` would have to
    /// remember to update it — a body merging at the wrong distance is a
    /// plausible-looking simulation that no conservation test would catch. The
    /// field would also cost Phase A only 0.8–2.7 % (RFC-002 §1.1 measures it),
    /// so performance is not what decides this.
    ///
    /// The merge rule, the tests, and the renderer all go through here, so none
    /// of them can disagree about how big a body is.
    pub fn radius(p: Particle, cfg: Config) f32 {
        return cfg.merge_radius_scale * @sqrt(p.mass);
    }
};

pub const Sim = struct {
    /// Capacity is the initial n; the live prefix is `particles[0..n]`.
    particles: []Particle,
    /// Phase A's output and Phase B's input, sized to capacity.
    ///
    /// These are *derived per-tick state*, not particle identity, which is why
    /// they live outside `Particle` from day one (RFC §2.1). The split is what
    /// makes the frozen-snapshot rule (Step 8) expressible at all, and it
    /// survives unchanged into Part 3's SoA layout.
    ax: []f32,
    ay: []f32,
    /// Phase C scratch: `Particle.radius` for each live body, refilled at the
    /// top of every merge pass (RFC-002 §5.1).
    ///
    /// Scratch, not state — rebuilt from `mass` each pass, so it cannot
    /// disagree with the masses it came from. It exists because the merge scan
    /// compares every pair: computing the radius per pair costs 2.08× the
    /// scan, and reading it from here costs 1.28×. Phase A never touches it,
    /// and allocating it does not move Phase A's timing (RFC-002 §5.1).
    radii: []f32,
    /// Live count. Shrinks when merging is enabled; never grows.
    n: usize,

    /// Allocates all buffers to `capacity` and starts empty. Merging only ever
    /// shrinks `n`, so nothing reallocates mid-run (RFC §2.1).
    pub fn initCapacity(gpa: std.mem.Allocator, cap: usize) !Sim {
        const particles = try gpa.alloc(Particle, cap);
        errdefer gpa.free(particles);
        const ax = try gpa.alloc(f32, cap);
        errdefer gpa.free(ax);
        const ay = try gpa.alloc(f32, cap);
        errdefer gpa.free(ay);
        const radii = try gpa.alloc(f32, cap);

        @memset(particles, std.mem.zeroes(Particle));
        @memset(ax, 0);
        @memset(ay, 0);
        @memset(radii, 0);

        return .{ .particles = particles, .ax = ax, .ay = ay, .radii = radii, .n = 0 };
    }

    /// Allocates to `cfg.n` and fills it with the configured preset (RFC §2.7).
    pub fn initSeeded(gpa: std.mem.Allocator, cfg: Config) !Sim {
        var sim = try initCapacity(gpa, cfg.n);
        errdefer sim.deinit(gpa);
        seed.populate(&sim, cfg);
        return sim;
    }

    pub fn deinit(sim: *Sim, gpa: std.mem.Allocator) void {
        gpa.free(sim.particles);
        gpa.free(sim.ax);
        gpa.free(sim.ay);
        gpa.free(sim.radii);
        sim.* = undefined;
    }

    pub fn capacity(sim: Sim) usize {
        return sim.particles.len;
    }

    /// Appends one particle. Used by seeding and by hand-built test setups.
    pub fn push(sim: *Sim, p: Particle) void {
        std.debug.assert(sim.n < sim.capacity());
        sim.particles[sim.n] = p;
        sim.n += 1;
    }

    pub fn live(sim: Sim) []Particle {
        return sim.particles[0..sim.n];
    }
};

test "initCapacity allocates and starts empty" {
    var sim = try Sim.initCapacity(std.testing.allocator, 16);
    defer sim.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), sim.n);
    try std.testing.expectEqual(@as(usize, 16), sim.capacity());
    try std.testing.expectEqual(@as(usize, 16), sim.ax.len);
    try std.testing.expectEqual(@as(usize, 16), sim.ay.len);
    try std.testing.expectEqual(@as(usize, 0), sim.live().len);
}

test "push extends the live prefix" {
    var sim = try Sim.initCapacity(std.testing.allocator, 4);
    defer sim.deinit(std.testing.allocator);

    sim.push(.{ .x = 1, .y = 2, .vx = 3, .vy = 4, .mass = 5, .heat = 0 });
    try std.testing.expectEqual(@as(usize, 1), sim.live().len);
    try std.testing.expectEqual(@as(f32, 1), sim.live()[0].x);
    try std.testing.expectEqual(@as(f32, 5), sim.live()[0].mass);
}

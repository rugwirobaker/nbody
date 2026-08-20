//! One running simulation, in either layout, with the bookkeeping the page
//! needs to describe it.
//!
//! The two kernels differ in memory layout and instruction width and in
//! nothing else (RFC Part 3), so the demo gives them identical structure here:
//! same accumulator, same timing, same packed output. Whatever the `stacked`
//! view shows as a difference between the panels is then a difference in the
//! kernels, not in the harness around them.

const std = @import("std");
const nbody = @import("nbody");

const timestep = @import("timestep.zig");
const FixedStep = timestep.FixedStep;

/// Lane width for the SoA build. On `wasm32-freestanding+simd128` this is 4,
/// the same width as NEON (RFC §3.2 forbids hard-coding it).
pub const lanes = nbody.simd.default_lanes;

/// Which build a `World` is running. `base` is Part 2's scalar AoS baseline —
/// the experimental control — and `simd` is Part 3's SoA vector kernel.
pub const Kernel = enum { base, simd };

/// A monotonic clock in milliseconds. `wasm32-freestanding` has no `Io`, so
/// `std.Io.Timestamp` is unavailable and the browser's `performance.now()` is
/// passed in instead (RFC §2.5 note).
pub const Clock = *const fn () callconv(.c) f64;

/// Floats written per particle into the render buffer: x, y, mass, heat.
pub const floats_per_particle = 4;

/// Ticks after which the timing window is halved.
///
/// A plain running total would average over the whole session and stop
/// responding to a change; a ring buffer would cost memory for no extra
/// fidelity. Halving both sums keeps a decaying window with two adds per tick.
const timing_window_ticks: f64 = 2048;

pub const World = struct {
    state: State,
    step: FixedStep,

    /// Packed `(x, y, mass, heat)` per live particle — the instance buffer the
    /// vertex shader reads. Allocated once to the initial n, since merging only
    /// ever shrinks the live count.
    render: []f32,

    /// Phase-A time and ticks, over the decaying window above. Kept apart from
    /// the frame clock because RFC §2.5 rule 2 reports Phase A alone.
    phase_a_ns: f64 = 0,
    phase_a_ticks: f64 = 0,

    /// Simulated and wall-clock seconds, over the same window. Their ratio is
    /// how far from real time this world is running.
    sim_seconds: f64 = 0,
    real_seconds: f64 = 0,

    /// Total simulated time since the run started, never decayed. Displayed
    /// per panel: under `stacked` the two clocks drift apart at exactly the
    /// ratio of the kernels' throughput, which is the comparison made visible.
    clock_seconds: f64 = 0,

    /// The last frame ended because the wall-clock budget ran out rather than
    /// because the accumulator was drained.
    starved: bool = false,

    /// Ticks run since the render buffer was last refreshed, and the number of
    /// refreshes so far. Together they pace the picture — see `advance`.
    ticks_since_pack: usize = 0,
    render_updates: u64 = 1,

    pub const State = union(Kernel) {
        base: nbody.Sim,
        simd: nbody.simd.Particles,
    };

    /// Seeds a world from `cfg`.
    ///
    /// Both kernels are seeded through an AoS `Sim` — `Particles.fromAoS` is
    /// documented as the only way a SoA sim gets populated, precisely so the
    /// two builds start bit-identical (RFC §3.5). `stacked` inherits that, so a
    /// visible divergence between the panels is chaos amplifying `@reduce`
    /// reordering rather than two seeding paths that merely look alike.
    pub fn init(gpa: std.mem.Allocator, cfg: nbody.Config, kernel: Kernel) !World {
        var sim = try nbody.Sim.initSeeded(gpa, cfg);
        errdefer sim.deinit(gpa);

        const state: State = switch (kernel) {
            .base => .{ .base = sim },
            .simd => blk: {
                const p = try nbody.simd.Particles.fromAoS(gpa, sim, lanes);
                sim.deinit(gpa); // the AoS copy was scaffolding for the seeding
                break :blk .{ .simd = p };
            },
        };

        var world: World = .{
            .state = state,
            .step = FixedStep.init(cfg.dt),
            .render = try gpa.alloc(f32, cfg.n * floats_per_particle),
        };
        world.pack();
        return world;
    }

    pub fn deinit(w: *World, gpa: std.mem.Allocator) void {
        switch (w.state) {
            .base => |*s| s.deinit(gpa),
            .simd => |*p| p.deinit(gpa),
        }
        gpa.free(w.render);
        w.* = undefined;
    }

    /// Live particle count. Shrinks as merges land (RFC §2.6).
    pub fn count(w: World) usize {
        return switch (w.state) {
            .base => |s| s.n,
            .simd => |p| p.n,
        };
    }

    /// Runs the ticks this frame owes and refreshes the render buffer.
    ///
    /// Phase A is timed on its own, exactly as `bench/main.zig` does it:
    /// integration and merging run outside the timer, because the n² claim is
    /// about Phase A and nothing else (RFC §2.5 rule 2). Phases stay separate
    /// and ordered — accelerations for the whole tick, then integration, then
    /// merging over a stable particle set (Steps 8 and §2.6).
    pub fn advance(
        w: *World,
        frame_seconds: f32,
        budget_seconds: f64,
        cfg: nbody.Config,
        clock: Clock,
    ) void {
        const owed = w.step.takeSteps(frame_seconds);
        const budget_ms = budget_seconds * std.time.ms_per_s;
        const started = clock();

        var steps: usize = 0;
        w.starved = false;
        while (steps < owed) : (steps += 1) {
            // A tick budget alone cannot show the comparison: both worlds would
            // take the same ten ticks per frame however fast they are, and the
            // two panels would advance in lockstep. Budgeting wall clock
            // instead lets the faster kernel fit more ticks into its share, so
            // the panels' simulated clocks separate at the ratio of their
            // throughput. Overshoot by at most one tick, since the cost of the
            // next one is only known after running it.
            if (steps > 0 and clock() - started >= budget_ms) {
                w.starved = true;
                // Hand the unrun time back so nothing is silently lost.
                w.step.accumulator += @floatCast(@as(f64, cfg.dt) * @as(f64, @floatFromInt(owed - steps)));
                break;
            }

            const t0 = clock();
            switch (w.state) {
                .base => |*s| nbody.scalar.computeAccelerations(s, cfg),
                .simd => |*p| nbody.simd.computeAccelerations(lanes, p, cfg),
            }
            const t1 = clock();
            w.phase_a_ns += (t1 - t0) * std.time.ns_per_ms;
            w.phase_a_ticks += 1;

            switch (w.state) {
                .base => |*s| nbody.scalar.integrate(s, cfg),
                .simd => |*p| nbody.simd.integrate(lanes, p, cfg),
            }

            if (cfg.merging) switch (w.state) {
                .base => |*s| _ = nbody.merge.mergeCollisions(s, cfg),
                .simd => |*p| _ = nbody.simd.mergeCollisions(lanes, p, cfg),
            };
        }

        const advanced = @as(f64, @floatFromInt(steps)) * @as(f64, cfg.dt);
        w.sim_seconds += advanced;
        w.clock_seconds += advanced;
        w.real_seconds += @max(0, @as(f64, frame_seconds));
        w.decayWindow();

        // Publish a picture once a full frame's worth of ticks is complete,
        // rather than after every frame.
        //
        // Refreshing every frame shows a starved kernel a smaller step of
        // motion sixty times a second, which reads as a deliberately slower
        // simulation. Holding the buffer until `owed` ticks are done makes
        // every published picture the same size of step, so a kernel that
        // cannot keep up publishes fewer of them — the reading the reference
        // implementation gets from its frame rate, which is the same quantity:
        // pictures per second, against a fixed amount of physics in each.
        w.ticks_since_pack += steps;
        if (owed > 0 and w.ticks_since_pack >= owed) {
            w.ticks_since_pack -= owed;
            w.render_updates += 1;
            w.pack();
        }
    }

    fn decayWindow(w: *World) void {
        if (w.phase_a_ticks > timing_window_ticks) {
            w.phase_a_ticks *= 0.5;
            w.phase_a_ns *= 0.5;
        }
        // One second of frames is a comparable window for the rate, and it is
        // the unit the figure is quoted in.
        if (w.real_seconds > 1.0) {
            w.real_seconds *= 0.5;
            w.sim_seconds *= 0.5;
        }
    }

    /// Phase-A nanoseconds per tick — the RFC's metric, never FPS.
    ///
    /// Honest in every mode: JavaScript is single threaded, so under `stacked`
    /// the two worlds tick in sequence and each is timed while it runs alone.
    pub fn nsPerTick(w: World) f64 {
        if (w.phase_a_ticks == 0) return 0;
        return w.phase_a_ns / w.phase_a_ticks;
    }

    /// Simulated seconds advanced per real second. 1.0 is real time; below
    /// that the world is in slow motion because its clamp keeps firing.
    ///
    /// Only meaningful when this world has the frame to itself. Under `stacked`
    /// the two split one frame budget, so both read low — the page labels it.
    pub fn simRate(w: World) f64 {
        if (w.real_seconds <= 0) return 0;
        return w.sim_seconds / w.real_seconds;
    }

    /// How many times real-time speed this kernel can sustain, from its own
    /// Phase-A cost alone: `dt / ns_per_tick`. At or above 1.0 the kernel can
    /// simulate a second of physics in a second of wall clock.
    ///
    /// This is the figure that separates the two panels, and it is the one the
    /// clamp cannot tell you. With `dt` = 1 ms and §2.4's ten-tick ceiling, a
    /// 60 Hz frame owes ~17 ticks and gets 10, so *every* world reports a
    /// clamped frame and a sim rate of 0.6 no matter how fast it is. Dividing
    /// dt by measured Phase-A time removes the frame budget from the question
    /// and leaves the kernel.
    pub fn realtimeFactor(w: World) f64 {
        const ns = w.nsPerTick();
        if (ns <= 0) return 0;
        return @as(f64, w.step.dt) * std.time.ns_per_s / ns;
    }

    /// Whether the last frame hit RFC §2.4's ten-tick ceiling. Expected to be
    /// true at 60 Hz for any kernel — see `realtimeFactor`.
    pub fn clamped(w: World) bool {
        return w.step.clamped;
    }

    /// Simulated seconds since the run began.
    pub fn clockSeconds(w: World) f64 {
        return w.clock_seconds;
    }

    /// Whether the last frame ran out of wall clock before running every tick
    /// it owed. Under `stacked` this is what the slower kernel reports.
    pub fn budgetLimited(w: World) bool {
        return w.starved;
    }

    /// How many pictures this world has published. Rising once per frame means
    /// the kernel is keeping up; rising slower is the visible deficit.
    pub fn renderUpdates(w: World) u64 {
        return w.render_updates;
    }

    /// Copies the live particles into the render buffer as `(x, y, mass, heat)`.
    ///
    /// Both layouts could feed WebGL directly — AoS through stride-24 attribute
    /// offsets, SoA through four tightly-packed arrays — but one packed buffer
    /// keeps the JavaScript identical across kernels and leaves the renderer
    /// ignorant of how the physics stores anything. The cost is O(n) against
    /// Phase A's O(n²).
    ///
    /// `heat` is what the shader colours by: divided by mass it gives a
    /// temperature, which sets both the tint and the brightness.
    pub fn pack(w: *World) void {
        var out = w.render;
        switch (w.state) {
            .base => |s| for (s.live(), 0..) |p, i| {
                const o = i * floats_per_particle;
                out[o + 0] = p.x;
                out[o + 1] = p.y;
                out[o + 2] = p.mass;
                out[o + 3] = p.heat;
            },
            .simd => |p| for (0..p.n) |i| {
                const o = i * floats_per_particle;
                out[o + 0] = p.x[i];
                out[o + 1] = p.y[i];
                out[o + 2] = p.mass[i];
                out[o + 3] = p.heat[i];
            },
        }
    }
};

// ---------------------------------------------------------------------------
// Tests. The pacing rule is what makes the demo's claim legible, so it is
// asserted here rather than judged by eye in a browser.

/// A clock frozen at zero, which makes the budget check exact: any elapsed
/// figure is `0 - 0`, so a budget of 0 starves every frame after the first tick
/// and a large budget never starves. No wall-clock timing enters the test.
fn frozenClock() callconv(.c) f64 {
    return 0;
}

test "a world that keeps up publishes once per frame" {
    const cfg: nbody.Config = .{ .n = 64, .merging = false };
    var w = try World.init(std.testing.allocator, cfg, .base);
    defer w.deinit(std.testing.allocator);

    const before = w.renderUpdates();
    const frames = 30;
    for (0..frames) |_| w.advance(cfg.dt * 20, 1e9, cfg, &frozenClock);

    try std.testing.expectEqual(before + frames, w.renderUpdates());
}

test "a starved world publishes in proportion to the ticks it finished" {
    const cfg: nbody.Config = .{ .n = 64, .merging = false };
    var w = try World.init(std.testing.allocator, cfg, .base);
    defer w.deinit(std.testing.allocator);

    const before = w.renderUpdates();
    const frames = 30;
    // A zero budget stops each frame after its first tick, so ten frames of
    // work land per frame's worth of demand.
    for (0..frames) |_| w.advance(cfg.dt * 20, 0, cfg, &frozenClock);

    try std.testing.expect(w.budgetLimited());
    try std.testing.expectEqual(
        before + frames / timestep.max_steps_per_frame,
        w.renderUpdates(),
    );
    // Every published picture still carries a full frame's worth of motion.
    try std.testing.expect(w.ticks_since_pack < timestep.max_steps_per_frame);
}

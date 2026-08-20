//! nbody-viz — the wasm half of the demo (RFC Part 4's demo client).
//!
//! The browser owns the window and the swapchain; JavaScript owns the WebGL2
//! calls; this module owns the physics and the clock. The interface between
//! them is deliberately thin: JavaScript calls `advance` once per animation
//! frame and then reads a packed float buffer straight out of wasm memory.
//!
//! RFC §2.4's outer loop is inverted here. The browser drives the frame, so the
//! `while (rendering)` of the spec becomes a callback — but the accumulator and
//! its normative ten-tick clamp are unchanged and stay on this side of the
//! boundary, in `timestep.zig`, where the rest of the spec lives.

const std = @import("std");
const nbody = @import("nbody");

const timestep = @import("timestep.zig");
const world_mod = @import("world.zig");
const Kernel = world_mod.Kernel;
const World = world_mod.World;

/// Traps instead of formatting a panic message. Keeps `std.fmt` and the
/// default panic handler — with its stack-trace machinery — out of the wasm,
/// which is most of the module's size.
pub const panic = std.debug.no_panic;

/// `performance.now()`, in milliseconds. `wasm32-freestanding` has no `Io`, so
/// `std.Io.Timestamp` (which `bench/main.zig` uses) is unavailable here.
///
/// Browsers coarsen this clock to roughly 100 µs outside cross-origin-isolated
/// contexts, which is a large fraction of one Phase A. `World` therefore
/// averages over a window of ticks rather than trusting any single reading.
extern "env" fn now() f64;

/// Which panels are on screen.
const Mode = enum(u32) {
    base = 0,
    simd = 1,
    stacked = 2,

    fn needs(m: Mode, k: Kernel) bool {
        return m == .stacked or @intFromEnum(m) == @intFromEnum(k);
    }
};

const gpa = std.heap.wasm_allocator;

var cfg: nbody.Config = .{};
var mode: Mode = .stacked;
var worlds: [2]?World = .{ null, null };

fn slot(k: Kernel) *?World {
    return &worlds[@intFromEnum(k)];
}

fn kernelFrom(which: u32) ?Kernel {
    return switch (which) {
        0 => .base,
        1 => .simd,
        else => null,
    };
}

fn teardown() void {
    for (&worlds) |*w| {
        if (w.*) |*live| live.deinit(gpa);
        w.* = null;
    }
}

/// Seeds the run. Returns false if the configuration is unusable or memory ran
/// out, in which case nothing is left running.
///
/// Called again on every control change, so the URL and the running simulation
/// never disagree. Both worlds are rebuilt from one config, which is what keeps
/// the two panels bit-identical at t = 0 (RFC §3.5).
export fn start(n: u32, seed: u32, preset: u32, merging: u32, mode_raw: u32) bool {
    teardown();

    if (n == 0) return false;
    if (preset > 1 or mode_raw > 2) return false;

    // Every constant is now a library default. The demo used to override `dt`
    // to a quarter of it and `d_merge2` to a hundredth, both because the fixed
    // merge threshold let bodies plunge to one tiny distance before merging and
    // the integrator could not resolve the encounters that made. Contact
    // merging (RFC-002) removes those encounters — a body swallows its
    // neighbour on contact, before the plunge — so both overrides went with it.
    //
    // Measured at the coarse step, the disk reaches a body holding 12.5 % of
    // its mass in 4,000 ticks rather than 16,000, with matching survivor counts
    // at every checkpoint and slightly better energy.
    cfg = .{
        .n = n,
        .seed = seed,
        .preset = if (preset == 0) .disk else .keplerian,
        // Demo mode: RFC §2.6 turns merging on here and off for benchmarks.
        .merging = merging != 0,
    };
    if (nbody.Config.validate(cfg) != null) return false;

    mode = @enumFromInt(mode_raw);

    for ([_]Kernel{ .base, .simd }) |k| {
        if (!mode.needs(k)) continue;
        slot(k).* = World.init(gpa, cfg, k) catch {
            teardown();
            return false;
        };
    }
    return true;
}

/// Advances every on-screen world by one frame's worth of wall clock.
///
/// Each world keeps its own accumulator and publishes a picture once it has
/// completed a frame's worth of ticks, which is the whole point of `stacked`:
/// handed the same frame time, the kernel that cannot keep up publishes fewer
/// pictures and falls behind on its clock while the other holds the frame rate.
///
/// `budget_seconds` is what the caller is willing to spend on this world before
/// cutting the frame short. Alone on screen a world gets a figure large enough
/// to never engage, so a long frame becomes a low frame rate exactly as it does
/// in a native window; under `stacked` the two split a frame so that neither
/// can freeze the page.
export fn advance(frame_seconds: f32, budget_seconds: f64) void {
    for (&worlds) |*w| {
        if (w.*) |*live| live.advance(frame_seconds, budget_seconds, cfg, &now);
    }
}

/// Simulated seconds this world has advanced since it was seeded.
///
/// Under `stacked` the two clocks separate at the ratio of the kernels'
/// throughput, because each world gets wall clock rather than a tick quota.
/// That divergence is the comparison, stated in the units the physics runs in.
export fn clockSeconds(which: u32) f64 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return w.clockSeconds();
}

/// Whether this world ran out of wall clock before finishing the frame's ticks.
export fn budgetLimited(which: u32) u32 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return @intFromBool(w.budgetLimited());
}

/// Simulated seconds per tick, so the page can label its clocks.
export fn dtSeconds() f64 {
    return cfg.dt;
}

/// The density constant behind `r(m) = k·√m` (RFC-002 §1.1).
///
/// Exported so the renderer draws bodies at the size the physics merges them
/// at. A copy of this number in JavaScript would be free to drift out of
/// agreement with the simulation, which is the failure RFC-002 exists to end.
export fn mergeRadiusScale() f64 {
    return cfg.merge_radius_scale;
}

export fn particleCount(which: u32) u32 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return @intCast(w.count());
}

/// Byte offset of the packed `(x, y, mass, heat)` instance buffer within wasm
/// memory, or 0 when that world is not running.
///
/// An offset rather than a pointer, so the "not running" case has a defined
/// value to return. Callers pair it with `particleCount`, which is 0 in the
/// same case.
///
/// Growing the wasm heap detaches every `Float32Array` JavaScript holds over
/// `memory.buffer`, so the caller must rebuild its views whenever the buffer
/// identity changes.
export fn renderBufferOffset(which: u32) u32 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return @intCast(@intFromPtr(w.render.ptr));
}

/// Pictures published since the run began.
///
/// The page reads it two ways: to skip the upload when nothing new exists, and
/// to show the rate. That rate is the reference implementation's FPS in another
/// spelling — each published picture carries the same fixed amount of physics,
/// so pictures per second is throughput. The reported *metric* stays Phase-A
/// ns/tick (RFC §2.5 rule 2); this is the symptom, not the measurement.
export fn renderUpdates(which: u32) f64 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    // f64 because JavaScript reads every number as one, and it cannot wrap.
    return @floatFromInt(w.renderUpdates());
}

/// Phase-A nanoseconds per tick (RFC §2.5 rule 2). Never FPS.
export fn nsPerTick(which: u32) f64 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return w.nsPerTick();
}

/// Simulated seconds per real second. 1.0 is real time.
export fn simRate(which: u32) f64 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return w.simRate();
}

/// Times real-time speed this kernel sustains, from Phase-A cost alone.
/// The figure that actually distinguishes the two panels — see `World`.
export fn realtimeFactor(which: u32) f64 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return w.realtimeFactor();
}

export fn clamped(which: u32) u32 {
    const k = kernelFrom(which) orelse return 0;
    const w = slot(k).* orelse return 0;
    return @intFromBool(w.clamped());
}

export fn running(which: u32) u32 {
    const k = kernelFrom(which) orelse return 0;
    return @intFromBool(slot(k).* != null);
}

/// The SIMD build's lane width on this target: 4 under `+simd128`. Reported so
/// the page can state what it is actually running (RFC §3.2).
export fn laneCount() u32 {
    return @intCast(world_mod.lanes);
}

/// Floats per particle in the render buffer, so the stride lives in one place.
export fn floatsPerParticle() u32 {
    return @intCast(world_mod.floats_per_particle);
}

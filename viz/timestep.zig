//! The fixed-timestep accumulator (RFC §2.4).
//!
//! Physics and rendering run on different clocks. `dt` is simulated time per
//! tick, fixed by config; the browser redraws at whatever rate the display
//! runs. This type is the bridge: hand it the wall-clock time since the last
//! frame, get back the number of ticks owed.
//!
//! It lives in its own file, free of any wasm-only import, so the clamp can be
//! tested on the host by `zig build test`. `main.zig` is the wasm-only part.

const std = @import("std");

/// RFC §2.4's spiral-of-death guard, normative: a frame may claim at most this
/// many ticks.
///
/// Without it a slow frame demands more ticks, which makes the next frame
/// slower, which demands more ticks. Under the clamp an overloaded simulation
/// degrades into slow motion instead of locking the tab up — and that slow
/// motion is exactly what the `stacked` view is built to show.
pub const max_steps_per_frame: usize = 10;

pub const FixedStep = struct {
    /// Simulated seconds owed but not yet ticked.
    accumulator: f32 = 0.0,
    /// Simulated seconds per tick. Mirrors `Config.dt`.
    dt: f32,
    /// Whether the last `takeSteps` hit the clamp — i.e. this world is no
    /// longer keeping up with real time. Drives the HUD indicator.
    clamped: bool = false,

    pub fn init(dt: f32) FixedStep {
        std.debug.assert(dt > 0);
        return .{ .dt = dt };
    }

    /// Consumes `frame_seconds` of wall clock and returns the ticks owed.
    ///
    /// Named for the mutation: it does not merely report what is due, it takes
    /// it out of the accumulator.
    pub fn takeSteps(s: *FixedStep, frame_seconds: f32) usize {
        // A negative or NaN frame time can only come from a broken clock;
        // treating it as zero keeps the accumulator monotone.
        if (!(frame_seconds > 0)) return s.drain();
        s.accumulator += frame_seconds;
        return s.drain();
    }

    fn drain(s: *FixedStep) usize {
        const ceiling = @as(f32, @floatFromInt(max_steps_per_frame)) * s.dt;

        // The RFC clamps the accumulator to `10 * dt` and then drains it, which
        // yields exactly `max_steps_per_frame` ticks and no remainder. Taking
        // that outcome directly avoids the float drift that repeated
        // subtraction of `dt` from `10 * dt` would leave behind.
        if (s.accumulator > ceiling) {
            s.accumulator = 0;
            s.clamped = true;
            return max_steps_per_frame;
        }

        s.clamped = false;
        var steps: usize = 0;
        while (s.accumulator >= s.dt) : (steps += 1) s.accumulator -= s.dt;
        return steps;
    }
};

const testing = std.testing;

test "a frame shorter than dt owes nothing and banks the time" {
    var s = FixedStep.init(0.001);
    try testing.expectEqual(@as(usize, 0), s.takeSteps(0.0004));
    try testing.expect(!s.clamped);
    // Banked, not discarded: two more of these cross the threshold.
    try testing.expectEqual(@as(usize, 0), s.takeSteps(0.0004));
    try testing.expectEqual(@as(usize, 1), s.takeSteps(0.0004));
}

test "a 60 fps frame at dt = 1 ms owes 16 ticks... and is clamped to 10" {
    var s = FixedStep.init(0.001);
    // 16.7 ms of simulated time is owed, but §2.4 caps a frame at 10 ticks.
    try testing.expectEqual(max_steps_per_frame, s.takeSteps(1.0 / 60.0));
    try testing.expect(s.clamped);
}

test "the clamp bounds every frame, however long (RFC §2.4)" {
    var s = FixedStep.init(0.001);
    for ([_]f32{ 0.02, 0.5, 10.0, 1000.0 }) |frame| {
        const steps = s.takeSteps(frame);
        try testing.expectEqual(max_steps_per_frame, steps);
        try testing.expect(s.clamped);
        // The backlog is dropped rather than carried, which is what stops the
        // spiral: the next frame starts even.
        try testing.expectEqual(@as(f32, 0), s.accumulator);
    }
}

test "an unclamped frame leaves less than one tick banked" {
    var s = FixedStep.init(0.001);
    const steps = s.takeSteps(0.005);
    try testing.expect(!s.clamped);
    // 4 or 5, depending on how 0.005 and 0.001 land in f32 — the count for a
    // single frame is not the invariant. What must hold is that the remainder
    // is smaller than one tick, so nothing is owed that could have been paid.
    try testing.expect(steps == 4 or steps == 5);
    try testing.expect(s.accumulator >= 0);
    try testing.expect(s.accumulator < s.dt);
}

test "unclamped frames lose no simulated time over a long run" {
    // The real guarantee. Any single frame may round down and bank the
    // remainder; across many frames the banked residue is paid out, so total
    // ticks tracks total wall-clock time to within one tick. If this drifts,
    // the demo runs systematically slow and every ns/tick figure sits on a
    // simulation that is not advancing at the rate it claims.
    var s = FixedStep.init(0.001);

    var elapsed: f64 = 0;
    var ticks: usize = 0;
    // 5 ms per frame: well under the 10-tick ceiling, so the clamp never fires.
    for (0..2000) |_| {
        ticks += s.takeSteps(0.005);
        elapsed += 0.005;
    }
    try testing.expect(!s.clamped);

    const expected = elapsed / @as(f64, s.dt);
    const drift = @abs(@as(f64, @floatFromInt(ticks)) - expected);
    try testing.expect(drift <= 1.0);
}

test "identical frame sequences produce identical tick counts" {
    const frames = [_]f32{ 0.016, 0.004, 0.0009, 0.031, 0.0005, 0.002 };

    var a = FixedStep.init(0.001);
    var b = FixedStep.init(0.001);
    for (frames) |f| {
        try testing.expectEqual(a.takeSteps(f), b.takeSteps(f));
        try testing.expectEqual(a.accumulator, b.accumulator);
    }
}

test "a broken clock cannot run the accumulator backwards" {
    var s = FixedStep.init(0.001);
    _ = s.takeSteps(0.0005);
    const banked = s.accumulator;
    try testing.expectEqual(@as(usize, 0), s.takeSteps(-1.0));
    try testing.expectEqual(banked, s.accumulator);
    try testing.expectEqual(@as(usize, 0), s.takeSteps(std.math.nan(f32)));
    try testing.expectEqual(banked, s.accumulator);
}

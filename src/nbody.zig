//! nbody — a 2D gravitational N-body simulation.
//!
//! The public surface of the core library. Renderer-free and I/O-free: this
//! module allocates, simulates, and reports numbers, and knows nothing about
//! screens, files, or stdout.
//!
//! The specification this implements is `docs/RFC-001.md`; comments here cite
//! it by step and section rather than restating the derivations.

const std = @import("std");

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const Preset = config.Preset;

pub const platform = @import("platform.zig");
pub const lane_count = platform.lane_count;

test {
    std.testing.refAllDecls(@This());
}

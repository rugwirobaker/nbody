//! Simulation configuration (RFC §2.2).
//!
//! Every constant the simulation uses lives here — nothing physical is
//! hard-coded in a kernel. A recorded measurement or run is meaningless
//! without the `Config` and seed that produced it.

const std = @import("std");

pub const Preset = enum {
    /// Uniform-density rotating disk (RFC §2.7).
    disk,
    /// `disk` plus one heavy central body: v_circ ∝ 1/√r, the solar-system look.
    keplerian,
};

pub const Config = struct {
    // ---- physics ----

    /// Gravitational coupling constant in *simulation units* (RFC Step 1).
    /// Not the SI value: once mass is 1.0 and length is 1.0 we have left the
    /// SI universe. Tuned so accelerations stay O(1).
    g: f32 = 5.0e-4,

    /// Simulated seconds per tick. Fixed, frame-independent (RFC §2.4).
    dt: f32 = 1.0e-3,

    /// Softening length squared, ε² (RFC "Softening"). Heuristic default is
    /// the typical near-neighbour spacing ε ≈ R/√n; with R = 1 and n = 2000
    /// that is ε ≈ 0.022, ε² ≈ 5e-4.
    eps2: f32 = 5.0e-4,

    // ---- population ----

    /// Initial particle count. The benchmark variable.
    n: usize = 2000,

    /// Disk radius R (RFC §2.7); the simulation's length unit.
    radius: f32 = 1.0,

    /// Per-particle mass, drawn uniformly from [mass_min, mass_max).
    mass_min: f32 = 0.5,
    mass_max: f32 = 1.5,

    /// `keplerian` only: central mass = factor × mean particle mass
    /// (RFC §2.7 suggests 10–100×).
    central_mass_factor: f32 = 50.0,

    /// Velocity noise as a fraction of v_circ (RFC §2.7). A perfectly cold
    /// disk is glassy; jitter seeds the encounters that drive merging.
    jitter: f32 = 0.03,

    /// Initial conditions.
    preset: Preset = .disk,

    /// The sole source of randomness: same seed + same config ⇒ same run.
    seed: u64 = 0xC0FFEE,

    // ---- merging (RFC §2.6) ----

    /// Enable Phase C. **False for benchmarks** (RFC §2.5 rule 3): performance
    /// numbers are only meaningful at fixed n.
    merging: bool = false,

    /// Merge threshold squared. Default ≈ eps2, so the simulation never
    /// lingers in the regime where softening distorts the force law.
    d_merge2: f32 = 5.0e-4,

    /// Per-tick multiplier on `heat` — radiative cooling, presentation only.
    /// **1.0 for the energy test** (RFC §2.5 test (d)).
    heat_decay: f32 = 0.999,

    /// Checks the parameters that would otherwise fail silently or violate a
    /// normative rule. Returns null when the config is usable.
    pub fn validate(cfg: Config) ?[]const u8 {
        if (!(cfg.eps2 > 0)) return "eps2 must be > 0: softening is what bounds the force " ++
            "law, zeroes the self-term, and keeps zero-mass padding NaN-free";
        if (!(cfg.dt > 0)) return "dt must be > 0";
        if (!(cfg.radius > 0)) return "radius must be > 0";
        if (!(cfg.mass_min > 0)) return "mass_min must be > 0";
        if (cfg.mass_max < cfg.mass_min) return "mass_max must be >= mass_min";
        if (cfg.d_merge2 < 0) return "d_merge2 must be >= 0";
        // Phase C (RFC §2.6) has not landed yet. Rejecting the flag here beats
        // silently ignoring it: a caller who asks for merging and gets a
        // fixed-n run would only find out via a failed conservation test.
        // Delete this rule when merge() exists.
        if (cfg.merging) return "merging (Phase C, RFC §2.6) is not implemented yet";
        return null;
    }
};

test "default config is valid" {
    try std.testing.expect(Config.validate(.{}) == null);
}

test "validate rejects an unsoftened config" {
    try std.testing.expect(Config.validate(.{ .eps2 = 0 }) != null);
}

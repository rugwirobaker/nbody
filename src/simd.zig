//! The SIMD build: SoA layout and vector kernels (RFC §3).
//!
//! Same algorithm as `scalar.zig`: same force law, same integration order,
//! same **ordered**-pair n² traversal. Only the memory layout and the
//! instruction width differ. Changing *what* is computed here, as opposed to
//! how many lanes wide it runs, breaks the comparison.
//!
//! Layout and kernels live in one file on purpose: the padding invariant
//! (§3.2) spans them. Phase A reads to `n_padded` *by design*, so the code
//! that maintains what lives out there belongs beside the code that trusts it.
//!
//! Vector vocabulary, all builtins: `@Vector(L, T)` is L values of T in one
//! register with `+ - * /` elementwise by language rule; `@splat(x)` broadcasts
//! one scalar into every lane; `@reduce(.Add, v)` collapses the lanes back into
//! one scalar. Splat fans out, elementwise runs parallel, reduce fans in.

const std = @import("std");

const Config = @import("config.zig").Config;
const platform = @import("platform.zig");
const sim_mod = @import("sim.zig");
const Particle = sim_mod.Particle;
const Sim = sim_mod.Sim;

/// The natural lane width for this target — 4 on NEON, 8 on AVX2, 16 on
/// AVX-512. Kernels below take `L` as a comptime parameter rather than baking
/// this in (RFC §3.2 calls hard-coding non-compliant), so the harness can also
/// instantiate them at other widths as a labeled experiment.
pub const default_lanes = platform.lane_count;

/// One cache-line-aligned f32 array (RFC §3.2 "Alignment").
///
/// The alignment lives in the *type*, not just at the allocation site, because
/// Zig's allocator interface needs it back at free time — storing these in a
/// plain `[]f32` erases it and `free` reports "allocation alignment 64 does not
/// match free alignment 4". Coercion to `[]f32` for reading is fine; it is
/// only ownership that needs the alignment preserved.
pub const Lanes = []align(platform.cache_line.toByteUnits()) f32;

/// Structure of arrays: one array per field, so the `L` values a vector load
/// needs are adjacent (RFC §3.2).
///
/// Under AoS, consecutive `x` values sit `sizeof(Particle)` apart — a vector
/// load would need a slow gather, and every cache line fetched would carry
/// unwanted `vy`/`heat` freight. Here `x[j..j+L]` is one unit-stride load and
/// every byte in the line is a byte the kernel wants.
pub const Particles = struct {
    x: Lanes,
    y: Lanes,
    vx: Lanes,
    vy: Lanes,
    mass: Lanes,
    heat: Lanes,

    /// Phase A's output, Phase B's input. Separate arrays, as in the AoS
    /// build: derived per-tick state, not particle identity (RFC §2.1). Owned
    /// here rather than by a separate struct so that one function can restore
    /// the whole padding invariant after a merge — see `mergePair`.
    ax: Lanes,
    ay: Lanes,

    /// Live count. Everything at index ≥ `n` is padding and has mass zero.
    n: usize,

    /// `n` rounded up to a multiple of `L`: the horizon Phase A reads to, and
    /// the reason no tail loop or masked load is needed (RFC §3.4).
    ///
    /// Tracked as `n` shrinks, which the RFC's struct comment glosses over by
    /// conflating it with capacity — they are only equal at t = 0. Keeping
    /// them apart is what makes merging actually shrink the inner loop instead
    /// of grinding through dead slots for the rest of the run.
    n_padded: usize,

    /// Allocated length of every array, fixed at construction. Never shrinks.
    capacity: usize,

    /// Builds the SoA layout from a seeded AoS sim.
    ///
    /// This is the **only** way a SoA sim gets populated, by design.
    /// `seed.populate` fills an AoS `Sim`; a second seeding path writing
    /// straight into these arrays would have to be *proved* to produce
    /// identical particles. §3.5 compares the two builds over a chaotic
    /// system, where a one-ULP difference in initial conditions diverges
    /// exponentially. Converting makes bit-identical initial conditions true
    /// by construction.
    pub fn fromAoS(gpa: std.mem.Allocator, sim: Sim, comptime L: usize) !Particles {
        const cap = platform.alignUp(sim.n, L);

        var arrays: [8]Lanes = undefined;
        var allocated: usize = 0;
        errdefer for (arrays[0..allocated]) |a| gpa.free(a);
        for (&arrays) |*a| {
            a.* = try platform.allocLanes(gpa, cap);
            allocated += 1;
            // Zeroing everything up front establishes the padding invariant
            // for free: any slot the scatter below does not touch is already
            // mass 0, position 0, velocity 0.
            @memset(a.*, 0);
        }

        var p: Particles = .{
            .x = arrays[0],
            .y = arrays[1],
            .vx = arrays[2],
            .vy = arrays[3],
            .mass = arrays[4],
            .heat = arrays[5],
            .ax = arrays[6],
            .ay = arrays[7],
            .n = sim.n,
            .n_padded = platform.alignUp(sim.n, L),
            .capacity = cap,
        };

        for (sim.live(), 0..) |particle, i| {
            p.x[i] = particle.x;
            p.y[i] = particle.y;
            p.vx[i] = particle.vx;
            p.vy[i] = particle.vy;
            p.mass[i] = particle.mass;
            p.heat[i] = particle.heat;
        }

        return p;
    }

    pub fn deinit(p: *Particles, gpa: std.mem.Allocator) void {
        gpa.free(p.x);
        gpa.free(p.y);
        gpa.free(p.vx);
        gpa.free(p.vy);
        gpa.free(p.mass);
        gpa.free(p.heat);
        gpa.free(p.ax);
        gpa.free(p.ay);
        p.* = undefined;
    }

    /// Gathers the live particles back into an AoS buffer.
    ///
    /// Exists so `properties.zig` works unchanged on this layout. Writing SoA
    /// versions of all nine property functions would double the surface where
    /// a conservation test can pass on one layout while the bug lives in the
    /// other. `buf` must hold at least `n` particles; the copy only ever runs
    /// in tests and diagnostics, never in the timed path.
    pub fn toAoS(p: Particles, buf: []Particle) []Particle {
        std.debug.assert(buf.len >= p.n);
        for (buf[0..p.n], 0..) |*particle, i| {
            particle.* = .{
                .x = p.x[i],
                .y = p.y[i],
                .vx = p.vx[i],
                .vy = p.vy[i],
                .mass = p.mass[i],
                .heat = p.heat[i],
            };
        }
        return buf[0..p.n];
    }

    /// Asserts the padding invariant: every slot at index ≥ `n` has mass zero.
    ///
    /// Cheap enough to call from tests after anything that touches `n`. The
    /// bug this catches is silent by construction — see `mergePair`.
    pub fn assertPaddingIsInert(p: Particles) void {
        for (p.mass[p.n..p.capacity]) |m| std.debug.assert(m == 0);
        for (p.ax[p.n..p.capacity]) |a| std.debug.assert(a == 0);
        for (p.ay[p.n..p.capacity]) |a| std.debug.assert(a == 0);
    }
};

/// Phase B (RFC Steps 4–5): semi-implicit Euler, `L` particles at a time.
///
/// The trivial half of the transform, and §3.3 says to implement it first:
/// pure elementwise streaming, no broadcast beyond the constants, no
/// reduction. Padded lanes compute `0 += 0 * dt` and stay zero, which is why
/// this can run to `n_padded` without a tail loop.
pub fn integrate(comptime L: usize, p: *Particles, cfg: Config) void {
    const V = platform.Vec(L);
    const dt: V = @splat(cfg.dt);
    const decay: V = @splat(cfg.heat_decay);

    var i: usize = 0;
    while (i < p.n_padded) : (i += L) {
        const ax: V = p.ax[i..][0..L].*;
        const ay: V = p.ay[i..][0..L].*;

        // Velocity first, then position from the *new* velocity (Step 5).
        var vx: V = p.vx[i..][0..L].*;
        var vy: V = p.vy[i..][0..L].*;
        vx += ax * dt;
        vy += ay * dt;

        var x: V = p.x[i..][0..L].*;
        var y: V = p.y[i..][0..L].*;
        x += vx * dt;
        y += vy * dt;

        const heat: V = p.heat[i..][0..L].*;

        p.vx[i..][0..L].* = vx;
        p.vy[i..][0..L].* = vy;
        p.x[i..][0..L].* = x;
        p.y[i..][0..L].* = y;
        p.heat[i..][0..L].* = heat * decay; // presentation, not physics
    }
}

/// Phase A (RFC Steps 7–8), Variant A: vectorize the *sources*.
///
/// The outer loop over accelerated particles stays scalar; each row broadcasts
/// its target coordinates once and consumes `L` sources per inner iteration
/// into lane accumulators, collapsed by one `@reduce` per component at the row
/// bottom. Variant B (vectorizing the targets) is a labeled experiment in
/// §3.3 and explicitly may not substitute for this.
///
/// Reads positions and masses, writes only `ax`/`ay` — the frozen snapshot
/// (Step 8), unchanged from the scalar build.
///
/// Two things make this loop straight-line, and both were decided in Part 1
/// for other reasons:
///   - **Softening** zeroes the self term (lane `i % L` computes dx = dy = 0
///     over a denominator of ε³), so no `if (i != j)` and no mask.
///   - **Zero-mass padding** makes the slots past `n` contribute `0 × finite`.
///     They cannot produce NaN, because softening guarantees the denominator
///     is at least ε³ — the two decisions click together (§3.2).
pub fn computeAccelerations(comptime L: usize, p: *Particles, cfg: Config) void {
    const V = platform.Vec(L);
    const g: V = @splat(cfg.g);
    const eps2: V = @splat(cfg.eps2);
    const one: V = @splat(@as(f32, 1.0));

    for (0..p.n) |i| {
        const xi: V = @splat(p.x[i]); // broadcast once per row
        const yi: V = @splat(p.y[i]);
        var axv: V = @splat(@as(f32, 0.0)); // L partial sums, one per lane
        var ayv: V = @splat(@as(f32, 0.0));

        var j: usize = 0;
        while (j < p.n_padded) : (j += L) {
            // Three vector loads, zero stores (RFC §3.3c). The `[j..][0..L].*`
            // chain is the vector load: runtime slice, comptime-length
            // pointer-to-array, dereference to [L]f32, which coerces to V.
            const xj: V = p.x[j..][0..L].*;
            const yj: V = p.y[j..][0..L].*;
            const mj: V = p.mass[j..][0..L].*;

            const dx = xj - xi; // the same fourteen operations as
            const dy = yj - yi; //   scalar.zig, with f32 -> V. That
            const d2 = dx * dx + dy * dy + eps2; //   substitution is the entire
            const inv_d3 = one / (d2 * @sqrt(d2)); //   transformation.
            const s = g * mj * inv_d3;
            axv += s * dx;
            ayv += s * dy;
        }

        // Horizontal: L partials -> one scalar. This is where Step 7's
        // summation gets reordered relative to the scalar build, and the
        // entire reason §3.5's comparison is tolerance-based rather than
        // bitwise. Float addition is not associative; both orderings are
        // equally correct and differ in the last bits.
        p.ax[i] = @reduce(.Add, axv);
        p.ay[i] = @reduce(.Add, ayv);
    }
}

/// One tick: advance simulated time by `cfg.dt` (RFC Step 8).
pub fn tick(comptime L: usize, p: *Particles, cfg: Config) void {
    computeAccelerations(L, p, cfg);
    integrate(L, p, cfg);
    if (cfg.merging) _ = mergeCollisions(L, p, cfg);
}

/// Phase C (RFC §2.6), deliberately **scalar even in the SIMD build** (§3.3).
///
/// Vectorizing a distance scan would buy nothing measurable — merging is
/// demo-only and every benchmark runs `merging = false` — while the scalar
/// pass keeps greedy-restart determinism trivially intact. A vectorized scan
/// is parked as a Part 4 exercise.
pub fn mergeCollisions(comptime L: usize, p: *Particles, cfg: Config) usize {
    var merges: usize = 0;

    var restart = true;
    while (restart) {
        restart = false;
        scan: for (0..p.n) |i| {
            for (i + 1..p.n) |j| {
                const dx = p.x[j] - p.x[i];
                const dy = p.y[j] - p.y[i];
                // Proximity test, not a force evaluation: no eps2 here.
                const d2 = dx * dx + dy * dy;
                if (d2 < cfg.d_merge2) {
                    mergePair(L, p, i, j);
                    merges += 1;
                    restart = true;
                    break :scan;
                }
            }
        }
    }

    return merges;
}

/// Merges `j` into `i` by Step 10's formulas, swap-removes `j`, and restores
/// the padding invariant.
///
/// **The trap this function exists to avoid (RFC §3.2).** In the scalar build,
/// the slot vacated by a swap-remove is stale memory no loop ever reads —
/// leaving garbage there is harmless and standard. Here Phase A reads to
/// `n_padded` *by design*, so after `n` decrements, the vacated slot sits
/// **inside the read region** still holding a full copy of a real particle.
/// Miss the re-zero and you get a ghost: invisible to any renderer drawing
/// `[0..n]`, never a merge candidate, and still pulling on everything. Every
/// test passes except momentum-with-merging.
///
/// Padding moved the boundary of "what memory is the algorithm's business"
/// outward, and every code path that shrinks `n` has to respect the new
/// boundary.
pub fn mergePair(comptime L: usize, p: *Particles, i: usize, j: usize) void {
    std.debug.assert(i < j);
    std.debug.assert(j < p.n);

    const mi = p.mass[i];
    const mj = p.mass[j];
    const m = mi + mj;

    // ΔKE = ½·μ·v_rel², computed before the overwrite: it needs both original
    // velocities (Step 10).
    const rvx = p.vx[i] - p.vx[j];
    const rvy = p.vy[i] - p.vy[j];
    const mu = (mi * mj) / m;
    const dke = 0.5 * mu * (rvx * rvx + rvy * rvy);

    p.x[i] = (mi * p.x[i] + mj * p.x[j]) / m; // centre of mass (Step 10)
    p.y[i] = (mi * p.y[i] + mj * p.y[j]) / m;
    p.vx[i] = (mi * p.vx[i] + mj * p.vx[j]) / m; // momentum      (Step 9)
    p.vy[i] = (mi * p.vy[i] + mj * p.vy[j]) / m;
    p.heat[i] = p.heat[i] + p.heat[j] + dke;
    p.mass[i] = m;

    // Swap-remove j. Self-assignment when j was last is harmless.
    p.n -= 1;
    p.x[j] = p.x[p.n];
    p.y[j] = p.y[p.n];
    p.vx[j] = p.vx[p.n];
    p.vy[j] = p.vy[p.n];
    p.mass[j] = p.mass[p.n];
    p.heat[j] = p.heat[p.n];
    p.ax[j] = p.ax[p.n];
    p.ay[j] = p.ay[p.n];

    // Restore the invariant at the vacated slot. `mass` is the one the RFC
    // names and the one that matters physically; `ax`/`ay` matter because
    // Phase A writes only [0..n] while Phase B streams to n_padded, so these
    // would otherwise keep integrating a dead particle's last acceleration.
    p.mass[p.n] = 0;
    p.x[p.n] = 0;
    p.y[p.n] = 0;
    p.vx[p.n] = 0;
    p.vy[p.n] = 0;
    p.heat[p.n] = 0;
    p.ax[p.n] = 0;
    p.ay[p.n] = 0;

    p.n_padded = platform.alignUp(p.n, L);
}

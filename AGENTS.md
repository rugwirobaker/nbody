# AGENTS.md

Instructions for coding agents working in this repository. Read this before
touching code; read [`docs/RFC-001.md`](docs/RFC-001.md) before touching
physics, and [`docs/RFC-002.md`](docs/RFC-002.md) before touching merging.

## What this project is

A 2D gravitational N-body simulation implemented **twice over the same
algorithm** — a scalar AoS baseline and a SIMD SoA kernel — so that the
measured speedup between them is honest. The RFCs in `docs/` are the
specification and the authority; their **normative** rules (marked *must* /
*must not*) are requirements, not preferences. When code and RFC disagree, the
RFC wins unless the user says otherwise. Cite RFC steps/sections in comments
rather than restating derivations.

Spec changes go in a **new numbered RFC**, never an edit to an old one, so
reasoning that was superseded stays readable. The amended section keeps its
original text and gains a pointer; the later amendment wins. `RFC-002` amends
RFC-001's merge trigger this way.

Three packages, dependencies pointing one way:

| Package | Depends on | Rule |
| --- | --- | --- |
| `nbody` (`src/`) | — | Renderer-free and I/O-free. No printing, no files, no GL. |
| `nbody-bench` (`bench/`) | `nbody` | Measurement only. |
| `nbody-viz` (`viz/`) | `nbody` | Wasm module + hand-written WebGL2. Browser only; the Zig side draws nothing. See [`docs/graphics-primer.md`](docs/graphics-primer.md) before touching the renderer. |

## Commands

```sh
zig build test                          # library tests — run before claiming anything works
zig build bench -Doptimize=ReleaseFast  # the harness; any other mode measures spills, not the algorithm
```

Zig 0.16.0. Verify claims by building, not by memory.

## Zig 0.16 API notes

0.15/0.16 broke a lot of `std`. These are verified in this repo — do not
"correct" them back to the pre-0.15 spellings:

- **Entry point:** `pub fn main(init: std.process.Init) !void`. I/O needs an
  explicit `Io` instance, which arrives as `init.io`; long-lived allocation
  comes from `init.arena.allocator()`; arguments from
  `init.minimal.args.toSlice(arena)`.
- **Files moved:** `std.Io.File`, not `std.fs.File`. Buffered stdout is
  `var w: std.Io.File.Writer = .init(.stdout(), io, &buf);` then
  `const out = &w.interface;` … `try out.flush();` — the flush is not optional.
- **Alignment is an enum:** `allocator.alignedAlloc(f32, .@"64", n)` takes
  `std.mem.Alignment`; convert with `.toByteUnits()`.
- **Build graph:** `b.addExecutable(.{ .name = …, .root_module = b.createModule(…) })`
  and `b.addTest(.{ .root_module = mod })`. Modules carry `target`/`optimize`.
- **Containers are unmanaged:** `std.ArrayList(T) = .empty`, `list.deinit(gpa)`,
  `list.append(gpa, x)`.
- **Random:** `std.Random.DefaultPrng.init(seed)`, then `prng.random()` and
  `r.float(f32)` / `r.intRangeLessThan(...)`.
- **`std.time.Timer` is gone** (`std/time.zig` is now only unit constants), so
  RFC §2.5's spelling no longer compiles. Monotonic timing goes through the
  `Io` clock: `const t0 = std.Io.Timestamp.now(io, .awake);` … then
  `t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds()`, which
  returns `i96`. `.awake` is `CLOCK_UPTIME_RAW` on macOS.

## Rules that protect the experiment

Violating any of these silently invalidates the benchmark or the physics. They
are the reason this project exists, so they outrank tidiness, cleverness, and
performance.

- **Never enable fast math.** No `@setFloatMode(.optimized)` anywhere, and no
  reordering of float reductions by hand. Strict FP semantics are what stop
  LLVM from auto-vectorizing the scalar baseline; a secretly-vectorized
  "scalar" build makes the whole comparison a lie. Re-check the disassembly
  when Phase A changes.
- **What the baseline's disassembly actually shows** (aarch64, ReleaseFast,
  verified 2026-08-19). The protection worked: `computeAccelerations` consumes
  **one source per iteration**, with a scalar `fsqrt s` and `fdiv s` per pair —
  strict FP kept LLVM from reordering the `ax +=` reduction across iterations,
  so the n² traversal is intact. LLVM does SLP-pair the two independent
  accumulators, emitting `fsub.2s` / `fmul.2s` / `faddp.2s` / `fadd.2s` for the
  x and y component arithmetic. Each accumulator is still summed in order, so
  this is what the natural code compiles to, and Zig 0.16 exposes no flag to
  disable the vectorizer anyway. The consequence for speedups: the baseline
  already gets 2-wide on the cheap component math. `docs/disassembly.md`
  explains how to read that output and what to look for.

  Part 3 confirmed this from the other side. The SIMD kernel at `L` = 1 runs
  **0.74×** the baseline on NEON and 0.85× on AVX2 — slower — because the
  baseline is not 1-wide. Measured against that 1-wide floor the native kernels
  reach 3.97× of a possible 4 (NEON) and 6.90× of a possible 8 (AVX2), close to
  the ceiling; measured against the SLP-paired baseline they reach **~2.9× on
  NEON and ~5.8× on AVX2** — the same 0.725 per lane on both, which is the
  result worth knowing. **Quote the figure for the machine you measured on**,
  against the SLP-paired baseline rather than the 1-wide floor, because that is
  the comparison against code someone would actually write. Run-to-run
  variation is a percent or two, so do not quote a third significant digit.
- **What the SIMD kernel's disassembly shows** (aarch64, ReleaseFast, verified
  2026-08-19). Exactly what RFC §3.3c specifies: **three vector loads (`ldr q`)
  and zero stores per inner iteration**, all fourteen operations in `.4s` form
  including `fsqrt.4s` and `fdiv.4s`, `j += 4`, and no stack traffic — every
  intermediate stayed in registers. The `@splat`s are hoisted out of the inner
  loop entirely, so each broadcast happens once per row as designed. If a future
  change introduces `str`/`ldr` against `sp` inside that loop, the register
  allocator is spilling and the §3.3c contract is broken.
- **Both builds run the same algorithm.** Same force law, same integration
  order, same **ordered**-pair n² traversal. Do not apply the pairwise-symmetry
  halving (RFC Step 6) to the baseline, even though it is a real optimization —
  it is explicitly rejected for comparison fairness.
- **The baseline stays scalar AoS.** It is the experimental control, not code
  to be improved.
- **Nothing new goes inside `Particle`.** The AoS row is 24 bytes and the
  scalar baseline streams it, so adding a field — an id, a radius, a colour —
  widens it and moves the memory traffic the measured speedup rests on. Derived
  per-tick state already lives outside it (`ax`, `ay`, `radii`, RFC §2.1);
  anything else a caller needs goes in a parallel array on `Sim`, which Phase A
  never reads and which therefore costs it nothing. This is why `nbody-viz`
  recovers particle identity for trails from the packed render buffer instead
  of being handed ids: the obvious fix would have quietly rewritten the
  benchmark.
- **Frozen snapshot (RFC Step 8).** Phase A reads positions/masses and writes
  only `ax[]`/`ay[]`. No particle state moves until every acceleration in the
  tick exists. Never fuse Phase A and Phase B.
- **Semi-implicit Euler (RFC Step 5).** Velocity first, then position using the
  *new* velocity. The reverse ordering costs the same and pumps energy in.
- **No `if (i != j)` in the inner loop.** Softening makes the self term exactly
  zero; the branch's absence is intentional and load-bearing for the lanes.
- **Merging's energy books do not close** (verified 2026-08-19). RFC §2.5 test
  (d) claims `KE + PE + Σheat` stays constant across merges. Step 10 banks the
  destroyed *kinetic* energy into `heat` and leaves out the merged pair's
  mutual potential, which vanishes with the pair and steps total energy **up**
  at every merge. RFC-002 §7.2 measures the size: a median of 0.019 % of the
  total per merge, a 90th percentile of 0.232 %, and a maximum of 4.5 %. The
  distribution is heavy-tailed, so quote the spread — no single figure
  describes it. The spec's wording overstates the invariant. Do not loosen a
  tolerance until the test passes, and do not fold the pair potential into
  `heat` without the user's say-so, because Step 10 is normative about what
  `heat` holds. The tests split the claim instead: momentum, total mass and
  determinism are asserted sharply across merges; energy flatness only on
  merge-free ticks (which is why `mergeCollisions` returns a count); and the
  full ledger is proved exactly in a two-body case where the vanished term is
  computable.
- **Padding invariant (RFC §3.2).** In the SoA build every slot at index ≥ `n`
  has mass 0. Any code path that shrinks `n` (i.e. merging's swap-remove) must
  re-zero the vacated slot, or an invisible ghost particle keeps pulling.
  Likewise, `ax`/`ay` beyond `n` must be zero if Phase B streams to `n_padded`.
- **Lane width comes from `std.simd.suggestVectorLength(f32)`** (4 here on
  NEON, 8 on AVX2). Hard-coding 8 is non-compliant; kernels take `L` as a
  comptime parameter.
- **Benchmarks:** `ReleaseFast`, `merging = false`, timing around **Phase A
  only**, in ns/tick, with the seed and full config recorded. Never FPS.
- **Scalar-vs-SIMD comparison is tolerance-based, never bitwise** — `@reduce`
  reorders the summation, and the system is chaotic. Compare over a short
  horizon plus the conservation invariants.

## Working style here

- Update the README status checklist as parts land. Nothing is claimed as
  working before its test passes; report failures with the output.
- Prefer conservation-law tests (momentum, centre-of-mass drift, energy,
  determinism) over eyeballing trajectories.
- Repository hygiene comes first on new work: scaffolding, README, and build
  wiring land before implementation code.
- Do not commit unless asked.

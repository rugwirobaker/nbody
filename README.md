# nbody

A 2D gravitational N-body simulation in Zig, implemented **twice over the same
algorithm** — once as an honest scalar baseline, once with SIMD — so the
speedup between them means something.

The physics, the normative rules, and the measurement methodology are specified
in [`docs/RFC-001.md`](docs/RFC-001.md). This README is the map; the RFC is the
contract.

## Why

Micro-optimization demos usually cheat: they compare a deliberately bad scalar
loop against a tuned vector one. This project's rule is the opposite. Both
builds implement the *identical* specified algorithm — same force law, same
integration order, same O(n²) ordered-pair traversal — and differ only in
memory layout (AoS → SoA) and instruction width. Everything that would make the
scalar side look worse than a reasonable engineer's first attempt is
off-limits, and every performance number is Phase-A nanoseconds per tick under
`-Doptimize=ReleaseFast`, never FPS.

## Layout

| Module | Depends on | What it is |
| --- | --- | --- |
| `nbody` | — | The core library: AoS + SoA layouts, scalar + SIMD kernels, `tick()`, presets, merging, conserved properties. Renderer-free, I/O-free. |
| `nbody-bench` | `nbody` | The measurement harness: ns/tick vs n, both kernels, same seed. |
| `nbody-viz` | `nbody`, raylib | The demo client. Not started — physics and measurement come first. |

Dependencies point one way. The core library never learns about a screen.

## The algorithm in one screen

Each tick advances simulated time by a fixed `dt` in phases:

- **Phase A — accelerate.** For every particle `i`, sum over every source `s`:
  `a = G·m_s·(x_s − x_i) / (d² + ε²)^1.5`. Reads positions and masses only;
  writes to scratch arrays `ax[]`, `ay[]`. **No particle state moves.**
- **Phase B — integrate.** Velocity first, then position using the *new*
  velocity (semi-implicit Euler).
- **Phase C — merge** (demo only, off for benchmarks). Pairs closer than
  `d_merge` become one particle whose mass, position and velocity are forced by
  conservation of mass, centre of mass, and momentum.

Three details carry more weight than they look:

- **Softening (`ε²`)** bounds the force law, makes the self-interaction term
  exactly zero (so the inner loop needs no `if (i != j)` branch), and
  guarantees the denominator is never zero — which is what later makes
  zero-mass SIMD padding safe.
- **The frozen snapshot.** Every acceleration in a tick comes from one
  position snapshot. This is what makes momentum conservation exact, runs
  reproducible, and lane reordering a legal transformation rather than a
  different simulation.
- **Ordered pairs, not unordered.** Phase A does the full n² work even though
  half of it is redundant by Newton's third law. Halving it would make the
  scalar baseline a different algorithm and the comparison a lie.

Correctness is defined by conservation laws, not by eyeballing: momentum
constant across merges, centre of mass drifting in a straight line, orbits that
don't spiral outward, bitwise-reproducible runs from a seed.

## Building

Requires Zig 0.16.0.

```sh
zig build test                          # library tests: the RFC's acceptance tests
zig build bench -Doptimize=ReleaseFast  # the measurement harness
```

Benchmarks in any mode other than `ReleaseFast` measure register spills rather
than the algorithm, so the build warns about it.

## What the baseline compiles to

Checked once against the disassembly, because "the scalar baseline is really
scalar" is the assumption the whole project rests on
(`objdump -d zig-out/bin/nbody-bench`, aarch64 ReleaseFast):

- The inner loop consumes **one source per iteration**, with one scalar
  `fsqrt s` and one scalar `fdiv s` per pair. Strict FP stopped LLVM from
  reordering the `ax +=` reduction across iterations, which is exactly the
  protection it is there for — the n² traversal is intact and un-widened.
- LLVM *did* pair the two independent accumulators, emitting 2-wide NEON
  (`fsub.2s`, `fmul.2s`, `faddp.2s`, `fadd.2s`) for the x and y component
  arithmetic. That is not a reordering of any single reduction, so nothing
  forbids it, and it is what the natural code honestly compiles to — but it
  means the baseline already gets 2-wide on the cheap operations, which is
  worth remembering when reading the eventual speedup.

## A note on lane width

The SIMD kernel is parameterized by `std.simd.suggestVectorLength(f32)` — 8 on
AVX2, 16 on AVX-512, **4 on Apple Silicon / NEON**. On a 4-wide target the
theoretical ceiling is 4×, so the RFC's "4–7× is success" figure (written for
AVX2) becomes roughly 2.5–3.5× here. Hard-coding 8 is non-compliant; the
harness sweeps lane width as a labeled experiment instead.

## Status

Nothing below is claimed until its test passes.

- [x] Part 2 — scalar baseline (AoS): `computeAccelerations`, `integrate`,
      `tick`
- [x] Seeding — `disk` and `keplerian` presets, uniform-area radii, tangential
      velocities, net momentum zeroed
- [x] Acceptance tests (a)–(e) in their merging-off forms: two-body asymmetry,
      momentum, CoM drift, symplectic energy, determinism
- [x] Benchmark: Phase-A ns/tick sweep vs n
- [x] Phase C — merging (RFC §2.6): `mergeCollisions`, `mergePair`, greedy
      restart, swap-remove
- [x] Tests (b)–(e) in their merging-on forms: momentum and total mass across
      merge events, CoM drift, determinism, and the single-merge energy ledger
- [ ] Part 3 — SoA layout, SIMD kernels, padding invariants
- [ ] Scalar-vs-SIMD short-horizon agreement + two-kernel benchmark sweep
- [ ] `nbody-viz` (raylib)

One caveat on what the green suite proves. RFC test (d) says
`KE + PE + Σheat` holds constant across merges; it doesn't, quite. Step 10
banks the destroyed *kinetic* energy into `heat`, but the merged pair's mutual
potential simply vanishes along with the pair, stepping total energy up at each
merge — about 0.3 % of the total per merge at default config. The
implementation follows Step 10 exactly and the tests split the claim
accordingly: momentum, total mass and determinism are asserted sharply across
merges, energy flatness is asserted only on merge-free ticks, and the full
ledger is proved exactly in a two-body case where the vanished term is
computable.

Reference: Core Dumped, *"Why compilers can't optimize this"* — used for
notation and motivation; all results here derive from the RFC's own first
principles.

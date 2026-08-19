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

## Results

One run on an aarch64-macos laptop, verbatim. The harness prints the seed and
the full config alongside every number, because a measurement without them is
not reproducible and therefore not a measurement.

```sh
$ zig build bench -Doptimize=ReleaseFast
nbody-bench — scalar baseline vs SIMD (RFC Parts 2–3)

  target      aarch64-macos
  optimize    ReleaseFast
  lane count  4 (f32; scalar baseline does not use it)
  seed        0xC0FFEE
  preset      disk, merging false
  g/dt/eps2   0.0005 / 0.001 / 0.0005
  mass/radius [0.5, 1.5] / 1, jitter 0.03

  Phase A only, ns/tick. Both kernels seeded from the same AoS sim.

         n   scalar ns/tick     simd ns/tick  simd/pair    speedup
  --------  ---------------  ---------------  ---------  ---------
       256          71788.0          24885.8      0.380      2.88x
       512         287241.2         100851.4      0.385      2.85x
      1024        1158602.5         402368.1      0.384      2.88x
      2048        4619606.1        1595205.0      0.380      2.90x
      4096       18352333.5        6338713.5      0.378      2.90x
      8192       73921866.8       25418516.6      0.379      2.91x
     16384      296128275.0      102773866.6      0.383      2.88x

  Lane-width experiment (labeled, non-normative), n = 4096:

         L     simd ns/tick    speedup
  --------  ---------------  ---------
         1       25168366.8      0.74x
         2       12680729.3      1.46x
         4        6361083.5      2.91x  <- native
         8        6542671.9      2.83x
        16        6240907.1      2.97x

  ns/pair = ns/tick / n². Flat across n means the kernel is
  compute-bound — and on this hardware it is flat all the way:
  no cache cliff appears even at n = 16384, where each row
  streams ~390 KB of AoS particles. The access pattern is pure
  sequential and the per-pair sqrt+div leaves the prefetcher
  ample time, so memory never becomes the limiter.
```

**~2.9×, flat across a 64× range in n.** Same algorithm, same force law, same
ordered-pair n² traversal — only the memory layout and the instruction width
differ. Run-to-run variation is a percent or two, so 2.9× is the figure; a
third significant digit here would be false precision.

### Why ~2.9× and not 4×

The lane-width experiment answers this, and it is the most interesting table
the harness prints. Each doubling of `L` buys almost exactly 2× — 1.99×, then
1.99× again — until the hardware runs out of lanes at 4, after which widening
buys nothing, because `@Vector` just splits the work across more registers.
That is as clean as this kind of scaling ever gets.

The striking row is `L` = 1, at **0.74× — slower than the "scalar" baseline**.
That is not a defect; it is the SLP finding above, measured from the other
side. The baseline is not truly 1-wide, because LLVM pairs its `ax`/`ay` chains
into 2-wide NEON, so a genuinely 1-wide kernel loses to it by about the factor
you would expect. Against that honest 1-wide floor, the native kernel runs
**3.96×** — essentially the theoretical ceiling for four lanes.

Both numbers are true and they measure different things. 3.96× is what the
vectorization achieves; ~2.9× is what it achieves *over code a reasonable
engineer would actually write*. The second is the one worth quoting, and
publishing only the first is exactly the cheat this project was built to avoid.

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
- [x] Part 3 — SoA layout with `n_padded`, `@Vector` kernels parameterized by
      `L`, padding invariants including the ghost-particle trap
- [x] Scalar-vs-SIMD short-horizon agreement + two-kernel benchmark sweep and
      the lane-width experiment
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

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

Two machines, same source, same seed:

| | native `L` | speedup | speedup / `L` |
| --- | --- | --- | --- |
| aarch64-macos, NEON | 4 | 2.90× | **0.725** |
| x86_64-linux, AVX2 | 8 | 5.80× | **0.725** |

The efficiency is identical to three digits on two unrelated
microarchitectures — different vendor, different ISA, different memory system,
and the kernel turns lane width into throughput at exactly the same rate. That
agreement is the best evidence available that the comparison measures what it
claims to; if either side were contaminated by layout luck or a compiler
artifact, these would not line up.

It also settles what 2.9× means. RFC §3.5 calls **"4–7× at L = 8"** success,
and AVX2 delivers 5.80× — the NEON figure was never a shortfall, just the same
result seen through half the lanes.

### The full run, verbatim

The harness prints the seed and the full config alongside every number, because
a measurement without them is not reproducible and therefore not a measurement.

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
       256          72772.7          25088.5      0.383      2.90x
       512         289154.2         100174.8      0.382      2.89x
      1024        1153092.4         397079.7      0.379      2.90x
      2048        4589706.4        1583431.0      0.378      2.90x
      4096       18469597.2        6318726.6      0.377      2.92x
      8192       74053266.6       25320383.8      0.377      2.92x
     16384      296199633.0      102383433.2      0.381      2.89x

  Lane-width experiment (labeled, non-normative), n = 4096:

         L     simd ns/tick    speedup
  --------  ---------------  ---------
         1       25169216.8      0.74x
         2       12659286.4      1.46x
         4        6341906.3      2.92x  <- native
         8        6503198.0      2.85x
        16        6226622.5      2.97x

  ns/pair = ns/tick / n². Flat across n means compute-bound;
  a rise at large n is the working set outgrowing cache.
  Per row, AoS streams 24n bytes and SoA 12n.

    scalar  best 1.094 at n=2048, 1.103 at n=16384  (+0.8%)
    simd    best 0.377 at n=4096, 0.381 at n=16384  (+1.3%)

  A few percent is run-to-run noise or the first hint of cache
  pressure; a large jump is the cliff itself.
```

And the same sweep on x86_64-linux (AVX2, `L` = 8):

```
         n   scalar ns/tick     simd ns/tick  simd/pair    speedup
  --------  ---------------  ---------------  ---------  ---------
       256         154041.4          28282.3      0.432      5.45x
       512         614398.7         109220.9      0.417      5.63x
      1024        2444725.5         429935.0      0.410      5.69x
      2048        9816729.5        1705079.1      0.407      5.76x
      4096       39320603.2        6791690.7      0.405      5.79x
      8192      157165899.8       27094294.6      0.404      5.80x
     16384      633980640.8      111062193.4      0.414      5.71x

         L     simd ns/tick    speedup
  --------  ---------------  ---------
         1       46823148.2      0.85x
         2       23377437.0      1.70x
         4       12201108.3      3.25x
         8        6786553.0      5.84x  <- native
        16        6723077.3      5.90x
```

Same algorithm, same force law, same ordered-pair n² traversal on both — only
the memory layout and the instruction width differ. Run-to-run variation is a
percent or two, so quote 2.9× and 5.8×; a third significant digit would be
false precision.

### Reading the lane-width experiment

Two things fall out of it, and the second only became visible with a second
machine.

**`L` = 1 is slower than the "scalar" baseline** — 0.74× on NEON, 0.85× on
AVX2. That is not a defect; it is the SLP finding above, measured from the
other side. The baseline is not truly 1-wide, because LLVM pairs its `ax`/`ay`
chains into 2-wide vectors, so a genuinely 1-wide kernel loses to it. Against
that honest 1-wide floor the native kernels reach **3.97× of a possible 4**
(NEON, 99 % of ceiling) and **6.90× of a possible 8** (AVX2, 86 %).

Both framings are true and they measure different things: the larger number is
what vectorization achieves, the smaller is what it achieves *over code a
reasonable engineer would actually write*. The smaller one is worth quoting,
and publishing only the larger is exactly the cheat this project was built to
avoid.

**The two ISAs scale differently below native width**, which is where the AVX2
shortfall comes from:

| doubling | NEON | AVX2 |
| --- | --- | --- |
| 1 → 2 | 1.98× | 2.00× |
| 2 → 4 | 1.99× | 1.92× |
| 4 → 8 | — | **1.80×** |

NEON scales essentially perfectly to its native 4. AVX2 decays, so widening to
256 bits returns less than 2× even before running out of lanes. That is the
`sqrt`/`div` signature RFC §3.5 predicted: `vsqrtps`/`vdivps` on `ymm` do not
have double the per-element throughput of the `xmm` form on most parts, and
those two operations dominate the fourteen. Past native width both flatten —
`@Vector` just splits the work across more registers.

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

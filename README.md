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
| `nbody-viz` | `nbody` | The demo client: a `wasm32-freestanding` build of the library plus a hand-written WebGL2 renderer. Runs both kernels side by side. |

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
zig build viz                           # the demo into zig-out/web
```

Benchmarks in any mode other than `ReleaseFast` measure register spills rather
than the algorithm, so the build warns about it. The demo always builds
`ReleaseFast` for the same reason: it reports Phase-A ns/tick, and a Debug wasm
would report spills.

The demo needs to be served over HTTP, because `fetch` refuses `file://`:

```sh
zig build viz && python3 -m http.server -d zig-out/web
```

## The demo

`zig build viz` produces three files and no dependencies: a 25 KB
`wasm32-freestanding` build of the library, one HTML page, and one JavaScript
file holding the WebGL2 renderer. If the renderer is unfamiliar territory,
[`docs/graphics-primer.md`](docs/graphics-primer.md) explains what a GPU is
asked to do and walks through this one. There is no emscripten in the toolchain and
nothing in `build.zig.zon`. The library being renderer-free and I/O-free is
what makes that possible — it imports `std.debug.assert`, `std.Random`, and an
`Allocator`, all of which exist on freestanding wasm.

The build targets `wasm32-freestanding+simd128`, which matters: without
`simd128`, `std.simd.suggestVectorLength(f32)` returns null and the SIMD kernel
falls back to an 8-wide vector that wasm emulates. With it the target reports
4, the same width as NEON, so the page runs the real Part 3 kernel.

Three modes, chosen in the page or the URL:

| Mode | What it shows |
| --- | --- |
| `base` | the scalar AoS baseline alone |
| `simd` | the SoA vector kernel alone |
| `stacked` | both at once, base left and simd right |

Both worlds are seeded through `Particles.fromAoS`, so they start
bit-identical and any divergence you see between the panels is `@reduce`
reordering amplified by a chaotic system (RFC §3.5), not two seeding paths.

The configuration lives in the URL —
`?n=4096&seed=0xC0FFEE&preset=disk&merging=1&mode=stacked` — and the controls
write to it, so a link reproduces a run exactly. That is the same reason
`nbody-bench` prints the seed and full config above every table.

The page opens paused. Space or the button starts it.

Each panel reports **Phase-A ns/tick**, never FPS (RFC §2.5 rule 2), and a
simulated clock. The clock is where the comparison shows: each world is given
a share of wall clock per frame rather than a tick quota, so the faster kernel
fits more ticks into its share and its clock pulls ahead. At the default
n = 1000, simd finishes the ten ticks §2.4 allows while base does not, and the
two clocks separate about 2.5×.

That default is chosen, not arbitrary. Below n ≈ 750 neither kernel is stressed
and the panels run in lockstep; above n ≈ 1400 both are starved and the whole
thing crawls at hundreds of real seconds per orbit.

### Two demo constants differ from `Config`'s defaults

Both were measured rather than guessed, and both exist because the library
defaults make merging destroy the simulation within a second.

| | `Config` | demo | why |
| --- | --- | --- | --- |
| `d_merge2` | 5e-4 | 5e-6 | The default merge radius is 0.0224 against a mean nearest-neighbour spacing of 0.0198, so at t = 0 nearly every particle is already touching one. Its merge discs cover 25 % of the disk. |
| `dt` | 1e-3 | 2.5e-4 | The default resolves ~6,300 ticks per orbit. That is not enough to integrate the close encounters merging creates. |

Measured over 8,000 ticks at the library defaults with merging on, 85 % of the
population merges in the first 500 ticks and total energy runs from −992 to
positive: the system crosses from bound to unbound and the survivors leave.
RFC Step 10 banks a merged pair's destroyed kinetic energy into `heat` and the
pair's mutual potential simply vanishes, so every merge steps total energy up —
an effect `AGENTS.md` already records at ~0.3 % per merge, compounded here
about 1,975 times.

At the demo's constants the same run holds energy to within 7 %, `n` decays
smoothly instead of collapsing, and the disk stays framed:

```
                        tick      n  merges   E/E0  r_p90
library defaults        4000     25    1975  0.323  2.459
demo constants          4000    306    1694  0.895  0.831
```

## What the baseline compiles to

Checked once against the disassembly, because "the scalar baseline is really
scalar" is the assumption the whole project rests on
(`objdump -d zig-out/bin/nbody-bench`, aarch64 ReleaseFast):

- The inner loop consumes **one source per iteration**, with one scalar
  `fsqrt s` and one scalar `fdiv s` per pair. Strict FP stopped LLVM from
  reordering the `ax +=` reduction across iterations, which is exactly the
  protection it is there for — the n² traversal is intact and un-widened.
- LLVM pairs the two independent accumulators, emitting 2-wide NEON
  (`fsub.2s`, `fmul.2s`, `faddp.2s`, `fadd.2s`) for the x and y component
  arithmetic. Each accumulator is still summed in order, so this is what the
  natural code compiles to. The baseline therefore gets 2-wide on the cheap
  operations, which sets the floor the SIMD kernel is measured against.

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

         n   scalar ns/tick  scalar/pair     simd ns/tick  simd/pair    speedup
  --------  ---------------  -----------  ---------------  ---------  ---------
       256          72307.4        1.103          25028.6      0.382      2.89x
       512         287764.8        1.098         100134.6      0.382      2.87x
      1024        1151873.6        1.099         398832.8      0.380      2.89x
      2048        4609210.2        1.099        1587517.1      0.378      2.90x
      4096       18355020.8        1.094        6336130.3      0.378      2.90x
      8192       73984058.4        1.102       25392908.2      0.378      2.91x
     16384      296170150.2        1.103      102156516.8      0.381      2.90x

  Lane-width experiment (labeled, non-normative), n = 4096:

         L     simd ns/tick    speedup
  --------  ---------------  ---------
         1       25115516.8      0.74x
         2       12601140.6      1.47x
         4        6332427.1      2.93x  <- native
         8        6494375.1      2.86x
        16        6227330.9      2.98x

  ns/pair = ns/tick / n², and it is the shape that matters.
  Flat across n means compute-bound; a rise at large n is the
  working set outgrowing cache (AoS streams 24n bytes per row,
  SoA 12n). Small n carries real per-row overhead instead: one
  @reduce per row, amortized over fewer sources.

  Each figure is the fastest of 3 runs, and run-to-run variation
  is still 1–2 %. A cache cliff would be a large jump, not a few
  percent — one sweep cannot resolve anything smaller.
```

And the same sweep on x86_64-linux (AVX2, `L` = 8). This excerpt predates
the `scalar/pair` column, so it shows the tables only:

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

### How to read the table

Each kernel gets three numbers. They are related: `ns/tick` divided by `n²` is
`ns/pair`. Each answers a different question.

Gravity superposes, so every particle sums over every source and the algorithm
is O(n²) (RFC Step 7). That makes `n²` the unit of work. Dividing by it gives
the cost of the implementation per unit of work, which should stay constant as
`n` grows. A rising `ns/pair` means something outside the arithmetic is costing
time.

| Column | What it tells you |
| --- | --- |
| `ns/tick` | How long one Phase A takes. Use it for budgeting. |
| `ns/pair` | Whether cost per unit of work stays constant as `n` grows. |
| `speedup` | Which kernel is faster at this `n`. |

`speedup` is a ratio of two times measured on the same clock, so the clock
cancels and the figure is comparable across machines. That is how the NEON and
AVX2 runs can be put side by side. `ns/pair` compares one kernel across `n`;
`speedup` compares two kernels at one `n`.

**Three regimes appear in `ns/pair`.** Small `n` sits high because each row pays
for two `@reduce` calls regardless of how many sources it summed. The middle is
the plateau: pure compute cost, and the figure to quote after a kernel change.
Large `n` can drift up as the working set outgrows cache. On the run above that
drift is about one percent, too small for a single sweep to confirm.

**The plateau converts to cycles.** At roughly 4 GHz, 1.10 ns/pair is 4.4
cycles for the scalar loop's fourteen float operations, including a `sqrt` and
a `div`. 0.38 ns/pair is 6.1 cycles for the SIMD loop's four pairs. Divided
into the disassembled instruction counts, both come to about 3.2 instructions
per cycle. That figure says each kernel is limited by issue rate. A much lower
figure, say 0.5, would indicate stalls on memory.

**The demo budget follows from `ns/tick`.** `dt` is fixed at 1 ms, so 60 fps in
real time needs about 17 ticks per frame, and RFC §2.4's clamp caps it at 10.
One SIMD tick costs 1.6 ms at n = 2048, 6.3 ms at n = 4096, and 25 ms at
n = 8192. Real time therefore holds to about n = 2048. Above that the
simulation runs in slow motion with the clamp engaged. A larger `dt` buys
headroom and costs integration accuracy.

### Reading the lane-width experiment

Two things fall out of it. The second only became visible with a second
machine.

**`L` = 1 is slower than the "scalar" baseline** — 0.74× on NEON, 0.85× on
AVX2. This is the SLP finding above, measured from the other side: LLVM pairs
the baseline's `ax`/`ay` chains into 2-wide vectors, so a genuinely 1-wide
kernel loses to it. Measured against that 1-wide floor the native kernels reach
**3.97× of a possible 4** (NEON, 99 % of ceiling) and **6.90× of a possible 8**
(AVX2, 86 %).

The two figures measure different things. The larger is what vectorization
achieves against a 1-wide kernel; the smaller is what it achieves against code
a reasonable engineer would write. Quote the smaller one.

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
- [x] `nbody-viz` — wasm + WebGL2, with `base` / `simd` / `stacked` modes and
      Phase-A ns/tick reported per panel

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

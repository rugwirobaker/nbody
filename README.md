# nbody

A 2D gravitational N-body simulation in Zig, implemented **twice over the same
algorithm** — once as an honest scalar baseline, once with SIMD — so the
speedup between them means something.

The physics, the normative rules, and the measurement methodology are specified
in [`docs/RFC-001.md`](docs/RFC-001.md), amended by
[`docs/RFC-002.md`](docs/RFC-002.md) on merging. This README is the map; the
RFCs are the contract.

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
`nbody-bench` prints the seed and full config above every table. The view is
part of that record: `zoom0`/`cx0`/`cy0` and `zoom1`/`cx1`/`cy1` restore each
panel's framing along with the run. A continuous gesture coalesces its writes
rather than making one per event, because `history.replaceState` is rate-limited
and a drag that spent the budget would leave the address bar quietly disagreeing
with the screen.

The page opens paused. Space or the button starts it. Drag the field to pan and
use the wheel to zoom about the pointer; double-click restores that panel's
opening view.

Each panel carries its own camera, and a gesture moves only the panel it lands
in. A shared one would only be worth having if the panels showed the same thing.
They do not: they start bit-identical and then diverge, so within a few merges
the same screen position is a different part of a different arrangement in each.
What the page compares is throughput, which the HUD reports in ns/tick — the
pictures are not in correspondence, and binding them to one view would only stop
you inspecting either. For the times you do want both on the same window,
shift-double-click propagates one panel's view to the other, and **reset view**
returns both to the opening framing.

There is no zoom control. Each panel's HUD carries a read-only gauge showing
where its own view sits between 0.04 R and 63 R, and the pointer is the only
thing that moves it.

### How the comparison reads on screen

Every frame owes ten ticks of physics — RFC §2.4's accumulator and its clamp —
and a panel publishes a picture once it has finished them. Each picture
therefore carries the same amount of simulated time, and a kernel that cannot
keep up publishes fewer of them per second. That rate is what the eye reads as
performance, and it is throughput in another spelling: pictures per second
against a fixed quantity of physics per picture.

Alone on screen a kernel gets the whole frame. Falling behind then shows up as
a falling frame rate, the same way it does in a native window. Under `stacked`
the two split one frame, so neither can freeze the page and each publishes at
its own rate:

```
                 alone            stacked
       n      base    simd      base    simd
    1000        54      60        28      57      <- default
    1500        28      57        12      25
    2000        11      27         6      12
```

Pictures per second, measured over the first two seconds of a run. The default
sits where `stacked` separates most clearly. Switching to a solo mode and
raising n to 1500 shows the other half of it: base collapses to 28 while simd
still holds the display's ceiling.

The panels also report **Phase-A ns/tick**, which is the measurement — RFC §2.5
rule 2 keeps the reported figure free of anything the renderer does — and a
simulated clock, which diverges at the same ratio for the same reason.

### What you are looking at

Three channels, each carrying one quantity, none of them overlapping:

| | shows | how |
| --- | --- | --- |
| Size | mass | `r(m) = k·√m` — the same radius the merge rule tests against |
| Colour | temperature | `heat / mass`, pale blue through white to orange |
| Brightness | temperature, and crowding | hotter bodies burn brighter; overlapping ones sum |
| Trails | nothing | one pixel wide, one colour, fading with age |

Size is the merge cross-section (RFC-002 §1.1), so two discs at the same
overlap are equally close to merging whatever they weigh. True radii are a
fraction of a pixel, so bodies are drawn **16× life size** above a 1.5 px
floor — the same factor for every body, which is what keeps the comparison
between them honest.

Colour is temperature rather than heat, because heat pools when bodies merge
and spans two decades; dividing by mass gives the intensive quantity a ramp can
use. A merge banks the kinetic energy it destroys into `heat`, which then
decays, so a body flares orange and cools back to blue over the next second or
so. Warm is the hot end deliberately: additive blending already turns a crowd
of cold particles white, so **white means crowded and orange means hot**.

Trails are the exception that proves the rule: they are the one thing on screen
carrying no quantity at all. Width is one device pixel and colour is one
constant, deliberately, so a trail says only *this body was here* and never
competes with the three channels above. The 128 heaviest bodies get one, which
is every body once accretion has done its work and a bound on the cost before
that.

A point is added to each trail once per **published picture** rather than once
per frame, which fixes its length in simulated time: base publishes far fewer
pictures than simd, and a per-frame trail would show that as a difference in
the physics. A trail holds 1,024 points, so 10.2 s of simulated time, and each
point's opacity halves every 256 — it fades out rather than ending, which is
what makes it read as a wake instead of a wire. Trails are stored in world
coordinates, so a zoom leaves them attached to their bodies.

Trails composite rather than add, the one place they depart from everything
else on screen. Additive light sums, and a hundred crossing trails would climb
to white — which already means *bodies are crowded here*. Compositing caps a
pixel at the tint however many trails cross it, so the two signals stay
separate.

#### Trails without particle identity

A trail has to stay attached to one body, and the simulation deliberately has
no body identity: `mergePair` swap-removes, so a slot silently becomes a
different body. The renderer recovers the lineage from what it can already see,
because giving `Particle` an id would widen the AoS row from 24 bytes to 28 and
move the memory traffic the whole 2.9× measurement rests on.

Two signatures give a stale trail away, and both are needed because they catch
different failures. **Position**: the slot was overwritten by the last live
particle, which is somewhere else entirely. **Mass**: `mergePair` puts the
product in the lower slot whatever the masses are, so a speck can swallow a
giant and the product inherits the speck's trail — and since the two were
touching, the position barely moves and no position test can ever see it. A
trail is dropped when its body's step exceeds six times the median step of the
same picture, or when its mass more than doubles. The threshold is a multiple
of the median rather than a fixed distance because the median rises with `n`
(orbital speed goes as `√(G·M_enc)`, and enclosed mass goes as `n`), while the
*shape* does not: `p999/p50` measured between 2.98 and 3.38 in every run below.

Measured against ground truth — a probe replicating `mergeCollisions`'s scan so
it can track real identity through every swap-remove, with lineage following
the heavier body:

| | merges | detection | false positives | worst missed step |
| --- | --- | --- | --- | --- |
| disk, n = 500 | 435 | 100.00 % | 0.000 % | — |
| disk, n = 1000 | 909 | 99.29 % | 0.000 % | 0.036 |
| disk, n = 2000 | 1887 | 100.00 % | 0.000 % | — |
| disk, n = 4000 | 3869 | 98.16 % | 0.006 % | 0.080 |
| keplerian, n = 1000 | 915 | 100.00 % | 0.000 % | — |
| keplerian, n = 2000 | 1878 | 100.00 % | 0.000 % | — |

Separately at n = 1000, position alone catches 85.7 % and mass alone 13.6 %;
together they reach 99.3 %, so the two are very nearly disjoint. What survives
is bounded rather than merely rare: stale steps run to a median of 0.70 and a
maximum of 21.6 world units, and **every one of those is caught in every
configuration**. The residual one percent are all short-range — the worst
missed step anywhere was 0.080 world units, about 14 px — so the visible
failure is a small kink where a trail hands over to a body a few pixels away,
never a line whipped across the screen. 6× rather than 4× because a false
positive is the more visible failure: a trail that keeps truncating is a
constant annoyance, a missed short-range handover is a rare kink.

### The demo runs on library defaults

It no longer overrides anything. Both of the constants it used to carry —
`dt` at a quarter of the default, and a merge threshold a hundredth of it —
existed because the fixed merge threshold let bodies plunge to one tiny
distance before merging, and the integrator could not resolve the encounters
that produced. [RFC-002](docs/RFC-002.md)'s contact rule removes those
encounters: a body swallows its neighbour when their discs touch, well before
the plunge.

Measured at the restored timestep, the disk reaches a body holding 12.5 % of
its mass in 4,000 ticks rather than 16,000, with matching survivor counts at
every checkpoint and slightly better energy — four times sooner in wall clock,
so the accretion is something you watch rather than wait for.

The system does not stay put. Every merge leaks a little energy (RFC-001
Step 10, quantified in RFC-002 §7.2), so the survivors drift outward for as
long as you watch:

```
   real s    sim t       n    r_p50    r_p90   in 1.4R   in 2.8R
        0      0.0    1000     0.71     0.95      100%      100%
       33     10.0     103     2.40     5.18       24%       59%
      100     30.0      84     7.25    20.44        8%       18%
      200     60.0      77    14.54    40.99        5%       13%
```

The default view is 2.8 disk radii, which frames the accretion phase; the zoom
range reaches 63 R to follow what comes after.

What that table does not show is any translation. Seeding zeroes net momentum,
the integrator conserves it, and merging conserves it too, so the centre of mass
never moves: the numbers above are a cloud expanding about a stationary point,
not one travelling away from a fixed camera. That is why zooming out is enough
to keep the whole system in frame, and why resetting the view to the origin is
the same thing as recentring on the field.

Panning earns its place at the other end of the range. At full zoom the view is
0.04 disk radii wide, and without a movable centre the only structure you could
magnify is whatever happened to sit at the origin.

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
- [x] Contact merging ([RFC-002](docs/RFC-002.md)): bodies merge when their
      discs touch, `r(m) = k·√m`, so the size on screen is the size that
      merges
- [x] Trails — the recent path of the heaviest bodies, stored in world
      coordinates and drawn as lines, with the body each one belongs to
      recovered from the packed buffer rather than from particle identity
- [x] Pan and zoom about a point — drag to move the view, wheel to scale it
      about the pointer, so structure away from the origin can be magnified
      rather than only centred structure. One camera per panel, since the two
      runs diverge and the same screen position is not the same thing in both
- [ ] Close the energy ledger (RFC-003) — bank the merged pair's vanished
      potential and account radiated heat rather than deleting it, turning
      RFC-001 test (d)'s claim into the sharp invariant it was written as
- [ ] Stellar formation and life (RFC-004) — ignition, the main sequence, mass
      loss as light at `E = mc²`, and the endpoints a body reaches by mass:
      white dwarf, neutron star, black hole. A radiating body loses mass, and
      mass is the only thing the force law reads, so the whole cycle feeds back
      into the orbits
- [ ] Phase A on the GPU as a compute shader, with WebGPU in place of WebGL2
- [ ] Native desktop, paired with the WebGPU work since Dawn and wgpu reach
      both

One caveat on what the green suite proves. RFC test (d) says
`KE + PE + Σheat` holds constant across merges; it doesn't, quite. Step 10
banks the destroyed *kinetic* energy into `heat`, but the merged pair's mutual
potential simply vanishes along with the pair, stepping total energy up at each
merge. RFC-002 §7.2 measures it: a median of 0.019 % of the total per merge,
a 90th percentile of 0.232 %, and a maximum of 4.5 % — heavy-tailed enough
that no single figure describes it. Over a run those steps sum to +89 % of the
initial energy, very nearly cancelled by the −80 % that radiative cooling
removes, which is why the visible drift is small and why `E/E0` is weak
evidence about anything. The implementation follows Step 10 exactly and the
tests split the claim accordingly: momentum, total mass and determinism are asserted sharply across
merges, energy flatness is asserted only on merge-free ticks, and the full
ledger is proved exactly in a two-body case where the vanished term is
computable.

Reference: Core Dumped, *"Why compilers can't optimize this"* — used for
notation and motivation; all results here derive from the RFC's own first
principles.

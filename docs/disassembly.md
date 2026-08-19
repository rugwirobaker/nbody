# Reading the disassembly (AArch64)

We have to re-check the disassembly whenever Phase A changes, and
RFC §3.3c makes it a *should* for both builds. This is the companion note that
makes that instruction actionable: how to get the output, how to find the loop
in it, and what the instructions mean.

Everything below is from this project's actual `ReleaseFast` binary on
aarch64-macos.

---

## 1. Getting the output and finding the loop

```sh
zig build bench -Doptimize=ReleaseFast
objdump -d zig-out/bin/nbody-bench > /tmp/bench.asm
```

**The symbol you want will not be there.** `ReleaseFast` inlines aggressively;
`nm zig-out/bin/nbody-bench | grep computeAccelerations` returns nothing,
because the whole kernel was folded into `_start.main`. Don't go looking for a
function — look for a **fingerprint**: a rare instruction the loop must
contain.

```sh
grep -n "fsqrt" /tmp/bench.asm     # Phase A has exactly one sqrt per pair
```

Then read ~40 lines around each hit and pick out the one whose backward branch
makes it a hot loop. In our binary there were 7 `fsqrt` sites; the Phase-A one
is the one wrapped in a two-level loop.

---

## 2. Anatomy of a line

```
100003c50: 1e21c085    	fsqrt	s5, s4
└────┬───┘  └───┬──┘     └─┬─┘  └──┬─┘
  address    raw bytes   mnemonic  operands (destination first)
```

AArch64 is fixed-width: **every instruction is exactly 4 bytes**, so addresses
step by 4. Destination register comes first, like `dst = op(src1, src2)`.

---

## 3. Registers — the one thing to internalize

**Integer registers.** `x0`–`x30` are 64-bit; `w0`–`w30` are the *same*
registers viewed as their low 32 bits. Used here for addresses, indices, and
loop counters. `sp` is the stack pointer, `xzr`/`wzr` read as zero.

**Float/SIMD registers.** There are 32 of them, each 128 bits wide, and the
name you see tells you *how much of the register the instruction touches*:

| Name | Width | Holds |
| --- | --- | --- |
| `s5` | 32 bits | one `f32` |
| `d5` | 64 bits | one `f64`, **or** two `f32` |
| `q5` | 128 bits | four `f32`, or two `f64` |
| `v5` | 128 bits | the vector view, used with a lane suffix |

`s5`, `d5`, `q5` and `v5` are all **the same physical register**. So the
difference between scalar and vector code is *not* different registers — it is
which name the instruction uses. That is the whole trick to reading this:

- `fmul s3, s3, s12` → scalar: one multiply, one value.
- `fmul.2s v2, v2, v3` → vector: two `f32` multiplied at once.

**Lane suffixes** (the arrangement specifier):

| Suffix | Means | In our terms |
| --- | --- | --- |
| `.2s` | 2 × 32-bit | two f32 — half a NEON register |
| `.4s` | 4 × 32-bit | **four f32 — full NEON, what our SIMD kernel should emit** |
| `.2d` | 2 × 64-bit | two f64 |
| `.8b` / `.16b` | 8 / 16 bytes | raw byte moves |

Apple's `objdump` puts the suffix on the *mnemonic* (`fmul.2s v2, v2, v3`).
ARM's own documentation puts it on the *operands*
(`FMUL V2.2S, V2.2S, V3.2S`). Same instruction — worth knowing when you look
things up.

---

## 4. Instruction vocabulary

**Floating-point arithmetic** — the `f` prefix means floating point.

| Instruction | Meaning |
| --- | --- |
| `fadd` / `fsub` / `fmul` / `fdiv` | the obvious four |
| `fsqrt` | square root — our `@sqrt`, one per pair |
| `fneg` / `fabs` | negate / absolute value |
| `fmla` / `fmls` | fused multiply-add / -subtract (`a += b*c`) |
| `faddp` | **pairwise (horizontal) add** — sums lanes *within* a register |
| `fcmp` | compare, sets condition flags |
| `fcvtl` / `fcvtn` | widen f32→f64 / narrow f64→f32 |

`faddp` is the one to recognize on sight: normal SIMD ops work lane-by-lane
across registers, while a *horizontal* op combines lanes inside one register.
It means a reduction happened. Zig's `@reduce(.Add, v)` compiles to this family.

**Data movement**

| Instruction | Meaning |
| --- | --- |
| `mov` | register to register |
| `movi` | move immediate — `movi d0, #0` is "zero this register" |
| `movk` | move-keep: patch 16 bits into a register without clearing the rest — how 32-bit constants get built, in pairs |
| `dup` | **broadcast** one value into every lane — this is Zig's `@splat` |
| `ldr` / `str` | load / store register |
| `ldur` / `stur` | same, unscaled offset (allows negative immediates) |
| `ldp` / `stp` | load / store a *pair* of registers in one go |
| `ld1` / `st1` | NEON structured access; can target one lane: `st1.s {v0}[1]` stores lane 1 only |

**Addressing modes** — the brackets matter as much as the mnemonic:

| Form | Meaning |
| --- | --- |
| `[x10]` | address in `x10` |
| `[x10, #-0x10]` | address + constant offset; `x10` unchanged |
| `[x10], #0x18` | **post-index**: use `x10`, *then* `x10 += 0x18` |
| `[x10, #0x18]!` | **pre-index**: `x10 += 0x18` first, then use |
| `[x25, x10]` | base + register offset |
| `[x26, x10, lsl #2]` | base + register offset × 4 — indexing an `f32` array |

Post-index is how a loop walks an array: the load and the pointer bump are one
instruction. **The post-index constant is the stride**, which tells you the
element size the loop is walking.

**Integer / control flow**

| Instruction | Meaning |
| --- | --- |
| `add` / `sub` / `mul` | integer arithmetic |
| `lsl` | shift left (`lsl #2` = ×4) |
| `subs` / `adds` | same, but **set condition flags** (the `s`) |
| `cmp` | compare (a `subs` that discards its result) |
| `b.ne` / `b.eq` | branch if not-equal / equal, per the flags |
| `cbnz` / `cbz` | compare-and-branch against zero, no flags needed |
| `bl` / `blr` | branch with link — a function call |

A `subs`/`cmp` immediately followed by a **backward** `b.ne` is a loop bottom.
That pattern is how you find loops without any symbols.

---

## 5. Our Phase-A inner loop, annotated

This is `computeAccelerations` from `src/scalar.zig`, as actually compiled:

```
; ---- outer loop over accelerated particle i ----
100003c24: mul   x10, x9, x13        ; byte offset of particle i (x13 = 24)
100003c28: ldr   d1, [x25, x10]      ; d1 = {x_i, y_i}  — 8 bytes = TWO f32
100003c2c: movi  d0, #0              ; {ax, ay} accumulators = 0
100003c30: mov   x10, x8             ; x10 = cursor over sources
100003c34: mov   x11, x23            ; x11 = n, the inner trip count

; ---- inner loop over source particle j ----
100003c38: ldur  d2, [x10, #-0x10]   ; d2 = {x_j, y_j}
100003c3c: fsub.2s v2, v2, v1        ; {dx, dy} = {x_j - x_i, y_j - y_i}   ← 2-wide
100003c40: ldr   s3, [x10], #0x18    ; s3 = mass_j; then cursor += 24 bytes
100003c44: fmul.2s v4, v2, v2        ; {dx*dx, dy*dy}                      ← 2-wide
100003c48: faddp.2s s4, v4           ; HORIZONTAL: dx*dx + dy*dy
100003c4c: fadd  s4, s4, s12         ; + eps2                        → d2   (scalar)
100003c50: fsqrt s5, s4              ; sqrt(d2)                             (scalar)
100003c54: fmul  s4, s4, s5          ; d2 * sqrt(d2)                        (scalar)
100003c58: fmul  s3, s3, s12         ; mass_j * g                           (scalar)
100003c5c: fdiv  s4, s13, s4         ; 1.0 / (d2*sqrt(d2)) = inv_d3         (scalar)
100003c60: fmul  s3, s3, s4          ; s = g * mass_j * inv_d3              (scalar)
100003c64: fmul.2s v2, v2, v3[0]     ; {s*dx, s*dy}  — v3[0] broadcasts s   ← 2-wide
100003c68: fadd.2s v0, v0, v2        ; {ax, ay} += that                     ← 2-wide
100003c6c: subs  x11, x11, #0x1      ; j counter--
100003c70: b.ne  0x100003c38         ; loop back  ─── ONE source per iteration

; ---- row epilogue ----
100003c74: lsl   x10, x9, #2         ; i * 4 (f32 index)
100003c78: str   s0, [x26, x10]      ; ax[i] = lane 0
100003c80: st1.s {v0}[1], [x10]      ; ay[i] = lane 1
100003c88: cmp   x9, x23
100003c8c: b.ne  0x100003c24         ; next i
```

Three things this tells you that the source alone does not:

**a) The AoS penalty is visible in the addressing.** The stride is `#0x18` =
24 bytes = `sizeof(Particle)`. Each iteration loads 8 bytes at offset 0
(`x`,`y`) and 4 bytes at offset 16 (`mass`) — and *skips* `vx`, `vy`, `heat`
entirely. Those 12 skipped bytes still ride into cache on every line fetched.
That gap between "bytes loaded" and "bytes used" is exactly what RFC §3.2's
SoA transform exists to close, and here it is as a literal instruction encoding.

**b) The loop was not vectorized over particles.** One source per iteration,
one `fsqrt`, one `fdiv`. Strict FP is what bought this: `ax += s*dx` is a
serial float reduction, and without fast-math LLVM may not reorder it across
iterations. The protection RFC §3.3c asks for is working.

**c) LLVM pairs the x and y components into 2-wide NEON.** `fsub.2s`,
`fmul.2s`, `fmul.2s`, `fadd.2s`, plus a `faddp.2s` for `dx²+dy²`. The SLP
vectorizer noticed that the `ax` and `ay` chains are two independent reductions
and ran them side by side. Each chain is still summed in order, so this is what
the natural code compiles to. The consequence: the baseline already gets 2-wide
on the cheap component arithmetic, and the headline speedup is measured against
a partly-widened control.

---

## 6. Traps that will bite you

**Inlining erases the symbol.** Covered above: search for a fingerprint
instruction, never a function name.

**A whole-binary histogram is meaningless.** Counting `grep -c "fmul\.2d"`
over the entire file mixes every function together. The `.2d` (f64) forms in
this binary come from seeding's momentum accumulation and the properties —
not Phase A at all. Always locate the loop first, then read it; a global count
answers a question you didn't ask.

**Identical constants get merged.** In the default config `g` and `eps2` are
*both* `5.0e-4`, so LLVM materializes one register (`s12`) and uses it for
both — which is why the same `s12` appears as "+ eps2" on one line and
"× g" three lines later. You cannot tell them apart here. To confirm which is
which, temporarily change one value and re-read.

Decoding a constant you see being built:

```
mov  w16, #0x126f          ; low 16 bits
movk w16, #0x3a03, lsl #16 ; high 16 bits → 0x3a03126f
```

```sh
python3 -c "import struct; print(struct.unpack('>f', bytes.fromhex('3a03126f'))[0])"
# 0.0005000000237487257
```

The three constants hoisted before our loop are `0x3a83126f` = `dt` (1e-3),
`0x3f7fbe77` = `heat_decay` (0.999), and `0x3a03126f` = `g` / `eps2` (5e-4).

---

## 7. The checklist this exists to serve

When Phase A changes, there are two questions to answer:

1. **Is the scalar baseline still one source per iteration?** Find its
   `fsqrt`. Note that since Part 3 landed, **the binary legitimately contains
   `fsqrt.4s`** — that is the SIMD kernel, and grepping the whole file will
   find it. This is the histogram trap above, so locate the scalar loop first
   and read only that: if *its* sqrt ever becomes `.4s`, or several appear in
   one loop body, LLVM widened the baseline and the comparison is dishonest.
2. **Does the SIMD kernel show `.4s` everywhere and no stack traffic?** RFC
   §3.3c: three vector loads, zero stores per iteration, all intermediates in
   registers. `str`/`ldr` against `sp` inside the inner loop means spills —
   investigate rather than shrug.

## 8. What Part 3 actually emitted

The prediction was `dup.4s`, `ldr q…`, the fourteen operations in `.4s`, and a
`faddp`/`addv` at the row bottom. Most of it held:

```
ldr     q4, [x10], #0x10   ; three vector loads — x, y, mass
ldr     q5, [x11], #0x10   ;   post-index 0x10 = 16 bytes = 4 f32
ldr     q6, [x12], #0x10
fsub.4s v4, v4, v1         ; dx — v1 holds the broadcast x_i
fsub.4s v5, v5, v3         ; dy
fmul.4s v7, v4, v4         ; the same fourteen operations as the
fmul.4s v16, v5, v5        ;   scalar loop, four pairs at a time
fadd.4s v7, v7, v16
fadd.4s v7, v7, v18        ; + eps2
fsqrt.4s v16, v7           ; now a vector sqrt
fmul.4s v7, v7, v16
fdiv.4s v7, v17, v7        ; and a vector divide
fmul.4s v6, v6, v18
fmul.4s v6, v6, v7
fmul.4s v4, v4, v6
fadd.4s v2, v2, v4         ; lane accumulators
fmul.4s v4, v5, v6
fadd.4s v0, v0, v4
add     x9, x9, #0x4       ; j += L
b.lo    <top>
```

**Three vector loads, zero stores, no stack traffic** — §3.3c's contract, met
exactly. Every intermediate stayed in a register.

Two things differed from the prediction, both for good reasons, and both worth
recognizing when you read your own kernels:

- **No `dup.4s` in the loop.** The broadcasts were hoisted above it into `v1`,
  `v3`, `v17` and `v18`. That is the "broadcast once per row" design showing up
  in the instruction schedule rather than in the loop body — the splat happens
  once per row, not once per iteration, which is the whole point of Variant A.
- **No `faddp`/`addv` either.** The `@reduce` lives in the row epilogue,
  outside this listing, executed once per row. Finding a horizontal op *inside*
  an inner loop is usually a sign something is being reduced too often.

Note `v18` doing double duty as both `eps2` and `g` — the same constant-merging
trap described in §6, since both are `5.0e-4` by default.

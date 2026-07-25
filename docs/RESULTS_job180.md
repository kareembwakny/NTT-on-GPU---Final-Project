# GPU-NTT — Final Results Reference (job-180)

Consolidated from the three server-side optimization session reports (Step 1, Step 2,
Step 3). This is the authoritative number set for the report, poster, and presentation.
**All published deliverables should cite job-180, not the old job-153.**

> Status note: reconstructed from the session reports pasted into chat while cluster
> access is down. Headline numbers below were reported consistently across Step 2 and
> Step 3 and are reliable. A few items flagged **[verify]** were garbled in transit and
> should be confirmed against the actual CSVs/docs once the files are reachable.

---

## 1. Headline result (the number that goes everywhere)

| Metric | Baseline (job-153) | Final (job-180) |
|---|---|---|
| Wins / ties / losses vs GPU-NTT (74 cells) | 48 / 4 / 22 | **70 / 1 / 3** |
| Max speedup vs GPU-NTT | 1.93× | **2.12×** |
| N=16384 best (worst regime at baseline) | 0.97 (never won) | **1.27** (wins) |
| Small-N saturated bandwidth | 272 GB/s (52.6% of roof) | **446 GB/s (86.2%)** |
| Mid-N saturated bandwidth | 204 GB/s (39.5% of roof) | **343 GB/s (66.3%)** |
| Correctness (bit-exact, all cells) | 74/74 | **74/74** |

- **Session kernels are selected in 66 of 74 cells.**
- Best-of-ours median gain over baseline: **1.476×**.
- Regret / correctness median: **1.000** (selector never worse than best available).
- "Achieved roofline" = **517 GB/s** measured (grid-stride uint4 copy) = 84% of the
  616 GB/s theoretical peak; D2D memcpy = 534 GB/s = 86.7%. **Report % against 517,
  keep 616 alongside labelled "theoretical."**

### Intermediate checkpoint (after Step 1, job-173)
70 / 1 / 3, max 2.08×, N=16384 crossed 1.0 for the first time (0.97 → 1.27).
Step 2 held the win/loss count and pushed bandwidth + max speedup further.

---

## 2. The narrative spine — three-link bottleneck chain

This is the architectural argument the report should be built around. Each regime is
bound by a *different* hardware resource, which is why one kernel design cannot be
optimal everywhere and regime-specialization is a necessity, not a convenience.

1. **Sync-bound** → radix-4 / radix-8 cut the *number* of `__syncthreads()` barriers
   (13→7 at radix-4, 13→5 at radix-8). Gain: 1.29× / 1.12× median.
2. **Barrier cost** → block size cut the *cost* of each barrier. At N=2048, a
   1024-thread block gets 1 block/SM, so one barrier stalls the whole SM; 256 threads
   gives 4 blocks/SM → **1.58× at b=512**.
3. **Twiddle-table footprint** → once sync was largely removed, a second bottleneck
   surfaced. Montgomery (single u32 twiddle) beats Shoup (uint2 {w, w′}) above N=4096
   *despite more ALU work*, because its table is half the size (32 KB vs 64 KB at
   N=8192) and the kernel's shared demand forces a large carveout / small L1. The
   speedup's growth with N (0.98× at N=512 → 1.07× at N=8192) is what proves the cause
   is table footprint, not arithmetic.

**Supporting negative result:** lazy reduction was **rejected** (median 0.994×) — the
conditional subtractions were never the constraint. This is what makes the chain
credible rather than a story fitted to the data.

**Two contrasting findings that anchor the "different limits" thesis:**
- Padded kernel: wider global access (uint4) did **nothing** → it was *barrier-bound*.
- Multi-block kernel: coalescing + block size gave 1.50× of a 1.65× total → it was
  *access-bound*.
Same optimization, opposite verdicts, because the kernels sit against different limits.

---

## 3. Kernels developed this session

### Step 1 — close the last saturated loss
- **`mbr4_step1` / `mbr4_step2`** (`tools/arch_opt/multiblock_r4_kernels.cu`).
  N=8192 b=8 went **0.84× → 1.26×** vs GPU-NTT. **1.65× over the old multi-block kernel.**
  Breakdown: radix-4 = 1.10×, coalescing/block-size = 1.50×.
  Fixed three defects the old multi-block kernel still carried (it predated Shoup):
  u64 `% MOD` arithmetic, 13 barriers, and a stride-N₁ gather feeding 32-thread blocks.
  Also flipped **N=16384** — 1.04× over old multi-block, past GPU-NTT at b=1–16.

### Step 2 — generalize + arithmetic re-test
| Kernel | vs its base | build |
|---|---|---|
| `ws_shoup_r4` (radix-4 tail on warp-shuffle) | median 1.054× | 22 regs, 0 spills |
| `padded_shoup_r8` (3 stages/barrier, 13→5 at N=8192) | median 1.097× | 43 regs, 0 spills |

- radix-8 is best-of-ours at 17 of 30 A/B cells.
- radix-16 **not** pursued — needs 16 data + 15 twiddle regs for only 5→4 barriers,
  hitting the register wall Phase 2 documented.
- **Montgomery** added to the selector (wins above N=4096, see chain link 3).
- Block size is a large lever (256 vs 1024 threads → 1.58× at N=2048); L1/shared
  carveout is negligible (1.016× once measured interleaved).

### Step 3 — fused polynomial multiplication
- **`fused_polymul`**: ACCEPT. Keeps both operands in shared memory; two forward
  transforms share one stage loop and its barriers. Traffic **9N → 3N words**,
  **4 launches → 1**.
  - vs unfused, inputs-preserved baseline: **1.58–2.43×**
  - vs unfused in-place (strongest baseline): **1.29–1.92×** (reported against this,
    the harder comparison, after catching that the in-place kernels needed input copies).
  - Gated against CPU schoolbook O(N²) cyclic convolution + multiply-by-x / by-1 edge
    cases: **20/20 PASS**.
  - **[verify]** shared-memory demand reported as 67.6 KB vs the 64 KB sm_75 cap — the
    exact figure came through garbled; confirm the real number and the size ceiling.

### Step 3b — tensor cores: investigated and rejected (good future-work material)
- Measured INT8 IMMA = **287.7 Tops/s** vs INT32 = **6.38 Tops/s** (≈45× compute edge).
- Rejected on quantitative grounds: a 30-bit modulus needs multi-digit decomposition,
  the matmul form is O(N²) vs O(N log N), and step 2e already showed the kernels are
  **memory-bound** — so the compute roof was never the wall. For a 128-point
  sub-transform, ~2.9× more arithmetic time than radix-8 on plain INT32.
- Scoped honestly against Sugizaki & Takahashi (SCA/HPCAsia 2026): their operating
  point has fewer digits, newer tensor hardware, and higher arithmetic intensity — not
  a contradiction, a different regime. **Frame as future work, not failure.**

### Step 3c — scaling study
- 84/84 gate-PASS. mbr4 holds its gains at N=16384 / 32768 / 65536 (1.93× / 1.78× /
  **[verify third value — garbled]**); beats GPU-NTT at b=1–2 at every size. Crossover
  moves down in batch as N grows (the known cost of the multi-block split — stated, not
  hidden).

---

## 4. Analysis artifacts already produced

- **Roofline plot**: `results/plots/roofline_archopt_20260720.png`. Ridge point
  13.0 ops/B; all measured sizes at arithmetic intensity 4.5–7.9 → **memory-bound at
  every size this project measures**. N=256 reaches 86% of the 517 GB/s roof.
- **Ablation (round-robin, all bit-exact)**, cumulative **1.94×**:
  padding 1.235× → Shoup 1.008× → uint2 1.012× → radix-4 1.293× → radix-8 1.122× →
  Montgomery 1.016× → block size 1.049×.

---

## 5. Methodology (for the measurement chapter)

- Hardware: **RTX 2080 Ti**, Turing **sm_75**, driver **550.120**, CUDA **12.4**.
- Modulus **q = 998244353**.
- Timing: CUDA events, **WARM=5 / REP=101**.
- **Timing-protocol correction (report this honestly):** the original baseline used
  WARM=3 / REP=9. With this GPU idling at 300 MHz, 9 reps land inside the boost ramp,
  making small-N numbers unreliable. All final data was re-measured at WARM=5 / REP=101.
  job-153 is retained for continuity only; **job-180 is authoritative.**
- **Three fixed-order timing artifacts** found and eliminated this session (40%, 34%,
  and an 11-cell false regression). Standing rule adopted: accept/reject comparisons are
  round-robin, and at N ≤ 512 cross-run absolute deltas are **not admissible** — the tell
  for a false regression is the *same backend* being selected on both sides.
- Profiler still blocked: `ncu 2024.1.1` ships in-tree and runs, but
  `/proc/driver/nvidia/params` shows `RmProfilingAdminOnly: 1` → `ERR_NVGPUCTRPERM`.
  One admin line (`NVreg_RestrictProfilingToAdminUsers=0`) would unblock it. Occupancy
  is read from the compiled binary in the meantime; stall reasons remain attributed by
  controlled experiment.

---

## 6. Regressions (state plainly, do not hide)

- vs job-153: **2 of 74 cells** slower at all, **1 by >3%** — N=16384 b=128, where
  `gpu_ntt` is selected on **both** sides, i.e. GPU-NTT's own run-to-run variance.
- vs same-day job-172: 0 real regressions.
- **No cell regressed by selecting a new kernel.**

---

## 7. Remaining GPU-NTT wins (the honest 3 losses)

N=16384 at b=32 / 64 / 128 — the DRAM-saturated large-N corner. One timeboxed attempt
at these is optional; 70/74 is a stronger honest number than a forced 73–74.

---

## 8. Git anchors (for the Project Documentation chapter)

- Baseline tag (untouchable): `pre-optimization-baseline-20260720` @ `e092747`
- Step 1: `103e0b5` (1a), `f5b67cb` (1b–1d)
- Step 2: `6d37df7`
- Step 3: `fd61627`; final summary `9aa8a10`
- Branch: `cuda-arch-optimization`
- Archived baseline data (untouched): `results/raw_baseline_20260720/`,
  `docs_baseline_20260720/`

---

## 9. Still needed before the documents are final

- The actual **job-180 CSVs and regenerated figures** (all plots rebuilt at job-180:
  backend selection map, throughput curves, bandwidth-by-regime, optimization
  progression, roofline) — download once cluster access is restored.
- Re-measured **CPU baseline** (the current "58.6× vs C++ CPU" is against the *old*
  kernels — now understated) and **energy** numbers.
- Two **student ID numbers** (still placeholder in both report covers).
- The **repository URL** (still placeholder in the Project Documentation chapter).

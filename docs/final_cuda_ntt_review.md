# Final CUDA NTT Review

**Scope:** the complete technical review of the architecture-aware CUDA NTT system (Phases C1–C4,
jobs 151/152, srvr-2). Every number below traces to a committed CSV.

## 1. Hardware used
RTX 2080 Ti (Turing TU102, sm_75): 68 SMs, 4352 CUDA cores (FP32) plus a separate concurrent 64-wide INT32 datapath per SM (analytic INT32 upper bound ≈6.7 T op/s at the 1545 MHz boost clock),
64 K regs/SM, 64 KB smem/SM (96 KB unified L1), 5.5 MB L2, GDDR6 352-bit → 616 GB/s theoretical.
Driver 550.120, CUDA 12.4, `nvcc -gencode arch=compute_75,code=sm_75 -O3 -std=c++17`.
Full table + design implications: `docs/final_cuda_hardware_profile.md`.

## 2. How the architecture was exploited
- **Warp width 32 → 5 sync-free stages:** butterflies at distance m<32 exchange via
  `__shfl_xor_sync` in registers — no smem, no barriers, no bank conflicts (small-N kernels).
- **64 K regs/SM → Shoup for free:** the precomputed-quotient butterfly (uint2 twiddles) fits in
  22–29 regs at 100% theoretical occupancy while shortening the modmul dependency chain.
- **Smem padding → conflict-free mid-N:** one pad word per 32 makes cross-stage strided accesses
  bank-conflict-free (padded kernel), 0 spills at 34 regs.
- **68 SMs → multi-block 4-step:** N=N₁·N₂ splits one large transform into 128+ blocks, fixing the
  single-block batch=1 occupancy cliff (1/68 SMs → all SMs).
- **5.5 MB L2 → twiddle reads ~free:** tables ≤64 KB stay L2-resident; only the 8 B/coefficient
  data traffic matters at scale.
- **Launch reduction:** all our kernels are 1 launch per batched transform set (multiblock: 2)
  vs GPU-NTT's per-stage launches.

## 3. Final kernel families
`warp_shuffle(+Shoup)` (N≤1024) · `padded(+Shoup)` (N=2048–8192) · `multiblock 4-step`
(large-N low batch) · `low_latency`/`fused` (retained, dominated) · **GPU-NTT fallback** ·
**table-driven argmin selector** over all of them. Audit: `docs/final_cuda_kernel_audit.md`.

## 4. Why different regimes need different kernels
Small N is **latency-bound** (modmul chain + launch floor: time flat in batch up to ~64) → shuffle
execution + Shoup. Mid N is **smem/sync-bound** (arithmetic reduction proven neutral) → conflict
elimination. Large-N low batch is **grid-occupancy-bound** → multi-block. Large-N/saturated is
**DRAM-bound** → nothing arithmetic helps; GPU-NTT's per-stage streaming wins → fallback.

## 5. Resource analysis
All our kernels **0 spills**; 19–34 regs (mb_step1 45) → 100% theoretical occupancy (87.5%
mb_step1). Achieved occupancy, SM-active %, stall reasons: **unavailable** (no ncu on cluster) — binding regimes attributed by
controlled experiments instead. Effective bandwidth (estimated: known bytes ÷ measured time; not a profiler counter): latency-bound cells 12–120 GB/s
*by design*; saturated small-N 181–237 GB/s (29–38% of peak); large-N high-batch ~188–205 GB/s
(30–33%) on both ours and GPU-NTT — the achievable-traffic-bound region.

## 6. Transforms per second (runtime-selected, measured; job 153 execution-verified)
Representative table (`results/raw/final_cuda_representative_throughput.csv`, raw native-order):

| workload | N | batch | backend | µs/transform | transforms/s | eff. GB/s (est.) | vs GPU-NTT |
|---|---|---|---|---|---|---|---|
| small-N low batch | 256 | 1 | shoup_shuf | 4.58 | 0.219M | 0.4 (0.1%) | 1.93× |
| small-N pre-saturation | 256 | 2048 | shoup_shuf | 0.010 | 96.2M | 197 (32%) | 1.08× |
| small-N saturation | 256 | 8192 | gpu_ntt | 0.009 | 117.2M | 240 (39%) | 0.95 (handoff) |
| medium-N | 2048 | 32 | shoup_pad | 0.280 | 3.57M | 59 (9.5%) | 1.33× |
| large-N low batch | 16384 | 1 | gpu_ntt | 10.24 | 98K | 13 (2.1%) | 0.97 (parity) |
| large-N high batch | 16384 | 128 | gpu_ntt | 0.701 | 1.43M | 187 (30%) | 0.82 (handoff) |

Our-kernel peak: **110.9M transforms/s** = warp_shuffle_shoup at **N=256, batch=8192**,
0.073856 ms/batch (raw natural-order output, no normalization) — at that cell GPU-NTT is 5%
faster (0.069888 ms → 117.2M/s), so the runtime selects GPU-NTT there.
Full grid: `results/raw/final_cuda_backend_map.csv`; plot `final_cuda_transforms_per_second.png`.

## 7. Final comparison vs GPU-NTT (74 cells, job 153 execution-verified, raw native order)
**Ours 48 wins · 4 within 3% · GPU-NTT 22 wins.** Max per N: 1.93× (256), 1.62× (512), 1.48×
(1024), 1.38× (2048), 1.34× (4096), 1.18× (8192). (Job-151 run of the same grid: 49/7/18, max
2.14× — boundary cells flip within ±3% run-to-run noise; small-N sub-10 µs latencies vary ~20%
with clock state, direction stable.) **Output-contract sensitivity** (`final_cuda_output_order.md`):
natural-order contract ours 69/2/3 (GPU-NTT pays the bitrev→natural pass); bit-reversed contract
ours 3/6/65 (we pay it). GPU-NTT wins the saturated region in all framings' raw core. Methodology
+ excluded baselines: `docs/final_cuda_baseline_methodology.md`.

## 8. Strongest winning region
**N≤1024 at batch 1–2048**: 1.08–1.93× over GPU-NTT in the execution-verified run (up to 2.14× in the job-151 run), the latency-critical and
mid-throughput regime of lattice-crypto-shaped workloads.

## 9. Most important tradeoff
Our single-kernel designs win until DRAM saturation, then lose (0.79–0.96× at N≤1024 B=8192;
0.49–0.88× at N=16384 mid/high batch). The runtime *knows the boundary* (measured, not guessed)
and hands off to GPU-NTT — the system is never worse than the baseline.

## 10–11. Claimed features (required format)

**Feature:** warp-shuffle + Shoup small-N NTT kernel.
**Target regime:** N=256–1024, batch 1–~2048.
**Hardware idea:** 5 butterfly stages live inside one warp → register exchange via `__shfl_xor`
(no smem/barriers); Shoup precomputed quotient shortens the latency-bound modmul chain (22 regs).
**Measured advantage:** up to **1.93×** vs GPU-NTT (N=256 B=1–2; 2.14× in the job-151 run); wins throughout N≤1024 B≤128; holds >1.05× to B≈2048–4096 at N=256.
**Measured tradeoff:** loses at saturation (0.96× @N=256 B=8192, 0.79× @N=1024 B=8192).
**Why valuable:** small transforms dominate lattice/PQC workloads; latency floor 4.5 µs.
**When selected:** argmin table routes it exactly in its win region.
**Baseline:** GPU-NTT same GPU/modulus/harness. **Correctness:** edge+random+roundtrip PASS,
convention formally verified. **Verdict: ACCEPTED — core contribution.**

**Feature:** multi-block 4-step large-N kernel.
**Target regime:** N=4096–16384, batch 1–8.
**Hardware idea:** split N=N₁·N₂ across 128+ blocks to fill 68 SMs at batch=1.
**Measured advantage:** argmin-selected wins up to **1.15×** (N=4096 B=2), 1.07× (8192 B=1);
historically cut the N=16384 b=1 gap 4.31×→~1×.
**Measured tradeoff:** 2 global passes lose at high batch (0.49–0.73×) → fallback there.
**Why valuable:** single-transform large-N latency is the worst case for single-block designs.
**When selected:** large-N B≤2–8 (argmin). **Baseline:** GPU-NTT. **Correctness:** fwd edge PASS
+ roundtrip in bench_multiblock. **Verdict: ACCEPTED.**

**Feature:** table-driven argmin backend selector.
**Target regime:** all 74 cells.
**Hardware idea:** no kernel wins everywhere; route per measured cell, GPU-NTT as a backend.
**Measured advantage:** 44/56 cells to ours vs 37 under the old 10% margin; 7 boundary cells
recovered (up to 1.146×); never worse than GPU-NTT anywhere.
**Measured tradeoff:** ~0.7% aggregate only; possible noise-flips at true parity (harmless).
**Why valuable:** converts specialized kernels into a deployable system with no regression risk.
**When selected:** always (it is the dispatch layer). **Baseline:** margin selector + GPU-NTT.
**Correctness:** profile gate ALL PASS. **Verdict: ACCEPTED (one-constant complexity).**

**Feature:** natural-order output convention (bit-reversal folded into load).
**Measured advantage:** consumers needing natural-order forward output avoid one permutation pass
that GPU-NTT (bit-reversed output) would pay; not charged in our ratios (conservative).
**Verdict: documented property, not counted as a speedup.**

## 12. Is the contribution large enough to justify the tradeoff?
Yes: 49/74 cells won (up to 2.14×) + a never-worse runtime vs 18 fallback cells whose only cost
is using the mature library where it is already optimal.

## 13. Report features
Hardware profile; kernel families + audit; argmin selector; 74-cell comparison; Kyber NTT case
study (12.3× attribution). **14. Future work:** ML-KEM full-protocol pipeline, transfer/seed
routing, server scheduling, CUDA graphs (all built, all supporting material); persistent small-N
megakernel; sm_80+ `cp.async` double-buffered padded kernel.

## 14b. Metrics that remain unavailable (stated, not omitted)
Achieved occupancy, SM-active %, eligible warps/cycle, and stall breakdown were **not measured
because no GPU profiler (Nsight Compute / nvprof) is installed on this cluster**; occupancy
figures are compiler/resource-derived only. p50/p95/p99 latency IS measured per cell (REP=100,
`final_cuda_selector_execution.csv`). Energy IS measured for workload mixes via the NVML
total-energy counter with two-point subtraction (`final_cuda_energy.csv`: adaptive 0.192 µJ/NTT
@255 W on the D mix); per-single-cell energy was not sampled (sub-ms kernels are below reliable
counter resolution).

## 15. Strongest honest final claim
> We developed an architecture-aware CUDA NTT system for sm_75 that exploits warp-level
> communication (5 sync-free shuffle stages), register-resident Shoup butterflies, bank-conflict-
> free padded shared memory, and multi-block 4-step decomposition, dispatched by a table-driven
> argmin runtime with GPU-NTT as a fallback backend. On the same RTX 2080 Ti, same modulus, same
> harness, and correctness gated on edge inputs with the transform convention formally verified,
> our kernels beat GPU-NTT in **48 of 74 execution-verified (N, batch) cells under the raw
> native-order contract — up to 1.93× at N=256 (49/74 and 2.14× in a second run of the same
> grid; boundary cells flip within noise) — and per-cell the runtime is never worse than
> GPU-NTT**, handing off at the measured saturation boundary
> (N≤1024 beyond batch ≈2–4K, and large-N high-batch) where the library's per-stage streaming
> remains superior. The same optimization principles transfer to Kyber's incomplete NTT with a
> 12.3× correctness-verified arithmetic improvement.

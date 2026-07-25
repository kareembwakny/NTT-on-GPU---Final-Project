# Final CUDA Kernel Audit

Per-kernel data: `results/raw/final_cuda_kernel_audit.csv` (ptxas resources from
`fable5_kernel_resource_audit.csv` + fresh ptxas capture `final_cuda/ptxas_verbose.txt`; timings
from the argmin map job 151 and the extended-batch sweep job 152, all CUDA-event, compute-only).

## Metric provenance (read before quoting any number)

| metric | status |
|---|---|
| register count / spills / static smem | **compiler-reported** (ptxas -v) |
| theoretical occupancy | **analytically derived** from ptxas registers/smem vs sm_75 limits |
| achieved occupancy | **unavailable** (no Nsight Compute / nvprof on this cluster) |
| SM-active %, active-SM count | **unavailable** |
| eligible warps per cycle, stall reasons, issue efficiency | **unavailable** |
| kernel latency (CUDA events) | **measured** |
| transforms/s, coefficients/s | **analytically derived** (batch ÷ measured runtime) |
| effective DRAM bandwidth | **estimated** (known bytes moved — 8 B/coefficient, one read + one write pass — ÷ measured runtime); *not* a profiler counter |
| % of theoretical peak bandwidth | **derived** (effective BW ÷ 616 GB/s theoretical) |
| binding-regime attribution | **inferred from controlled experiments** (batch scaling flatness, arithmetic-reduction neutrality, block-count scaling) |

**Effective-bandwidth percentage is not "GPU utilization"** — it is a traffic-based estimate of how
close a kernel runs to the DRAM roof, and says nothing about SM occupancy or issue-slot usage.

## Kernel families and their regimes

| kernel | regs | smem | spills | occ (theo) | syncs/transform | selected where |
|---|---|---|---|---|---|---|
| `warp_shuffle_ntt` | 19 | N·4 staging | 0 | 100% | logN−4 | dominated by +Shoup |
| **`warp_shuffle_ntt_shoup`** | 22 | N·4 | 0 | 100% | logN−4 | **N≤1024, B up to ~2048–4096** |
| `low_latency_ntt` | 32–34 | N·4 | 0 | 100% | logN | fallback (dominated) |
| `low_latency_ntt_shoup2` | 26 | N·4 | 0 | 100% | logN | not selected (subsumed) |
| `padded_ntt` | 34 | SPAD(N)·4 | 0 | 100% | logN | N=8192 high batch |
| **`padded_ntt_shoup`** | 27–29 | SPAD(N)·4 | 0 | 100% | logN | **N=2048–4096 mid/high batch** |
| **`mb_step1`/`mb_step2`** | 45/37 | tile·4 | 0 | 87.5% (step1) | logN₁+logN₂ | **large-N batch 1–8** |
| `fused_ntt` | 42 | N·4 | 0 | 100% | logN | dominated |
| GPU-NTT (Merge NTT) | ext | ext | — | — | per-stage launches | **large-N / saturated fallback** |

All our kernels: **0 spills** (ptxas), single launch per transform batch (multiblock: 2 launches),
natural-order in/out with bit-reversal folded into the load.

## Measured behavior per selected kernel (representative cells)

- **warp_shuffle_shoup, N=256:** latency floor ~4.5 µs (B=1) — launch+chain-latency bound;
  saturates at **110.9 M transforms/s = 227 GB/s (36.9% of peak)** at B=8192 (job 152). Wins vs
  GPU-NTT up to **2.14×** (B=2), holds >1.05× to B≈4096, loses 0.96× at B=8192 → runtime hands off.
- **padded(+Shoup), N=2048–8192:** conflict-free smem; sync-bound at mid batch (arith-reduction
  proven neutral in prior passes); at N=8192 B=128 runs 205 GB/s (33% peak) vs GPU-NTT 188 GB/s —
  both in the achievable-traffic-bound region.
- **multiblock 4-step, large-N low batch:** turns a 1-SM single-block kernel into 128+ blocks;
  argmin-selected at N=4096–16384 B=1–2 (up to **1.15×** vs GPU-NTT at N=4096 B=2); its 2 global
  passes make it lose at high batch (0.49–0.73×) → handoff.
- **GPU-NTT fallback:** wins the saturated region (N≥512 at very high batch; N=16384 mid/high
  batch) — its per-stage Merge kernels stream better at full DRAM saturation. The argmin runtime
  routes there; the system is never worse than GPU-NTT.

## Primary bottleneck per regime (evidence-based)

| regime | binding | evidence |
|---|---|---|
| N≤1024, B≤~64 | launch + modmul-chain latency | time ~flat in B; Shoup (chain-shortening) wins; occupancy irrelevant |
| N≤1024, B≥~2048 | DRAM/issue saturation | GB/s plateaus 181–237; GPU-NTT converges/overtakes |
| N=2048–8192 mid batch | smem+sync | Karatsuba/Montgomery/lazy all neutral (prior E6); padding (conflicts) is what helped |
| large-N B=1–8 | grid occupancy | block-count scaling experiment (multi-block 3.6× over single-block) |
| large-N high batch | DRAM bandwidth | ours and GPU-NTT within ~3% at ~30% peak; nothing arithmetic moves it |

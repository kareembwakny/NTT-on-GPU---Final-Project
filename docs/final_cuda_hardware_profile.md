# Final CUDA Hardware Profile — RTX 2080 Ti (sm_75)

Measured on the actual benchmark node (srvr-2, job 151) via `tools/cuda_hw_profile/hw_profile.cu`
(`cudaGetDeviceProperties`). Raw CSV: `results/raw/final_cuda_hardware_profile.csv`.

| property | value | consequence for NTT design |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti (Turing TU102), compute capability 7.5 | no async-copy (`cp.async` is sm_80+), no tensor-core integer path used → all results are plain INT32/SIMT |
| SMs | **68** | a single-block kernel at batch=1 uses 1/68 SMs → the multi-block 4-step kernel exists precisely to spread one large transform across ≥128 blocks |
| CUDA cores | **4352** (= 68 SM × 64 FP32 cores). Turing additionally has a **separate 64-wide INT32 datapath per SM**, concurrent with FP32 | modular butterflies are integer IMAD chains executed on the INT32 units; FP32 cores are idle in these kernels |
| peak INT32 throughput | **6.72 T int32 op/s** = 68 SM × 64 INT32 units × 1.545 GHz (boost clock as reported by `cudaDeviceProp.clockRate`; assumes 1 integer op issued per unit per cycle — an upper bound, IMAD throughput on Turing is lower) | analytically derived upper bound, **not measured**; used only to argue small-N kernels are latency-bound, never as an achieved figure |
| warp size / warps per SM | 32 / **32 resident** (1024 thr/SM) | 5 NTT stages (m=1..16) fit inside one warp → warp-shuffle kernel does stages 1–5 with `__shfl_xor` and **zero** shared memory or barriers |
| max blocks/SM | 16 | small blocks (≤64 threads) cannot fill an SM alone; our kernels use 128–1024 threads/block |
| register file | 64 K regs/SM | ≤32 regs/thread keeps 100% occupancy at 1024 thr/SM; all selected kernels are 19–34 regs, **0 spills** |
| shared memory | 64 KB/SM (opt-in per block), unified with L1 (96 KB) | N=16384 × 4 B = 64 KB is exactly the single-block ceiling → padded kernel capped at N≤8192, N=16384 needs multi-block or opt-in low-latency variant |
| L2 | 5.5 MB | twiddle tables (≤64 KB) are L2-resident after first touch → twiddle traffic effectively free at steady state |
| memory | GDDR6, 352-bit @ 7 GHz eff. → **616 GB/s theoretical** | one NTT reads+writes 8 B/coefficient → BW roof = 616/(8·N) transforms/s; large-N high-batch measured ≈30% of peak on both ours and GPU-NTT (achievable-traffic-bound region) |
| clocks | boost 1545 MHz core / 7000 MHz mem | timings taken at steady clocks after warm-up (WARM≥3) |
| driver / runtime | 550.120 / CUDA 12.4 (nvcc 12.9 front-end, cudart 12.4 static — see `docs/gpu_ntt_build_log.md`) | |
| flags | `nvcc -gencode arch=compute_75,code=sm_75 -O3 -std=c++17` (+`--ptxas-options=-v` for audits) | |

## How each hardware feature is exploited

1. **Warp = 32 lanes ⇒ 5 sync-free butterfly stages.** Stages with butterfly distance m<32 exchange
   partners with `__shfl_xor_sync` in registers: no shared memory, no bank conflicts, no
   `__syncthreads()`. This is the core of the small-N win (up to 1.9–2.0× over GPU-NTT).
2. **64 K regs/SM ⇒ Shoup butterflies are free real estate.** The Shoup variant carries a second
   precomputed quotient per twiddle (uint2 table) yet still fits 22–29 regs → 100% occupancy while
   shortening the modmul dependency chain — the win in the latency-bound regime.
3. **64 KB smem/SM + padding ⇒ conflict-free mid-N.** The padded layout inserts one word per 32 so
   cross-stage strided accesses hit distinct banks; `ptxas` confirms 0 spills at ≤34 regs.
4. **68 SMs ⇒ multi-block decomposition for batch=1 large-N.** N=N₁·N₂ 4-step splits one transform
   into N₂+N₁ blocks (128×128 at N=16384), turning a 1-SM kernel into a 68-SM kernel: measured
   large-N batch=1 parity-to-win vs GPU-NTT (up to 1.15× at N=4096 B=2 under argmin selection).
5. **5.5 MB L2 ⇒ twiddle bandwidth is not a bottleneck** for N≤16384 (tables ≤64 KB); measured
   achieved-BW analysis therefore counts only the 8 B/coefficient data traffic.
6. **No cp.async / no tensor cores on sm_75** — nothing in the design depends on them; the
   contribution targets plain-SIMT INT32 GPUs.

## Honest measurement limitation

The cluster has **no ncu/nvprof**; *achieved* occupancy, stall reasons, and issue efficiency are not
directly measurable. We report **theoretical occupancy** (from ptxas regs/smem), **achieved bandwidth**
(bytes-moved ÷ measured time), and binding-regime attribution from controlled experiments
(arithmetic-reduction neutrality, batch scaling, block-count scaling) instead — each labeled as such.

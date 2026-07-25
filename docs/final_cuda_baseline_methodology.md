# Final CUDA Baseline Methodology

The main comparison is **CUDA NTT vs CUDA NTT on the same GPU** (RTX 2080 Ti, sm_75, driver
550.120, CUDA 12.4). All numbers compute-only (data GPU-resident), CUDA-event timed, correctness
gated before timing.

## Primary baseline: GPU-NTT (Merge NTT)

| item | value |
|---|---|
| repository | https://github.com/Alisah-Ozcan/GPU-NTT |
| commit | `95c739c48d11827277e132f5eec4d4e454d60835` ("Fix bug in LowRingNTT") |
| build | CMake → `external/GPU-NTT/build/src/libntt-1.0.a`, sm_75, same nvcc/flags as ours (`docs/gpu_ntt_build_log.md`); 32-bit modulus pool pinned to 998244353 (documented edit) |
| modulus | q = 998244353 (31-bit NTT prime), Data32 |
| transform | forward cyclic NTT, X^N−1, `PerPolynomial` layout |
| input/output ordering | **input natural, output BIT-REVERSED** (library convention) |
| our kernels' ordering | **input natural, output NATURAL** (bit-reversal folded into load) — verified analytically: ours ≡ direct DFT at ω=3^((P−1)/N) ≡ bitrev(NTTCPU) (`tools/final_correctness/convention_diag.cu`) |
| sizes / batches | N=256…16384; batch 1…128 (all N) + 256…8192 (N≤1024, extended sweep) |
| warm-up / reps | WARM 3–10, REP 9–30 per harness; median reported (mean/std/min/max in CSVs) |
| timing | `cudaEventRecord` pairs around the launch, `cudaEventSynchronize` |
| scope | compute-only; H2D-inclusive reported separately where measured |
| transfers / preprocessing | excluded on both sides; twiddle/root tables prebuilt on both sides |
| correctness | GPU-NTT: forward vs its own `NTTCPU` reference. Ours: forward vs bitrev-permuted `NTTCPU` (same values, natural order) on {random×2, zeros, one-hot, all-(q−1)} + inverse round trip — ALL PASS, N=256…16384 (jobs 151/152) |

**Comparability note.** Both sides compute the identical size-N cyclic NTT over the identical
modulus with the identical root ω=3^((P−1)/N); the only difference is output ordering. For
NTT→pointwise→INTT pipelines both conventions are equally usable (each library's inverse consumes
its own forward order), so no reordering penalty is charged to either side. For consumers that
need natural-order forward output, GPU-NTT would additionally pay a permutation pass that our
kernels do not — our reported ratios do **not** include that (conservative in GPU-NTT's favor).

## External-baseline scope (read first)

**GPU-NTT is the only external CUDA NTT implementation reproduced experimentally on the same
hardware with the same transform scope.** No claim in this project implies that all existing GPU
NTT implementations were benchmarked. Baselines considered and excluded, each with the exact
limitation and why quoting published numbers instead would be unfair:

| baseline | source | limitation | why published numbers can't be quoted fairly |
|---|---|---|---|
| **ICICLE** (Ingonyama) | github.com/ingonyama-zk/icicle (built in `external/icicle`; `docs/icicle_build_log.md`) | GPU NTT backend **closed-source** (private CUDA repo; only the CPU backend builds); small-field NTT compile-time pinned to **babybear (2013265921)**, not 998244353 | different field, different (usually newer) GPUs in published figures |
| **cuPQC / NVIDIA ML-KEM CUDA** | developer.nvidia.com/cupqc | distributed for **sm_80+ (Ampere+)** — cannot run on this sm_75 card; targets q=3329 Kyber transforms, not the 998244353 benchmark modulus | different hardware class and transform scope |
| **HI-Kyber** (GPU Kyber, Sun et al.) | publication only; no public same-scope source that builds on sm_75 | no buildable artifact for this GPU/modulus | paper reports end-to-end Kyber on different GPUs, not isolated cyclic NTT at this modulus |
| **Tensor-Core Kyber/NTT works** (e.g., Wan et al.) | publications | Tensor-Core INT paths are a **different arithmetic model**; most need sm_80+ INT MMA shapes | different execution model, hardware, and reported scope |
| **pq-crystals AVX2 (CPU)** | github.com/pq-crystals/kyber @4768bd3 | CPU — secondary context only for the Kyber case study (`docs/cpu_mlkem_baseline_methodology.md`) | not a CUDA NTT baseline |

## Final comparison result (74 measured cells; `results/raw/final_cuda_vs_existing.csv`)

| region | cells | reason |
|---|---|---|
| **ours wins (>1.03×)** | **49** — max 2.14× (N=256 B=2), 1.99× (512), 1.52× (1024), 1.45× (2048), 1.35× (4096), 1.21× (8192) | latency-bound small/mid N: register/shuffle execution + Shoup chain-shortening + conflict-free smem; large-N low batch: multi-block SM fill |
| within 3% | 7 | boundary/parity cells |
| GPU-NTT wins (<0.97×) | 18 | saturated high-batch region (N≥512 at B≥2048–8192; N=16384 mid/high batch): per-stage Merge kernels stream better at full DRAM saturation |

The argmin runtime executes the per-cell winner (GPU-NTT is one of its backends), so the deployed
system is **never worse than GPU-NTT** on any measured cell.
Plots: `final_cuda_vs_existing_heatmap.png`, `final_cuda_speedup_curves.png`,
`final_cuda_transforms_per_second.png`, `final_cuda_utilization.png`, `final_cuda_backend_map.png`.

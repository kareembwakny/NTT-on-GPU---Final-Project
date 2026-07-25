# GPU-Accelerated Number Theoretic Transform (NTT)

Architecture-aware CUDA kernels for the Number Theoretic Transform on an NVIDIA RTX 2080 Ti
(Turing, sm_75), with a compiled adaptive runtime that dispatches the measured-best kernel
per problem size.

**Final project 3363 — Tel Aviv University, Iby and Aladar Fleischman Faculty of Engineering**
Kareem Bwakny · Hakeem Najjar · Advisor: Mr. Oren Ganon

---

## Results

Modulus `q = 998244353`, primitive root `g = 3`, sizes `N = 256–16384`, batches `1–8192`.
All numbers below are from the authoritative `job-180` run (CUDA events, WARM=5 / REP=101),
correctness-gated bit-exactly before timing.

| Metric | Value |
|---|---|
| Peak throughput | **110.9 M transforms/s** (N=256, batch 8192) |
| Small-N latency floor | **4.5 µs** |
| Cumulative kernel tuning | **1.94×** end to end |
| Saturated-regime bandwidth | **86%** of the measured 517 GB/s DRAM roofline (was 53%) |
| vs optimized C++ CPU | up to **58.6×** |
| vs external GPU NTT reference | **70 / 74** cells faster, up to **2.12×** |
| Correctness | **74 / 74** configurations bit-exact |

The measured DRAM roofline (517 GB/s, STREAM-style probe) is used throughout instead of the
616 GB/s datasheet peak; both are reported where relevant.

## The central finding

The NTT is not bound by one bottleneck but by a **chain** of them — removing one exposes
the next:

1. **Barrier count** — radix-4/8 butterflies perform 2–3 stages per barrier, cutting
   `__syncthreads()` from 13 to 5 at N=8192.
2. **Barrier cost** — block-size tuning puts several blocks on each SM, so a barrier stalls
   part of an SM instead of all of it (1.58× at N=2048).
3. **Twiddle-table footprint** — with synchronization largely removed, Montgomery's single
   32-bit constant per twiddle beats Shoup's pair above N=4096 *despite more ALU work*,
   because the table is half the size.

Lazy reduction was tested and **rejected** (0.994×), confirming the conditional subtractions
were never the constraint.

## Kernel families

| Kernel | Regime | Key idea |
|---|---|---|
| Warp-shuffle + Shoup | N ≤ 1024 | Early stages register-resident via `__shfl_xor_sync`; no shared memory, no barriers |
| Padded shared memory + radix-8 | N = 2048–8192 | Bank-conflict-free `SPAD(i)=i+⌊i/32⌋`; 3 stages per barrier |
| Multi-block 4-step (radix-4) | large N, low batch | `N = N₁·N₂` spread across all 68 SMs; 1.65× over prior design |
| Fused polynomial multiply | NTT · pointwise · INTT | One launch replaces four; DRAM traffic 9N → 3N words |
| Adaptive selector | all | Compiled dispatch table, ~1.25 ns lookup |

## Architecture

![Pipeline](figures/pipeline_arch2.png)

End-to-end pipeline with the GPU stage opened to its memory hierarchy and the per-regime
on-chip dataflow. See `figures/gpu_interior.png` for the SM-level resource mapping.

## Repository layout

```
docs/                   Methodology, kernel audit, hardware profile, selector design,
                        and RESULTS_job180.md (the consolidated final number set)
results/raw/            Benchmark CSVs (baseline generation — see note below)
results/plots/          Plots generated from the raw CSVs
figures/                Figures used in the report and poster
figures/src/            Matplotlib scripts that regenerate them
deliverables/           Project report (EN/HE), poster, presentation
```

> **Note on scope.** This repository currently contains the documentation, measurement data,
> figures, and written deliverables. The CUDA kernel sources live in the cluster working tree
> and are pending a push from that machine; `results/raw/` here is the baseline-generation
> data, while the final `job-180` numbers are documented in `docs/RESULTS_job180.md`.

## Methodology

- Hardware: RTX 2080 Ti (Turing TU102, sm_75), driver 550.120, CUDA 12.4
- Build: `nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75`
- Timing: CUDA events, WARM=5 / REP=101, steady clocks after warm-up
- Correctness gate: edge inputs (zeros, one-hot, all-`q−1`), random inputs, and inverse
  round-trip — **no timing is reported for a kernel that has not passed it**
- Accept/reject comparisons are run **round-robin** (interleaved with their baseline). At
  N ≤ 512, cross-run absolute deltas are not treated as admissible evidence; three
  fixed-order timing artifacts were identified and eliminated during the work.

## Known limitations

- Hardware performance counters are restricted to administrators on the benchmark cluster,
  so achieved occupancy and stall reasons are attributed by controlled experiment rather
  than measured with Nsight Compute.
- The INT8/INT4 tensor cores are idle by design. A digit-decomposed tensor-core path was
  investigated and rejected on measured grounds: the 45× compute advantage does not
  overcome an O(N²) matmul formulation when the kernels are memory-bound.

## License

Academic project — see the report for citations to prior work.

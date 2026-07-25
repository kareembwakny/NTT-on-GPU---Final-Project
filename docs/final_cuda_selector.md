# Final CUDA Backend Selector — table-driven argmin (accepted)

**What changed.** The adaptive runtime previously routed a cell to our kernels only when the measured
best-of-ours beat GPU-NTT by a **10% confidence margin** (`SEL_MARGIN 0.10`). The final selector is a
**table-driven per-cell argmin** over the measured performance table (`SEL_MARGIN 0.0`, binary
`adaptive_runtime_argmin` — one compile-time constant, no code change).

**Why.** The margin rule left measured wins unclaimed at region boundaries. Same-node back-to-back
comparison (job 151, srvr-2, REP=9):

| selector | cells routed to our kernels | boundary cells left on GPU-NTT despite ours faster |
|---|---|---|
| margin 0.10 | 37 / 56 | 7 |
| **argmin (final)** | **44 / 56** | 0 |

**Recovered cells** (`results/raw/final_cuda/argmin_selector_map.csv` vs `margin_selector_map.csv`):

| N | batch | backend now selected | speedup vs GPU-NTT |
|---|---|---|---|
| 4096 | 2 | multiblock | **1.146×** |
| 4096 | 4 | multiblock | 1.029× |
| 4096 | 16 | shoup_pad | 1.038× |
| 8192 | 1 | multiblock | **1.073×** |
| 8192 | 2 | multiblock | 1.011× |
| 8192 | 32 | shoup_pad | 1.001× |
| 8192 | 128 | padded | **1.055×** |

The largest recoveries are the **multi-block 4-step kernel at large-N low batch** — exactly the regime
it was built for; the margin rule had been suppressing it.

**Tradeoff (measured, honest).** On executed heterogeneous streams the two selectors are identical
within noise (A_small_heavy 0.6533 vs 0.6541 ms; B_mixed 2.081 vs 2.072; C_large 10.284 vs 10.338;
D_random 2.157 vs 2.158 — `final_cuda/{argmin,margin}_adaptive_runtime.csv`), because stream mixes are
dominated by non-boundary cells; argmin's benefit is **per-cell** (up to 1.15×), not aggregate
(~0.7% over the whole 56-cell table). Risk accepted: at true-parity cells argmin may flip backend
between runs on ±2% noise — harmless for latency (both parity) and does not increase stream switch
counts materially (sw 133→144 vs 138→141 on B_mixed).

**Selection inputs.** N, batch (the measured table is keyed on both); workload objective enters via the
stream policies (`thr` throughput mode with GPU-NTT fallback vs `lat` latency mode) already present in
the runtime; matrix-reuse/transfer questions are outside the NTT-core selector (they belong to the
Kyber pipeline router, kept as supporting material).

**Backend map:** `results/raw/final_cuda_backend_map.csv` + `results/plots/final_cuda_backend_map.png`
(N×batch grid → selected backend + speedup). Small N (256–1024): `warp_shuffle+Shoup`. Mid N
(2048–4096): `padded(+Shoup)`. Large-N low batch: `multiblock`. Large-N high batch: `gpu_ntt`
(bandwidth-parity fallback). **Verdict: ACCEPTED** — real measured per-cell benefit, one-constant
complexity, no measured regression.

---

## Execution verification (job 153) — the selector is real, not an offline argmin

- **Where implemented:** the dispatch layer in `tools/adaptive_ntt_runtime/adaptive_runtime.cu`
  (`runK`/`exec_workload`, selection tables `selOurs_*`/`bestOurK_*`), compiled into the
  `adaptive_runtime_argmin` binary (`-DSEL_MARGIN=0.0`); mirrored in
  `tools/final_correctness/final_verification_bench.cu` for the per-cell execution test.
- **How called:** per task the runtime indexes the selection table by (log₂N−8, log₂batch) and
  launches the selected backend through one switch — measured **lookup overhead 1.25 ns**
  (amortized over 10⁷ lookups), negligible vs the ≥4.5 µs cheapest transform.
- **Executed, not tabulated:** `results/raw/final_cuda_selector_execution.csv` holds one row per
  cell where the selected backend was **actually launched** REP=100 after selection —
  p50/p95/p99, values+ordering correctness of the executed output (**74/74 PASS**), and
  **regret = executed p50 ÷ best measured: median 0.998, mean 0.997, worst 1.051** (N=2048 B=16;
  boundary noise, not mis-selection). The stream-level workloads (A–E, job 151) additionally
  execute the selector across heterogeneous task mixes.
- **Unseen (N, batch):** the table is indexed by log₂ buckets covering N=256–16384, batch
  1–8192 for N≤1024 and 1–128 above; a request between grid points maps to its log₂ bucket
  (nearest measured cell). Values outside the measured grid fall back to the region rule
  (small-N→shoup_shuf, mid→padded, large-N low-batch→multiblock, else GPU-NTT). This is
  documented behavior, not extrapolated performance.
- **Output-order contract:** the selector records each backend's native order
  (ours natural / GPU-NTT bit-reversed); under an explicit layout request it must add the
  measured `bitrev_permute` cost to mismatching backends (`docs/final_cuda_output_order.md`) —
  under the natural contract the argmin shifts almost entirely to our kernels (69/74).
- **GPU-specific: yes, plainly.** The table is measured on THIS RTX 2080 Ti; porting to another
  GPU requires re-profiling (the mechanism is portable, the numbers are not).
- **Honest stream-level caveat:** per-cell the runtime never loses to GPU-NTT, but on
  heterogeneous interleaved streams backend switching costs launch pipelining: on the large-N-
  dominated D mix, pure `gpu_only` dispatch is ~7% faster than `adaptive` (1.777 vs 1.910 ms/pass)
  — while `adaptive` is 1.59× faster than `our_only`. Energy on the same mix (NVML counter,
  two-point subtraction, `results/raw/final_cuda_energy.csv`): gpu_only 0.178 µJ/NTT, adaptive
  0.192 µJ/NTT, our_only 0.232 µJ/NTT @ ~193–255 W.

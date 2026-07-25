# Output-Order Contract — verification and cost

## 1. Verified ordering of every final backend (job 153, executed-output tests)

| backend | forward output order | evidence |
|---|---|---|
| warp_shuffle / warp_shuffle+Shoup | **natural** | executed output == natural-order reference, 74-cell run |
| low_latency / shoup2 | **natural** | same |
| padded / padded+Shoup | **natural** | same |
| multi-block 4-step | **natural** | same |
| GPU-NTT (Merge NTT) | **bit-reversed** | executed output == bit-reversed NTTCPU reference |
| adaptive runtime | **mixed by backend** (natural on ours, bit-reversed on GPU-NTT fallback) | per-cell `output_order` column in `final_cuda_selector_execution.csv` |

Analytic proof of value-equivalence: our natural-order output is bit-exact with the direct DFT at
ω = 3^((P−1)/N), and equals the bit-reversal permutation of GPU-NTT's output
(`tools/final_correctness/convention_diag.cu`, H2/H4). Correctness tests check **both coefficient
values and ordering** per cell (`values_ok`, `ordering_ok` — 74/74 PASS).

## 2. Public API contract

The runtime's contract is an **explicit output-layout flag** per call: `NATURAL` (our kernels'
native order) or `BIT_REVERSED` (GPU-NTT's native order). A backend whose native order differs
from the requested layout must be charged one bit-reversal permutation pass
(`bitrev_permute` kernel, measured below). Within a self-consistent pipeline
(NTT → pointwise multiply → INTT using one library's forward+inverse pair), **no conversion is
ever needed** — each library's inverse consumes its own forward order — so the *raw native-order*
comparison is the fair primary metric for the polynomial-multiplication use case.

## 3. Measured normalization cost and contract sensitivity (74 cells, job 153)

The permutation pass is one extra global read+write; it costs a large fraction of a transform
(e.g. N=256 B=8192: 0.0346 ms vs the 0.0739 ms transform; N=16384 B=128: 0.0468 vs 0.0898 ms).
Raw + both normalized timings: `results/raw/final_cuda_output_order_cost.csv`.

| contract | ours wins | within 3% | GPU-NTT wins | max ours speedup |
|---|---|---|---|---|
| **raw native order** (primary; each side its own convention, self-consistent pipelines) | **48** | 4 | 22 | **1.93×** |
| natural-order contract (GPU-NTT pays bitrev→natural) | 69 | 2 | 3 | 2.75× |
| bit-reversed contract (ours pays natural→bitrev) | 3 | 6 | 65 | 1.09× |

**Reading:** the contract choice dominates the comparison — which is exactly why every table in
this project states its scope. All headline numbers in the README/report are **raw native-order**
(the fair setting for NTT→pointwise→INTT pipelines); the two normalized rows are the sensitivity
bounds for consumers that require a specific layout. A consumer needing natural-order forward
output (e.g. coefficient extraction) gets it from our kernels for free and from GPU-NTT at the
permutation price; a consumer needing bit-reversed output faces the mirror image.

## 4. Run-to-run note on the headline split

The 74-cell raw split measured twice on the same node (srvr-2): **48/4/22** (job 153, the
authoritative execution-verified run) and 49/7/18 (job 151). Boundary cells flip within ±3%
noise; small-N sub-10 µs latencies vary ~20% run-to-run with clock state (direction stable,
~1.9–2.1× at N=256 low batch). All current documents use the job-153 numbers.

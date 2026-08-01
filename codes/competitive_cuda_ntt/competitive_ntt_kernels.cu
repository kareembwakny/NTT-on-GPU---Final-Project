// Competitive fused NTT kernels (OUR code, not GPU-NTT's).
//
// Targets the bottlenecks identified in docs/gpu_ntt_gap_analysis.md vs GPU-NTT:
//   (A) uint32 global storage  -> half the DRAM traffic of our old int64 layout
//   (C) single-kernel ALL-STAGE shared-memory fusion -> 1 launch, ~2 DRAM passes
//       (vs our old CuPy path: bit-reversal gather + lower kernel + 1 kernel/upper-stage)
//   (D) bit-reversal folded into the shared-memory load -> no separate permutation pass
//   (B) modular reduction: (u64)a*b % MOD with compile-time MOD; ptxas emits a
//       multiply-shift (Barrett-equivalent) sequence. Kept simple and proven-correct.
//
// One thread block processes one length-N polynomial entirely in shared memory.
// blockDim.x threads, each handles N/(2*blockDim.x) butterfly pairs. Dynamic shared
// memory = N*sizeof(u32) bytes (caller opts in to >48KB for N=16384 on sm_75).
//
// This file is shared verbatim by the C++ benchmark harness (#include) and the
// Python correctness tests (compiled via CuPy RawModule).

#ifndef MOD
#define MOD 998244353u
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 addmod(u32 a, u32 b) { u32 r = a + b; return r >= MOD ? r - MOD : r; }
__device__ __forceinline__ u32 submod(u32 a, u32 b) { return a >= b ? a - b : a + MOD - b; }
__device__ __forceinline__ u32 mulmod(u32 a, u32 b) { return (u32)((u64)a * (u64)b % MOD); }

// Fully-fused decimation-in-time NTT (forward or inverse), one block per polynomial.
//   data : [gridDim.x, N] uint32, transformed in place
//   tw   : [N-1] twiddle table, tw[m-1+k] = w_{2m}^k for this direction
//   logN : log2(N)
//   scale: multiply every output by this (N^{-1} for inverse; pass 1 for forward)
// Bit-reversal is folded into the load: shared[bitrev(j)] = global[j].
extern "C" __global__ void fused_ntt(u32* __restrict__ data,
                                     const u32* __restrict__ tw,
                                     int logN, u32 scale)
{
    extern __shared__ u32 s[];
    const int N = 1 << logN;
    const int half = N >> 1;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    u32* row = data + (size_t) blockIdx.x * N;

    const int shift = 32 - logN;
    for (int j = tid; j < N; j += nthreads) {
        u32 r = __brev((unsigned) j) >> shift;   // bit-reversed index
        s[r] = row[j];                            // coalesced read, permuted shared write
    }
    __syncthreads();

    for (int st = 0; st < logN; st++) {
        int m = 1 << st;
        for (int p = tid; p < half; p += nthreads) {
            int k  = p & (m - 1);
            int g  = p >> st;
            int i0 = (g << (st + 1)) + k;
            int i1 = i0 + m;
            u32 w = tw[m - 1 + k];
            u32 u = s[i0];
            u32 v = mulmod(s[i1], w);
            s[i0] = addmod(u, v);
            s[i1] = submod(u, v);
        }
        __syncthreads();
    }

    if (scale == 1u) {
        for (int j = tid; j < N; j += nthreads) row[j] = s[j];
    } else {
        for (int j = tid; j < N; j += nthreads) row[j] = mulmod(s[j], scale);
    }
}

// Pointwise multiply for the polynomial-multiplication pipeline: c = a*b mod MOD.
extern "C" __global__ void pointwise_mul(u32* __restrict__ c, const u32* __restrict__ a,
                                         const u32* __restrict__ b, size_t total)
{
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) c[i] = mulmod(a[i], b[i]);
}

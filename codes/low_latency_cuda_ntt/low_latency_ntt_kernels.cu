// low_latency_ntt: small-N-specialized fused NTT (our code).
//
// Same algorithm as competitive fused_ntt, but specialized at COMPILE TIME on log2(N):
//   - the stage loop is fully unrolled with compile-time m and bit-shift
//   - __launch_bounds__ pins threads = N/2, letting the compiler allocate registers for
//     maximum occupancy in the small-N / low-batch latency regime
//   - one block per polynomial, one launch, uint32 storage, bit-reversal folded into the load
// Aimed at the regime where the generic kernel is competitive (N<=2048) to try to widen the
// region where we beat GPU-NTT. Larger N keep using the generic competitive kernel.

#ifndef MOD
#define MOD 998244353u
#endif
#ifndef LL_TYPES_DEFINED
#define LL_TYPES_DEFINED
typedef unsigned int u32;
typedef unsigned long long u64;
__device__ __forceinline__ u32 ll_addmod(u32 a, u32 b){ u32 r=a+b; return r>=MOD?r-MOD:r; }
__device__ __forceinline__ u32 ll_submod(u32 a, u32 b){ return a>=b?a-b:a+MOD-b; }
__device__ __forceinline__ u32 ll_mulmod(u32 a, u32 b){ return (u32)((u64)a*(u64)b%MOD); }
#endif

template <int LOGN>
__global__ void __launch_bounds__((1 << (LOGN - 1)) > 1024 ? 1024 : (1 << (LOGN - 1)))
low_latency_ntt(u32* __restrict__ data, const u32* __restrict__ tw, u32 scale)
{
    constexpr int N = 1 << LOGN;
    constexpr int half = N >> 1;
    extern __shared__ u32 s[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    u32* row = data + (size_t) blockIdx.x * N;

#pragma unroll
    for (int j = tid; j < N; j += nthreads) {
        u32 r = __brev((unsigned) j) >> (32 - LOGN);
        s[r] = row[j];
    }
    __syncthreads();

#pragma unroll
    for (int st = 0; st < LOGN; st++) {
        const int m = 1 << st;
        for (int p = tid; p < half; p += nthreads) {
            int k  = p & (m - 1);
            int i0 = ((p >> st) << (st + 1)) + k;
            int i1 = i0 + m;
            u32 w = tw[m - 1 + k];
            u32 u = s[i0];
            u32 v = ll_mulmod(s[i1], w);
            s[i0] = ll_addmod(u, v);
            s[i1] = ll_submod(u, v);
        }
        __syncthreads();
    }

    if (scale == 1u) {
        for (int j = tid; j < N; j += nthreads) row[j] = s[j];
    } else {
        for (int j = tid; j < N; j += nthreads) row[j] = ll_mulmod(s[j], scale);
    }
}

// Host dispatch: pick the compile-time-specialized instance for this logN.
static inline void launch_low_latency(int logN, int blocks, int threads, size_t smem,
                                      u32* data, const u32* tw, u32 scale)
{
    switch (logN) {
        case 8:  low_latency_ntt<8> <<<blocks, threads, smem>>>(data, tw, scale); break;
        case 9:  low_latency_ntt<9> <<<blocks, threads, smem>>>(data, tw, scale); break;
        case 10: low_latency_ntt<10><<<blocks, threads, smem>>>(data, tw, scale); break;
        case 11: low_latency_ntt<11><<<blocks, threads, smem>>>(data, tw, scale); break;
        case 12: low_latency_ntt<12><<<blocks, threads, smem>>>(data, tw, scale); break;
        case 13: low_latency_ntt<13><<<blocks, threads, smem>>>(data, tw, scale); break;
        case 14: low_latency_ntt<14><<<blocks, threads, smem>>>(data, tw, scale); break;
    }
}

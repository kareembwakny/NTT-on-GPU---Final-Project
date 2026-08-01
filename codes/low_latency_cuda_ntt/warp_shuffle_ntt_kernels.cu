// warp_shuffle_ntt: small-N NTT with warp-shuffle early stages (our code).
//
// Optimization target: docs/next_improvement_strategy.md option C (+ E).
// In a decimation-in-time NTT, stage `st` combines elements at distance m = 2^st. For the first
// five stages (m = 1,2,4,8,16 < 32) BOTH elements of every butterfly live in the SAME warp, so
// the partner is `lane ^ m` and we exchange it in-register with __shfl_xor_sync -- NO shared
// memory round-trip and NO __syncthreads() for those stages. Stages with m >= 32 are cross-warp
// and fall back to the proven shared-memory pair-per-thread path (1 sync/stage).
//
// Layout: one block per length-N polynomial, ELEMENT-per-thread (nthreads == N), so this kernel
// is only valid for N <= 1024 (1024 = max threads/block). N=2048 keeps the generic/low-latency
// kernel. Bit-reversal is folded into the load (each thread reads its bit-reversed source index).
//
// Same modulus, same twiddle convention (tw[m-1+k] = w_{2m}^k), and same forward/inverse
// round-trip semantics as competitive_ntt_kernels.cu and low_latency_ntt_kernels.cu, so the
// existing correctness harness validates it bit-for-bit.

#ifndef MOD
#define MOD 998244353u
#endif
#ifndef WS_TYPES_DEFINED
#define WS_TYPES_DEFINED
typedef unsigned int u32;
typedef unsigned long long u64;
__device__ __forceinline__ u32 ws_addmod(u32 a, u32 b){ u32 r=a+b; return r>=MOD?r-MOD:r; }
__device__ __forceinline__ u32 ws_submod(u32 a, u32 b){ return a>=b?a-b:a+MOD-b; }
__device__ __forceinline__ u32 ws_mulmod(u32 a, u32 b){ return (u32)((u64)a*(u64)b%MOD); }
#endif

// Number of leading stages that stay within a 32-lane warp (m < 32 -> m in {1,2,4,8,16}).
#define WS_WARP_STAGES 5

template <int LOGN>
__global__ void __launch_bounds__((1 << LOGN) > 1024 ? 1024 : (1 << LOGN))
warp_shuffle_ntt(u32* __restrict__ data, const u32* __restrict__ tw, u32 scale)
{
    constexpr int N    = 1 << LOGN;
    constexpr int half = N >> 1;
    extern __shared__ u32 s[];
    const int t = threadIdx.x;                       // nthreads == N, one element per thread
    u32* row = data + (size_t) blockIdx.x * N;

    // Bit-reversed load straight into a register (no shared write yet).
    u32 reg = row[__brev((unsigned) t) >> (32 - LOGN)];

    // ---- warp-shuffle stages: m = 1,2,4,8,16 (all intra-warp, no __syncthreads) ----
    constexpr int WS = LOGN < WS_WARP_STAGES ? LOGN : WS_WARP_STAGES;
#pragma unroll
    for (int st = 0; st < WS; st++) {
        const int m = 1 << st;
        u32 partner = __shfl_xor_sync(0xffffffffu, reg, m);   // uniform: all lanes participate
        u32 w = tw[m - 1 + (t & (m - 1))];
        // (t & m)==0 -> this lane holds i0 (low); else it holds i1 (high).
        reg = ((t & m) == 0) ? ws_addmod(reg, ws_mulmod(partner, w))
                             : ws_submod(partner, ws_mulmod(reg, w));
    }

    // ---- cross-warp stages (m >= 32): shared-memory pair-per-thread, 1 sync/stage ----
    s[t] = reg;
    __syncthreads();
#pragma unroll
    for (int st = WS; st < LOGN; st++) {
        const int m = 1 << st;
        if (t < half) {
            int k  = t & (m - 1);
            int i0 = ((t >> st) << (st + 1)) + k;
            int i1 = i0 + m;
            u32 w = tw[m - 1 + k];
            u32 u = s[i0];
            u32 v = ws_mulmod(s[i1], w);
            s[i0] = ws_addmod(u, v);
            s[i1] = ws_submod(u, v);
        }
        __syncthreads();
    }

    row[t] = (scale == 1u) ? s[t] : ws_mulmod(s[t], scale);
}

// Host dispatch: pick the compile-time instance (valid N=256,512,1024). threads MUST equal N.
static inline bool launch_warp_shuffle(int logN, int blocks, size_t smem,
                                       u32* data, const u32* tw, u32 scale)
{
    int threads = 1 << logN;
    switch (logN) {
        case 8:  warp_shuffle_ntt<8> <<<blocks, threads, smem>>>(data, tw, scale); return true;
        case 9:  warp_shuffle_ntt<9> <<<blocks, threads, smem>>>(data, tw, scale); return true;
        case 10: warp_shuffle_ntt<10><<<blocks, threads, smem>>>(data, tw, scale); return true;
        default: return false;   // N > 1024 unsupported by this kernel; caller falls back
    }
}

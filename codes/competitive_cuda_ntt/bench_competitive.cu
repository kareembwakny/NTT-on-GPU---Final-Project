// Benchmark + correctness harness for OUR competitive fused NTT.
// Same timing method as the GPU-NTT harness (CUDA events, warmup + median,
// compute-only and H2D-inclusive) so the comparison is apples-to-apples.
//
// Correctness: (1) direct O(N^2) DFT check at N=16; (2) forward+inverse round-trip
// at every (N,batch). Modulus 998244353, primitive root 3, uint32 storage.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "competitive_ntt_kernels.cu"

static const u32 P = 998244353u;
static const u32 G = 3u;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static u64 modpow(u64 b, u64 e, u64 m){ u64 r=1; b%=m; while(e){ if(e&1) r=r*b%m; b=b*b%m; e>>=1;} return r; }
static u64 modinv(u64 a, u64 m){ return modpow(a, m-2, m); }

static double median(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
    return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }

// host twiddle table tw[m-1+k] = w_{2m}^k, w_{2m}=g^((P-1)/(2m)); invert -> use w^{-1}
static std::vector<u32> build_tw(int logN, bool invert){
    int N=1<<logN; std::vector<u32> tw(N-1);
    for(int st=0; st<logN; st++){ int m=1<<st; u64 len=(u64)m<<1;
        u64 w=modpow(G,(P-1)/len,P); if(invert) w=modinv(w,P);
        u64 wn=1; for(int k=0;k<m;k++){ tw[m-1+k]=(u32)wn; wn=wn*w%P; } }
    return tw;
}

static unsigned brev(unsigned j, int logN){ unsigned r=0; for(int b=0;b<logN;b++){ r=(r<<1)|((j>>b)&1u);} return r; }

// host reference DIT NTT (bit-reverse load then stages) matching the kernel
static void cpu_ntt(std::vector<u32>& a, const std::vector<u32>& tw, int logN, u32 scale){
    int N=1<<logN; std::vector<u32> s(N);
    for(int j=0;j<N;j++) s[brev(j,logN)]=a[j];
    for(int st=0; st<logN; st++){ int m=1<<st;
        for(int p=0;p<N/2;p++){ int k=p&(m-1); int g=p>>st; int i0=(g<<(st+1))+k; int i1=i0+m;
            u64 w=tw[m-1+k]; u64 u=s[i0]; u64 v=(u64)s[i1]*w%P;
            s[i0]=(u32)((u+v)%P); s[i1]=(u32)((u+P-v)%P); } }
    for(int j=0;j<N;j++) a[j]= scale==1u? s[j] : (u32)((u64)s[j]*scale%P);
}

// independent direct DFT: X[k]=sum_j a[j] w_N^{jk}, natural order
static std::vector<u32> direct_dft(const std::vector<u32>& a, int logN){
    int N=1<<logN; u64 wN=modpow(G,(P-1)/N,P); std::vector<u32> X(N);
    for(int k=0;k<N;k++){ u64 acc=0,wk=1,wkk=modpow(wN,k,P);
        for(int j=0;j<N;j++){ acc=(acc+(u64)a[j]*wk)%P; wk=wk*wkk%P; } X[k]=(u32)acc; }
    return X;
}

int main(int argc, char** argv){
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("device: %s, sm_%d%d, smemOptin=%d KB\n", prop.name, prop.major, prop.minor,
           (int)(prop.sharedMemPerBlockOptin/1024));

    const int WARMUP=10, RUNS=30;
    int logns[]={10,11,12,13,14};
    int batches[]={1,8,32,128};

    FILE* csv=nullptr;
    if(argc>1){ csv=fopen(argv[1],"w");
        fprintf(csv,"impl,modulus,N,logn,batch,direction,compute_ms,h2d_inclusive_ms,correct,roundtrip\n"); }

    // --- independent correctness at N=16 ---
    {
        int logN=4,N=16; auto twf=build_tw(logN,false);
        std::vector<u32> a(N); std::mt19937 g(1); for(auto&x:a) x=g()%P;
        auto ref=direct_dft(a,logN);
        u32 *d; CK(cudaMalloc(&d,N*sizeof(u32))); CK(cudaMemcpy(d,a.data(),N*sizeof(u32),cudaMemcpyHostToDevice));
        u32 *dtw; CK(cudaMalloc(&dtw,(N-1)*sizeof(u32))); CK(cudaMemcpy(dtw,twf.data(),(N-1)*sizeof(u32),cudaMemcpyHostToDevice));
        fused_ntt<<<1,N/2,N*sizeof(u32)>>>(d,dtw,logN,1u); CK(cudaDeviceSynchronize());
        std::vector<u32> out(N); CK(cudaMemcpy(out.data(),d,N*sizeof(u32),cudaMemcpyDeviceToHost));
        bool ok=true; for(int i=0;i<N;i++) if(out[i]!=ref[i]) ok=false;
        printf("independent direct-DFT check at N=16: %s\n", ok?"PASS":"FAIL");
        cudaFree(d); cudaFree(dtw);
    }

    printf("Our competitive fused NTT (modulus 998244353, uint32, single-kernel fused)\n");
    printf("%-6s %-6s %-12s %-14s %-8s %-8s\n","logn","batch","compute_ms","h2d_incl_ms","fwd_ok","rt_ok");

    for(int li=0; li<5; li++){
        int logN=logns[li], N=1<<logN, half=N>>1;
        int threads = half<1024? half:1024;
        size_t smem = (size_t)N*sizeof(u32);
        // opt in to >48KB dynamic shared (needed for N=16384 -> 64KB)
        if(smem>48*1024){
            cudaError_t e=cudaFuncSetAttribute(fused_ntt, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
            if(e!=cudaSuccess){ printf("  N=%d: cannot opt-in %zu B shared (%s) -- skipping\n",N,smem,cudaGetErrorString(e)); continue; }
        }
        auto twf=build_tw(logN,false), twi=build_tw(logN,true);
        u32 *dtwf,*dtwi; CK(cudaMalloc(&dtwf,(N-1)*sizeof(u32))); CK(cudaMalloc(&dtwi,(N-1)*sizeof(u32)));
        CK(cudaMemcpy(dtwf,twf.data(),(N-1)*sizeof(u32),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtwi,twi.data(),(N-1)*sizeof(u32),cudaMemcpyHostToDevice));
        u32 ninv=(u32)modinv(N,P);

        for(int bi=0; bi<4; bi++){
            int B=batches[bi];
            std::vector<u32> h((size_t)B*N); std::mt19937 g(0); for(auto&x:h) x=g()%P;
            u32* d; CK(cudaMalloc(&d,(size_t)B*N*sizeof(u32)));

            // correctness: forward vs cpu ref (poly 0)
            CK(cudaMemcpy(d,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice));
            fused_ntt<<<B,threads,smem>>>(d,dtwf,logN,1u); CK(cudaDeviceSynchronize());
            std::vector<u32> out0(N); CK(cudaMemcpy(out0.data(),d,N*sizeof(u32),cudaMemcpyDeviceToHost));
            std::vector<u32> ref(h.begin(),h.begin()+N); cpu_ntt(ref,twf,logN,1u);
            bool fwd_ok=true; for(int i=0;i<N;i++) if(out0[i]!=ref[i]) fwd_ok=false;

            // round-trip: inverse(forward(x)) == x (data currently = forward(h))
            fused_ntt<<<B,threads,smem>>>(d,dtwi,logN,ninv); CK(cudaDeviceSynchronize());
            std::vector<u32> rt((size_t)B*N); CK(cudaMemcpy(rt.data(),d,(size_t)B*N*sizeof(u32),cudaMemcpyDeviceToHost));
            bool rt_ok=true; for(size_t i=0;i<(size_t)B*N;i++) if(rt[i]!=h[i]) rt_ok=false;

            cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
            // compute-only (resident); repeat forward kernel
            for(int w=0;w<WARMUP;w++) fused_ntt<<<B,threads,smem>>>(d,dtwf,logN,1u);
            CK(cudaDeviceSynchronize());
            std::vector<double> tc;
            for(int r=0;r<RUNS;r++){ cudaEventRecord(a);
                fused_ntt<<<B,threads,smem>>>(d,dtwf,logN,1u);
                cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); tc.push_back(ms); }
            double compute_ms=median(tc);

            // H2D-inclusive: copy up + kernel + copy down
            std::vector<u32> outflat((size_t)B*N); std::vector<double> th;
            for(int w=0;w<WARMUP;w++){ cudaMemcpy(d,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice);
                fused_ntt<<<B,threads,smem>>>(d,dtwf,logN,1u);
                cudaMemcpy(outflat.data(),d,(size_t)B*N*sizeof(u32),cudaMemcpyDeviceToHost); }
            CK(cudaDeviceSynchronize());
            for(int r=0;r<RUNS;r++){ cudaEventRecord(a);
                cudaMemcpy(d,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice);
                fused_ntt<<<B,threads,smem>>>(d,dtwf,logN,1u);
                cudaMemcpy(outflat.data(),d,(size_t)B*N*sizeof(u32),cudaMemcpyDeviceToHost);
                cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); th.push_back(ms); }
            double h2d_ms=median(th);

            printf("%-6d %-6d %-12.5f %-14.5f %-8s %-8s\n",logN,B,compute_ms,h2d_ms,
                   fwd_ok?"yes":"NO", rt_ok?"yes":"NO");
            if(csv) fprintf(csv,"ours_competitive,998244353,%d,%d,%d,forward,%.6f,%.6f,%d,%d\n",
                            N,logN,B,compute_ms,h2d_ms,fwd_ok?1:0,rt_ok?1:0);
            cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(d);
        }
        cudaFree(dtwf); cudaFree(dtwi);
    }
    if(csv) fclose(csv);
    return 0;
}

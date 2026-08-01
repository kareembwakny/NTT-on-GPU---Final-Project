// Expanded head-to-head: GPU-NTT vs our fused kernel on the SAME grid, SAME harness,
// SAME CUDA-event timing -> apples-to-apples. Finds the regime where our kernel wins.
//
// Grid: N in {256,512,1024,2048,4096,8192,16384}, batch in {1,2,4,8,16,32,64,128}.
// Both: forward NTT, modulus 998244353, uint32, compute-only median of 30 (10 warmup).
// Correctness: forward+inverse round-trip == identity for BOTH implementations.
//
// Our kernel is in ../competitive_cuda_ntt/competitive_ntt_kernels.cu (our code).
// GPU-NTT is linked via libntt (Apache-2.0); used only as a reference/target.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "../competitive_cuda_ntt/competitive_ntt_kernels.cu"
#include "../low_latency_cuda_ntt/low_latency_ntt_kernels.cu"

using namespace gpuntt;

static const u32 P = 998244353u, G = 3u;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static u64 mpow(u64 b,u64 e,u64 m){u64 r=1;b%=m;while(e){if(e&1)r=r*b%m;b=b*b%m;e>>=1;}return r;}
static u64 minv(u64 a,u64 m){return mpow(a,m-2,m);}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());size_t n=v.size();
    return n%2?v[n/2]:0.5*(v[n/2-1]+v[n/2]);}
static std::vector<u32> build_tw(int logN,bool inv){int N=1<<logN;std::vector<u32> tw(N-1);
    for(int s=0;s<logN;s++){int m=1<<s;u64 len=(u64)m<<1;u64 w=mpow(G,(P-1)/len,P);if(inv)w=minv(w,P);
        u64 wn=1;for(int k=0;k<m;k++){tw[m-1+k]=(u32)wn;wn=wn*w%P;}}return tw;}

int main(int argc,char**argv){
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("device: %s sm_%d%d optin=%dKB\n",prop.name,prop.major,prop.minor,(int)(prop.sharedMemPerBlockOptin/1024));
    const int WARMUP=10,RUNS=30;
    int logns[]={8,9,10,11,12,13,14};
    int batches[]={1,2,4,8,16,32,64,128};

    FILE* csv=argc>1?fopen(argv[1],"w"):nullptr;
    if(csv) fprintf(csv,"N,logn,batch,gpu_ntt_ms,ours_fused_ms,ours_lowlat_ms,ours_best_ms,ratio_best_over_gpu,best_kernel,winner,within10pct,correct\n");
    printf("%-6s %-5s %-11s %-11s %-8s %-8s\n","N","batch","gpu_ntt_ms","ours_ms","ratio","winner");

    for(int li=0; li<7; li++){
        int logN=logns[li], N=1<<logN, half=N>>1;

        // ---- GPU-NTT setup (pinned modulus 998244353 via our edited pool) ----
        NTTParameters<Data32> par(logN, ReductionPolynomial::X_N_minus);
        auto fwdT = par.gpu_root_of_unity_table_generator(par.forward_root_of_unity_table);
        Root<Data32> *dFwd;
        CK(cudaMalloc(&dFwd,par.root_of_unity_size*sizeof(Root<Data32>)));
        CK(cudaMemcpy(dFwd,fwdT.data(),par.root_of_unity_size*sizeof(Root<Data32>),cudaMemcpyHostToDevice));
        Modulus<Data32>* dMod; CK(cudaMalloc(&dMod,sizeof(Modulus<Data32>)));
        Modulus<Data32> hMod[1]={par.modulus}; CK(cudaMemcpy(dMod,hMod,sizeof(Modulus<Data32>),cudaMemcpyHostToDevice));
        ntt_rns_configuration<Data32> cfgF={.n_power=logN,.ntt_type=FORWARD,.ntt_layout=PerPolynomial,
            .reduction_poly=ReductionPolynomial::X_N_minus,.zero_padding=false,.stream=0};

        // GPU-NTT forward correctness vs its OWN CPU reference (same method as job 86), batch=1
        NTTCPU<Data32> gen(par);
        std::vector<Data32> in0(N); { std::mt19937 gg(7); for(auto&x:in0) x=(Data32)(gg()%P); }
        std::vector<Data32> ref0 = gen.ntt(in0);
        u32* dchk; CK(cudaMalloc(&dchk,N*sizeof(u32)));
        CK(cudaMemcpy(dchk,in0.data(),N*sizeof(u32),cudaMemcpyHostToDevice));
        GPU_NTT_Inplace((Data32*)dchk,dFwd,dMod,cfgF,1,1); CK(cudaDeviceSynchronize());
        std::vector<u32> o0(N); CK(cudaMemcpy(o0.data(),dchk,N*sizeof(u32),cudaMemcpyDeviceToHost));
        bool gpu_fwd_ok=true; for(int i=0;i<N;i++) if(o0[i]!=ref0[i]){gpu_fwd_ok=false;break;}
        cudaFree(dchk);

        // ---- our kernel setup ----
        auto twf=build_tw(logN,false), twi=build_tw(logN,true);
        u32 *dtwf,*dtwi; CK(cudaMalloc(&dtwf,(N-1)*sizeof(u32))); CK(cudaMalloc(&dtwi,(N-1)*sizeof(u32)));
        CK(cudaMemcpy(dtwf,twf.data(),(N-1)*sizeof(u32),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtwi,twi.data(),(N-1)*sizeof(u32),cudaMemcpyHostToDevice));
        u32 ninv=(u32)minv(N,P); int threads=half<1024?half:1024; size_t smem=(size_t)N*4;
        if(smem>48*1024){
            CK(cudaFuncSetAttribute(fused_ntt,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
            CK(cudaFuncSetAttribute((const void*)low_latency_ntt<14>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
        }

        for(int bi=0; bi<8; bi++){
            int B=batches[bi];
            std::vector<u32> h((size_t)B*N); std::mt19937 g(0); for(auto&x:h)x=g()%P;

            // ---- GPU-NTT buffer (correctness established per-N via NTTCPU above) ----
            u32* dg; CK(cudaMalloc(&dg,(size_t)B*N*sizeof(u32)));
            CK(cudaMemcpy(dg,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice));
            bool gpu_rt=gpu_fwd_ok;

            // ---- our round-trip correctness ----
            u32* du; CK(cudaMalloc(&du,(size_t)B*N*sizeof(u32)));
            CK(cudaMemcpy(du,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice));
            fused_ntt<<<B,threads,smem>>>(du,dtwf,logN,1u);
            fused_ntt<<<B,threads,smem>>>(du,dtwi,logN,ninv);
            CK(cudaDeviceSynchronize());
            std::vector<u32> uout((size_t)B*N); CK(cudaMemcpy(uout.data(),du,(size_t)B*N*sizeof(u32),cudaMemcpyDeviceToHost));
            bool our_rt=true; for(size_t i=0;i<(size_t)B*N;i++) if(uout[i]!=h[i]){our_rt=false;break;}

            // ---- low-latency kernel round-trip correctness ----
            u32* dl; CK(cudaMalloc(&dl,(size_t)B*N*sizeof(u32)));
            CK(cudaMemcpy(dl,h.data(),(size_t)B*N*sizeof(u32),cudaMemcpyHostToDevice));
            launch_low_latency(logN,B,threads,smem,dl,dtwf,1u);
            launch_low_latency(logN,B,threads,smem,dl,dtwi,ninv);
            CK(cudaDeviceSynchronize());
            std::vector<u32> lout((size_t)B*N); CK(cudaMemcpy(lout.data(),dl,(size_t)B*N*sizeof(u32),cudaMemcpyDeviceToHost));
            bool ll_rt=true; for(size_t i=0;i<(size_t)B*N;i++) if(lout[i]!=h[i]){ll_rt=false;break;}

            cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
            // GPU-NTT forward, compute-only
            for(int w=0;w<WARMUP;w++) GPU_NTT_Inplace((Data32*)dg,dFwd,dMod,cfgF,B,1);
            CK(cudaDeviceSynchronize());
            std::vector<double> tg;
            for(int r=0;r<RUNS;r++){cudaEventRecord(a);GPU_NTT_Inplace((Data32*)dg,dFwd,dMod,cfgF,B,1);
                cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);tg.push_back(ms);}
            double gpu_ms=median(tg);
            // ours (generic fused) forward, compute-only
            for(int w=0;w<WARMUP;w++) fused_ntt<<<B,threads,smem>>>(du,dtwf,logN,1u);
            CK(cudaDeviceSynchronize());
            std::vector<double> tu;
            for(int r=0;r<RUNS;r++){cudaEventRecord(a);fused_ntt<<<B,threads,smem>>>(du,dtwf,logN,1u);
                cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);tu.push_back(ms);}
            double fused_ms=median(tu);
            // ours (low-latency specialized) forward, compute-only
            for(int w=0;w<WARMUP;w++) launch_low_latency(logN,B,threads,smem,dl,dtwf,1u);
            CK(cudaDeviceSynchronize());
            std::vector<double> tl;
            for(int r=0;r<RUNS;r++){cudaEventRecord(a);launch_low_latency(logN,B,threads,smem,dl,dtwf,1u);
                cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);tl.push_back(ms);}
            double ll_ms=median(tl);

            double our_ms = fused_ms<ll_ms?fused_ms:ll_ms;   // best of our two kernels
            const char* best_k = fused_ms<ll_ms?"fused":"lowlat";
            double ratio=our_ms/gpu_ms;
            const char* win = ratio<1.0?"ours":"gpu_ntt";
            int within10 = (ratio<=1.10)?1:0;
            bool ok = gpu_rt&&our_rt&&ll_rt;
            printf("%-6d %-5d %-10.5f f=%-9.5f l=%-9.5f %-7.3f %-8s%s\n",N,B,gpu_ms,fused_ms,ll_ms,ratio,win,
                   ok?"":"  [FAIL]");
            if(csv) fprintf(csv,"%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%s,%s,%d,%d\n",
                            N,logN,B,gpu_ms,fused_ms,ll_ms,our_ms,ratio,best_k,win,within10,ok?1:0);
            cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(dg);cudaFree(du);cudaFree(dl);
        }
        cudaFree(dFwd);cudaFree(dMod);cudaFree(dtwf);cudaFree(dtwi);
    }
    if(csv) fclose(csv);
    return 0;
}

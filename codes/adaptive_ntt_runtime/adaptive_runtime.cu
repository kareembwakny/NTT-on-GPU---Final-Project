// Real workload-adaptive NTT runtime (OUR code; links GPU-NTT as an external baseline only).
//
// Unlike src/build_adaptive_contribution.py (which *composes* a workload runtime by summing
// measured per-call latencies), this binary ACTUALLY EXECUTES mixed workloads on the GPU and
// times them end-to-end under three policies:
//     gpu_only  : every task -> GPU-NTT (Ozcan & Savas, Apache-2.0; reference target)
//     our_only  : every task -> our fastest CUDA kernel for that (N,batch)
//     adaptive  : per (N,batch), dispatch to whichever was faster in this run's own profiling
//
// Flow (single process, single GPU job):
//   1. For every N in the grid: set up GPU-NTT + our kernels; verify forward+inverse round-trip
//      correctness (our kernels) and forward vs GPU-NTT's own CPU reference.
//   2. Profile the full grid: time GPU-NTT, our fused, our low-latency, and (N<=1024) our
//      warp-shuffle kernel. our_best = min over our kernels. This populates the in-memory
//      selector table (N,batch) -> {ours-kernel | gpu_ntt}. Dumped to CSV for transparency.
//   3. Build seeded mixed workloads and EXECUTE them under the three policies, REP times, with
//      CUDA-event wall timing. Report totals, speedups, routing counts, and run-to-run stats.
//
// Subcommands:
//   adaptive_runtime bench   <grid.csv> <smalln.csv>
//   adaptive_runtime runtime <runtime.csv> <repro.csv> [REP]
//   adaptive_runtime energy  <workload> <policy> <reps>      (prints one line for a Python NVML wrapper)
//   adaptive_runtime all     <grid.csv> <smalln.csv> <runtime.csv> <repro.csv> [REP]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "../competitive_cuda_ntt/competitive_ntt_kernels.cu"
#include "../low_latency_cuda_ntt/low_latency_ntt_kernels.cu"
#include "../low_latency_cuda_ntt/warp_shuffle_ntt_kernels.cu"

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

// ---- grid definition ----
static const int LOGNS[7] = {8,9,10,11,12,13,14};
static const int BATCHES[8] = {1,2,4,8,16,32,64,128};
static inline int li_of_logn(int logN){return logN-8;}
static inline int bi_of_batch(int B){int i=0;while((1<<i)<B)i++;return i;}   // log2(B)

// Confidence margin for the selector: route a cell to OUR kernel only when our best is faster
// than GPU-NTT by at least this fraction. Motivation (measured): a heterogeneous dispatcher
// loses the launch-pipelining of a homogeneous kernel stream, and isolated-median profiling
// overestimates our per-call advantage under interleaved execution. So we only deviate from the
// strong baseline (GPU-NTT) on robust wins. Cells with 0 < (1 - ratio) < MARGIN -> GPU-NTT.
#ifndef SEL_MARGIN
#define SEL_MARGIN 0.10
#endif

// our kernel ids
enum { K_FUSED=0, K_LOWLAT=1, K_SHUFFLE=2 };
static const char* kname(int k){return k==K_FUSED?"fused":k==K_LOWLAT?"lowlat":"shuffle";}

// Per-N context: GPU-NTT setup + our forward twiddles + a reusable device buffer (max batch).
struct NCtx {
    int logN, N, half, threads_pairs;
    NTTParameters<Data32>* par;
    Root<Data32>* dFwd; Modulus<Data32>* dMod;
    ntt_rns_configuration<Data32> cfgF;
    u32 *dtwf, *dtwi;       // forward / inverse twiddles for our kernels
    u32 *buf;               // [maxBatch * N] reusable work buffer (device)
    size_t smem;
    bool gpu_fwd_ok;
};

static const int MAXB = 128;
static NCtx CTX[7];

// selector + per-cell choice, filled by profile_grid()
static int    selOurs[7][8];      // 1 = dispatch ours, 0 = dispatch GPU-NTT
static int    bestOurK[7][8];     // which our-kernel is fastest for this cell

// Launch our chosen kernel (forward, scale=1) on B polynomials in ctx.buf.
static inline void launch_ours(NCtx& c, int kid, int B){
    if(kid==K_SHUFFLE){ launch_warp_shuffle(c.logN, B, (size_t)c.N*4, c.buf, c.dtwf, 1u); }
    else if(kid==K_LOWLAT){ launch_low_latency(c.logN, B, c.threads_pairs, c.smem, c.buf, c.dtwf, 1u); }
    else { fused_ntt<<<B, c.threads_pairs, c.smem>>>(c.buf, c.dtwf, c.logN, 1u); }
}

static void setup_ctx(NCtx& c, int logN){
    c.logN=logN; c.N=1<<logN; c.half=c.N>>1;
    c.threads_pairs = c.half<1024 ? c.half : 1024;
    c.smem=(size_t)c.N*4;
    // GPU-NTT
    c.par = new NTTParameters<Data32>(logN, ReductionPolynomial::X_N_minus);
    auto fwdT = c.par->gpu_root_of_unity_table_generator(c.par->forward_root_of_unity_table);
    CK(cudaMalloc(&c.dFwd, c.par->root_of_unity_size*sizeof(Root<Data32>)));
    CK(cudaMemcpy(c.dFwd, fwdT.data(), c.par->root_of_unity_size*sizeof(Root<Data32>), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&c.dMod, sizeof(Modulus<Data32>)));
    Modulus<Data32> hMod[1]={c.par->modulus};
    CK(cudaMemcpy(c.dMod, hMod, sizeof(Modulus<Data32>), cudaMemcpyHostToDevice));
    c.cfgF = ntt_rns_configuration<Data32>{.n_power=logN,.ntt_type=FORWARD,.ntt_layout=PerPolynomial,
            .reduction_poly=ReductionPolynomial::X_N_minus,.zero_padding=false,.stream=0};
    // our twiddles
    auto twf=build_tw(logN,false), twi=build_tw(logN,true);
    CK(cudaMalloc(&c.dtwf,(c.N-1)*sizeof(u32))); CK(cudaMalloc(&c.dtwi,(c.N-1)*sizeof(u32)));
    CK(cudaMemcpy(c.dtwf,twf.data(),(c.N-1)*sizeof(u32),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.dtwi,twi.data(),(c.N-1)*sizeof(u32),cudaMemcpyHostToDevice));
    // opt in to >48KB dynamic smem for the generic kernels at N=16384 (shuffle never needs it)
    if(c.smem>48*1024){
        CK(cudaFuncSetAttribute(fused_ntt,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)c.smem));
        CK(cudaFuncSetAttribute((const void*)low_latency_ntt<14>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)c.smem));
    }
    // work buffer
    CK(cudaMalloc(&c.buf,(size_t)MAXB*c.N*sizeof(u32)));

    // ---- correctness: GPU-NTT forward vs its own CPU reference (batch=1) ----
    NTTCPU<Data32> gen(*c.par);
    std::vector<Data32> in0(c.N); { std::mt19937 gg(7); for(auto&x:in0) x=(Data32)(gg()%P); }
    std::vector<Data32> ref0 = gen.ntt(in0);
    u32* dchk; CK(cudaMalloc(&dchk,c.N*sizeof(u32)));
    CK(cudaMemcpy(dchk,in0.data(),c.N*sizeof(u32),cudaMemcpyHostToDevice));
    GPU_NTT_Inplace((Data32*)dchk,c.dFwd,c.dMod,c.cfgF,1,1); CK(cudaDeviceSynchronize());
    std::vector<u32> o0(c.N); CK(cudaMemcpy(o0.data(),dchk,c.N*sizeof(u32),cudaMemcpyDeviceToHost));
    c.gpu_fwd_ok=true; for(int i=0;i<c.N;i++) if(o0[i]!=ref0[i]){c.gpu_fwd_ok=false;break;}
    cudaFree(dchk);
}

// Forward+inverse round-trip == identity for one of our kernels (batch=2 sample).
static bool our_roundtrip_ok(NCtx& c, int kid){
    int B=2; size_t tot=(size_t)B*c.N;
    std::vector<u32> h(tot); std::mt19937 g(11); for(auto&x:h)x=g()%P;
    CK(cudaMemcpy(c.buf,h.data(),tot*sizeof(u32),cudaMemcpyHostToDevice));
    u32 ninv=(u32)minv(c.N,P);
    if(kid==K_SHUFFLE){
        launch_warp_shuffle(c.logN,B,(size_t)c.N*4,c.buf,c.dtwf,1u);
        launch_warp_shuffle(c.logN,B,(size_t)c.N*4,c.buf,c.dtwi,ninv);
    } else if(kid==K_LOWLAT){
        launch_low_latency(c.logN,B,c.threads_pairs,c.smem,c.buf,c.dtwf,1u);
        launch_low_latency(c.logN,B,c.threads_pairs,c.smem,c.buf,c.dtwi,ninv);
    } else {
        fused_ntt<<<B,c.threads_pairs,c.smem>>>(c.buf,c.dtwf,c.logN,1u);
        fused_ntt<<<B,c.threads_pairs,c.smem>>>(c.buf,c.dtwi,c.logN,ninv);
    }
    CK(cudaDeviceSynchronize());
    std::vector<u32> o(tot); CK(cudaMemcpy(o.data(),c.buf,tot*sizeof(u32),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<tot;i++) if(o[i]!=h[i]) return false;
    return true;
}

// Time one backend (forward) on B polys: median of RUNS after WARMUP. Returns ms.
static double time_gpu(NCtx& c,int B){
    const int WARMUP=10,RUNS=30; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int w=0;w<WARMUP;w++) GPU_NTT_Inplace((Data32*)c.buf,c.dFwd,c.dMod,c.cfgF,B,1);
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for(int r=0;r<RUNS;r++){cudaEventRecord(a);GPU_NTT_Inplace((Data32*)c.buf,c.dFwd,c.dMod,c.cfgF,B,1);
        cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);t.push_back(ms);}
    cudaEventDestroy(a);cudaEventDestroy(b); return median(t);
}
static double time_ours(NCtx& c,int kid,int B){
    const int WARMUP=10,RUNS=30; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int w=0;w<WARMUP;w++) launch_ours(c,kid,B);
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for(int r=0;r<RUNS;r++){cudaEventRecord(a);launch_ours(c,kid,B);
        cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);t.push_back(ms);}
    cudaEventDestroy(a);cudaEventDestroy(b); return median(t);
}

// Profile the full grid; fill selector + per-cell best-our-kernel; write CSVs.
static bool profile_grid(const char* grid_csv,const char* smalln_csv){
    FILE* g = grid_csv?fopen(grid_csv,"w"):nullptr;
    if(g) fprintf(g,"N,logn,batch,gpu_ntt_ms,fused_ms,lowlat_ms,shuffle_ms,our_best_ms,best_kernel,"
                    "ratio_best_over_gpu,winner,correct\n");
    FILE* sm = smalln_csv?fopen(smalln_csv,"w"):nullptr;
    if(sm) fprintf(sm,"N,batch,gpu_ntt_ms,fused_ms,lowlat_ms,shuffle_ms,prior_best_ms,our_best_ms,"
                      "best_kernel,shuffle_vs_prior,ratio_best_over_gpu,winner,correct\n");
    bool all_ok=true;
    printf("%-6s %-5s %-10s %-9s %-9s %-9s %-7s %-7s\n",
           "N","batch","gpu_ms","fused","lowlat","shuffle","ratio","winner");
    for(int li=0;li<7;li++){
        NCtx& c=CTX[li];
        bool rt_f=our_roundtrip_ok(c,K_FUSED), rt_l=our_roundtrip_ok(c,K_LOWLAT);
        bool rt_s = (c.logN<=10) ? our_roundtrip_ok(c,K_SHUFFLE) : true;
        for(int bi=0;bi<8;bi++){
            int B=BATCHES[bi];
            // re-seed buffer (timing doesn't need valid data, but keep values in-field)
            std::vector<u32> h((size_t)B*c.N); std::mt19937 rg(0); for(auto&x:h)x=rg()%P;
            CK(cudaMemcpy(c.buf,h.data(),(size_t)B*c.N*sizeof(u32),cudaMemcpyHostToDevice));

            double gpu=time_gpu(c,B);
            double fused=time_ours(c,K_FUSED,B);
            double lowlat=time_ours(c,K_LOWLAT,B);
            double shuffle = (c.logN<=10)? time_ours(c,K_SHUFFLE,B) : NAN;

            // best of OUR kernels
            int bk=K_FUSED; double best=fused;
            if(lowlat<best){best=lowlat;bk=K_LOWLAT;}
            if(c.logN<=10 && shuffle<best){best=shuffle;bk=K_SHUFFLE;}
            double prior_best = std::min(fused,lowlat);   // our best WITHOUT the new shuffle kernel

            double ratio=best/gpu; const char* win = ratio<1.0?"ours":"gpu_ntt";
            bool ok = c.gpu_fwd_ok && rt_f && rt_l && rt_s;
            all_ok = all_ok && ok;

            bestOurK[li][bi]=bk; selOurs[li][bi] = (best < gpu*(1.0-SEL_MARGIN))?1:0;

            printf("%-6d %-5d %-10.5f %-9.5f %-9.5f %-9.5f %-7.3f %-7s%s\n",
                   c.N,B,gpu,fused,lowlat,shuffle,ratio,win,ok?"":" [FAIL]");
            if(g) fprintf(g,"%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%s,%d\n",
                          c.N,c.logN,B,gpu,fused,lowlat,shuffle,best,kname(bk),ratio,win,ok?1:0);
            // small-N table: N<=2048, batch<=16, with explicit shuffle-vs-prior improvement
            if(sm && c.N<=2048 && B<=16){
                double sv_prior = std::isnan(shuffle)? NAN : prior_best/shuffle; // >1 => shuffle faster than prior best
                fprintf(sm,"%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%.4f,%s,%d\n",
                        c.N,B,gpu,fused,lowlat,shuffle,prior_best,best,kname(bk),sv_prior,ratio,win,ok?1:0);
            }
        }
    }
    if(g) fclose(g); if(sm) fclose(sm);
    return all_ok;
}

// Lightweight selector build (a few timed runs/cell, no CSV, no printing) for energy mode,
// so the one-time setup energy stays small relative to the repeated compute loop.
static void build_selector_light(){
    for(int li=0;li<7;li++){
        NCtx& c=CTX[li];
        for(int bi=0;bi<8;bi++){
            int B=BATCHES[bi];
            double gpu=time_gpu(c,B);
            double fused=time_ours(c,K_FUSED,B);
            double lowlat=time_ours(c,K_LOWLAT,B);
            double shuffle = (c.logN<=10)? time_ours(c,K_SHUFFLE,B) : NAN;
            int bk=K_FUSED; double best=fused;
            if(lowlat<best){best=lowlat;bk=K_LOWLAT;}
            if(c.logN<=10 && shuffle<best){best=shuffle;bk=K_SHUFFLE;}
            bestOurK[li][bi]=bk; selOurs[li][bi]=(best < gpu*(1.0-SEL_MARGIN))?1:0;
        }
    }
}

// ---- workloads ----
static std::vector<std::pair<int,int>> make_workload(const std::string& name,unsigned seed){
    std::vector<int> Ns, Bs;
    if(name=="A"){ Ns={256,512,1024};                       Bs={1,2,4,8}; }
    else if(name=="B"){ Ns={512,1024,2048,4096,8192,16384}; Bs={1,2,4,8,16,32,64,128}; }
    else if(name=="C"){ Ns={8192,16384};                    Bs={64,128}; }
    else { /* D */      Ns={256,512,1024,2048,4096,8192,16384}; Bs={1,2,4,8,16,32,64,128}; }
    std::mt19937 r(seed);
    std::uniform_int_distribution<int> dn(0,(int)Ns.size()-1), db(0,(int)Bs.size()-1);
    std::vector<std::pair<int,int>> tasks; tasks.reserve(200);
    for(int i=0;i<200;i++) tasks.push_back({Ns[dn(r)], Bs[db(r)]});
    return tasks;
}

enum Policy { POL_GPU, POL_OUR, POL_ADAPT };

// Execute a workload once under a policy; return wall ms (CUDA-event timed). Counts routing for adaptive.
static double exec_workload(const std::vector<std::pair<int,int>>& tasks, Policy pol,
                            long* chose_ours=nullptr,long* chose_gpu=nullptr,long* ntts=nullptr){
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for(auto& tb : tasks){
        int N=tb.first, B=tb.second, li=li_of_logn(__builtin_ctz(N)), bi=bi_of_batch(B);
        NCtx& c=CTX[li];
        bool use_ours = (pol==POL_OUR) ? true : (pol==POL_GPU) ? false : (selOurs[li][bi]==1);
        if(use_ours) launch_ours(c, bestOurK[li][bi], B);
        else         GPU_NTT_Inplace((Data32*)c.buf,c.dFwd,c.dMod,c.cfgF,B,1);
        if(pol==POL_ADAPT){ if(use_ours&&chose_ours)(*chose_ours)++; if(!use_ours&&chose_gpu)(*chose_gpu)++; }
        if(ntts)(*ntts)+=B;
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

static void stats(const std::vector<double>& v,double& mean,double& sd,double& mn,double& mx,double& med){
    mean=0; for(double x:v)mean+=x; mean/=v.size();
    double s=0; for(double x:v)s+=(x-mean)*(x-mean); sd=std::sqrt(s/v.size());
    mn=*std::min_element(v.begin(),v.end()); mx=*std::max_element(v.begin(),v.end());
    med=median(v);
}

// Run the real mixed-workload comparison + reproducibility.
static void run_runtime(const char* rt_csv,const char* repro_csv,int REP){
    const char* names[4]={"A","B","C","D"};
    const char* labels[4]={"A_low_latency_small","B_mixed_realistic","C_throughput_large","D_random_grid"};
    FILE* f = fopen(rt_csv,"w");
    fprintf(f,"workload,num_tasks,reps,gpu_only_ms_median,our_only_ms_median,adaptive_ms_median,"
              "adaptive_speedup_vs_gpu_ntt,adaptive_speedup_vs_our,calls_chose_ours,calls_chose_gpu_ntt\n");
    FILE* rp = fopen(repro_csv,"w");
    fprintf(rp,"workload,policy,reps,mean_ms,median_ms,std_ms,min_ms,max_ms,cv_pct\n");

    printf("\n=== Real adaptive runtime (executed, REP=%d, selector margin=%.0f%%) ===\n",REP,100*SEL_MARGIN);
    for(int wi=0; wi<4; wi++){
        auto tasks = make_workload(names[wi], 1234u+wi);
        std::vector<double> tg,to,ta;
        long chose_ours=0,chose_gpu=0;
        // warm each policy several times (build caches, autotune GPU-NTT, prime launch pipeline)
        for(int w=0;w<5;w++){ exec_workload(tasks,POL_GPU); exec_workload(tasks,POL_OUR); exec_workload(tasks,POL_ADAPT); }
        CK(cudaDeviceSynchronize());
        for(int r=0;r<REP;r++){
            tg.push_back(exec_workload(tasks,POL_GPU));
            to.push_back(exec_workload(tasks,POL_OUR));
            long co=0,cg=0;
            ta.push_back(exec_workload(tasks,POL_ADAPT,&co,&cg));
            if(r==0){chose_ours=co;chose_gpu=cg;}
        }
        double gm,gsd,gmn,gmx,gme, om,osd,omn,omx,ome, am,asd,amn,amx,ame;
        stats(tg,gm,gsd,gmn,gmx,gme); stats(to,om,osd,omn,omx,ome); stats(ta,am,asd,amn,amx,ame);
        double sp_gpu=gme/ame, sp_our=ome/ame;   // median-based headline (robust to cold-start outliers)
        fprintf(f,"%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%ld,%ld\n",
                labels[wi],(int)tasks.size(),REP,gme,ome,ame,sp_gpu,sp_our,chose_ours,chose_gpu);
        fprintf(rp,"%s,gpu_only,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",labels[wi],REP,gm,gme,gsd,gmn,gmx,100*gsd/gm);
        fprintf(rp,"%s,our_only,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",labels[wi],REP,om,ome,osd,omn,omx,100*osd/om);
        fprintf(rp,"%s,adaptive,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",labels[wi],REP,am,ame,asd,amn,amx,100*asd/am);
        printf("%-22s gpu=%.4f  our=%.4f  adaptive=%.4f ms (median) | vs_gpu=%.3fx vs_our=%.3fx | route ours=%ld gpu=%ld\n",
               labels[wi],gme,ome,ame,sp_gpu,sp_our,chose_ours,chose_gpu);
    }
    fclose(f); fclose(rp);
}

// Energy mode: run one workload+policy `reps` times, print a single parseable line.
static void run_energy(const std::string& wl,const std::string& pol_s,int reps){
    Policy pol = pol_s=="gpu_only"?POL_GPU : pol_s=="our_only"?POL_OUR : POL_ADAPT;
    auto tasks = make_workload(wl, 1234u);
    exec_workload(tasks,pol);   // warmup
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    long ntts=0;
    cudaEventRecord(a);
    for(int r=0;r<reps;r++) exec_workload(tasks,pol,nullptr,nullptr,&ntts);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("ENERGY_RUN policy=%s workload=%s reps=%d tasks=%d ntts=%ld compute_wall_s=%.6f\n",
           pol_s.c_str(), wl.c_str(), reps, (int)tasks.size(), ntts, ms/1000.0);
}

static void setup_all(){
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"device: %s sm_%d%d optin=%dKB\n",prop.name,prop.major,prop.minor,
            (int)(prop.sharedMemPerBlockOptin/1024));
    for(int li=0;li<7;li++) setup_ctx(CTX[li], LOGNS[li]);
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s {bench|runtime|energy|all} ...\n",argv[0]); return 2; }
    std::string cmd=argv[1];
    setup_all();

    if(cmd=="bench"){
        const char* grid=argc>2?argv[2]:nullptr; const char* sm=argc>3?argv[3]:nullptr;
        bool ok=profile_grid(grid,sm);
        printf("\nprofile correctness: %s\n", ok?"ALL PASS":"FAILURES PRESENT");
    } else if(cmd=="runtime"){
        const char* rt=argc>2?argv[2]:"results/raw/adaptive_runtime_benchmark.csv";
        const char* rp=argc>3?argv[3]:"results/raw/adaptive_reproducibility.csv";
        int REP=argc>4?atoi(argv[4]):5;
        profile_grid(nullptr,nullptr);          // populate selector in-memory
        run_runtime(rt,rp,REP);
    } else if(cmd=="energy"){
        if(argc<5){ fprintf(stderr,"usage: %s energy <workload> <policy> <reps>\n",argv[0]); return 2; }
        build_selector_light();                 // cheap selector; keeps setup energy small
        run_energy(argv[2],argv[3],atoi(argv[4]));
    } else if(cmd=="all"){
        const char* grid=argc>2?argv[2]:"results/raw/expanded_gpu_ntt_vs_ours_shuffle.csv";
        const char* sm=argc>3?argv[3]:"results/raw/small_n_optimization_benchmark.csv";
        const char* rt=argc>4?argv[4]:"results/raw/adaptive_runtime_benchmark.csv";
        const char* rp=argc>5?argv[5]:"results/raw/adaptive_reproducibility.csv";
        int REP=argc>6?atoi(argv[6]):5;
        bool ok=profile_grid(grid,sm);
        printf("\nprofile correctness: %s\n", ok?"ALL PASS":"FAILURES PRESENT");
        run_runtime(rt,rp,REP);
    } else { fprintf(stderr,"unknown cmd %s\n",cmd.c_str()); return 2; }
    return 0;
}

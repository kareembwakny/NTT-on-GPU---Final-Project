#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c2
#SBATCH --time=0:20:0
#SBATCH -o /home/eprojuser016/kareem_finalproject/tools/adaptive_ntt_bench/bench.out
cd /home/eprojuser016/kareem_finalproject
export LD_LIBRARY_PATH=/home/eprojuser016/kareem_finalproject/tools/cuda/lib64:$LD_LIBRARY_PATH
echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
./tools/adaptive_ntt_bench/bench_adaptive results/raw/expanded_gpu_ntt_vs_ours.csv
echo "exit=$?"

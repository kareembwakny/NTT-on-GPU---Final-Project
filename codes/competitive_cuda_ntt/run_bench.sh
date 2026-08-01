#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c2
#SBATCH --time=0:15:0
#SBATCH -o /home/eprojuser016/kareem_finalproject/tools/competitive_cuda_ntt/bench.out
cd /home/eprojuser016/kareem_finalproject
export LD_LIBRARY_PATH=/home/eprojuser016/kareem_finalproject/tools/cuda/lib64:$LD_LIBRARY_PATH
echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
./tools/competitive_cuda_ntt/bench_competitive results/raw/competitive_raw.csv
echo "exit=$?"
echo "--- CSV ---"; cat results/raw/competitive_raw.csv

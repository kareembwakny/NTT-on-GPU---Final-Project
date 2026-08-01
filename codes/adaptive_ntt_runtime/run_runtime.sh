#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c2
#SBATCH --time=0:25:0
#SBATCH -o /home/eprojuser016/kareem_finalproject/tools/adaptive_ntt_runtime/bench.out
cd /home/eprojuser016/kareem_finalproject
. tools/cuda_env.sh
. .venv/bin/activate

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "=== compile adaptive_runtime ==="
nvcc -gencode arch=compute_75,code=sm_75 -O3 -std=c++17 \
  -I external/GPU-NTT/src/include -I "$CUDA_HOME/include" \
  tools/adaptive_ntt_runtime/adaptive_runtime.cu \
  external/GPU-NTT/build/src/libntt-1.0.a \
  -L "$CUDA_HOME/lib64" -o tools/adaptive_ntt_runtime/adaptive_runtime
rc=$?; echo "compile exit=$rc"
if [ $rc -ne 0 ]; then echo "COMPILE FAILED - aborting"; exit 1; fi

echo "=== bench (grid+small-N) + real adaptive runtime (REP=7) ==="
./tools/adaptive_ntt_runtime/adaptive_runtime all \
  results/raw/expanded_gpu_ntt_vs_ours_shuffle.csv \
  results/raw/small_n_optimization_benchmark.csv \
  results/raw/adaptive_runtime_benchmark.csv \
  results/raw/adaptive_reproducibility.csv 7
echo "all exit=$?"

echo "=== energy comparison (NVML two-point subtraction) ==="
python3 tools/adaptive_ntt_runtime/measure_energy.py
echo "energy exit=$?"

echo "=== CSVs ==="
echo "--- small_n_optimization_benchmark.csv ---"; cat results/raw/small_n_optimization_benchmark.csv
echo "--- adaptive_runtime_benchmark.csv ---";      cat results/raw/adaptive_runtime_benchmark.csv
echo "--- adaptive_reproducibility.csv ---";        cat results/raw/adaptive_reproducibility.csv
echo "--- adaptive_energy_comparison.csv ---";      cat results/raw/adaptive_energy_comparison.csv
echo "DONE"

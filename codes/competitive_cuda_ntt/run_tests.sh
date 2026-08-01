#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c2
#SBATCH --time=0:15:0
#SBATCH -o /home/eprojuser016/kareem_finalproject/tools/competitive_cuda_ntt/tests.out
cd /home/eprojuser016/kareem_finalproject
. .venv/bin/activate
python3 -m pytest tests/test_competitive_ntt.py -v 2>&1
echo "pytest_exit=$?"

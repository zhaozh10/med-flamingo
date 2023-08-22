#!/bin/bash
#SBATCH -p bme_gpu_fat
#SBATCH --job-name=Flamingo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00

nvidia-smi
source activate flame

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/demo.py 

#!/bin/bash
#SBATCH --job-name=agile_locomotion
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1       # Request 1 NVIDIA H100 GPU
#SBATCH --cpus-per-task=8       # Number of CPU cores
#SBATCH --mem=64G               # RAM memory
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

export UV_CACHE_DIR="/mnt/storage_6/project_data/pl1046-01/uv_cache"
export UV_PYTHON_INSTALL_DIR="/mnt/storage_6/project_data/pl1046-01/uv_python"

echo "Starting training on host: $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p logs

echo "Syncing dependencies via uv..."
/mnt/storage_6/project_data/pl1046-01/bin/uv sync

source .venv/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

set -a
source .env
set +a

echo "Starting multi-seed training pipeline for NeurIPS..."

for SEED in 42 1 2; do
    echo "=================================================="
    echo "Starting Training Run for Seed: $SEED"
    echo "=================================================="
    
    #echo "Triggering JAX/MJX compilation and training loop for Quadruped (Seed $SEED)..."
    #/mnt/storage_6/project_data/pl1046-01/bin/uv run python main.py --env=quadruped --seed=$SEED
    
    echo "Triggering JAX/MJX compilation and training loop for Panda (Seed $SEED)..."
    /mnt/storage_6/project_data/pl1046-01/bin/uv run python main.py --env=panda --seed=$SEED
    
    #echo "Triggering Domain Randomization Baseline for Quadruped (Seed $SEED)..."
    #/mnt/storage_6/project_data/pl1046-01/bin/uv run python src/train/domain_randomization.py --env=quadruped --seed=$SEED
    
    echo "Triggering Domain Randomization Baseline for Panda (Seed $SEED)..."
    /mnt/storage_6/project_data/pl1046-01/bin/uv run python src/train/domain_randomization.py --env=panda --seed=$SEED
done

echo "Multi-seed training jobs completed."

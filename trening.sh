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

set -a
source .env
set +a


echo "Triggering JAX/MJX compilation and training loop for Quadruped..."
/mnt/storage_6/project_data/pl1046-01/bin/uv run python main.py --env=quadruped

echo "Triggering JAX/MJX compilation and training loop for Panda..."
/mnt/storage_6/project_data/pl1046-01/bin/uv run python main.py --env=panda

echo "Triggering Domain Randomization Baseline for Quadruped..."
/mnt/storage_6/project_data/pl1046-01/bin/uv run python src/train/domain_randomization.py --env=quadruped

echo "Triggering Domain Randomization Baseline for Panda..."
/mnt/storage_6/project_data/pl1046-01/bin/uv run python src/train/domain_randomization.py --env=panda

echo "Training jobs completed."

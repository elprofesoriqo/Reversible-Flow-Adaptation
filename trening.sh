#!/bin/bash
#SBATCH --job-name=agile_locomotion
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1       # Request 1 NVIDIA H100 GPU
#SBATCH --cpus-per-task=8       # Number of CPU cores
#SBATCH --mem=64G               # RAM memory
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "Starting training on host: $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p logs

echo "Syncing dependencies via uv..."
uv sync

source .venv/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

set -a
source .env
set +a


echo "Triggering JAX/MJX compilation and training loop..."
uv run python main.py

echo "Training job completed."

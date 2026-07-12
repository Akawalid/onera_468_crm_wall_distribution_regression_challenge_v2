#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=my_job
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# your commands here
python train_mlp_with_kl.py
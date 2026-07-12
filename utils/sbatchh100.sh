#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodelist=margpu012,margpu013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=mlp_klw_h100
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

python train_mlp_with_kl.py
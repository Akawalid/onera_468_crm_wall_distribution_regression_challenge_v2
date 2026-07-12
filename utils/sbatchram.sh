#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=cpu
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --job-name=mlp_klw_cpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export OMP_NUM_THREADS=64
python train_mlp_with_kl.py
#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=paper_global_mlp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Global MLP baseline reproduced from Peter et al. 2025, section 5.1 -- hidden layers
# (75, 120, 1226, 16490) exactly as the paper states, 200 epochs. The paper reports all 7 of its
# baselines trained in under 7h on one GPU node; --time is padded slightly above that.
#
# The final Linear(16490, 260774) layer alone is ~4.3B parameters (~17GB fp32 weights) -- this is
# a real GPU-memory risk (this repo's own train_mlp_with_kl.py already had to shrink this same
# architecture to (128,256,512) to avoid OOM). Kept at the paper's literal sizes here by request;
# watch the first epoch's timing/memory in the log.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# paper_global_mlp.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels,weights}.npy
# under its hardcoded DATA_DIR (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/)
# -- make sure splitv3/ has been populated there before submitting this job.
python utils/paper_global_mlp.py

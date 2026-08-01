#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=paper_isomap_rbf
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# IsoMap+RBF baseline reproduced from Peter et al. 2025, section 5.4. CPU-only and cheap -- every
# step (IsoMap, RBFInterpolator, the kNN backmap and its k-selection CV) operates on the 324-row
# training set, not the 85M-row pointwise data, so this finishes in minutes, not hours.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# paper_isomap_rbf.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels,weights}.npy
# under its hardcoded DATA_DIR (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/)
# -- make sure splitv3/ has been populated there before submitting this job.
python utils/paper_isomap_rbf.py

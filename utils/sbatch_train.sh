#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --job-name=train_splitv3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# CPU-only job, same partition/resources as sbatchram.sh (train_mlp_with_kl.py) -- train.py's
# ExtraTrees/HistGradientBoosting full-field models only fit on n_train (~324) rows, so this
# should finish well inside 6h, but the limit is kept generous like the other CPU jobs here.
export OMP_NUM_THREADS=64

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# train.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels}.npy under its hardcoded
# DATA_DIR (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/) -- make sure
# that splitv3/ has been populated there (e.g. via utils/train_test_splitting_v3.py, or by
# copying data/splitv3/ from the laptop) before submitting this job.
python utils/train.py

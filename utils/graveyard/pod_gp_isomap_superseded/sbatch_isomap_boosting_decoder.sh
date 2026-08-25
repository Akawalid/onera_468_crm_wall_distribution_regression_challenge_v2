#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --job-name=isomap_boosting_decoder
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# IsoMap embedding + gradient-boosting decoder (replaces the kNN backmap in paper_isomap_rbf.py).
# CPU-only, needs real cores: MultiOutputRegressor fits one independent
# HistGradientBoostingRegressor per output point (260,774 of them -- boosting can't share a
# single factorization across outputs the way the GP-decoder variant could), measured at
# ~28ms/output on 12 local cores -> ~124 minutes there. 64 cpus-per-task here should cut that
# roughly in proportion (joblib parallelizes cleanly across the independent per-output fits).
# Same partition/nodelist/mem spec as sbatch_train.sh (proven for big CPU jobs on this account) --
# per-model memory footprint at 260,774 separate small tree ensembles is hard to bound precisely
# in advance, so this errs generous rather than risk an OOM discovered hours in.
#
# The script pins each joblib worker to 1 BLAS thread internally (OMP/OPENBLAS/MKL_NUM_THREADS=1)
# before numpy/sklearn ever load -- without that, 64 worker processes each also trying to
# multithread their own BLAS calls would oversubscribe the node far worse than the laptop ever
# could. No env var needed here; the script sets it itself.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# isomap_boosting_decoder.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels,weights}.npy
# under paper_isomap_rbf.py's hardcoded DATA_DIR
# (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/) -- make sure splitv3/
# has been populated there before submitting this job.
python utils/isomap_boosting_decoder.py

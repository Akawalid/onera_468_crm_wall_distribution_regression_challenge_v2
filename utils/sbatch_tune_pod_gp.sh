#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=tune_pod_gp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# CPU-only hyperparameter search for POD+GP (tune_pod_gp.py): grid over the POD variance
# threshold x GP kernel smoothness (nu), leave-2-Machs-out CV within the training set only.
# Same partition/nodelist as sbatch_train.sh (proven for big CPU jobs on this account) -- 64
# cpus-per-task lets joblib's per-mode GP parallelism (up to ~50-90 modes depending on the
# variance threshold) actually spread across that many workers at once instead of the ~5 rounds of
# 12 this took on a 12-core laptop (measured there: ~5.4min per config x fold, ~45-50min for the
# full 3x3 grid). --mem=64G is generous headroom -- POD+GP's working set is tiny (n_train ~ 324
# rows, not the full nwallp field per worker), nowhere near sbatch_train.sh's 400G.
#
# tune_pod_gp.py itself pins each joblib worker to 1 BLAS thread (OMP/OPENBLAS/MKL_NUM_THREADS=1)
# before numpy/sklearn ever load -- without that, 64 worker PROCESSES each *also* trying to use 64
# BLAS threads would oversubscribe the node far worse than the laptop ever could and this would run
# SLOWER than 12 cores, not faster. No env var needed here; the script sets it internally so this
# also does the right thing if run directly (not just from this sbatch file).

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# tune_pod_gp.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels,weights}.npy under
# paper_pod_gp.py's hardcoded DATA_DIR (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/)
# -- make sure splitv3/ has been populated there before submitting this job.
python utils/tune_pod_gp.py

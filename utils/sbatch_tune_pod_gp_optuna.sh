#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=tune_pod_gp_optuna
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Optuna hyperparameter search for POD+GP -- wider than tune_pod_gp.py's fixed 3x3 grid: directly
# tunes n_modes (10-150, not a coarse variance threshold), kernel_nu, alpha, and the GP kernel's
# own length-scale/noise-level upper bounds. Same partition/nodelist/thread-pinning reasoning as
# sbatch_tune_pod_gp.sh.
#
# --timeout (in seconds, given to the script) leaves ~30min of buffer before this job's own SLURM
# --time limit so the study can wrap up (print best trial, do the final refit + evaluation) instead
# of getting SIGKILLed mid-trial. --n_trials is set high as a ceiling that realistically won't be
# reached before --timeout -- same convention as sbatch_tune_simple_gnn.sh.
#
# --storage persists the study to sqlite on the shared drive (not utils/ -- same reasoning as
# every other output this job produces), so it survives this job's time limit and can be resumed
# by just resubmitting this same script (load_if_exists=True in the script), or inspected with
# optuna-dashboard.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

python utils/tune_pod_gp_optuna.py \
    --n_restarts 2 \
    --n_trials 300 \
    --timeout 12600 \
    --study_name pod_gp_splitv3 \
    --storage sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_pod_gp_splitv3.db

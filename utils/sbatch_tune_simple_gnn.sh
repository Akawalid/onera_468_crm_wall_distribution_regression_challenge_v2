#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=tune_simple_gnn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: same reasoning as the other sbatch scripts here --
# under sbatch, SLURM executes a copy of this script from /var/spool/slurmd/..., so $0-based path
# tricks break.
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

pip show optuna > /dev/null 2>&1 || pip install optuna

# 50 trials x up to 60 epochs each, MedianPruner cuts unpromising trials early -- 12h is a guess
# since there's no prior timing for this workload; watch the first few trials in the log (same
# caveat as the other GPU sbatch scripts here) and adjust --n_trials/--time accordingly.
# --storage persists trials to sqlite so the study survives a job timeout/preemption and can be
# resumed by rerunning this script (load_if_exists=True) or inspected with optuna-dashboard.
python utils/tune_simple_gnn.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --split_dir splitv3 \
    --k 20 \
    --epochs 60 \
    --patience 10 \
    --n_trials 50 \
    --study_name simple_gnn_splitv3 \
    --storage sqlite:///utils/optuna_simple_gnn_splitv3.db

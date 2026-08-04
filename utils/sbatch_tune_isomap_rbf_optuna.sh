#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=tune_isomap_rbf_optuna
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Optuna hyperparameter search for IsoMap+RBF -- tunes n_components (latent dim, fixed at 3 by
# hand in paper_isomap_rbf.py), isomap_k (neighbor graph size), rbf_kernel, RBFInterpolator's
# smoothing (0 in the deployed script -- no regularization at all in the p->z step), and
# backmap_k (folded into this outer search instead of a redundant nested CV -- see module
# docstring). Measured ~28s/trial on a 12-core laptop, so this is far cheaper than the POD+GP
# tuning jobs -- 16 cpus-per-task is already generous headroom, not a hard requirement.
#
# --storage persists the study to sqlite on the shared drive (not utils/), so it survives this
# job's time limit and can be resumed by just resubmitting this same script.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

python utils/tune_isomap_rbf_optuna.py \
    --n_trials 300 \
    --timeout 9000 \
    --study_name isomap_rbf_splitv3 \
    --storage sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_isomap_rbf_splitv3.db

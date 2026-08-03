#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=simple_gnn_tuned
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# gpu-best is a mixed fleet (per https://clusters-saclay.gitlabpages.inria.fr/clusters-docs/docs/margaret/hardware/),
# some cards as small as 11 GB (2080Ti). Pinned to the >=24 GB nodes, same as the other GNN jobs.
#SBATCH --nodelist=margpu001,margpu002,margpu003,margpu004,margpu005,margpu006,margpu007,margpu008,margpu009,margpu010,margpu011,margpu012,margpu013,margpu014,margpu015,margpu016

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# Retrains the best Optuna trial found so far (trial 52, val MSE=0.001164, as of the log from
# 2026-08-02) for real -- full epoch budget, periodic honest test_phase1 KLw (via the already-
# fixed base_models.evaluate), and an actual saved checkpoint, none of which the tuner itself
# produces (it only tracks validation MSE + hyperparameters, no checkpoint, see
# summarize_optuna_study.py's docstring for why). Double-check these are still the best params
# with `python utils/summarize_optuna_study.py` before submitting -- more trials may have
# completed since this was written, and if a better one has, swap the five values below for it.
python utils/train_simple_gnn.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --split_dir splitv3 \
    --k 20 \
    --hidden_dim 64 \
    --n_layers 5 \
    --lr 0.0020900516244633854 \
    --dropout 0.06609130049699566 \
    --weight_decay 1.3707483439908248e-06 \
    --epochs 100 \
    --eval_every 5 \
    --out_prefix /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/simple_gnn_tuned

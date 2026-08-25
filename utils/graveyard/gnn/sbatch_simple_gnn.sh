#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=simple_gnn_splitv3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# gpu-best is a mixed fleet (per https://clusters-saclay.gitlabpages.inria.fr/clusters-docs/docs/margaret/hardware/),
# some cards as small as 11 GB (2080Ti). Pinned to the >=24 GB nodes (RTX6000/V100/A40/A100/H100)
# to avoid landing on one of those, same as sbatch_tune_simple_gnn.sh.
#SBATCH --nodelist=margpu001,margpu002,margpu003,margpu004,margpu005,margpu006,margpu007,margpu008,margpu009,margpu010,margpu011,margpu012,margpu013,margpu014,margpu015,margpu016

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: same reasoning as sbatch_offset_gnn.sh -- under sbatch,
# SLURM executes a copy of this script from /var/spool/slurmd/..., so $0-based path tricks break.
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# --split_dir splitv3 is the only change from the splitv2 baseline run; --data_dir points at the
# shared folder where splitv3/ needs to already exist (populate it there first, e.g. via
# utils/train_test_splitting_v3.py or by copying data/splitv3/ from the laptop).
# --k/--hidden_dim/--epochs match the defaults this baseline was previously run with; watch the
# first couple of epoch timings in the log and adjust --time if margaret's GPUs differ from the
# dev-laptop timings noted in sbatch_offset_gnn.sh.
python utils/train_simple_gnn.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --split_dir splitv3 \
    --k 20 \
    --hidden_dim 64 \
    --n_layers 3 \
    --epochs 100 \
    --lr 1e-3 \
    --eval_every 5 \
    --out_prefix /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/simple_gnn_splitv3

#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=simple_gnn2_splitv3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: same reasoning as the other sbatch scripts here --
# under sbatch, SLURM executes a copy of this script from /var/spool/slurmd/..., so $0-based path
# tricks break.
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# Same k/hidden_dim/n_layers as sbatch_simple_gnn.sh's baseline run, so the two are comparable --
# the only real difference is the loss (KLw here vs MSE there). The KLw loss is heavier per
# simulation (O(n_points*n_bins) soft-histogram vs O(n_points) for MSE), so --time is padded up
# from sbatch_simple_gnn.sh's 8h; watch the first couple of epoch timings in the log and adjust.
python utils/train_simple_gnn2.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --split_dir splitv3 \
    --k 20 \
    --hidden_dim 64 \
    --n_layers 3 \
    --epochs 100 \
    --lr 1e-3 \
    --eval_every 5 \
    --out_prefix utils/simple_gnn2

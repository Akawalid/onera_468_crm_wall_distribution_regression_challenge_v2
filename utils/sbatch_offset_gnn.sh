#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=offset_gnn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# repo root, so `utils` is importable as a package (train_offset_gnn.py does `from utils.* import`)
cd "$(dirname "$0")/.."

# --k 10 and --hidden_dim 256 are the paper's section 4.2 values (kNN neighbors, velocity/pressure
# hidden width -- the 2 conv layers w/ skip connections and 3-5 layer decoder are hardcoded in
# OffsetGNN's defaults, also matching the paper). --epochs/--lr/--time aren't given in the paper
# excerpt -- these are reasonable starting points; watch the first few epoch timings in the log
# and adjust --time accordingly since 260k-node / 2.6M-edge graphs at hidden_dim=256 are not cheap.
python -m utils.train_offset_gnn \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --k 10 \
    --hidden_dim 256 \
    --noise_std 0.01 \
    --epochs 100 \
    --lr 1e-3 \
    --eval_every 10 \
    --out_prefix utils/offset_gnn

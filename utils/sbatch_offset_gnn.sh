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

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 points into the spool dir, not the repo
# (this is what broke the previous version of this script -- "can't open file
# '/var/spool/slurmd/utils/train_offset_gnn.py'"). $0-based path tricks that work for a script
# run directly via bash/./script.sh do not work under sbatch for this reason.
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# Invoke by script path, NOT `python -m utils.train_offset_gnn` -- `-m` resolves the `utils`
# package from cwd *before* any script code runs, so it silently breaks if the job's cwd isn't
# exactly the repo root. Running the file directly sidesteps that: train_offset_gnn.py inserts
# the repo root into sys.path itself (via its own __file__, which IS reliable under sbatch)
# before importing `utils.*`.
#
# The model (utils/offset_gnn.py) is now a port of the paper's actual reference implementation
# (CoordConv/CoordGNN from SolarisAdams/ML4CFD-Offset-based-Graph-Convolution), not the paper
# text's simpler description: inverse-L1-distance softmax attention (not Gaussian), ELU, a
# 4-layer fusion MLP per conv layer, an extra raw-input concat-skip before the decoder, and
# training-time-only Gaussian noise injected inside the model's forward(). Self-loops (which
# their code adds) are deliberately left off for this dataset -- verified empirically that this
# mesh's ~68x nearest-neighbor-distance spread (p10 to p90) makes a self-loop dominate the
# softmax for most nodes regardless of how eps is calibrated, which would zero out real spatial
# aggregation. --k 10 and --hidden_dim 256 match the paper; --epochs 12 matches the reference
# code's own default (its --noise_std default of 1e-5 is used automatically -- not overridden
# here). This deeper model is slower per epoch than an earlier from-scratch version: on this
# project's dev laptop (a memory-constrained 8GB GPU, likely far weaker than whatever margaret's
# gpu-best partition provides) it measured ~7.5 min/epoch at hidden_dim=256 -- watch the first
# couple of epoch timings in the log and adjust --time if margaret's GPUs are comparably slow.
python utils/train_offset_gnn.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --k 10 \
    --hidden_dim 256 \
    --epochs 12 \
    --lr 1e-3 \
    --eval_every 4 \
    --out_prefix utils/offset_gnn

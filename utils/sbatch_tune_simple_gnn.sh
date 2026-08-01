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

# Complements tune_simple_gnn.py's own per-trial cleanup (del model/optimizer + empty_cache()):
# lets the allocator grow/shrink segments instead of relying on exactly-sized cached blocks, which
# reduces fragmentation across trials that build differently-shaped models (hidden_dim/n_layers
# both vary) -- this exact flag is what the CUDA OOM error message itself suggested.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# First run (50 trials requested, 12h limit) measured ~45-90 min/trial and OOM'd at trial 9/50 --
# 50 trials at that rate is ~40+ hours, so this would have hit the SLURM time limit around trial
# 14 even without the crash. Switched from a --n_trials target to a --timeout budget (in seconds,
# with an hour of buffer before the SLURM --time limit for the study to wrap up cleanly) so the
# run just does as many trials as actually fit instead of overshooting; --n_trials is left high as
# a ceiling that won't realistically be reached. --storage persists trials to sqlite (with
# load_if_exists=True in the script) so this resumes the existing study -- the 9 trials already
# completed are not lost, and rerunning this script picks up where it left off.
python utils/tune_simple_gnn.py \
    --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
    --split_dir splitv3 \
    --k 20 \
    --epochs 60 \
    --patience 10 \
    --n_trials 200 \
    --timeout 39600 \
    --study_name simple_gnn_splitv3 \
    --storage sqlite:///utils/optuna_simple_gnn_splitv3.db

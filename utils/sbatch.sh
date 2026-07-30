#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=gpu-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mlp_klw_splitv3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Previously ran on --partition=normal-best with a CPU-only nodelist and no --gres=gpu, so
# torch.cuda.is_available() in train_mlp_with_kl.py silently fell back to CPU (hence the old
# job-name mlp_klw_cpu) -- for a net whose last layer alone is Linear(512, 260774) (~134M params),
# that's a real cost, not a rounding error. Switched to an actual GPU allocation.

source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: same reasoning as sbatch_offset_gnn.sh -- under sbatch,
# SLURM executes a copy of this script from /var/spool/slurmd/..., so $0-based path tricks break.
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# train_mlp_with_kl.py now reads splitv3/ (see DATA_DIR block in its main()) -- make sure
# splitv3/ already exists under the shared data folder before submitting.
python utils/train_mlp_with_kl.py
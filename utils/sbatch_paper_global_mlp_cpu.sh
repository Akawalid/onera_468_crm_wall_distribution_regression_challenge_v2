#!/bin/bash
#SBATCH --account=tau
#SBATCH --partition=normal-best
#SBATCH --nodelist=marg037,marg038,marg042,marg043,marg044,marg045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --job-name=paper_global_mlp_cpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# CPU fallback for paper_global_mlp.py: the gpu-best partition's GPU only has ~15.89GB total VRAM,
# and the paper's literal final layer -- Linear(16490, 260774), ~4.3B params -- needs ~16.02GB just
# for that one weight tensor (confirmed by a real CUDA OOM on gpu-best: "Tried to allocate 16.02
# GiB. GPU 0 has a total capacity of 15.89 GiB"). It cannot fit on that GPU at all, independent of
# batch size. A big-memory CPU node (same partition/nodelist as sbatch_train.sh, already validated
# for large-memory jobs on this account) sidesteps the VRAM ceiling entirely -- 400G of system RAM
# is enormously more headroom than the ~70GB or so this layer's weights + Adam moments + gradients
# actually need.
#
# paper_global_mlp.py picks CPU automatically when torch.cuda.is_available() is False (no code
# change needed) -- just don't request --gres=gpu here.
#
# CPU matmul on a 16490x260774 layer is memory-bandwidth-bound like the GPU case, but system RAM
# bandwidth is far below GPU VRAM bandwidth, so this will run much slower than the paper's <7h GPU
# figure -- --time is padded generously to 24h. Watch the first couple of epoch timings in the log
# (printed every 10 epochs) and adjust --time if it's not on track to finish.
export OMP_NUM_THREADS=64

# sbatch_train.sh (the closest analogous CPU big-mem script) skips this and relies on whatever
# python is on PATH for that CPU partition -- not safe to assume here since torch is a much
# heavier/less-likely-to-be-preinstalled dependency than train.py's numpy/sklearn, so activate the
# repo's own venv explicitly, same as every GPU launcher in this repo does.
source /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/.venv/bin/activate

# Hardcoded, not derived from $0/dirname: under sbatch, SLURM copies the submitted script into
# /var/spool/slurmd/... and executes THAT copy, so $0 does not point into the repo (see
# sbatch_offset_gnn.sh for the same issue/reasoning).
cd /home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2

# paper_global_mlp.py reads splitv3/{train,test_phase1,test_phase2}_{data,labels,weights}.npy
# under its hardcoded DATA_DIR (/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/)
# -- make sure splitv3/ has been populated there before submitting this job.
python utils/paper_global_mlp.py

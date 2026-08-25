# ONERA 468 CRM Challenge

![logo](bundle/logo.png)

Design and implementation of a machine learning challenge built on 468 RANS CFD
simulations of the NASA/Boeing Common Research Model (CRM), hosted on
[Codabench](https://www.codabench.org/competitions/17632/). Given flow conditions
(Mach number, angle of attack, stagnation pressure) and aircraft geometry, the task is
to predict the surface volumetric density field over ~260k wall points per simulation.

For the full write-up (challenge design, train/test split rationale, metrics, baselines,
results), see **[report/main.pdf](report/main.pdf)**. This README only maps out the
repository itself.

## Repository structure

- **`bundle/`**: the actual Codabench competition package: ingestion program, scoring
  program, competition pages (`bundle/pages/*.md`), and the starting kit
  (`bundle/starting_kit/`, with baseline code and five ready-made submission zips: mean,
  KNN, POD+GP, IsoMap+RBF, global MLP). This is what gets uploaded to Codabench. See
  `bundle/starting_kit/README.md` for participant-facing instructions.

- **`report/`**: the internship report. `main.typst` is the source (compile with
  `python3 -c "import typst; typst.compile('main.typst', output='main.pdf')"` from
  inside `report/`), `refs.bib` the bibliography, `main.pdf` the compiled report, and
  `figures/` its images.

- **`utils/`**: active scripts behind the report and the baselines:
  - Train/test split generation: `train_test_splitting_v3.py`.
  - Main baselines: `paper_pod_gp.py` (POD + Gaussian Process), `train_mlp_with_kl_v2.py`
    (full-field MLP trained on the differentiable KL loss).
  - Other baselines: `paper_isomap_rbf.py`, `paper_global_mlp.py`.
  - Hyperparameter tuning (Optuna): `tune_pod_gp_optuna.py`, `tune_isomap_rbf_optuna.py`,
    plus their `sbatch_*.sh` cluster job scripts.
  - Diagnostics used in the report: `pca_plots.py` (error-weighted PCA slices),
    `plane_segmentation.py`, `plane_visualization.py`.
  - `prepare_codabench_data.py`: packages data for upload to the platform.
  - `base_models.py`: kept as a reference (documented in `bundle/starting_kit/README.md`).
  - **`utils/graveyard/`**: superseded or dead-end code, kept for traceability rather
    than deleted outright, not part of the active pipeline:
    - `gnn/`: graph neural network baseline, abandoned.
    - `pod_gp_isomap_superseded/`, `mlp_klw_superseded/`: earlier iterations of the two
      main baselines, before the versions actually used in the report.
    - `tree_baselines_superseded/`: early tree-ensemble baselines (CatBoost, random
      forest, LightGBM, ...), superseded by the current baseline set.
    - `dead_end_models/`: wavelet- and boosting-based experiments that didn't pan out.
    - `superseded_infra/`: earlier train/split scripts replaced by the current ones.

- **`logs/`**: curated Margaret (Inria cluster) SLURM job logs cited as provenance for
  the numbers reported in `report/main.typst` (linked there via clickable
  `logs/<job-id>.out` references). Only the jobs actually cited are kept; everything else
  was pruned.

- **`data/`**: small, versioned reference data
  (`fullfiles_PiMinfAoA_with_scores.csv`: per-simulation flow conditions). The raw
  simulation arrays, generated train/test splits, and model checkpoints are not tracked
  in git (see `.gitignore`); they live on the cluster / local disk only.

- **`bot.py`**: Discord bot assigning the profile-role buttons (researcher, PhD
  candidate, Master's student, ...) on the challenge's Discord server, used for
  participants to team up.

- **`important_notes.txt`**: running personal notes from the internship (environment
  setup, dataset gotchas, ideas), not a polished document.

- **`.gitignore`**: excludes raw data (`*.npy`, `FILES*`, `*.cgns`, ...), model
  checkpoints (`*.pt`, `*.joblib`), and generic zips, except the starting-kit submission
  zips which are intentionally shipped as zips (see the `!bundle/starting_kit/*.zip`
  exceptions).

## Downloading the data

The raw simulation arrays are not tracked in git; they are pulled from Codabench's stable
per-dataset download endpoints (the same URLs used by
`bundle/starting_kit/kit_utils/download.py`). Each endpoint 302-redirects to a freshly-signed
URL every time it's hit, so the two links below don't expire even though the redirect target
does.

```bash
mkdir -p data/splitv3
cd data/splitv3

# Phase 1 (interpolation): train_data.npy, train_labels.npy, test_data.npy (unlabeled),
# component_labels_unique.npy, component_map.json
wget -O feedback_phase_input_data.zip \
  "https://www.codabench.org/datasets/download/6c284e9f-482b-4015-8c83-200b41f66674/"
unzip -o feedback_phase_input_data.zip -d feedback_phase
rm feedback_phase_input_data.zip

# Phase 2 (extrapolation): same train_data.npy/train_labels.npy, plus its own test_data.npy
wget -O final_phase_input_data.zip \
  "https://www.codabench.org/datasets/download/e32b5c9b-dfd8-4f39-9a58-9fd5bc7af33d/"
unzip -o final_phase_input_data.zip -d final_phase
rm final_phase_input_data.zip

# Lay both out to match the data/splitv3 naming used by utils/train_test_splitting_v3.py
mv feedback_phase/train_data.npy train_data.npy
mv feedback_phase/train_labels.npy train_labels.npy
mv feedback_phase/test_data.npy test_phase1_data.npy
mv final_phase/test_data.npy test_phase2_data.npy
mv feedback_phase/component_labels_unique.npy component_labels_unique.npy
mv feedback_phase/component_map.json component_map.json
rm -r feedback_phase final_phase
```

This gives `data/splitv3/{train_data,train_labels,test_phase1_data,test_phase2_data,
component_labels_unique}.npy` and `component_map.json`, which is what the baseline scripts
under `utils/` and `bundle/starting_kit/` expect to load. One caveat: the two Codabench
downloads intentionally withhold `test_phase1_labels.npy` and `test_phase2_labels.npy` (the
test labels are held out server-side for scoring), along with the per-split `*_weights.npy`
confidence weights and `component_nn_distances.npy`; those four files only exist in the
internal copy of `data/splitv3/` used to produce the numbers in `report/main.pdf`, generated
from the full raw dataset directly on the Margaret cluster (see
`utils/train_test_splitting_v3.py`).

## Where to start

- Read **[report/main.pdf](report/main.pdf)** for the full story.
- The Codabench-uploadable package lives entirely under **`bundle/`**.
- To regenerate a baseline or the split, see the corresponding script under **`utils/`**.

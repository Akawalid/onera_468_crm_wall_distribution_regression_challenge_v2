# ONERA 468 CRM Challenge rho: Starting Kit

## Goal

Predict the non-dimensionalized volumetric density $\rho$ at every point of the fluid
domain, for aerodynamic conditions not seen during training.

## Data structure (`input_data/`, downloaded from Codabench)

| File | Shape | Description |
|---|---|---|
| `train_data.npy` | (n_train x 260,774, 9) | Train inputs |
| `train_labels.npy` | (n_train x 260,774, 1) | Train target (rho) |
| `test_data.npy` | (n_test x 260,774, 9) | Test inputs (no labels) |
| `component_labels_unique.npy` | (260,774,) | Component id per wall point |
| `component_map.json` | -- | `{id: component_name}` (wing/pylon/fuselage/nacelle) |

The 9 columns of X are: `x, y, z, nx, ny, nz, Minf, AoA, Pi`

- 260,774 wall points per aerodynamic condition
- `Pi` is the stagnation pressure factor (1, 2, or 4)
- **You never have access to the test labels** (`reference_data/`) -- the whole
  notebook evaluates models via cross-validation on `train` only.

## Contents of the kit

- `starting_kit.ipynb` -- the main notebook, meant to be read in order.
- `kit_utils/` -- all the reusable code (metrics, models, plots), imported by the
  notebook. Look here directly if you want to understand or tweak an
  implementation detail.
  - `metrics.py` -- R2, wrMAE, KLw (mean_KL) and their bootstrap confidence
    intervals, formulas identical to `scoring_program/scoring.py`.
  - `data.py` -- loading the train data and component masks, splitting into
    cross-validation folds (leave-two-Machs-out).
  - `lgbm_baseline.py` -- pointwise LightGBM baseline, same model as `utils/base_models.py`.
  - `pca_plots.py` -- visual diagnostics (error on the aircraft, PCA directions
    of the error, slices).
- `baselines/` -- heavier reference baselines, run on their own (not from the
  notebook).
  - `mlp_klw.py` -- production-scale full-field MLP trained with the same
    differentiable KLw loss, see section 2.2 of the notebook.

## Submission

Your submission must be a zip file containing `model.py` (at the root of the
zip, not in a subfolder) with a `Model` class implementing `fit(X, y)` and
`predict(X)`. The notebook generates `submission/model.py` (the LightGBM
baseline) and `submission.zip` in section 5.

## Metrics (see `bundle/scoring_program/scoring.py` for the official code)

- **KLw** (`mean_KL`) -- **the leaderboard's primary metric**, closer to 0 is
  better. Measures the (KL-divergence) distance between the distribution of your
  residuals and a narrow reference Gaussian, weighted per component (wing/pylon
  0.3, fuselage/nacelle 0.2). See section 2.2 of the notebook.
- **R^2** (confidence-weighted) -- closer to 1 is better.
- **wrMAE** (worst-case relative MAE on confidence=1 cases) -- closer to 0 is
  better.
- `score` = 5 x R2 + 5 x (1 - wrMAE) -- kept for reference, KLw remains the
  metric the leaderboard is sorted on.

## Running the starting kit

```bash
pip install -r requirements.txt
jupyter notebook starting_kit.ipynb
```

The LightGBM cross-validation trains on a subsample of simulations (a handful per
Mach, see section 2 of the notebook) and takes roughly 10-15 minutes on CPU --
`Model` itself uses the same full-size, production hyperparameters as
`utils/base_models.py`, so most of that time is the 500 trees themselves, not the
data volume. `baselines/mlp_klw.py` is heavier still (production-scale network,
hundreds of epochs) and is meant to be run separately, ideally on a GPU.

# ONERA 468 CRM - Wall distribution regression challenge: data

## Database

The database consists of nf = 468 CFD simulations, each containing np = 260,774 skin points described by their Cartesian coordinates (x, y, z) and the normal vector (nx, ny, nz). Combined with the three flow condition variables (Minf, AoA, Pi), respectively the Mach number ranging from 0.30 to 0.96, the angle of attack ranging from -15 to +15 degrees, and the stagnation pressure taking values 1e5, 2e5 or 4e5 Pa, the entire dataset (train and test splits merged) is of shape (np x nf, 9).

> **Note:** Each simulation is assigned a confidence weight: 1.0 for well-converged simulations (|AoA| < 10 degrees) and 0.5 for low-confidence ones (|AoA| >= 10 degrees). These weights are provided in the Files tab and should be used when computing the evaluation metric.

## Target variable

The volumetric density rho has been selected as the output quantity of interest for the regression exercise.

## Train/test split

The split is done along the Mach number axis, in order to test the model's ability to generalize and extrapolate to unseen Mach values:

- Mach numbers 0.50, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90 belong to the **train set**
- Mach numbers 0.85 and 0.93 belong to the **test set of phase 1** (interpolation)
- Mach numbers 0.30 and 0.96 belong to the **test set of phase 2** (extrapolation)

## Input format

Depending on your model architecture, X can be used in two ways:

- **Full-field regressor** (e.g. a neural network that outputs all np values at once): the geometry columns are not needed. You can simplify X into Xg of size [nf, 3], where each row is just (Minf, AoA, Pi). The output then becomes Yg of shape [nf, np], a simple reordering of Y.
- **Pointwise regressor**: use the full X as-is. Each row has the 9 inputs (geometry + flow conditions) and you predict the output for that point.

## Files

From the **Files** tab, one can download:

- `input_data.zip`
  - `train_data.npy` train input matrix of size [np x n_train, 9]
  - `train_labels.npy` train output matrix of size [np x n_train]
  - `train_weights.npy` confidence weights per train simulation of size [n_train]
  - `test_data.npy` test input matrix of size [np x n_test, 9]
  - `test_weights.npy` confidence weights per test simulation of size [n_test]

- `solution.zip`
  - `scoring_program/`
    - `scoring.py` main script used by the platform to score submissions

- `starting_kit/`
  - `solution/` provides a template of a dummy model to use when submitting
    - `model.py`
  - `baseline.ipynb` a notebook showing how to load the data and reproduce the baseline results
  - `using_extra_packages/` a demonstration of how to use extra packages not available on Codabench
    - `codalab-env`
    - `environment.yml`

<!-- - `starting_kit/`
  - `solution/` — template model to use as a base for your submission
    - `model.py` — edit this file with your model and submit it as a zip
  - `starting_kit.ipynb` — notebook showing how to load the data, train a baseline model and generate the `Yhat.npy` submission file
  - `using_extra_packages/` — if your model depends on packages not available by default on Codabench (e.g. LightGBM, PyTorch, etc.), this folder shows you how to bundle them with your submission so they are available at runtime
    - `environement.yml` — conda environment file reproducing the Codabench environment locally
    - `codalab-env/` — uv-based environment setup, an alternative to conda
    - `conda_submission_example.py` — example showing how to install extra packages into a `python_packages/` folder using conda, and include them in your submission zip
    - `uv_submission_example.py` — same as above but using uv instead of conda -->

Each is a numpy array with type float32 (single precision). A starting kit is also available, providing a model template and demonstrating how to generate the submission file.

**Sizes:**

- np = 260,774, number of skin points per simulation
- nf = 468, total number of simulations
- n_train = 324, number of conditions in the train set (9 Mach x 3 Pi x 12 AoA)
- n_test_phase1 = 72, number of conditions in the test set of phase 1 (2 Mach x 3 Pi x 12 AoA)
- n_test_phase2 = 72 number of conditions in the test set of phase 2 (2 Mach x 3 Pi x 12 AoA)
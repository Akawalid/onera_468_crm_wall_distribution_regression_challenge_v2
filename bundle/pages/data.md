# ONERA 468 CRM Wall Distribution Regression Challenge: Data

## Database

The database consists of **nf = 468 CFD simulations**[^cfd], each simulation contains **np = 260,774 wall points** computed on the **same** CRM[^crm] surface mesh[^surfacemesh] (fixed geometry).

The input features are the follwing:
- **Cartesian coordinates (x, y, z):** the standard three-dimensional coordinate system used to
locate each wall point in space. Together with the surface normal, they fully describe the
manifold that models the aircrafs wall geometry.
- **Surface normal vector (nx, ny, nz):** a unit vector perpendicular to the aircraft surface at a given point,
pointing outward into the flow. It encodes the local orientation of the surface and is a key
geometric input for predicting how the flow interacts with the surface at that location.
- **Aerodynamic conditions (Minf, AoA, Pi):**
  - **Mach number (Minf)[^mach]:** ranges from 0.30 to 0.96, covering two flow regimes: subsonic
    (Minf < 0.80) and transonic[^transonic] (0.80 $\leq$ Minf < 1.0).
  - **Angle of attack (AoA)[^aoa]:** ranges from $-15^\circ$ to $+15^\circ$, covering conditions from attached
    flow to full flow separation[^separation].
  - **Stagnation pressure (Pi)[^pi]:** takes three values : 100 kPa, 200 kPa, and 400 kPa,
    which control the Reynolds number[^reynolds] of the flow.

The dataset is stored as a single `.npy` file of shape **(np $\times$ nf, 9) = (260,774*468, 9)**, where
**np = 260,774** is the number of surface mesh points and **nf = 468** is the number of simulations.
The simulations are stacked row-wise, which means, the first `np` rows correspond to the first simulation.

Each row describes one surface point under one set of flow conditions, with 9 columns split into two groups:

- **Geometric features (columns 0-5: x, y, z, nx, ny, nz):** the position and surface normal of
  the mesh point. These are identical across all simulations since the aircraft geometry is fixed
  the same np geometric rows repeat for every simulation.
- **Flow condition features (columns 6-8 respectively: Minf, AoA, Pi):** the flwo conditions. These are constant within a simulation (all np rows share the same values) but vary from one simulation to the next.

> **Confidence weights:** Each simulation is assigned a confidence weight based on its convergence[^convergence]
> quality: **1.0** for well-converged simulations (|AoA| < $10^\circ$) and **0.5** for low-confidence
> ones (|AoA| $\geq$ $10^\circ$). These weights are provided in the Files section below and must be used when
> computing the evaluation metric (see the **Evaluation** tab).
>
> **Notes:**
> - Do not modify the weights, as the test set follows the same convention defined here.
> - The columns of the `.npy` file are ordered as follows: `(x, y, z, nx, ny, nz, Minf, AoA, Pi)`

## Target Variable

The target variable is the **volumetric density $\rho$ (rho)** as stated in the overview page, it is evaluated at each of the np = 260,774 surface points. For a given simulation, the
output is a vector of shape **(np,) = (260,774,)**, and the full training label matrix is of
shape **(np $\times$ n_train,)**.

Density is a fundamental thermodynamic quantity, through the ideal gas law, it is directly linked
to local pressure and temperature, making it a physically meaningful summary of the aerodynamic
state at each surface point. Its distribution over the aircraft surface is particularly sensitive
to compressibility effects and shock wave[^shock] locations.

## Phases and Train/Test Split

You have two main phases, 
  - Phase 1 tests interpolation (easy)
  - phase 2 tests extrapolation (hard)

The training data is the same for phase 1 and phase 2,
You wil get only training data with their labels, the test data and their labels are only visible for the scoring program, not for you.

The split is performed along the **Mach number axis**, in order to evaluate the model's ability
to generalize and extrapolate[^extrapolation] to unseen compressibility regimes:

| Split | Mach numbers | Size |
|---|---|---|
| **Train** | 0.50, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93 | n_train = 324 |
| **Phase 1 test** | 0.84, 0.86 | n_test_phase1 = 72 |
| **Phase 2 test** | 0.30, 0.96 | n_test_phase2 = 72 |

Phase 1 tests **interpolation**: these Mach numbers fall between train values already seen
(0.84 between 0.82 and 0.85, 0.86 between 0.85 and 0.88).
Phase 2 tests **extrapolation**: these Mach numbers fall outside the full train
range, covering the lowest and highest compressibility regimes in the dataset.

Here is how it works:
train your model locally on the training data, then submit it to
be evaluated on the phase 1 test data. When phase 1 closes, submit the same code again
for phase 2. Your model is retrained on the same training data as before, then evaluated
on the phase 2 test data.

Since the training data does not change between the two phases, retraining is wasted
compute. To avoid it, save your big model as a `.pt` file, include it in your
submission, and load it instead of training again. A submission example is available at the starting kit
`starting_kit/submission_mlp_klw_with_dotpt_file.zip`

## Different ways of using the dataset

Depending on your model architecture, the input matrix X can be used in two ways:

- **Full-field regressor** Here, the aircraft geometry is encoded implicitly in the *ordering* of the predicted values. The intuition is as follows: imagine unfolding the aircraft surface onto a flat 2D plane, like unwrapping a gift box, and visualizing it as an image where each pixel's value represents the volumetric density at the corresponding surface point, scaled to an RGB value. The prediction task then becomes analogous to **image generation**: instead of predicting a single scalar per simulation, the model must predict an entire "image" which is simply the target vector of 260,774 values, where the geometry is not an explicit input but is instead **baked into the structure of the output itself**. Each position in that vector always corresponds to the same surface point, so the spatial ordering between points are preserved through ordering rather than through coordinate features, and the inter-points influence is preservsed through the covariance between the different dimensions.

- **Pointwise regressor:** use the full X as-is. Each row contains all 9 features (geometry +
  flow conditions) and the model predicts the density at that individual point. The model sees
  260,774 $\times$ n_train independent input-output pairs during training.

## Starting kit & data

When you run the starting kit, the data files will be automatically downloaded, and the folder organization is as the following:
- **`input_data/`**, containing:
  - `train_data.npy`: train input matrix, shape [np * n_train, 9]
  - `train_labels.npy`: train output matrix, shape [np * n_train]
  - `test_data.npy`: test input matrix, shape [np * n_test, 9]
  - `component_labels_unique.npy`: component id per wall point, shape [np]
  - `component_map.json`: maps component id to name (wing, pylon, fuselage, nacelle)

> **The component files.**
> tell you which aircraft part each wall point belongs to, for example, the point (x1, x_2, x_3) belongs to the wing.
> `component_labels_unique.npy`  is of shape (260774), it maps each index to an id between 0 and 3
> and `component_map.json` maps the id to the name of the component (id -> name: wing, pylon, fuselage, nacelle) together




All `.npy` files are stored as **float32 (single precision)** numpy arrays.


### The `starting_kit/` folder

- `starting_kit.ipynb`: main notebook: download and load data, train, evaluate, plot, submit.
- `kit_utils/`: code imported by the notebook.
  - `data.py`: data loading, plus cross-validation splitting (available, not used by default).
  - `metrics.py`: R2, wrMAE, KLw, same formulas as the scoring program.
  - `mean_baseline.py`: Baseline A, predicts the global training mean.
  - `pod_gp_baseline.py`: Baseline B, POD + Gaussian Process (Optuna-tuned).
  - `lgbm_baseline.py`: pointwise LightGBM baseline (reference, not used by the notebook by default).
  - `pca_plots.py`: visual diagnostics.
- `baselines/`: heavier reference baselines, run on their own, not from the notebook.
  - `mlp_klw.py`: production-scale full-field MLP with a KL-aware loss.
- `submission.zip`: the baseline written out by the notebook, ready to submit (unzip for `model.py`).
- `submission_pod_gp.zip`, `submission_knn.zip`, `submission_isomap_rbf.zip`,
  `submission_global_mlp.zip`, `submission_mlp_klw_with_dotpt_file.zip`: ready-made alternative
  submissions, one per baseline explored in the parent project -- POD+GP, input-space kNN,
  IsoMap+RBF, global-field MLP, and the KL-loss MLP (this last one ships a pretrained `.pt` file
  so it skips retraining between phases, see above).
- `using_extra_packages/`: how to bundle extra Python packages in a submission.
  - ...
  - `submission_example/`
    - ...
    - `conda_tuto.txt`: tutorial for installing extra packages using conda
    - `uv_tuto.txt`: tutorial for installing extra packages using uv
- `README.md`: quick reference for all of the above.

> The starting kit is also browsable directly on GitHub (no download needed to look
> around first): [bundle/starting_kit](https://github.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/tree/main/bundle/starting_kit).

## Dataset Sizes

| Symbol | Value | Description |
|---|---|---|
| np | 260,774 | Number of wall points per simulation |
| nf | 468 | Total number of simulations |
| n_train | 324 | Number of simulations in the train set |
| n_test_phase1 | 72 | Number of simulations in the phase 1 test set |
| n_test_phase2 | 72 | Number of simulations in the phase 2 test set |

## Terminology

[^cfd]: **CFD (Computational Fluid Dynamics):** numerical simulation of fluid flow, used here to
compute the aerodynamic quantities (pressure, velocity, density) at every wall point without a
physical wind-tunnel test. See the **Overview** tab for more detail.

[^crm]: **CRM (Common Research Model):** a publicly available aircraft geometry developed jointly
by NASA and Boeing, used here as the fixed surface mesh across all simulations. See the
**Overview** tab for more detail.

[^surfacemesh]: **Surface mesh:** a discrete representation of the aircraft surface as a collection
of points and connecting elements. The CRM mesh used here has 260,774 points and is identical
across all simulations.

[^mach]: **Mach number (Minf):** ratio of flow speed to the speed of sound. Determines the
compressibility regime of the flow.

[^transonic]: **Transonic:** the flight regime (0.80 $\leq$ Minf < 1.0) where subsonic and supersonic
zones coexist around the aircraft, producing shock waves.

[^aoa]: **Angle of attack (AoA):** angle between the incoming airflow and the aircraft reference
axis. Controls lift generation and can trigger flow separation at large values.

[^separation]: **Flow separation:** the airflow detaching from the aircraft surface instead of
following it, typically at large angle of attack. Causes a sharp rise in drag and loss of lift.

[^pi]: **Stagnation pressure (Pi):** the pressure a fluid element would reach if brought to rest
isentropically. Used here as a proxy to control the Reynolds number.

[^reynolds]: **Reynolds number:** a dimensionless number characterizing the ratio of inertial to
viscous forces in the flow. Controlled here via the stagnation pressure Pi.

[^convergence]: **Convergence:** in CFD, a simulation's convergence is assessed by running it
multiple times and measuring the standard deviation of the lift and drag forces -- the smaller
they are, the better the simulation converges.

[^shock]: **Shock wave:** a thin region of abrupt change in pressure, density, and velocity that
forms when a flow locally exceeds the speed of sound.

[^extrapolation]: **Extrapolation:** predicting outputs for input conditions that lie outside the
range seen during training. The train/test split in this challenge is specifically designed to
test extrapolation across Mach numbers.

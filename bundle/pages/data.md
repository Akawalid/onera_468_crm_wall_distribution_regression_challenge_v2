# ONERA 468 CRM Wall Distribution Regression Challenge: Data

## Database

The database consists of **nf = 468 CFD simulations**[^cfd], each simulation contains **np = 260,774 wall points** computed on the same CRM[^crm] surface mesh[^surfacemesh].

The input features are the follwing:
- **Cartesian coordinates (x, y, z):** the standard three-dimensional coordinate system used to
locate each wall point in space. Together with the surface normal, they fully describe the
manifold that models the aircrafs wall geometry.
- **Surface normal vector (nx, ny, nz):** a unit vector perpendicular to the aircraft surface at a given point,
pointing outward into the flow. It encodes the local orientation of the surface and is a key
geometric input for predicting how the flow interacts with the surface at that location.
- **Aerodynamic conditions (Minf, AoA, Pi):**
  - **Mach number (Minf)[^mach]:** ranges from 0.30 to 0.96, covering two flow regimes: subsonic
    (Minf < 0.80) and transonic[^transonic] (0.80 ≤ Minf < 1.0).
  - **Angle of attack (AoA)[^aoa]:** ranges from −15° to +15°, covering conditions from attached
    flow to full flow separation[^separation].
  - **Stagnation pressure (Pi)[^pi]:** takes three values : 100 kPa, 200 kPa, and 400 kPa,
    which control the Reynolds number[^reynolds] of the flow.

The dataset is stored as a single `.npy` file of shape **(np × nf, 9) = (260,774*468, 9)**, where
**np = 260,774** is the number of surface mesh[^surfacemesh] points and **nf = 468** is the number of simulations.
The simulations are stacked row-wise, which means, the first `np` rows correspond to the first simulation.

Each row describes one surface point under one set of flow conditions, with 9 columns split into two groups:

- **Geometric features (columns 0-5: x, y, z, nx, ny, nz):** the position and surface normal of
  the mesh point. These are identical across all simulations since the aircraft geometry is fixed
  the same np geometric rows repeat for every simulation.
- **Flow condition features (columns 6-8 respectively: Minf[^mach], AoA[^aoa], Pi[^pi]):** the flwo conditions. These are constant within a simulation (all np rows share the same values) but vary from one simulation to the next.

> **Confidence weights:** Each simulation is assigned a confidence weight based on its convergence[^convergence]
> quality: **1.0** for well-converged simulations (|AoA[^aoa]| < 10°) and **0.5** for low-confidence
> ones (|AoA[^aoa]| ≥ 10°). These weights are provided in the Files section below and must be used when
> computing the evaluation metric (see the **Evaluation** tab).
>
> **Notes:**
> - Do not modify the weights, as the test set follows the same convention defined here.
> - The columns of the `.npy` file are ordered as follows: `(x, y, z, nx, ny, nz, Minf, AoA, Pi)`

## Target Variable

The target variable is the **volumetric density ρ (rho)** which represents the mass of fluid per unit volume
(kg/m3), it is evaluated at each of the np = 260,774 surface points. For a given simulation, the
output is a vector of shape **(np,) = (260,774,)**, and the full training label matrix is of
shape **(np × n_train,)**.

Density is a fundamental thermodynamic quantity, through the ideal gas law, it is directly linked
to local pressure and temperature, making it a physically meaningful summary of the aerodynamic
state at each surface point. Its distribution over the aircraft surface is particularly sensitive
to compressibility effects and shock wave[^shock] locations.

## Train/Test Split

The split is performed along the **Mach number[^mach] axis**, in order to evaluate the model's ability
to generalize and extrapolate[^extrapolation] to unseen compressibility regimes:

| Split | Mach numbers | Size |
|---|---|---|
| **Train** | 0.50, 0.70, 0.75, 0.80, 0.88, 0.90, 0.93 | n_train = 252 |
| **Phase 1 test** | 0.82, 0.84, 0.85, 0.86 | n_test_phase1 = 144 |
| **Phase 2 test** | 0.30, 0.96 | n_test_phase2 = 72 |

Phase 1 tests **interpolation**: these Mach numbers fall between the train values 0.80 and 0.88.
Phase 2 tests **extrapolation**[^extrapolation]: these Mach numbers fall outside the full train
range, covering the lowest and highest compressibility[^transonic] regimes in the dataset.

## Different ways of using the dataset

Depending on your model architecture, the input matrix X can be used in two ways:

- **Full-field regressor** Here, the aircraft geometry is encoded implicitly in the *ordering* of the predicted values. The intuition is as follows: imagine unfolding the aircraft surface onto a flat 2D plane, like unwrapping a gift box, and visualizing it as an image where each pixel's value represents the volumetric density at the corresponding surface point, scaled to an RGB value. The prediction task then becomes analogous to **image generation**: instead of predicting a single scalar per simulation, the model must predict an entire "image" which is simply the target vector of 260,774 values, where the geometry is not an explicit input but is instead **baked into the structure of the output itself**. Each position in that vector always corresponds to the same surface point, so the spatial ordering between points are preserved through ordering rather than through coordinate features, and the inter-points influence is preservsed through the covariance between the different dimensions.

- **Pointwise regressor:** use the full X as-is. Each row contains all 9 features (geometry +
  flow conditions) and the model predicts the density at that individual point. The model sees
  260,774 × n_train independent input-output pairs during training.

## Files

From the **Files** tab, one can download:

- `input_data.zip`
  - `train_data.npy`: train input matrix, shape [np * n_train, 9]
  - `train_labels.npy`: train output matrix, shape [np * n_train]
  - `test_data.npy`: test input matrix, shape [np * n_test, 9]
  - `component_labels_unique.npy`: component id per wall point, shape [np]
  - `component_map.json`: maps component id to name (wing, pylon, fuselage, nacelle)
- `submission_example.zip`: example of a valid submission file for Codabench
  - `submission_example/`
    - `python_packages/`
    - `model.py`
    - `conda_tuto.txt`: tutorial for installing extra packages using conda
    - `uv_tuto.txt`: tutorial for installing extra packages using uv

All `.npy` files are stored as **float32 (single precision)** numpy arrays.

> The starting kit is also browsable directly on GitHub (no download needed to look
> around first): [bundle/starting_kit](https://github.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/tree/main/bundle/starting_kit).

## Dataset Sizes

| Symbol | Value | Description |
|---|---|---|
| np | 260,774 | Number of wall points per simulation |
| nf | 468 | Total number of simulations |
| n_train | 252 | Number of simulations in the train set |
| n_test_phase1 | 144 | Number of simulations in the phase 1 test set |
| n_test_phase2 | 72 | Number of simulations in the phase 2 test set |

## Terminology
**Surface mesh:**[^surfacemesh] a discrete representation of the aircraft surface as a collection of points
and connecting elements. The CRM mesh used here has 260,774 points and is identical across all
simulations.

**Convergence:**[^convergence] in CFD, a simulation convergence is asseced by running it multiple times then measure the standard diviation of the lift[^liftdrag] and drag[^liftdrag] forces, the smaller they are, the better the simulation converges.

**Extrapolation:**[^extrapolation] predicting outputs for input conditions that lie outside the range seen during
training. The train/test split in this challenge is specifically designed to test extrapolation
across Mach numbers[^mach].

**Reynolds number:**[^reynolds] a dimensionless number characterizing the ratio of inertial to viscous forces
in the flow. Controlled here via the stagnation pressure Pi[^pi].

**Stagnation pressure (Pi):**[^pi] the pressure a fluid element would reach if brought to rest
isentropically. Used here as a proxy to control the Reynolds number[^reynolds].

**Mach number (Minf):**[^mach] ratio of flow speed to the speed of sound. Determines the compressibility
regime of the flow.

**Angle of attack (AoA):**[^aoa] angle between the incoming airflow and the aircraft reference axis.
Controls lift[^liftdrag] generation and can trigger flow separation[^separation] at large values.
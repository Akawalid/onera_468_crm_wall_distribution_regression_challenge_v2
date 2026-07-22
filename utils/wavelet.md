# Wavelet Residual Correction of the Global MLP

## 1. Principle

The global MLP $f_\theta$ captures the smooth, large scale behavior of the wall density field but blurs localized structures such as shocks. The wavelet pipeline learns a correction on top of it. For each training simulation, the residual field

$$
r = y - f_\theta(c) \in \mathbb{R}^N
$$

is encoded into a small set of wavelet coefficients, a regressor learns how those coefficients vary with the flight conditions $c$, and at test time the predicted coefficients are decoded back into a correction field:

$$
\hat{y} = f_\theta(c) + \mathcal{W}^{-1}\!\big( h(c) \big).
$$

The residual is the ideal target for a sparse representation: it is close to zero over most of the surface and concentrated near shocks and leading edges.

## 2. From the mesh to images

Wavelets require a regular grid, while the residuals live on an unstructured 3D surface mesh. Each scored component (wing, pylon, fuselage, nacelle) is flattened as follows. The 3D coordinates of its points are projected onto their two principal directions by PCA, giving planar coordinates $(u, v)$. The plane is divided into an $n \times n$ grid ($n = 64$), and each cell takes the average residual of the points falling in it; empty cells are filled from the nearest filled cell. This defines a linear encoder $E$ (mesh values to image) and a decoder $D$ (image to mesh values, each point reading the value of its cell). The pair $(E, D)$ is lossy: the binning averages within cells, which sets a reconstruction ceiling independent of any learning.

## 3. The Haar transform

Each residual image $G \in \mathbb{R}^{n \times n}$ is decomposed with the 2D Haar wavelet, applied separably to rows then columns. The 1D transform recursively splits a signal into averages and differences of adjacent pairs,

$$
a_k = \frac{x_{2k} + x_{2k+1}}{\sqrt{2}}, \qquad
d_k = \frac{x_{2k} - x_{2k+1}}{\sqrt{2}},
$$

iterated on the averages until a single coefficient remains. The result is an orthogonal change of basis $W : \mathbb{R}^{n \times n} \to \mathbb{R}^{n^2}$: each coefficient describes the local contrast of the image at a given position and a given dyadic scale, and the transform is exactly invertible. Because the residual is mostly flat with a few localized bumps, its energy concentrates in few coefficients, which is the compression the pipeline exploits.

## 4. Coefficient selection and regression

Stacking the transforms of the $n_{\text{train}}$ training simulations gives a matrix $F \in \mathbb{R}^{n_{\text{train}} \times n^2}$ per component. Two quantities are extracted:

* the mean $\bar{F}$, used as a frozen baseline for all coefficients;
* the empirical variance of each coefficient across simulations, used to keep only the $K = 200$ coefficients that actually change with the conditions.

Coefficients outside the top $K$ carry either static structure (already in the baseline) or noise, and are not predicted. The selection acts as a low pass filter on what the model is allowed to express; the earlier experiment that predicted all coefficients with a smooth interpolator degraded the results precisely because it reinjected the noisy fine scale coefficients.

The regression is then small: a gradient boosted model (XGBoost, 300 trees of depth 4) maps $c \in \mathbb{R}^3$ to the $K$ selected coefficients, one independent model per coefficient, trained on the $n_{\text{train}}$ pairs. Trees are chosen because coefficient trajectories under shock motion are sharp and nonlinear in Mach.

## 5. Prediction

For a test condition $c$: predict the $K$ coefficients, insert them into the baseline vector $\bar{F}$, invert the Haar transform, and read the reconstructed image back onto the mesh through $D$. The correction is added to the MLP prediction. All evaluation uses the same metrics (weighted $R^2$, worst rMAE, component weighted KL with the global $\sigma$ reference) as every other baseline, so the comparison against the MLP alone isolates exactly what the correction contributes.

## 6. Division of labor and limits

The MLP models the global field, the wavelet basis fixes *where and at which scale* corrections can live, and the boosting model learns *how they vary with the conditions*. Three known limits bound the approach: the binning ceiling of the grid encoding; the folding of upper and lower surfaces into the same cells by the PCA projection, which averages extrados and intrados; and the fact that a shock moving with Mach makes many coefficients vary jointly and nonlinearly, which independent per coefficient regressors capture only approximately. Each limit maps to a tested variant (finer grids, per surface splitting, registration or per scale regressors).
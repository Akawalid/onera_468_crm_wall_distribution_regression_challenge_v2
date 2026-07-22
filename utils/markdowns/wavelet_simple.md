# The Wavelet Residual Correction, in Short

## 1. Principle

The global MLP captures the smooth part of the wall density field but blurs localized structures such as shocks. The wavelet pipeline learns a correction on top of it. For each training simulation the residual

$$
r = y - f_\theta(c)
$$

is compressed into a small set of wavelet coefficients, a boosting model learns how these coefficients vary with the flight conditions, and at test time the predicted coefficients are decoded back into a correction field:

$$
\hat{y} = f_\theta(c) + \mathcal{W}^{-1}\!\big( h(c) \big).
$$

The residual is the ideal target for this: it is near zero over most of the surface and concentrated around shocks, exactly the kind of signal wavelets compress well.

## 2. From the mesh to an image

Wavelets need a regular grid, but the residuals live on an unstructured 3D mesh. Each component is therefore flattened by PCA onto its principal plane $(u, v)$, and the plane is divided into a $64 \times 64$ grid where each cell takes the average residual of its points (empty cells copy their nearest filled neighbor). Applying the Haar transform to the resulting image concentrates the energy of the residual into a small number of coefficients.

![Mesh residuals, grid binning, Haar coefficients](figures/wav1_encoding.png)

The binning is lossy: averaging inside cells sets a reconstruction ceiling that no regressor can exceed. This ceiling is measured explicitly (encode the true fields, decode them back, score) before any learning.

## 3. The Haar transform

The 1D Haar transform recursively replaces a signal by averages and differences of adjacent pairs,

$$
a_k = \frac{x_{2k} + x_{2k+1}}{\sqrt{2}}, \qquad
d_k = \frac{x_{2k} - x_{2k+1}}{\sqrt{2}},
$$

iterated on the averages; in 2D it is applied to rows then columns. It is an orthogonal, exactly invertible change of basis in which each coefficient describes the local contrast at a given position and scale: coarse coefficients carry the overall shape, fine coefficients carry sharp local jumps such as a shock.

![One signal seen at three scales](figures/wav2_scales.png)

## 4. Selection and regression

Stacking the coefficient vectors of all training simulations, only the $K = 200$ coefficients with the highest variance across simulations are kept for prediction; the rest are frozen at their training mean. This acts as a low pass filter: static structure stays in the baseline, noisy fine scale coefficients are simply not predicted. An earlier variant that predicted all coefficients confirmed the need for this filter, as reinjecting the noisy coefficients degraded the results.

![Variance based selection of the coefficients](figures/wav3_selection.png)

The remaining learning problem is small: XGBoost (300 trees, depth 4) maps the 3 conditions to the $K$ coefficients, one model per coefficient, trained on a few hundred simulations. Trees are used because coefficient trajectories under shock motion are sharp and nonlinear in Mach. Training takes seconds.

## 5. Prediction and limits

At test time: predict the $K$ coefficients, insert them into the baseline, invert the transform, read the image back onto the mesh, add to the MLP prediction. Evaluation uses the same metrics as all baselines, so the comparison with the MLP alone isolates exactly what the correction contributes.

The known limits are the binning ceiling, the folding of upper and lower surfaces into the same cells by the PCA projection, and the fact that a moving shock makes many coefficients vary jointly, which independent per coefficient regressors capture only approximately.

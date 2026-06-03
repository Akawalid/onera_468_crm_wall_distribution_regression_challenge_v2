# ONERA 468 CRM - Wall distribution regression challenge: Evaluation

## Evaluation metrics

Submissions are evaluated using two complementary metrics:

### R2 (coefficient of determination)
The **R^2 score** (also known as the coefficient of determination) measures the proportion of variance in the true inflection points that is explained by the model.

### wrMAE (worst-case relative mean absolute error)
Measures the worst-case relative error across all test conditions that have a corresponding weight of 1.0 (well converged simulations), notice that it's the relative error, which means, it takes values between 0.0 and 1.0.

$$
rMAE^f = \frac{\sum_{i=1}^{n_p} |\rho_i^f - \hat{\rho}_i^f|}{\sum_{i=1}^{n_p} |\rho_i^f|}, \quad f \in \mathcal{W}
$$

$$
wrMAE = \max_{f \in \mathcal{W}} rMAE^f
$$

where:
- $\mathcal{W}$ is the set of well-converged test conditions (those with confidence weight $w_f = 1.0$)
- $n_p = 260,774$ is the number of skin points per simulation
- $\rho_i^f$ is the true volumetric density at point $i$ under condition $f$
- $\hat{\rho}_i^f$ is the predicted volumetric density at point $i$ under condition $f$

A detailed description of the metrics is available in the [associated paper](http://www.arxiv.org/abs/2505.06265).
The final score aggregates both metrics:
```
score = 5 × R2_mean + 5 × (1 - wrMAE_mean)
```

## Evaluation procedure

The evaluation follows a standard machine learning competition workflow:

1. **Model initialization:** the model defined in `model.py` is initialized.
2. **Training:**
   - The model receives training data (`X_train`, `Y_train`)
   - The model's `fit()` method is called with these inputs
3. **Prediction:**
   - The model receives test data (`X_test`) without labels
   - The model's `predict()` method is called to generate predictions
   - Predictions are saved as `Yhat.npy` in the output directory
4. **Scoring:**
   - Predictions are compared against ground truth labels
   - R2 and wrMAE scores are computed
   - Execution time is also recorded (Training, prediction and scoring)
5. **Leaderboard ranking:**
   - Models are ranked by score (a grade over 10, higher is better)
   - R2 and wrMAE are provided as additional columns for reference

## Submitting a solution

Edit the `model.py` file from the starting kit without changing the file name, and compress it into a zip file. Submit the zip file in the **My Submissions** tab.

- You can review the logs of each submission to identify errors.
- Execution time is recorded and displayed on the leaderboard.
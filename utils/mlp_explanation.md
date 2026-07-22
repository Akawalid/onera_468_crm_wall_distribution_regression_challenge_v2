# The KL Divergence Loss of the Global MLP, in Short

The MLP is trained directly on a differentiable version of the evaluation metric of the challenge: the component weighted KL divergence between the distribution of prediction residuals and a narrow Gaussian reference. Instead of minimizing a generic error like MSE and hoping the metric follows, the network optimizes the quantity on which it is judged.

## 1. The model

The network maps the three flight conditions $c = (M_\infty, \alpha, P_i)$ to the full wall density field of $N = 260{,}774$ points in one shot:

$$
\hat{y} = \bar{y} + g_\theta(c),
$$

where $\bar{y}$ is the mean training field and $g_\theta$ the learned deviation. Each surface point carries the weight of its component (wing 0.3, pylon 0.3, fuselage 0.2, nacelle 0.2), normalized to sum to one.

## 2. The metric

For one simulation, the residuals $\hat{y}_j - y_j$ are collected into a weighted histogram over $B = 200$ bins spanning $[-5\sigma_y,\, 5\sigma_y]$, where $\sigma_y$ is the standard deviation of the true field of that simulation. This gives the residual distribution $p$. It is compared to a reference $q$: a centered Gaussian of standard deviation $\sigma_{\text{ref}} = 1\%$ of the global mean density. The score of the simulation is

$$
\mathrm{KL}_w(p \,\|\, q) = \sum_{b} p_b \log \frac{p_b}{q_b},
\qquad \text{score} = \frac{1}{1 + \mathrm{KL}_w}.
$$

A perfect model has all residuals concentrated in a narrow, centered spike, so $p \approx q$ and KL is near zero. Bias shifts the histogram, fat tails put mass where $q$ is essentially zero: both are penalized.

![Pipeline: field, residual histogram, KL score](fig1_pipeline.png)

## 3. The obstacle: a histogram is not differentiable

Putting a residual in a bin is an all or nothing operation, written with an indicator function:

$$
p_b = \sum_j \omega_j \, \mathbf{1}\{ e_b \le \varepsilon_j < e_{b+1} \}.
$$

As a residual moves inside its bin, nothing changes; as it crosses an edge, $p$ jumps. The derivative is zero almost everywhere, so backpropagation receives no signal. The metric, as written, cannot train a network.

![One residual's contribution to a fixed bin: step vs smooth](fig3_step_vs_smooth.png)

## 4. The fix: soft assignment by softmax

The hard indicator is replaced by a softmax over the distances to all bin centers $t_b$:

$$
s_{jb} = \operatorname{softmax}_b\!\left( -\tfrac{1}{2} \left( \tfrac{\varepsilon_j - t_b}{\tau} \right)^{\!2} \right),
\qquad
p_b = \sum_j \omega_j \, s_{jb},
$$

with the temperature $\tau$ set to one bin width. Each residual now spreads its mass over its two or three nearest bins, and this spread varies smoothly with the residual value, so gradients flow. As $\tau \to 0$ the softmax collapses back to the hard indicator: the soft histogram is a genuine relaxation of the metric, not a different objective.

![Hard one hot assignment vs soft softmax assignment](fig2_soft_assignment.png)

The loss of a simulation is then the same KL formula applied to the soft $p$, and a batch simply sums it over its simulations. Note that this is a distributional loss: it couples all points of the field at once, rather than treating each point independently as MSE does.

## 5. What minimizing it does

The gradient pushes residuals sitting in overpopulated regions of the histogram, the tails and any off center bulk, toward the center, and leaves already well placed residuals almost untouched. The training effort concentrates automatically on the points responsible for tail mass, typically the shock regions.

![Effect of minimizing the KL](fig4_effect.png)

Two practical remarks. The loss constrains the distribution of errors, not their location, so rMAE and $R^2$ are monitored alongside it. And since the reference is centered, any global bias is penalized immediately, which is why the trained MLP shows near zero bias in the component tables.

## 6. Training summary

MLP $(128, 256, 512)$ with LeakyReLU and dropout $0.2$; Adam, learning rate $10^{-3}$ with cosine annealing; batches of 2 simulations (the $2 \times N \times B$ soft assignment tensor is the memory limit); $10\%$ validation split with early stopping (patience 30) on the same soft loss; the best checkpoint is saved and reused as the base model of the residual correction pipelines.
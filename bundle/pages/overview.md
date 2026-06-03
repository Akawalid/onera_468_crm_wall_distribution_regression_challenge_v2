# ONERA 468 CRM - Wall distribution regression challenge
This benchmark proposes a regression problem based on a new Computational Fluid Dynamics (CFD) database, developed at ONERA, to support the advancement of machine learning techniques for aerodynamic field prediction. High-fidelity CFD simulations are computationally expensive, making them impractical in memory or time-constrained settings. Machine learning surrogate models offer a promising alternative, therefore, instead of running full simulations, they learn the underlying physical patterns and provide much faster and lighter predictions.

The database contains 468 Reynolds-Averaged Navier-Stokes simulations using the Spalart-Allmaras turbulence model, performed on the NASA/Boeing Common Research Model wing-body-pylon-nacelle configuration. It spans a wide range of flow conditions, varying Mach number (including transonic regimes), angle of attack (capturing flow separation), and Reynolds number (based on three stagnation pressures, with one setting matching wind tunnel experiments). The quality of the database is assessed through checking the convergence level of each computation.

Based on these data, a regression challenge is defined. It consists in predicting the wall distributions of volumetric density, a physical quantity derived from pressure and flow conditions. See the **Data** and **Evaluation** tabs for details on the dataset and scoring metric.

**References and credits:**
- Jacques Peter et al. ["ONERA's CRM WBPN database for machine learning activities, related regression challenge and first results."](https://doi.org/10.1016/j.compfluid.2025.106838) *Computers & Fluids*, Volume 302, 2025.
- [arXiv preprint arXiv:2505.06265 (2025).](http://www.arxiv.org/abs/2505.06265)
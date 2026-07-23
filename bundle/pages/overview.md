# ONERA 468 CRM Wall Distribution Regression Challenge

> 💡 **New here?** The [starting kit](https://github.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/tree/main/bundle/starting_kit)
> walks through loading the data, training two baseline models, evaluating them with
> the challenge's own metrics, and preparing a submission.

## Introduction to Aircraft Physics
Since the Wright brothers' first powered flight in 1903, understanding how air flows around a wing has been the central problem of aeronautics.

Aircraft fly thanks to **the Bernoulli effect** that occurs on the wings. The Bernoulli effect states that in a moving flow, where the air speeds up, its pressure drops.

A wing is shaped and tilted so that air travels faster over its upper surface than under its lower one, therefore, the pressure above the wing becomes lower than the pressure below, according to the Bernoulli effect. Because air is a fluid, like water, it always tries to fill the void (a difference in pressure creates a void), therefore, the air pushes the wing upward, which causes the "flying". Increasing the angle of attack accentuates this asymmetry and increases the lift even more.

This pressure imbalance produces the lift, but it is not the only force at play. Physically speaking, in a steady flight, four forces act on an aircraft, as shown in Figure 1:

- **Weight:** gravity, pulling the aircraft down.
- **Lift:** the aerodynamic force perpendicular to the incoming flow, balancing the weight.
- **Thrust:** produced by the engines, pushing the aircraft forward.
- **Drag:** the aerodynamic force parallel to the incoming flow, opposing the motion and balanced by thrust.

<figure style="text-align: center;">
<img src="https://raw.githubusercontent.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/main/bundle/pages/figures/physics.svg" alt="Illustration of forces acting on an aircraft"/>
<figcaption>
<em>Figure 1: Illustration of the angle of attack and The four forces acting on an aircraft in flight, weight, lift, thrust, and drag.</em>
</figcaption>
</figure>

When the angle of attack gets large in magitude (very positive or very negative) a flow separation phenoma appears, which is the moment when the flow above the wing struggles to meet flow below the wing as shown in Figure 2.

<figure style="text-align: center;">
<img src="https://raw.githubusercontent.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/main/bundle/pages/figures/flow_separation.jpg" width=60% alt="Illustration of flow separation state"/>
<figcaption>
<em>Figure 2: Illustration of the flow separation state which occures at extrem angle of attack valuess.</em>
</figcaption>
</figure>

lift drops sharply and drag rises. This is *flow separation* (leading to stall), one of the flow phenomena that make aerodynamic fields very hard to predict.

Speed also brings its own complications. As the aircraft approaches the speed of sound, the air can no longer adjust smoothly: it gets compressed, and its density starts to vary strongly from one point of the surface to another. Where the flow locally exceeds the speed of sound, these variations become brutal — shock waves[^shock] form, thin regions where pressure and density jump abruptly. This flight regime, where subsonic and supersonic zones coexist around the aircraft, is called the *transonic regime*.

<figure style="text-align: center;">
  <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
    <img src="https://raw.githubusercontent.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/main/bundle/pages/figures/supersonic.png" style="height: 220px; object-fit: contain;" alt="Illustration of shock waves in transonic flight"/>
    <img src="https://raw.githubusercontent.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/main/bundle/pages/figures/aircrafspsonic.jpg" style="height: 220px; object-fit: contain;" alt="Illustration of shock waves in transonic flight"/>
  </div>
  <figcaption>
    <em>Figure 3: Progression of shock waves with increasing Mach number.</em>
  </figcaption>
</figure>

In this challenge, we are very interested in these critical conditions, since in some parts of the aircraft, unusual aerodynamic phenomena can happen. In fact, the challenge has two main goals:

- Having machine learning models that can generalize well to unseen critical conditions.
- Having models that work well across all the regions of the aircraft, not only the easy ones.

## The challenge

This benchmark proposes a regression problem based on a new Computational Fluid Dynamics (CFD)[^cfd]
database, developed at ONERA, to support the advancement of machine learning techniques for
aerodynamic field prediction. High-fidelity CFD simulations are computationally expensive, making
them impractical in memory- or time-constrained settings. Machine learning surrogate models offer
a promising alternative, that, instead of running full simulations, they learn the underlying physical
patterns and provide much faster predictions.

The database contains 468 Reynolds-Averaged Navier-Stokes (RANS)[^rans] simulations using the
Spalart-Allmaras[^sa] turbulence model, performed on the NASA/Boeing Common Research Model (CRM)[^crm]
a wing-body-pylon-nacelle configuration representative of a modern commercial aircraft.
The aircraft geometry is **fixed across all simulations**, only the aerodynamic operating
conditions vary from one simulation to another, the observations don't include time variation.

## Flow Condition Variables

Each simulation is characterized by three flow condition parameters:

- **Mach number (Minf):** the ratio of the flow speed to the speed of sound. The dataset spans
  Mach numbers from approximately 0.3 to 0.96, covering a wide range of qualitatively different
  aerodynamic regimes:
  - **Subsonic (Minf < 0.3):** air behaves nearly incompressibly and the flow is relatively smooth.
  - **Transonic (0.3 < Minf < 0.96):** compressibility effects become significant as the flow
    speed approaches the speed of sound, causing the local air density to vary markedly from point
    to point. Both subsonic and supersonic flow regions can coexist around the aircraft, often
    accompanied by shock waves[^shock] that produce abrupt pressure jumps on the surface.

- **Angle of attack (AoA):** the angle between the incoming airflow and the aircraft's reference
  axis. Small AoA values correspond to typical cruise conditions, where the flow remains attached
  to the surface. Large positive or negative AoA values tilt the aircraft more aggressively relative
  to the flow, which can significantly change the pressure distribution and may lead to *flow
  separation*, a phenomenon where the airflow detaches from the surface, causing a
  sharp increase in drag[^drag] and a loss of lift[^drag].
  
  <!-- <figure style="text-align: center;">
    <img src="https://raw.githubusercontent.com/Akawalid/onera_468_crm_wall_distribution_regression_challenge_v2/main/bundle/pages/figures/Airfoil_angle_of_attack.jpg" alt="Illustration of angle of attack"/>
    <figcaption>
      <em>Figure 1: Effect of angle of attack on surface flow. Left: low AoA, flow remains
      attached. Right: high AoA, flow separation occurs near the leading edge.</em>
    </figcaption>
  </figure> -->

- **Stagnation pressure (Pi)[^pi] and Reynolds number[^reynolds]:** the Reynolds number characterizes the ratio of
  inertial to viscous forces in the flow. At high Reynolds numbers the flow tends to be more
  turbulent, which affects how momentum and energy are exchanged near the aircraft surface. In this
  dataset, the Reynolds number is controlled through the stagnation pressure Pi: higher stagnation
  pressure increases the air density and therefore the Reynolds number, without changing the Mach
  number or AoA. Three stagnation pressure levels are used, one of which matches conditions from
  wind tunnel experiments[^windtunnel], allowing direct comparison with experimental data.

## Target Variable

The quantity to predict is the **wall distribution of volumetric density**. Unpacking this phrase:

- **Wall distribution** means the value of a quantity evaluated at every point on the aircraft
  surface mesh[^mesh], as opposed to the full three-dimensional flow field. Each of the 260,774 surface
  points of the CRM mesh has an associated density value for a given simulation.
- **Volumetric density** (kg/m3) is the mass of fluid per unit volume at a given location. It is
  a fundamental thermodynamic quantity: through the ideal gas law, density is directly linked to
  both pressure and temperature, meaning it encodes key information about the local aerodynamic
  state of the flow.
- **Why density on the wall?** The density distribution on the aircraft surface reflects the
  combined effect of Mach number, angle of attack, and Reynolds number on the flow. It is
  particularly sensitive to compressibility effects and shock wave locations, making it a
  physically meaningful and challenging quantity to predict.

The goal of this challenge is therefore to learn a surrogate model that, given a set of flow
conditions (Minf, AoA, Pi), accurately predicts the density value at each of the 260,774 surface
points of the CRM mesh, for new unseen flow conditions. See the **Data** and **Evaluation** tabs
for details on the dataset structure and scoring metrics.

## Terminology

[^cfd]: **CFD (Computational Fluid Dynamics):** a branch of fluid mechanics that uses numerical
methods and algorithms to simulate how fluids (liquids and gases) flow around or through objects.
In aerodynamics, CFD is used to compute quantities such as pressure, velocity, temperature, and
density at thousands or millions of points around an aircraft, without requiring a physical
experiment. A single high-fidelity CFD simulation can take hours to days on a supercomputer,
which motivates the use of machine learning surrogates.

[^rans]: **RANS (Reynolds-Averaged Navier-Stokes):** a family of equations used in CFD to model
turbulent flows. Instead of resolving every turbulent eddy (which would be computationally
prohibitive), RANS equations work with time-averaged quantities and use a *turbulence model* to
approximate the effect of small-scale fluctuations. RANS simulations are the standard approach
in industrial aerodynamics.

[^sa]: **Spalart-Allmaras turbulence model:** a widely used one-equation turbulence model designed
specifically for aerodynamic applications. It solves a single transport equation for a turbulent
viscosity variable, making it computationally efficient while remaining accurate for attached
boundary-layer flows typical of cruise conditions.

[^crm]: **CRM (Common Research Model):** a publicly available aircraft geometry developed jointly
by NASA and Boeing, widely used in the aerodynamics research community as a standard benchmark
configuration. It represents a realistic transonic transport aircraft (wing-body-pylon-nacelle)
and enables fair comparison of results across different research groups and tools. See the
[NASA CRM page](https://commonresearchmodel.larc.nasa.gov/) for geometry files and further details.

[^shock]: **Shock wave:** a thin region of abrupt change in pressure, density, and velocity that
forms when a flow locally exceeds the speed of sound. On a transonic aircraft, shock waves
typically appear on the upper wing surface and can cause a sudden increase in drag
and potentially trigger flow separation downstream.

[^pi]: **Stagnation pressure (Pi):** the pressure that a fluid element would reach if brought to
rest isentropically (without losses). It is a measure of the total energy of the flow. In this
dataset, Pi is used as a control parameter to vary the Reynolds number while keeping the Mach
number and AoA fixed.

[^reynolds]: **Reynolds number (Re):** a dimensionless number that characterizes the relative
importance of inertial forces to viscous forces in a flow. At low Reynolds numbers the flow is
laminar and orderly; at high Reynolds numbers it becomes turbulent. In aerodynamics, full-scale
aircraft typically operate at very high Reynolds numbers (Re ~ 10⁷–10⁸), while wind tunnel
models often operate at lower values due to size and pressure constraints.

[^windtunnel]: **Wind tunnel experiment:** a controlled laboratory test in which a scaled model
(or sometimes a full-scale component) is placed in an artificial airflow to measure aerodynamic
forces, pressures, or flow patterns. Wind tunnel data are used to validate CFD simulations.
One of the three stagnation pressure settings in this dataset is matched to wind tunnel
conditions, enabling direct CFD-vs-experiment comparisons.

[^wall]: **Wall distribution:** the spatial distribution of a physical quantity (here, density)
evaluated at points lying on the solid surface (wall) of the aircraft, as opposed to points in
the surrounding flow field. Wall quantities are particularly important in aerodynamics because
they directly determine forces (lift, drag) and heat transfer acting on the aircraft.

[^mesh]: **Surface mesh:** a discrete representation of the aircraft surface as a collection of
points and connecting elements (triangles or quadrilaterals). The CRM surface mesh used here
contains 260,774 points. Each point has a fixed location in 3D space and a surface normal
vector. The mesh is identical across all simulations, since the geometry does not change.

## References and Credits

- Jacques Peter et al. ["ONERA's CRM WBPN database for machine learning activities, related
  regression challenge and first results."](https://doi.org/10.1016/j.compfluid.2025.106838)
  *Computers & Fluids*, Volume 302, 2025.
- [arXiv preprint arXiv:2505.06265 (2025).](http://www.arxiv.org/abs/2505.06265)
# PROXIMA (`proxima-warp`)

[![PyPI version](https://img.shields.io/pypi/v/proxima-warp.svg?color=blue)](https://pypi.org/project/proxima-warp/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build & Tests](https://img.shields.io/badge/tests-128%20passed-brightgreen.svg)]()
[![JAX Accelerated](https://img.shields.io/badge/accelerated-JAX%20vmap%20%7C%20jit-red.svg)](https://github.com/google/jax)

**PROXIMA** (`proxima-warp`) is an end-to-end scientific computing framework and physics-informed simulation engine written in Python and accelerated via JAX. It enables modeling, geodesic integration, risk evaluation, and particle swarm optimization (PSO) for relativistic interstellar travel using modified spacetime warp metrics based on the formal **Bobrick & Martire (2021)** class of geometries within a $3+1$ Arnowitt-Deser-Misner (ADM) formulation.

---

## Table of Contents

- [Key Architectural Features](#key-architectural-features)
- [System Architecture & Physics Pipeline](#system-architecture--physics-pipeline)
  - [1. 3+1 ADM Metric Tensor Kernel](#1-31-adm-metric-tensor-kernel)
  - [2. Relativistic Geodesic Simulator & Fermi Tidal Tensor](#2-relativistic-geodesic-simulator--fermi-tidal-tensor)
  - [3. Hawking-Ellis Energy Conditions Classification](#3-hawking-ellis-energy-conditions-classification)
  - [4. Astrophysical Risk & Environmental Data Layer](#4-astrophysical-risk--environmental-data-layer)
  - [5. JAX-Vectorized Particle Swarm Optimizer (PSO)](#5-jax-vectorized-particle-swarm-optimizer-pso)
- [Installation Guide](#installation-guide)
  - [Basic Installation](#basic-installation)
  - [Installation with Astronomical Connectors](#installation-with-astronomical-connectors)
  - [Development Installation](#development-installation)
- [Quickstart & Usage Examples](#quickstart--usage-examples)
  - [1. Evaluating the Metric at Spacetime Points](#1-evaluating-the-metric-at-spacetime-points)
  - [2. Integrating Timelike Geodesics & Observables](#2-integrating-timelike-geodesics--observables)
  - [3. Querying Real Celestial Environments & In-Transit Risk](#3-querying-real-celestial-environments--in-transit-risk)
  - [4. Global Multi-Particle Warp Optimization](#4-global-multi-particle-warp-optimization)
  - [5. Running Calibration, Sensitivity & Non-Regression Suites](#5-running-calibration-sensitivity--non-regression-suites)
- [Technical Specifications & Conventions](#technical-specifications--conventions)
- [Author & Academic Citation](#author--academic-citation)
- [License](#license)

---

## Key Architectural Features

- **Covariant $3+1$ ADM Formulation**: Metric lapse function $\alpha(t, \mathbf{x})$, longitudinal and transverse shift vector $\boldsymbol{\beta}(t, \mathbf{x})$, and spatial 3-metric $\gamma_{ij}$.
- **Hyperbolic Shape Regularization**: Eliminates coordinate singularities at the bubble core via $r_{s,\text{reg}} = \sqrt{r_s^2 + (\epsilon_{\text{rel}} R)^2}$.
- **Adaptive Geodesic Integration**: 8-dimensional covariant phase-space integration $(X^\mu, u^\mu)$ driven by Dormand-Prince (DP5/RK45) with strict 4-velocity preservation ($|g_{\mu\nu}u^\mu u^\nu + 1| < 10^{-6}$).
- **Riemann & Fermi Tidal Stress Guard**: Covariant local spatial tidal tensor $\mathcal{E}^i{}_j = c^2 R^i{}_{0 j 0}$ with automatic cut-off preventing spaghettification ($\sigma_{\text{tidal}} \le 9.81\text{ s}^{-2}$).
- **Hawking-Ellis Type-I Energy Condition Classifiers**: Eigenvalue diagonalization verifying Weak (WEC), Null (NEC), Dominant (DEC), and Strong (SEC) Energy Conditions.
- **Real Astronomical Field Modeling**: Bundled ICRS-normalized catalogs (Sharpless H II regions, MWSC stellar clusters, and Supernova Remnants) plus live TAP/ADQL clients (Gaia DR3, NASA Exoplanet Archive, SIMBAD, and Bayestar19 dust extinction).
- **SIMD/JIT Accelerated PSO**: Particle Swarm Optimization fully vectorized via `jax.vmap` and `jax.jit` supporting Global Star and Local Ring communication topologies.

---

## System Architecture & Physics Pipeline

The library is organized into cleanly decoupled, typed contract layers:

```
┌────────────────────────────────────────────────────────┐
│               PROXIMA WARP ARCHITECTURE                │
└────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│  1. Metric Core (ADM / BM)   │ │  4. Astronomical Risk Layer  │
│  - Lapse alpha, Shift beta   │ │  - ICRS / SI Normalization   │
│  - Hyperbolic Shape Function │ │  - Radiative Fluence & Gas   │
│  - Curvature & Energy T_mn   │ │  - SHA-256 L1/L2 Multilevel  │
└──────────────────────────────┘ └──────────────────────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│  2. Relativistic Simulator   │ │  3. Hawking-Ellis Classifier │
│  - 8D Geodesic Integration   │ │  - WEC, NEC, DEC, SEC Checks │
│  - Fermi-Walker Tidal Stress │ │  - Algebraic Diagonalization │
└──────────────────────────────┘ └──────────────────────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌────────────────────────────────────────────────────────┐
│     5. Vectorized Particle Swarm Optimizer (PSO)       │
│     - Global / Ring Topology, Inertia Scheduling       │
│     - JAX vmap Batch Evaluation of Mission Fitness     │
└────────────────────────────────────────────────────────┘
```

### 1. 3+1 ADM Metric Tensor Kernel

Following standard General Relativity conventions with signature $(-, +, +, +)$, spacetime intervals satisfy:
$$ds^2 = -(\alpha^2 - \beta_i \beta^i) c^2 dt^2 + 2 \beta_i c dt dx^i + \gamma_{ij} dx^i dx^j$$

The Bobrick-Martire parameter vector $\boldsymbol{\theta} \in \mathbb{R}^5$ is defined by:
- `bubble_radius_m` ($R > 0$): Radius of the flat central bubble interior.
- `shell_thickness_m` ($\sigma > 0$): Transition wall thickness.
- `velocity_fraction_c` ($v_s / c$): Bubble coordinate translation speed.
- `shift_amplitude` ($A_s$): Shift modulation amplitude.
- `lapse_floor` ($\alpha_0 > 0$): Minimum allowed lapse inside the wall.

### 2. Relativistic Geodesic Simulator & Fermi Tidal Tensor

Geodesics are integrated in normalized affine time $\lambda = c\tau$:
$$\frac{dX^\mu}{d\lambda} = u^\mu, \qquad \frac{du^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta} u^\alpha u^\beta$$

The ship's structural safety is monitored through the spatial tidal tensor in Fermi-Walker coordinates:
$$\sigma_{\text{tidal}} = \max_k |\lambda_k(\mathcal{E}_{ij})| \quad [\si{\per\second\squared}]$$

### 3. Hawking-Ellis Energy Conditions Classification

The effective stress-energy-momentum tensor $T_{\mu\nu} = \frac{c^4}{8\pi G} G_{\mu\nu}$ is diagonalized to inspect physical matter energy constraints:
- **WEC**: $\rho \ge 0 \quad \land \quad \rho + p_i \ge 0$
- **NEC**: $\rho + p_i \ge 0$
- **DEC**: $\rho \ge 0 \quad \land \quad \rho \ge |p_i|$
- **SEC**: $\rho + \sum p_i \ge 0 \quad \land \quad \rho + p_i \ge 0$

### 4. Astrophysical Risk & Environmental Data Layer

Transit segments are evaluated for four cumulative physical hazards:
1. **Integrated Radiative Fluence** ($\si{\joule\per\meter\squared}$): $F = \sum_k \frac{L_k}{4\pi v} \int \frac{ds}{r_k(s)^2 + R_k^2}$
2. **Neutral Hydrogen Column & Extinction** ($\si{\per\meter\squared}$): Background ISM ($n_H = 10^5\text{ m}^{-3}$) plus chord lengths through H II clouds and supernova remnants.
3. **Local Mass Density Field** ($\si{\kilo\gram\per\meter\cubed}$).
4. **External Gravitational Tidal Gradients** ($\si{\per\second\squared}$).

### 5. JAX-Vectorized Particle Swarm Optimizer (PSO)

The swarm explores the warp configuration hyperspace using linearly decreasing inertia scheduling $w(t)$ and configurable social attraction topologies (Global Star or Local Ring), evaluating the composite scalar objective:
$$\mathcal{F}_{\text{total}} = \mathcal{J}_{\text{mission}} + \mathcal{P}_{\text{physical}} + \mathcal{P}_{\text{risk}} + \mathcal{P}_{\text{operational}}$$

---

## Installation Guide

### Basic Installation

Install the production wheel from PyPI:
```bash
pip install proxima-warp
```

### Installation with Astronomical Connectors

To enable live remote queries to NASA Exoplanet Archive, Gaia DR3 TAP services, SIMBAD, and 3D dust models:
```bash
pip install "proxima-warp[astro]"
```

### Development Installation

To install testing utilities and contribute to the repository:
```bash
git clone https://github.com/Leonardo-Zamudio/proxima.git
cd proxima
pip install -e ".[dev,astro]"
pytest -ra
```

---

## Quickstart & Usage Examples

### 1. Evaluating the Metric at Spacetime Points

```python
from proxima import (
    BobrickMartireMetricCore,
    MetricStatePoint,
    WarpParameterVector,
)

# 1. Instantiate canonical Bobrick-Martire warp parameters
params = WarpParameterVector(
    bubble_radius_m=100.0,       # 100-meter interior habitat
    shell_thickness_m=5.0,        # 5-meter wall thickness
    velocity_fraction_c=0.5,      # 0.5 c subluminal transit
    shift_amplitude=1.0,
    lapse_floor=0.01,
)

# 2. Define an evaluation event (t, x, y, z, tau)
point = MetricStatePoint(t_s=0.0, x_m=10.0, y_m=0.0, z_m=0.0, tau_s=0.0)

# 3. Evaluate ADM metric components and diagnostics
core = BobrickMartireMetricCore()
eval_metric = core.evaluate_metric(params, point)

print("Lapse function alpha:", eval_metric.lapse)
print("Shift vector beta (m/s):", eval_metric.shift_vector)
print("Spatial metric gamma_ij:", eval_metric.spatial_metric)
print("Eulerian energy density (J/m^3):", eval_metric.diagnostics["rho_eulerian_J_m3"])
```

### 2. Integrating Timelike Geodesics & Observables

```python
from proxima import (
    RelativisticSimulator,
    SimulationRequest,
    ScientificConventions,
    MetricStatePoint,
    WarpParameterVector,
    Frame,
    MetricSignature,
)

# Set scientific conventions (SI units, -+++ signature)
conventions = ScientificConventions(
    phase0_version="1.0.0",
    metric_signature=MetricSignature.MINUS_PLUS_PLUS_PLUS,
    global_frame=Frame.ICRS,
    local_frame=Frame.LOCAL_CARTESIAN_3P1,
    c_m_per_s=299792458.0,
    g_m3_per_kg_s2=6.67430e-11,
    atol=1e-10,
    rtol=1e-8,
)

sim = RelativisticSimulator()
request = SimulationRequest(
    params=params,
    initial_state=point,
    conventions=conventions,
    max_steps=500,
)

result = sim.simulate(request)

if result.status.value == "success":
    obs = result.observables
    print(f"Elapsed coordinate time Delta t: {obs.delta_t_s:.6e} s")
    print(f"Elapsed proper time Delta tau:    {obs.delta_tau_s:.6e} s")
    print(f"Maximum tidal stress:             {obs.tidal_stress_max_si:.6e} s^-2")
    print(f"Energy conditions report:         WEC={result.energy_conditions.wec_ok}")
```

### 3. Querying Real Celestial Environments & In-Transit Risk

```python
from proxima import (
    default_local_risk_provider,
    RouteSegment,
    SkyCoordinates,
    Frame,
)
from proxima.contracts.types import DataQueryContext

# Load real bundled astronomical environments (Sharpless, MWSC, SNRs)
provider = default_local_risk_provider()
print(f"Loaded {len(provider.records)} real astronomical bodies into memory.")

# Define an interstellar route segment (in meters, Cartesian 3+1 relative to target)
segment = RouteSegment(
    start_xyz_m=(0.0, 0.0, 0.0),
    end_xyz_m=(1.0e16, 0.0, 0.0),
    frame=Frame.LOCAL_CARTESIAN_3P1,
)

# Target celestial coordinates: Lambda Orionis ring (Sh2-264)
target_sky = SkyCoordinates(ra_deg=83.8, dec_deg=9.9, distance_pc=400.0, frame=Frame.ICRS)
context = DataQueryContext(
    dataset_version="v1.0.0",
    extraction_timestamp_utc="2026-11-09T00:00:00Z",
    cache_key="mission_alpha",
)

risk = provider.evaluate_segment_risk(segment, target_sky, context)
print("Radiative flux score [0, 1]:", risk.radiative_flux_score)
print("H column / extinction score:", risk.extinction_h_column_score)
print("Total composite risk score: ", risk.total_risk_score)
```

### 4. Global Multi-Particle Warp Optimization

```python
from proxima import VectorizedPSO, PSOConfig

# Configure swarm parameters
pso_config = PSOConfig(
    n_particles=30,
    n_iterations=40,
    topology="global",    # Or "ring" for enhanced local exploration
    seed=101,
    w_max=0.9,
    w_min=0.4,
    c1=1.5,
    c2=1.5,
)

pso = VectorizedPSO(config=pso_config)

# Run full relativistic optimization coupled to geodesic simulations
opt_result = pso.optimize_warp(seed=101)

print("\n--- OPTIMIZATION CONVERGED ---")
print("Best fitness score:", opt_result.best_fitness)
print("Optimized bubble radius (m):     ", opt_result.best_params.bubble_radius_m)
print("Optimized shell thickness (m):   ", opt_result.best_params.shell_thickness_m)
print("Optimized velocity fraction (c): ", opt_result.best_params.velocity_fraction_c)
print("Optimized shift amplitude:       ", opt_result.best_params.shift_amplitude)
print("Optimized lapse floor:           ", opt_result.best_params.lapse_floor)
```

### 5. Running Calibration, Sensitivity & Non-Regression Suites

```python
from proxima.validation import CalibrationEngine, IntegralValidator

# 1. Run canonical non-regression tests
validator = IntegralValidator()
report = validator.run_regression_suite()
assert report.all_passed, "Numerical divergence detected across canonical baselines!"
print("Integral validation: All canonical benchmarks passed within tolerance.")

# 2. Compute elasticity gradient sensitivities (dF/F) / (dTheta/Theta)
engine = CalibrationEngine()
nominal_vec = (100.0, 5.0, 0.5, 1.0, 0.01)
sens = engine.analyze_warp_sensitivity(
    lambda vec: (vec[0] / 100.0)**2 + (vec[2] / 0.5)**3,
    nominal_vec,
    delta_rel=0.05,
)

for param_name, elasticity in sens.sensitivities.items():
    print(f"Normalized sensitivity of {param_name:22s}: {elasticity:.4f}")
```

---

## Technical Specifications & Conventions

| Property | Standard / Value | Notes |
| :--- | :--- | :--- |
| **Metric Signature** | $(-, +, +, +)$ | Modern General Relativity convention |
| **System of Units** | International System (SI) | Distances in $\si{\meter}$, times in $\si{\second}$, energies in $\si{\joule}$ |
| **Coordinate System** | $X^\mu = (ct, x, y, z)$ | $x^0 = ct$ in meters, 4-velocities $u^\mu = \frac{dX^\mu}{c d\tau}$ |
| **Global Celestial Frame** | ICRS (J2000.0) | Rotations from Galactic $(l, b)$ via IAU 1958/Hipparcos matrix |
| **Physical Constants** | $c = 299\,792\,458.0\text{ m/s}$<br>$G = 6.67430 \times 10^{-11}\text{ m}^3\text{kg}^{-1}\text{s}^{-2}$ | Frozen in Phase 0 contract specs |
| **Core Regularization** | $\epsilon_{\text{rel}} = 1.0 \times 10^{-9}$ | $r_{s,\text{reg}} = \sqrt{r_s^2 + (\epsilon_{\text{rel}} R)^2}$ |
| **Tidal Limit Threshold** | $\sigma_{\text{tidal}} \le 9.81\text{ s}^{-2}$ | Evaluated via local spatial Fermi-Walker frame |
| **Energy Classifier** | Hawking-Ellis Type-I | Real eigenvalue stress-energy tensor decomposition |
| **Cache Architecture** | L1 LRU + L2 Disk JSON | SHA-256 cryptographic digests with TTL invalidation |
| **Acceleration Engine** | Google JAX | `jax.vmap` batching with `jax.jit` compiled execution |

---

## Author & Academic Citation

**PROXIMA** was developed by **Leonardo Zamudio**.

If you use **PROXIMA** in scientific research, academic publications, or numerical studies of spacetime warp metrics, please cite:

```bibtex
@software{zamudio_proxima_2026,
  author       = {Leonardo Zamudio},
  title        = {PROXIMA: Relativistic Optimization and Simulation Engine for Bobrick-Martire Warp Spacetimes},
  year         = {2026},
  publisher    = {PyPI},
  version      = {1.0.0},
  url          = {https://pypi.org/project/proxima-warp/}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

# PROXIMA

PROXIMA is a scientific Python library for modeling, simulating, and optimizing relativistic interstellar missions using warp-drive-inspired spacetime geometries grounded in the formal Bobrick–Martire family of solutions in a 3+1 ADM formulation.

It is designed for research workflows that combine:

- relativistic metric evaluation,
- geodesic propagation and energy-condition checks,
- astrophysical risk assessment from local catalogs,
- batched optimization with JAX-based particle swarm methods.

The library targets researchers, engineers, and advanced Python users working on theoretical propulsion, mission design, and high-performance scientific simulation.

---

## Overview

PROXIMA is built around a relativistic metric engine that evaluates a class of warp-like spacetime geometries derived from a 3+1 decomposition of the metric, with support for the canonical Bobrick–Martire parameterization. The project offers the following core capabilities:

1. Metric and curvature evaluation in a relativistic ADM-style framework.
2. Geodesic integration with adaptive numerical propagation.
3. Energy-condition classification using Hawking–Ellis reasoning.
4. Astrophysical data ingestion and risk evaluation using local stellar catalogs.
5. Vectorized PSO optimization for mission parameter search.
6. Validation layers for reproducibility, regression safety, and scientific sanity checks.

The implementation uses JAX for accelerated numerical paths and vectorized batch evaluation. This makes it suitable for parameter sweeps, optimization loops, and simulation-heavy workloads where performance matters.

---

## Technical specifications

### Supported Python

- Python >= 3.11
- JAX >= 0.4.30
- NumPy >= 1.26
- PyYAML >= 6.0

### Installation targets

The library is intended to run in a standard scientific Python environment, with optional extras for astrophysical workflows.

### Core scientific model

PROXIMA models a warp-like spacetime using a 3+1 decomposition of the metric, with:

- lapse function,
- shift vector,
- spatial metric,
- temporal coordinate evolution,
- relativistic geodesic propagation,
- tidal stress constraints,
- energy-condition diagnostics.

The metric engine computes geometric objects such as:

- Christoffel symbols,
- Riemann curvature tensor,
- Ricci tensor,
- Einstein tensor,
- ADM-style energy terms,
- local validity checks for the parameterized domain.

### Numerical behavior

The simulation layer is designed to be robust under physical plausibility checks and numerical regularization, with explicit handling for:

- invalid parameter domains,
- tidal-stress failure,
- energy-condition violations,
- non-finite state values,
- solver termination states.

### Optimization engine

The optimizer uses a vectorized particle swarm optimization (PSO) implementation with configurable topology:

- global topology,
- ring topology,
- inertia decay,
- acceleration coefficients,
- velocity clamping,
- fitness penalties for physical violations and simulation failures.

### Data and risk layer

The data layer includes local catalog connectors and risk evaluation for astronomical conditions, including:

- normalized celestial record ingestion,
- cross-match resolution,
- source metadata quality tracking,
- risk scoring for radiative flux, extinction, mass density, and tidal stress,
- persistent snapshot and cache support.

---

## Package architecture

The project is organized as follows:

- `src/proxima/contracts` — public dataclasses, protocol interfaces, and contract definitions.
- `src/proxima/core` — metric kernels, geometric operators, and ADM-related computations.
- `src/proxima/simulator` — geodesic simulation, energy analysis, and numerical integration.
- `src/proxima/data` — catalog ingestion, risk models, and data pipeline components.
- `src/proxima/optimizer` — PSO optimizer, fitness evaluation, and topology handling.
- `src/proxima/validation` — validation and regression-oriented checks.
- `configs/` — packaged YAML configuration defaults for scientific and optimizer stages.

The public package entrypoint exposes the main runtime objects through the top-level import path:

```python
import proxima

print(proxima.__version__)
```

---

## Installation

### From PyPI

```bash
pip install proxima-warp
```

### With astronomy extras

The optional extras provide additional astrophysical dependencies for catalog-driven risk workflows:

```bash
pip install "proxima-warp[astro]"
```

### For development and tests

```bash
pip install "proxima-warp[dev]"
```

### From source

If you are installing directly from a local checkout:

```bash
git clone https://github.com/your-org/proxima.git
cd proxima
pip install -e .
```

### Verify installation

```bash
python -c "import proxima; print(proxima.__version__)"
```

---

## Quick start

### 1. Create a warp parameter vector

```python
from proxima import WarpParameterVector

params = WarpParameterVector(
    bubble_radius_m=100.0,
    shell_thickness_m=5.0,
    velocity_fraction_c=0.5,
    shift_amplitude=1.0,
    lapse_floor=0.01,
    metadata={"mission": "demo"},
)
```

### 2. Define initial state and scientific conventions

```python
from proxima.contracts.types import (
    MetricSignature,
    MetricStatePoint,
    ScientificConventions,
    Frame,
)

state = MetricStatePoint(
    t_s=0.0,
    x_m=0.0,
    y_m=0.0,
    z_m=0.0,
    tau_s=0.0,
)

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
```

### 3. Evaluate the metric directly

```python
from proxima import BobrickMartireMetricCore

core = BobrickMartireMetricCore()
result = core.evaluate_metric(params, state)

print(result.lapse)
print(result.shift_vector)
print(result.valid_domain)
```

The metric evaluation returns a structured result with:

- lapse function,
- shift vector,
- spatial metric,
- validity flag,
- diagnostic metadata.

---

## Geodesic simulation example

```python
from proxima import RelativisticSimulator, SimulationRequest

simulator = RelativisticSimulator()
request = SimulationRequest(
    params=params,
    initial_state=state,
    conventions=conventions,
    max_steps=200,
)

result = simulator.simulate(request)

print(result.status)
if result.observables is not None:
    print(result.observables.delta_tau_s)
    print(result.observables.delta_t_s)
    print(result.observables.total_energy_j)
    print(result.observables.tidal_stress_max_si)
```

### Simulation output semantics

The simulation result contains:

- status: success or failure,
- observables: accumulated proper time, coordinate time, total energy, and maximum tidal stress,
- energy-condition diagnostics,
- failure_reason if the run terminates outside acceptable physical conditions,
- diagnostics with solver metadata and auxiliary information.

This is the primary runtime interface for evaluating candidate warp configurations under relativistic propagation.

---

## PSO optimization example

```python
from proxima import VectorizedPSO, PSOConfig

cfg = PSOConfig(
    n_particles=20,
    n_iterations=30,
    seed=42,
    topology="global",
)

pso = VectorizedPSO(config=cfg)
result = pso.optimize_warp(seed=42)

print(result.best_params)
print(result.best_fitness)
print(result.history.gbest_fitness_history[-1])
```

### Optimization workflow

The optimizer can be used to search over warp parameter vectors by minimizing a fitness objective that accounts for:

- mission duration,
- proper-time efficiency,
- physical penalties,
- risk terms,
- invalid-domain penalties,
- simulation failure penalties.

The optimizer is vectorized with JAX and is designed to operate efficiently on batched particle populations.

---

## Data layer and risk provider

PROXIMA includes a data ingestion layer for celestial and astrophysical sources. This component can be used to build risk-aware mission planning workflows.

```python
from proxima.data.pipeline import default_local_risk_provider

risk_provider = default_local_risk_provider()
print(type(risk_provider))
```

This risk provider integrates catalog data and produces a structured risk assessment used by the mission model to penalize routes that pass through high-radiation or high-extinction regions.

### Typical risk components

- radiative flux score,
- extinction score,
- mass-density score,
- tidal-stress score,
- aggregated total risk score,
- data source quality metadata.

---

## Public API reference

The most common imports are exposed at the package root:

```python
from proxima import (
    __version__,
    WarpParameterVector,
    MetricStatePoint,
    ScientificConventions,
    SimulationRequest,
    SimulationResult,
    BobrickMartireMetricCore,
    RelativisticSimulator,
    VectorizedPSO,
    PSOConfig,
    RelativisticFitnessEvaluator,
    AstronomicalRiskProvider,
    default_local_risk_provider,
)
```

For lower-level types and enums, use:

```python
from proxima.contracts.types import (
    MetricSignature,
    Frame,
    EnergyCondition,
    SimulationStatus,
)
```

### Core dataclasses

- `WarpParameterVector`: parametric description of the warp field.
- `MetricStatePoint`: spacetime state, including time and spatial coordinates.
- `ScientificConventions`: physical constants and numerical tolerances.
- `SimulationRequest`: model input for a geodesic propagation run.
- `SimulationResult`: outcome of a full simulation pass.
- `RiskBreakdown`: aggregated risk metadata.
- `FitnessBreakdown`: decomposed objective-function components.

---

## Scientific assumptions and conventions

PROXIMA is a research library and assumes explicit scientific conventions rather than hidden defaults. Users are expected to provide or configure:

- physical constants,
- metric signature,
- frame conventions,
- solver tolerances,
- regularization parameters,
- domain validity thresholds.

The default configuration sources are packaged under the project config tree, and are loaded automatically when needed.

### Units

The library follows a physically explicit convention where relevant variables are expressed in SI-like units:

- length in meters,
- time in seconds,
- energy in joules,
- speeds in meters per second,
- gravitational constant in SI units,
- c in meters per second.

### Numerical safety

The sim engine explicitly checks for:

- non-finite initial values,
- invalid metric domains,
- unphysical state propagation,
- energy-condition violations,
- excessive tidal stress.

This is intentional: the package is designed for scientific exploration, not only for “black-box” optimization.

---

## Typical usage patterns

### Research loop

1. Configure mission parameters.
2. Evaluate local metric behavior.
3. Propagate candidate trajectories.
4. Classify energy conditions.
5. Assess route risk from data providers.
6. Optimize the warp parameter vector with PSO.
7. Repeat with refined bounds or scientific assumptions.

### Parameter sweeps

Because the numerical kernels are JAX-based and batch-capable, PROXIMA is suitable for parameter sweeps and sensitivity scans across a grid of warp parameters.

### Mission heuristics

The library can be integrated into larger analytical or optimization workflows that combine:

- route planning,
- astrophysical environment assessment,
- physical condition screening,
- mission objective scoring.

---

## Error handling and diagnostics

The library is designed to fail explicitly when inputs are invalid or when a simulation crosses a domain or stability threshold. Typical failure modes include:

- invalid parameter values,
- non-finite state components,
- solver termination due to instability,
- excessive tidal stress,
- energy-condition rejection.

Each simulation result provides diagnostics for downstream workflows and post-processing analysis.

---

## Performance notes

PROXIMA is built with JAX to enable:

- array-based batch simulation,
- vectorized parameter sweeps,
- compiled numerical evaluation,
- higher throughput in optimization loops.

For scientific work, this is especially useful when evaluating many candidate parameter sets during a search or convergence study.

---

## Testing and validation

The project includes a substantial test suite covering:

- contract integrity,
- metric and energy validation,
- convergence benchmarks,
- optimization end-to-end runs,
- risk assessment behavior,
- cache and reproducibility checks,
- astrophysical data schema checks.

This makes PROXIMA suitable for validating research pipelines before using them in more elaborate experimental workflows.

---

## Advanced configuration

Most scientific defaults are shipped through YAML configuration files under `configs/`. This includes phase-based defaults for metric, simulator, data, and optimizer behavior.

Configuration can be loaded programmatically through the package helpers, making it easy to maintain consistent scientific assumptions across a project or experiment suite.

---

## Example: full end-to-end mission evaluation

```python
from proxima import (
    BobrickMartireMetricCore,
    RelativisticSimulator,
    VectorizedPSO,
    PSOConfig,
    WarpParameterVector,
    MetricStatePoint,
    SimulationRequest,
)
from proxima.contracts.types import (
    MetricSignature,
    ScientificConventions,
    Frame,
)

params = WarpParameterVector(
    bubble_radius_m=120.0,
    shell_thickness_m=6.0,
    velocity_fraction_c=0.7,
    shift_amplitude=1.2,
    lapse_floor=0.02,
)

state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

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

core = BobrickMartireMetricCore()
metric_eval = core.evaluate_metric(params, state)
print("Metric valid:", metric_eval.valid_domain)

sim = RelativisticSimulator()
request = SimulationRequest(
    params=params,
    initial_state=state,
    conventions=conventions,
    max_steps=200,
)
result = sim.simulate(request)
print("Simulation status:", result.status)

cfg = PSOConfig(n_particles=12, n_iterations=12, seed=7)
pso = VectorizedPSO(config=cfg)
opt = pso.optimize_warp(seed=7)
print("Best fitness:", opt.best_fitness)
print("Best parameters:", opt.best_params)
```

---

## License

This project is distributed under the MIT license.

---

## Support and research context

PROXIMA is a research-oriented library for advanced modeling of relativistic warp-drive-inspired mission concepts. It is designed to be transparent, inspectable, and configurable, so that users can evaluate assumptions, tune numerical settings, and integrate the model into larger experimentation pipelines.

For production research workflows, it is recommended to:

- maintain version pinning for scientific dependencies,
- store configuration snapshots alongside experiment results,
- keep risk and catalog provenance data with your analyses,
- validate outputs against established regression tests before publication or downstream use.

---

## Summary

PROXIMA provides a complete scientific stack for relativistic mission modeling and optimization in Python, integrating:

- warp geometry evaluation,
- relativistic geodesic simulation,
- energy-condition diagnostics,
- astrophysical risk assessment,
- parameter optimization with JAX-based PSO.

It is intended as a research-grade toolkit for users working at the intersection of physics, numerical analysis, and space mission design.

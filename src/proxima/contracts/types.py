from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricSignature(str, Enum):
    MINUS_PLUS_PLUS_PLUS = "-+++"


class Frame(str, Enum):
    ICRS = "ICRS"
    BCRS = "BCRS"
    LOCAL_CARTESIAN_3P1 = "LOCAL_CARTESIAN_3P1"


class EnergyCondition(str, Enum):
    WEC = "WEC"
    NEC = "NEC"
    DEC = "DEC"
    SEC = "SEC"


class SimulationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class ScientificConventions:
    phase0_version: str
    metric_signature: MetricSignature
    global_frame: Frame
    local_frame: Frame
    c_m_per_s: float
    g_m3_per_kg_s2: float
    atol: float
    rtol: float


@dataclass(frozen=True)
class WarpParameterVector:
    bubble_radius_m: float
    shell_thickness_m: float
    velocity_fraction_c: float
    shift_amplitude: float
    lapse_floor: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricStatePoint:
    t_s: float
    x_m: float
    y_m: float
    z_m: float
    tau_s: float


@dataclass(frozen=True)
class MetricEvaluation:
    lapse: float
    shift_vector: tuple[float, float, float]
    spatial_metric: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    valid_domain: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnergyConditionReport:
    wec_ok: bool
    nec_ok: bool
    dec_ok: bool
    sec_ok: bool
    violation_magnitudes: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteSegment:
    start_xyz_m: tuple[float, float, float]
    end_xyz_m: tuple[float, float, float]
    frame: Frame


@dataclass(frozen=True)
class SkyCoordinates:
    ra_deg: float
    dec_deg: float
    distance_pc: float | None
    frame: Frame


@dataclass(frozen=True)
class DataQueryContext:
    dataset_version: str
    extraction_timestamp_utc: str
    cache_key: str


@dataclass(frozen=True)
class DataQualityMetadata:
    source_name: str
    source_version: str
    confidence_score: float
    notes: str = ""


@dataclass(frozen=True)
class RiskBreakdown:
    radiative_flux_score: float
    extinction_h_column_score: float
    mass_density_score: float
    tidal_stress_score: float
    total_risk_score: float
    quality: DataQualityMetadata


@dataclass(frozen=True)
class SimulationRequest:
    params: WarpParameterVector
    initial_state: MetricStatePoint
    conventions: ScientificConventions
    max_steps: int


@dataclass(frozen=True)
class SimulationObservables:
    delta_tau_s: float
    delta_t_s: float
    total_energy_j: float
    tidal_stress_max_si: float


@dataclass(frozen=True)
class SimulationResult:
    status: SimulationStatus
    observables: SimulationObservables | None
    energy_conditions: EnergyConditionReport | None
    failure_reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitnessBreakdown:
    mission_term: float
    physical_penalty_term: float
    risk_penalty_term: float
    operational_penalty_term: float
    total_fitness: float

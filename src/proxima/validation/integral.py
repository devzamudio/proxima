"""Suite de validacion integral, no-regresion numerica y reproducibilidad entre snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxima.contracts.types import (
    DataQueryContext,
    Frame,
    MetricSignature,
    MetricStatePoint,
    RouteSegment,
    ScientificConventions,
    SimulationRequest,
    SimulationStatus,
    SkyCoordinates,
    WarpParameterVector,
)
from proxima.core.metric_core import BobrickMartireMetricCore
from proxima.data.pipeline import default_local_risk_provider
from proxima.data.snapshot import SnapshotStore
from proxima.optimizer.fitness import RelativisticFitnessEvaluator
from proxima.simulator.config import default_config as default_sim_config
from proxima.simulator.simulator import RelativisticSimulator


@dataclass(frozen=True)
class CanonicalCaseResult:
    case_name: str
    status: str
    delta_t_s: float
    delta_tau_s: float
    total_energy_j: float
    tidal_stress_max_si: float
    total_fitness: float
    risk_score: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionReport:
    n_cases: int
    all_passed: bool
    cases: list[CanonicalCaseResult]
    differences: dict[str, dict[str, float]] = field(default_factory=dict)


class IntegralValidator:
    """Validador integral end-to-end de no-regresion y consistencia entre corridas."""

    def __init__(
        self,
        conventions: ScientificConventions | None = None,
    ) -> None:
        self.conventions = conventions or ScientificConventions(
            phase0_version="1.0.0",
            metric_signature=MetricSignature.MINUS_PLUS_PLUS_PLUS,
            global_frame=Frame.ICRS,
            local_frame=Frame.LOCAL_CARTESIAN_3P1,
            c_m_per_s=299792458.0,
            g_m3_per_kg_s2=6.67430e-11,
            atol=1e-10,
            rtol=1e-8,
        )
        self.core = BobrickMartireMetricCore(self.conventions)
        self.sim = RelativisticSimulator(config=default_sim_config())
        self.risk_provider = default_local_risk_provider()
        self.fitness_eval = RelativisticFitnessEvaluator()

    def run_canonical_case(
        self,
        name: str,
        params: WarpParameterVector,
        sky: SkyCoordinates,
        segment: RouteSegment,
    ) -> CanonicalCaseResult:
        """Ejecuta una evaluacion completa end-to-end (Metrica -> Simulacion -> Riesgo -> Fitness)."""
        init_state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)
        req = SimulationRequest(
            params=params,
            initial_state=init_state,
            conventions=self.conventions,
            max_steps=100,
        )
        sim_res = self.sim.simulate(req)

        ctx = DataQueryContext(
            dataset_version="validation-1.0.0",
            extraction_timestamp_utc="2026-11-09T00:00:00Z",
            cache_key=f"val_{name}",
        )
        risk_res = self.risk_provider.evaluate_segment_risk(segment, sky, ctx)
        fit_res = self.fitness_eval.evaluate_fitness(sim_res, risk_res)

        obs = sim_res.observables
        delta_t = obs.delta_t_s if obs else 0.0
        delta_tau = obs.delta_tau_s if obs else 0.0
        e_total = obs.total_energy_j if obs else 0.0
        tidal_max = obs.tidal_stress_max_si if obs else 0.0

        return CanonicalCaseResult(
            case_name=name,
            status=sim_res.status.value,
            delta_t_s=delta_t,
            delta_tau_s=delta_tau,
            total_energy_j=e_total,
            tidal_stress_max_si=tidal_max,
            total_fitness=fit_res.total_fitness,
            risk_score=risk_res.total_risk_score,
            metrics={
                "mission_term": fit_res.mission_term,
                "physical_penalty": fit_res.physical_penalty_term,
                "risk_penalty": fit_res.risk_penalty_term,
            },
        )

    def run_regression_suite(
        self,
        baseline_file: Path | None = None,
        tol_rel: float = 1e-6,
    ) -> RegressionReport:
        """Corre los casos canonicos del sistema y los compara contra la linea base."""
        # 1. Caso Canonico Alcubierre/BM (Fase 1 baseline)
        c1_params = WarpParameterVector(
            bubble_radius_m=100.0,
            shell_thickness_m=5.0,
            velocity_fraction_c=0.5,
            shift_amplitude=1.0,
            lapse_floor=0.01,
        )
        c1_sky = SkyCoordinates(ra_deg=83.8, dec_deg=9.9, distance_pc=400.0, frame=Frame.ICRS)
        c1_seg = RouteSegment(start_xyz_m=(0, 0, 0), end_xyz_m=(1e15, 0, 0), frame=Frame.LOCAL_CARTESIAN_3P1)

        # 2. Caso Estatico Plano (Lapse=1, Shift=0 -> Espacio-tiempo de Minkowski)
        c2_params = WarpParameterVector(
            bubble_radius_m=100.0,
            shell_thickness_m=5.0,
            velocity_fraction_c=0.0,
            shift_amplitude=0.0,
            lapse_floor=1.0,
        )
        c2_sky = SkyCoordinates(ra_deg=0.0, dec_deg=0.0, distance_pc=10.0, frame=Frame.ICRS)
        c2_seg = RouteSegment(start_xyz_m=(0, 0, 0), end_xyz_m=(1e10, 0, 0), frame=Frame.LOCAL_CARTESIAN_3P1)

        r1 = self.run_canonical_case("canonical_bm_0.5c", c1_params, c1_sky, c1_seg)
        r2 = self.run_canonical_case("minkowski_static", c2_params, c2_sky, c2_seg)

        current_cases = [r1, r2]
        differences = {}
        all_passed = True

        if baseline_file and baseline_file.exists():
            baseline_data = json.loads(baseline_file.read_text(encoding="utf-8"))
            for case in current_cases:
                if case.case_name in baseline_data:
                    base = baseline_data[case.case_name]
                    diffs = {}
                    for k in ("delta_t_s", "delta_tau_s", "total_fitness", "risk_score"):
                        curr_v = getattr(case, k)
                        base_v = base.get(k, 0.0)
                        diff = abs(curr_v - base_v) / max(abs(base_v), 1e-12)
                        diffs[k] = diff
                        if diff > tol_rel:
                            all_passed = False
                    differences[case.case_name] = diffs

        return RegressionReport(
            n_cases=len(current_cases),
            all_passed=all_passed,
            cases=current_cases,
            differences=differences,
        )

    def verify_snapshot_reproducibility(
        self,
        snapshot_store: SnapshotStore,
        snapshot_id: str,
    ) -> bool:
        """Verifica que recargar un snapshot genere un hash y resultados exactamente identicos."""
        recs1, m1 = snapshot_store.load(snapshot_id)
        recs2, m2 = snapshot_store.load(snapshot_id)

        if m1.records_checksum != m2.records_checksum:
            return False
        if len(recs1) != len(recs2):
            return False
        return True

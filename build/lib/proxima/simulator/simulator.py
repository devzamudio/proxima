"""RelativisticSimulator — implementacion de RelativisticSimulatorContract."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp

from proxima.contracts.interfaces import RelativisticSimulatorContract
from proxima.contracts.types import (
    EnergyConditionReport,
    MetricStatePoint,
    SimulationObservables,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    WarpParameterVector,
)
from proxima.contracts.exceptions import DomainError

from .config import SimulatorConfig, default_config
from .state import pack_state, unpack_state, normalize_4velocity
from .geodesic import geodesic_rhs
from .integrator import (
    integrate_geodesic,
    IntegrationResult,
    TERMINATION_REASONS,
    TERM_SUCCESS,
    TERM_TIDAL_LIMIT,
)
from .tidal import tidal_tensor, max_tidal_stress
from .energy_integral import _build_grid, total_energy_fast
from .energy import (
    EnergyReport,
    aggregate_point_reports,
    aggregate_segment_reports,
    classify_energy_conditions,
    energy_margins,
    warpax_energy_conditions,
)
from .risk import RiskAssessor


def _termination_reason(code: int) -> str | None:
    if code == TERM_SUCCESS:
        return None
    return TERMINATION_REASONS.get(code, f"unknown code {code}")


class RelativisticSimulator:
    def __init__(self, config: SimulatorConfig | None = None):
        self.config = config or default_config()
        self.c = 299792458.0
        self.G = 6.67430e-11

    def _validate_request(self, params: WarpParameterVector, state: MetricStatePoint) -> None:
        from proxima.core.params import load_core_config, validate_params

        config = load_core_config()
        try:
            validate_params(params, config)
        except Exception as e:
            raise DomainError(str(e))

        for name, val in [("t_s", state.t_s), ("x_m", state.x_m),
                          ("y_m", state.y_m), ("z_m", state.z_m),
                          ("tau_s", state.tau_s)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise DomainError(f"estado inicial {name} invalido: {val}")

    def _params_to_array(self, params: WarpParameterVector) -> jnp.ndarray:
        return jnp.asarray(
            [
                params.bubble_radius_m,
                params.shell_thickness_m,
                params.velocity_fraction_c,
                params.shift_amplitude,
                params.lapse_floor,
            ],
            dtype=jnp.float64,
        )

    def _initial_state(
        self, params: WarpParameterVector, state: MetricStatePoint
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        X0 = jnp.asarray(
            [self.c * state.t_s, state.x_m, state.y_m, state.z_m],
            dtype=jnp.float64,
        )

        v_frac = params.velocity_fraction_c
        u0_raw = jnp.asarray([1.0, v_frac, 0.0, 0.0], dtype=jnp.float64)

        from proxima.core.kernels import adm
        from proxima.core.params import load_core_config

        core_config = load_core_config()
        eps_rel = core_config.radius_regularization_rel

        p = self._params_to_array(params)
        g4 = adm.metric4(p, X0, eps_rel)
        u0 = normalize_4velocity(u0_raw, g4, self.c)

        return X0, u0

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        params = request.params
        state = request.initial_state
        conventions = request.conventions
        max_steps = request.max_steps

        try:
            self._validate_request(params, state)
        except DomainError as e:
            return SimulationResult(
                status=SimulationStatus.FAILURE,
                observables=None,
                energy_conditions=None,
                failure_reason=f"DOMAIN: {e}",
                diagnostics={"error_type": "DomainError"},
            )

        p = self._params_to_array(params)
        from proxima.core.params import load_core_config

        core_config = load_core_config()
        eps_rel = core_config.radius_regularization_rel

        X0, u0 = self._initial_state(params, state)
        y0 = pack_state(X0, u0)

        try:
            integ_result: IntegrationResult = integrate_geodesic(
                y0, p, eps_rel, self.c, self.config,
                f=geodesic_rhs, max_steps=max_steps,
            )
        except Exception as e:
            return SimulationResult(
                status=SimulationStatus.FAILURE,
                observables=None,
                energy_conditions=None,
                failure_reason=f"INTEGRATION_ERROR: {e}",
                diagnostics={"error_type": "IntegrationError"},
            )

        term_code = int(integ_result.termination_code)

        X_final = integ_result.y_final[:4]
        delta_tau = float(integ_result.tau_final)
        delta_t = float(integ_result.t_final) - float(state.t_s)

        R_bubble = params.bubble_radius_m
        extent = self.config.energy_extent_rel
        n_grid = self.config.energy_n
        Xs, pesos = _build_grid(n_grid, extent, jnp.zeros(4), R_bubble)

        E_total, E_negative = total_energy_fast(
            p, Xs, pesos, eps_rel, self.c, self.G, max_points=64
        )

        traj = integ_result.trajectory
        mask = integ_result.valid_mask

        from proxima.core.kernels import adm as adm_mod, geometry as geom_mod

        def compute_tidal_at_point(y):
            X = y[:4]
            u = y[4:]
            g4 = adm_mod.metric4(p, X, eps_rel)
            u_norm = normalize_4velocity(u, g4, self.c)
            E_tidal = tidal_tensor(u_norm, X, p, eps_rel)
            return max_tidal_stress(E_tidal)

        # Marea maxima: solo evaluar en puntos validos, limitado para velocidad
        n_tidal_points = min(64, int(jnp.sum(mask)))
        if n_tidal_points > 0:
            valid_indices = jnp.where(mask)[0]
            tidal_idx = valid_indices[
                jnp.linspace(0, len(valid_indices) - 1, n_tidal_points).astype(jnp.int32)
            ]
            tidal_states = traj[tidal_idx]
            tidal_vals = jax.vmap(compute_tidal_at_point)(tidal_states)
            tidal_max = float(jnp.max(jnp.abs(tidal_vals)))
        else:
            tidal_max = 0.0

        if term_code == TERM_SUCCESS and tidal_max > self.config.tidal_limit_per_s2:
            term_code = TERM_TIDAL_LIMIT

        failure_reason = _termination_reason(term_code)
        status = (
            SimulationStatus.SUCCESS if term_code == TERM_SUCCESS else SimulationStatus.FAILURE
        )

        n_energy_points = min(32, int(jnp.sum(mask)))
        if n_energy_points > 0:
            valid_indices = jnp.where(mask)[0]
            sample_idx = valid_indices[
                jnp.linspace(0, len(valid_indices) - 1, n_energy_points).astype(jnp.int32)
            ]
            sample_states = traj[sample_idx]

            def compute_tensors_at_point(y):
                X = y[:4]
                g4 = adm_mod.metric4(p, X, eps_rel)
                g_inv = jnp.linalg.inv(g4)
                Gmn, _ = geom_mod.einstein_tensor(p, X, eps_rel)
                T_mn = (self.c**4 / (8.0 * jnp.pi * self.G)) * Gmn
                return T_mn, g4, g_inv

            T_batch, g4_batch, g_inv_batch = jax.vmap(compute_tensors_at_point)(sample_states)

            tol = min(
                self.config.wec_tolerance,
                self.config.nec_tolerance,
                self.config.dec_tolerance,
                self.config.sec_tolerance,
            )
            point_reports: list[EnergyReport] = []
            for idx_pt in range(n_energy_points):
                T_pt = T_batch[idx_pt]
                g4_pt = g4_batch[idx_pt]
                g_inv_pt = g_inv_batch[idx_pt]

                wp_res = None
                if self.config.energy_backend == "warpax":
                    wp_res = warpax_energy_conditions(T_pt, g4_pt, p, sample_states[idx_pt, :4])

                if wp_res is not None:
                    point_reports.append(
                        EnergyReport(
                            point_index=int(sample_idx[idx_pt]),
                            wec_ok=wp_res["wec_ok"],
                            nec_ok=wp_res["nec_ok"],
                            dec_ok=wp_res["dec_ok"],
                            sec_ok=wp_res["sec_ok"],
                            violation_magnitudes=wp_res["violation_magnitudes"],
                        )
                    )
                else:
                    he_res = classify_energy_conditions(T_pt, g_inv_pt, tol=tol)
                    m_res = energy_margins(T_pt, g_inv_pt)
                    point_reports.append(
                        EnergyReport(
                            point_index=int(sample_idx[idx_pt]),
                            wec_ok=bool(he_res["wec"]),
                            nec_ok=bool(he_res["nec"]),
                            dec_ok=bool(he_res["dec"]),
                            sec_ok=bool(he_res["sec"]),
                            violation_magnitudes=he_res.get("violation_magnitudes", {}),
                            margins={k: float(v) for k, v in m_res.items()},
                            hawking_ellis_type=str(he_res.get("type", "unknown")),
                        )
                    )

            agg_energy = aggregate_point_reports(point_reports, backend=self.config.energy_backend)
            seg_energy = aggregate_segment_reports(point_reports, n_segments=min(4, n_energy_points))

            energy_report = EnergyConditionReport(
                wec_ok=agg_energy.wec_ok_all,
                nec_ok=agg_energy.nec_ok_all,
                dec_ok=agg_energy.dec_ok_all,
                sec_ok=agg_energy.sec_ok_all,
                violation_magnitudes=agg_energy.max_violation_magnitudes,
            )
        else:
            agg_energy = aggregate_point_reports([], backend=self.config.energy_backend)
            seg_energy = []
            energy_report = EnergyConditionReport(
                wec_ok=True, nec_ok=True, dec_ok=True, sec_ok=True, violation_magnitudes={}
            )

        risk_assessor = RiskAssessor()
        risk_breakdown, segment_risks = risk_assessor.assess_trajectory(
            traj, mask, n_segments=10
        )

        diagnostics = {
            "termination_code": term_code,
            "steps_taken": int(integ_result.steps_taken),
            "lambda_final": float(integ_result.lambda_final),
            "energy_negative_J": E_negative,
            "tidal_max_per_s2": tidal_max,
            "risk_total_score": risk_breakdown.total_risk_score,
            "n_energy_points": n_energy_points,
            "energy_aggregate": agg_energy,
            "energy_segments": seg_energy,
        }

        version_fase0 = conventions.phase0_version if conventions is not None else "1.0.0"
        diagnostics["traceability"] = {
            "version_fase0": version_fase0,
            "firma_metrica": "-+++",
            "constantes_fisicas": {"c_m_per_s": self.c, "G_m3_per_kg_s2": self.G},
            "unidades_activas": {
                "tau": "s", "t": "s", "energia": "J", "marea": "1/s^2",
            },
            "tolerancias": {"atol": self.config.atol, "rtol": self.config.rtol},
        }

        if status == SimulationStatus.SUCCESS:
            observables = SimulationObservables(
                delta_tau_s=delta_tau,
                delta_t_s=delta_t,
                total_energy_j=E_total,
                tidal_stress_max_si=tidal_max,
            )
        else:
            observables = None

        return SimulationResult(
            status=status,
            observables=observables,
            energy_conditions=energy_report,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
        )
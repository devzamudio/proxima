"""Adaptador de contrato: BobrickMartireMetricCore implementa MetricCoreContract."""

from __future__ import annotations

import math

import jax.numpy as jnp

from proxima.contracts.exceptions import NumericalStabilityError
from proxima.contracts.types import (
    EnergyConditionReport,
    MetricEvaluation,
    MetricStatePoint,
    ScientificConventions,
    WarpParameterVector,
)

from .batch import evaluate_point_jit
from .params import CoreConfig, allows_superluminal, load_core_config, params_to_array, validate_params

C_DEFAULT = 299792458.0
G_DEFAULT = 6.67430e-11


class BobrickMartireMetricCore:
    """Nucleo metrico BM/ADM (caso canonico: shift tanh tipo Alcubierre).

    Unidades de salida:
      lapse: adimensional; shift_vector: m/s (SI); spatial_metric: adimensional.
    """

    def __init__(
        self,
        conventions: ScientificConventions | None = None,
        config: CoreConfig | None = None,
    ) -> None:
        self.config = config or load_core_config()
        self.conventions = conventions
        self.c = float(conventions.c_m_per_s) if conventions else C_DEFAULT
        self.G = float(conventions.g_m3_per_kg_s2) if conventions else G_DEFAULT
        self.atol = float(conventions.atol) if conventions else 1e-10

    # -- interno -----------------------------------------------------------------
    def _evaluate(self, params: WarpParameterVector, state: MetricStatePoint) -> dict:
        validate_params(params, self.config)
        p = params_to_array(params)
        X = jnp.asarray([self.c * state.t_s, state.x_m, state.y_m, state.z_m], dtype=jnp.float64)
        out = evaluate_point_jit(
            p,
            X,
            self.config.bounds_array(allows_superluminal(params)),
            eps_rel=self.config.radius_regularization_rel,
            c=self.c,
            G_newton=self.G,
        )
        if not bool(out["finite_ok"]):
            raise NumericalStabilityError(f"valores no finitos en nucleo metrico para {params} @ {state}")
        return out

    def _traceability(self) -> dict:
        return {
            "version_fase0": self.conventions.phase0_version if self.conventions else "1.0.0",
            "version_fase1": self.config.version,
            "firma_metrica": "-+++",
            "constantes_fisicas": {"c_m_per_s": self.c, "G_m3_per_kg_s2": self.G},
            "unidades_activas": {"lapse": "1", "shift_vector": "m/s", "spatial_metric": "1"},
            "tolerancias": {"atol": self.atol},
        }

    # -- contrato ----------------------------------------------------------------
    def evaluate_metric(self, params: WarpParameterVector, state: MetricStatePoint) -> MetricEvaluation:
        out = self._evaluate(params, state)
        beta_si = tuple(float(b) * self.c for b in out["shift"])
        gamma = tuple(tuple(float(x) for x in row) for row in out["spatial_metric"])
        diagnostics = {
            "shift_function": float(out["shift_function"]),
            "r_s_m": float(out["r_s_m"]),
            "rho_eulerian_J_m3": float(out["rho_eulerian_J_m3"]),
            "trace_K_per_m": float(out["trace_K_per_m"]),
            "KijKij_per_m2": float(out["KijKij_per_m2"]),
            "ricci_scalar_per_m2": float(out["ricci_scalar_per_m2"]),
            "traceability": self._traceability(),
        }
        return MetricEvaluation(
            lapse=float(out["lapse"]),
            shift_vector=beta_si,  # type: ignore[arg-type]
            spatial_metric=gamma,  # type: ignore[arg-type]
            valid_domain=bool(out["valid_domain"]),
            diagnostics=diagnostics,
        )

    def evaluate_energy_conditions(
        self, params: WarpParameterVector, state: MetricStatePoint
    ) -> EnergyConditionReport:
        """WEC preliminar para observador Euleriano (rho >= 0).

        NEC/SEC aun no se certifican (Fase 2, Warpax): se reportan False ("no certificado").
        DEC implica WEC; tambien queda no certificado.
        """
        out = self._evaluate(params, state)
        rho = float(out["rho_eulerian_J_m3"])
        # Tolerancia escalada: atol relativo a la escala de densidad c^4/(8 pi G R^2).
        scale = self.c**4 / (8.0 * math.pi * self.G * params.bubble_radius_m**2)
        wec_ok = rho >= -self.atol * scale
        return EnergyConditionReport(
            wec_ok=bool(wec_ok),
            nec_ok=False,
            dec_ok=False,
            sec_ok=False,
            violation_magnitudes={
                "WEC": max(0.0, -rho),
                "rho_eulerian_J_m3": rho,
            },
        )

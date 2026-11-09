"""Evaluador de Fitness que satisface FitnessEvaluatorContract y funciones JAX puras."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from proxima.contracts.exceptions import ContractValidationError
from proxima.contracts.types import (
    FitnessBreakdown,
    RiskBreakdown,
    SimulationResult,
    SimulationStatus,
)
from .config import PSOConfig


class RelativisticFitnessEvaluator:
    """Implementa FitnessEvaluatorContract para evaluación individual o fuera de JIT.

    Garantiza contractualmente que:
        total_fitness == mission_term + physical_penalty_term + risk_penalty_term + operational_penalty_term
    """

    def __init__(self, config: PSOConfig | None = None) -> None:
        self.config = config or PSOConfig.default_config()

    def evaluate_fitness(
        self,
        result: SimulationResult,
        risk: RiskBreakdown | None,
    ) -> FitnessBreakdown:
        cfg = self.config
        
        # 1. Termino de mision
        if result.status == SimulationStatus.SUCCESS and result.observables is not None:
            obs = result.observables
            # Tiempo: t_coord + w_tau * tau
            t_term = (obs.delta_t_s + cfg.mission_tau_weight * obs.delta_tau_s) / max(cfg.mission_time_s, 1e-12)
            # Energia (positiva o negativa/cero)
            e_term = cfg.energy_weight * (abs(obs.total_energy_j) / max(cfg.energy_scale_j, 1e-12))
            mission_term = float(t_term + e_term)
            operational_penalty = 0.0
        else:
            mission_term = 1.0e5
            operational_penalty = float(cfg.penalty_simulation_failure)

        # 2. Penalizaciones fisicas
        physical_penalty = 0.0
        if result.energy_conditions is not None:
            ec = result.energy_conditions
            v = ec.violation_magnitudes
            if not ec.wec_ok:
                physical_penalty += cfg.penalty_wec * (1.0 + float(v.get("WEC", 0.0)))
            if not ec.nec_ok:
                physical_penalty += cfg.penalty_nec * (1.0 + float(v.get("NEC", 0.0)))
            if not ec.sec_ok:
                physical_penalty += cfg.penalty_sec * (1.0 + float(v.get("SEC", 0.0)))
            if not ec.dec_ok:
                physical_penalty += cfg.penalty_dec * (1.0 + float(v.get("DEC", 0.0)))

        if result.observables is not None and result.observables.tidal_stress_max_si > 9.81:
            excess = result.observables.tidal_stress_max_si - 9.81
            physical_penalty += cfg.penalty_tidal * excess

        # 3. Penalizacion por riesgo ambiental / astronomico
        risk_penalty = 0.0
        if risk is not None:
            risk_penalty = float(cfg.penalty_risk * risk.total_risk_score)

        total_fitness = mission_term + physical_penalty + risk_penalty + operational_penalty

        return FitnessBreakdown(
            mission_term=mission_term,
            physical_penalty_term=physical_penalty,
            risk_penalty_term=risk_penalty,
            operational_penalty_term=operational_penalty,
            total_fitness=total_fitness,
        )


@jax.jit
def jax_fitness_breakdown(
    delta_t: jnp.ndarray,
    delta_tau: jnp.ndarray,
    total_energy: jnp.ndarray,
    tidal_max: jnp.ndarray,
    wec_margin: jnp.ndarray,
    nec_margin: jnp.ndarray,
    sec_margin: jnp.ndarray,
    dec_margin: jnp.ndarray,
    success: jnp.ndarray,
    risk_score: jnp.ndarray,
    mission_time_s: float,
    mission_tau_weight: float,
    energy_scale_j: float,
    energy_weight: float,
    penalty_wec: float,
    penalty_nec: float,
    penalty_sec: float,
    penalty_dec: float,
    penalty_tidal: float,
    penalty_risk: float,
    penalty_sim_failure: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculo puramente JAX vmappeable para lotes de particulas (N,).
    
    Retorna: (total_fitness, mission_term, physical_penalty, risk_penalty, operational_penalty)
    """
    # 1. Termino de mision
    t_term = (delta_t + mission_tau_weight * delta_tau) / jnp.maximum(mission_time_s, 1e-12)
    e_term = energy_weight * (jnp.abs(total_energy) / jnp.maximum(energy_scale_j, 1e-12))
    mission_success = t_term + e_term
    mission_fail = 1.0e5
    mission_term = jnp.where(success, mission_success, mission_fail)

    operational_penalty = jnp.where(success, 0.0, penalty_sim_failure)

    # 2. Penalizaciones fisicas: margenes negativos implican violacion
    wec_viol = jnp.maximum(0.0, -wec_margin)
    nec_viol = jnp.maximum(0.0, -nec_margin)
    sec_viol = jnp.maximum(0.0, -sec_margin)
    dec_viol = jnp.maximum(0.0, -dec_margin)

    p_wec = jnp.where(wec_viol > 0.0, penalty_wec * (1.0 + wec_viol), 0.0)
    p_nec = jnp.where(nec_viol > 0.0, penalty_nec * (1.0 + nec_viol), 0.0)
    p_sec = jnp.where(sec_viol > 0.0, penalty_sec * (1.0 + sec_viol), 0.0)
    p_dec = jnp.where(dec_viol > 0.0, penalty_dec * (1.0 + dec_viol), 0.0)

    tidal_excess = jnp.maximum(0.0, tidal_max - 9.81)
    p_tidal = penalty_tidal * tidal_excess

    physical_penalty = p_wec + p_nec + p_sec + p_dec + p_tidal

    # 3. Penalizacion de riesgo
    risk_penalty = penalty_risk * risk_score

    total_fitness = mission_term + physical_penalty + risk_penalty + operational_penalty
    return total_fitness, mission_term, physical_penalty, risk_penalty, operational_penalty

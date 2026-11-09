"""Optimizador PSO vectorizado con soporte de topologias y evaluación relativista."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np

from proxima.contracts.types import WarpParameterVector
from proxima.core.params import CoreConfig, PARAM_ORDER, load_core_config
from proxima.simulator.config import SimulatorConfig, default_config as default_sim_config
from proxima.simulator.batch import simulate_batch

from .config import PSOConfig
from .fitness import jax_fitness_breakdown, RelativisticFitnessEvaluator
from .topology import get_global_best, get_ring_best
from .swarm import SwarmState, init_swarm, update_particle_dynamics, update_pbest


@dataclass(frozen=True)
class OptimizationHistory:
    """Trazabilidad completa de la convergencia del enjambre."""
    gbest_fitness_history: list[float]
    mean_fitness_history: list[float]
    gbest_position_history: list[tuple[float, ...]]


@dataclass(frozen=True)
class OptimizationResult:
    """Resultado final de una optimización PSO."""
    best_params: WarpParameterVector
    best_vector: tuple[float, ...]
    best_fitness: float
    history: OptimizationHistory
    n_iterations: int
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


class VectorizedPSO:
    """Motor de optimización PSO vectorizado sobre espacios de parámetros Warp."""

    def __init__(
        self,
        config: PSOConfig | None = None,
        core_config: CoreConfig | None = None,
        sim_config: SimulatorConfig | None = None,
        *,
        bounds: jnp.ndarray | None = None,
        allow_superluminal: bool = False,
    ) -> None:
        self.config = config or PSOConfig.default_config()
        self.core_config = core_config or load_core_config()
        self.sim_config = sim_config or default_sim_config()
        self.allow_superluminal = allow_superluminal

        if bounds is not None:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)
        else:
            self.bounds = self.core_config.bounds_array(allow_superluminal)

    def optimize_function(
        self,
        fitness_fn: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        seed: int | None = None,
        n_particles: int | None = None,
        n_iterations: int | None = None,
    ) -> OptimizationResult:
        """Optimiza una función genérica vmappeada f(P) -> fitness (N,)."""
        cfg = self.config
        n_p = n_particles or cfg.n_particles
        n_iter = n_iterations or cfg.n_iterations
        s = seed if seed is not None else cfg.seed

        key = jax.random.PRNGKey(s)
        positions, velocities, key = init_swarm(key, n_p, self.bounds, cfg.v_clamp_ratio)

        # Eval inicial
        current_fit = fitness_fn(positions)
        pbest_positions = positions
        pbest_fitness = current_fit

        best_idx = jnp.argmin(pbest_fitness)
        gbest_pos = pbest_positions[best_idx]
        gbest_fit = pbest_fitness[best_idx]

        gbest_fit_history = [float(gbest_fit)]
        mean_fit_history = [float(jnp.mean(current_fit))]
        gbest_pos_history = [tuple(float(x) for x in gbest_pos)]

        for it in range(n_iter):
            # Inercia lineal decreciente w(t)
            w = cfg.w_max - (cfg.w_max - cfg.w_min) * (it / max(n_iter - 1, 1))

            # Determinar atractor social segun topologia
            if cfg.topology == "ring":
                attractor_pos, _ = get_ring_best(pbest_positions, pbest_fitness, cfg.ring_neighborhood_size)
            else:
                attractor_pos, _ = get_global_best(pbest_positions, pbest_fitness)

            # Actualizar particulas
            positions, velocities, key = update_particle_dynamics(
                positions,
                velocities,
                pbest_positions,
                attractor_pos,
                self.bounds,
                w,
                cfg.c1,
                cfg.c2,
                cfg.v_clamp_ratio,
                key,
            )

            # Evaluar
            current_fit = fitness_fn(positions)
            pbest_positions, pbest_fitness = update_pbest(
                positions, current_fit, pbest_positions, pbest_fitness
            )

            best_idx = jnp.argmin(pbest_fitness)
            if pbest_fitness[best_idx] < gbest_fit:
                gbest_pos = pbest_positions[best_idx]
                gbest_fit = pbest_fitness[best_idx]

            gbest_fit_history.append(float(gbest_fit))
            mean_fit_history.append(float(jnp.mean(current_fit)))
            gbest_pos_history.append(tuple(float(x) for x in gbest_pos))

        best_vec = tuple(float(x) for x in gbest_pos)
        if len(best_vec) >= 5:
            warp_params = WarpParameterVector(
                bubble_radius_m=best_vec[0],
                shell_thickness_m=best_vec[1],
                velocity_fraction_c=best_vec[2],
                shift_amplitude=best_vec[3],
                lapse_floor=best_vec[4],
                metadata={"allow_superluminal": self.allow_superluminal},
            )
        else:
            warp_params = WarpParameterVector(
                bubble_radius_m=best_vec[0] if len(best_vec) > 0 else 100.0,
                shell_thickness_m=best_vec[1] if len(best_vec) > 1 else 5.0,
                velocity_fraction_c=best_vec[2] if len(best_vec) > 2 else 0.5,
                shift_amplitude=best_vec[3] if len(best_vec) > 3 else 1.0,
                lapse_floor=best_vec[4] if len(best_vec) > 4 else 0.01,
                metadata={"allow_superluminal": self.allow_superluminal, "synthetic_dim": len(best_vec)},
            )

        history = OptimizationHistory(
            gbest_fitness_history=gbest_fit_history,
            mean_fitness_history=mean_fit_history,
            gbest_position_history=gbest_pos_history,
        )

        return OptimizationResult(
            best_params=warp_params,
            best_vector=best_vec,
            best_fitness=float(gbest_fit),
            history=history,
            n_iterations=n_iter,
            converged=True,
            diagnostics={
                "seed": s,
                "n_particles": n_p,
                "topology": cfg.topology,
            },
        )

    def optimize_warp(
        self,
        initial_states: jnp.ndarray | None = None,
        risk_scores: jnp.ndarray | None = None,
        *,
        seed: int | None = None,
        n_particles: int | None = None,
        n_iterations: int | None = None,
        c: float = 299792458.0,
        G_newton: float = 6.67430e-11,
        eps_rel: float = 1.0e-9,
    ) -> OptimizationResult:
        """Ejecuta PSO completa acoplada al simulador relativista via simulate_batch."""
        cfg = self.config
        n_p = n_particles or cfg.n_particles

        if initial_states is None:
            # Estado por defecto en reposo en el origen
            # X = (0, 0, 0, 0), u = (1, 0, 0, 0)
            single_state = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
            y0_list = jnp.repeat(single_state[None, :], n_p, axis=0)
        else:
            y0_list = initial_states

        if risk_scores is None:
            r_scores = jnp.zeros(n_p, dtype=jnp.float64)
        else:
            r_scores = risk_scores

        def warp_fitness(P: jnp.ndarray) -> jnp.ndarray:
            # 1. Simular lote
            batch_res = simulate_batch(
                P,
                y0_list,
                self.sim_config,
                c=c,
                G_newton=G_newton,
                eps_rel=eps_rel,
            )

            # 2. Computar fitness descompuesto vectorizado
            fit, _, _, _, _ = jax_fitness_breakdown(
                batch_res["delta_t"],
                batch_res["delta_tau"],
                batch_res["total_energy"],
                batch_res["tidal_max"],
                batch_res["wec_margin"],
                batch_res["nec_margin"],
                batch_res["sec_margin"],
                batch_res["dec_margin"],
                batch_res["success"],
                r_scores,
                cfg.mission_time_s,
                cfg.mission_tau_weight,
                cfg.energy_scale_j,
                cfg.energy_weight,
                cfg.penalty_wec,
                cfg.penalty_nec,
                cfg.penalty_sec,
                cfg.penalty_dec,
                cfg.penalty_tidal,
                cfg.penalty_risk,
                cfg.penalty_simulation_failure,
            )
            return fit

        return self.optimize_function(
            warp_fitness,
            seed=seed,
            n_particles=n_p,
            n_iterations=n_iterations,
        )

"""Modulo de calibracion y analisis de sensibilidad para hiperparametros y parametros Warp."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from proxima.contracts.types import WarpParameterVector
from proxima.core.params import PARAM_ORDER, CoreConfig, load_core_config
from proxima.optimizer.config import PSOConfig
from proxima.optimizer.optimizer import VectorizedPSO, OptimizationResult


@dataclass(frozen=True)
class HyperparameterSweepResult:
    """Resultado del barrido de configuraciones de hiperparametros de la PSO."""
    best_config: dict[str, Any]
    best_fitness: float
    records: list[dict[str, Any]]
    ranking: list[dict[str, Any]]


@dataclass(frozen=True)
class WarpSensitivityResult:
    """Sensibilidad del fitness y observables ante perturbaciones en cada parametro warp."""
    baseline_params: tuple[float, ...]
    baseline_fitness: float
    sensitivities: dict[str, float]  # d(Fitness) / d(theta_i) normalizado
    perturbations: dict[str, list[dict[str, float]]]


@dataclass(frozen=True)
class DataRobustnessResult:
    """Evaluacion de robustez del fitness ante perturbaciones estocasticas en datos astronomicos."""
    nominal_fitness: float
    perturbed_fitness_mean: float
    perturbed_fitness_std: float
    variance_ratio: float
    stability_score: float  # [0, 1] donde 1 es perfectamente robusto


class CalibrationEngine:
    """Motor de calibracion para el ajuste de hiperparametros y analisis de sensibilidad."""

    def __init__(
        self,
        core_config: CoreConfig | None = None,
        base_pso_config: PSOConfig | None = None,
    ) -> None:
        self.core_config = core_config or load_core_config()
        self.base_pso_config = base_pso_config or PSOConfig.default_config()

    def sweep_pso_hyperparameters(
        self,
        objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
        param_grid: dict[str, list[Any]],
        *,
        n_seeds: int = 2,
        base_seed: int = 42,
    ) -> HyperparameterSweepResult:
        """Realiza un barrido en rejilla sobre combinaciones de hiperparametros de la PSO.
        
        Promedia el rendimiento sobre n_seeds para aislar el factor estocastico.
        """
        import itertools

        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        records = []
        bounds = self.core_config.bounds_array(allow_superluminal=False)

        for combo in combos:
            overrides = dict(zip(keys, combo))
            # Construir PSOConfig con overrides
            cfg_dict = {
                "w_max": overrides.get("w_max", self.base_pso_config.w_max),
                "w_min": overrides.get("w_min", self.base_pso_config.w_min),
                "c1": overrides.get("c1", self.base_pso_config.c1),
                "c2": overrides.get("c2", self.base_pso_config.c2),
                "topology": overrides.get("topology", self.base_pso_config.topology),
                "n_particles": overrides.get("n_particles", 15),
                "n_iterations": overrides.get("n_iterations", 15),
            }
            cfg = PSOConfig(**cfg_dict)
            pso = VectorizedPSO(config=cfg, bounds=bounds)

            fitness_samples = []
            for s_idx in range(n_seeds):
                res = pso.optimize_function(
                    objective_fn,
                    seed=base_seed + s_idx,
                    n_particles=cfg.n_particles,
                    n_iterations=cfg.n_iterations,
                )
                fitness_samples.append(res.best_fitness)

            mean_fit = float(np.mean(fitness_samples))
            std_fit = float(np.std(fitness_samples))

            entry = {
                "params": overrides,
                "mean_fitness": mean_fit,
                "std_fitness": std_fit,
                "samples": fitness_samples,
            }
            records.append(entry)

        ranking = sorted(records, key=lambda x: x["mean_fitness"])
        best = ranking[0]

        return HyperparameterSweepResult(
            best_config=best["params"],
            best_fitness=best["mean_fitness"],
            records=records,
            ranking=ranking,
        )

    def analyze_warp_sensitivity(
        self,
        eval_fn: Callable[[tuple[float, ...]], float],
        nominal_vector: tuple[float, ...],
        delta_rel: float = 0.05,
    ) -> WarpSensitivityResult:
        """Calcula la sensibilidad relativa del fitness perturbando cada parametro warp +- delta_rel."""
        base_fit = eval_fn(nominal_vector)
        sensitivities = {}
        perturbation_records: dict[str, list[dict[str, float]]] = {}

        bounds = self.core_config.bounds_array(allow_superluminal=False)

        for i, name in enumerate(PARAM_ORDER):
            val = nominal_vector[i]
            low, high = float(bounds[i, 0]), float(bounds[i, 1])
            step = max(abs(val) * delta_rel, 1e-6)

            val_plus = min(high, val + step)
            val_minus = max(low, val - step)

            vec_plus = list(nominal_vector)
            vec_plus[i] = val_plus
            fit_plus = eval_fn(tuple(vec_plus))

            vec_minus = list(nominal_vector)
            vec_minus[i] = val_minus
            fit_minus = eval_fn(tuple(vec_minus))

            # Gradiente central normalizado: (dF/F) / (dx/x)
            effective_dx = val_plus - val_minus
            if effective_dx > 0 and abs(base_fit) > 1e-12:
                df = fit_plus - fit_minus
                grad = (df / effective_dx) * (val / base_fit)
            else:
                grad = 0.0

            sensitivities[name] = float(abs(grad))
            perturbation_records[name] = [
                {"delta": -step, "val": val_minus, "fitness": fit_minus},
                {"delta": 0.0, "val": val, "fitness": base_fit},
                {"delta": step, "val": val_plus, "fitness": fit_plus},
            ]

        return WarpSensitivityResult(
            baseline_params=nominal_vector,
            baseline_fitness=base_fit,
            sensitivities=sensitivities,
            perturbations=perturbation_records,
        )

    def evaluate_data_robustness(
        self,
        eval_risk_fn: Callable[[float], float],  # funcion que recibe noise_level y evalua
        nominal_noise: float = 0.0,
        perturbation_sigma: float = 0.1,
        n_mc_samples: int = 20,
        seed: int = 42,
    ) -> DataRobustnessResult:
        """Evalua la robustez frente a incertidumbre en datos mediante Monte Carlo."""
        rng = np.random.default_rng(seed)
        nominal_val = eval_risk_fn(nominal_noise)

        samples = []
        for _ in range(n_mc_samples):
            noise = float(rng.normal(0.0, perturbation_sigma))
            samples.append(eval_risk_fn(noise))

        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        var_ratio = std_val / max(abs(nominal_val), 1e-12)
        stability = float(math.exp(-var_ratio))

        return DataRobustnessResult(
            nominal_fitness=nominal_val,
            perturbed_fitness_mean=mean_val,
            perturbed_fitness_std=std_val,
            variance_ratio=var_ratio,
            stability_score=stability,
        )

"""Configuracion del Optimizador PSO Vectorizado (Fase 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

def _resolve_default_config() -> Path:
    pkg_cfg = Path(__file__).resolve().parent.parent / "configs" / "phase4" / "optimizer_defaults.yaml"
    if pkg_cfg.exists():
        return pkg_cfg
    repo_cfg = Path(__file__).resolve().parents[3] / "configs" / "phase4" / "optimizer_defaults.yaml"
    return repo_cfg


_DEFAULT_CONFIG = _resolve_default_config()


@dataclass(frozen=True)
class PSOConfig:
    version: str = "1.0.0"
    based_on_phase0: str = "1.0.0"

    # Dinamica PSO
    n_particles: int = 30
    n_iterations: int = 50
    topology: str = "global"  # "global" o "ring"
    ring_neighborhood_size: int = 3
    seed: int = 42

    # Inercia y coeficientes de aceleracion
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.5
    c2: float = 1.5
    v_clamp_ratio: float = 0.2

    # Ponderaciones de la funcion objetivo
    mission_time_s: float = 1.0
    mission_tau_weight: float = 0.5
    energy_scale_j: float = 1.0e20
    energy_weight: float = 0.1

    # Penalizaciones fisicas
    penalty_wec: float = 1.0e4
    penalty_nec: float = 1.0e4
    penalty_sec: float = 1.0e3
    penalty_dec: float = 1.0e3
    penalty_tidal: float = 1.0e5

    # Penalizaciones operativas y de riesgo
    penalty_risk: float = 1.0e3
    penalty_invalid_domain: float = 1.0e7
    penalty_simulation_failure: float = 1.0e6

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PSOConfig":
        raw = yaml.safe_load(Path(path or _DEFAULT_CONFIG).read_text(encoding="utf-8"))
        p4 = raw.get("phase4", {})
        pso = raw.get("pso", {})
        ow = raw.get("objective_weights", {})
        return cls(
            version=str(p4.get("version", "1.0.0")),
            based_on_phase0=str(p4.get("based_on_phase0", "1.0.0")),
            n_particles=int(pso.get("n_particles", 30)),
            n_iterations=int(pso.get("n_iterations", 50)),
            topology=str(pso.get("topology", "global")),
            ring_neighborhood_size=int(pso.get("ring_neighborhood_size", 3)),
            seed=int(pso.get("seed", 42)),
            w_max=float(pso.get("w_max", 0.9)),
            w_min=float(pso.get("w_min", 0.4)),
            c1=float(pso.get("c1", 1.5)),
            c2=float(pso.get("c2", 1.5)),
            v_clamp_ratio=float(pso.get("v_clamp_ratio", 0.2)),
            mission_time_s=float(ow.get("mission_time_s", 1.0)),
            mission_tau_weight=float(ow.get("mission_tau_weight", 0.5)),
            energy_scale_j=float(ow.get("energy_scale_j", 1.0e20)),
            energy_weight=float(ow.get("energy_weight", 0.1)),
            penalty_wec=float(ow.get("penalty_wec", 1.0e4)),
            penalty_nec=float(ow.get("penalty_nec", 1.0e4)),
            penalty_sec=float(ow.get("penalty_sec", 1.0e3)),
            penalty_dec=float(ow.get("penalty_dec", 1.0e3)),
            penalty_tidal=float(ow.get("penalty_tidal", 1.0e5)),
            penalty_risk=float(ow.get("penalty_risk", 1.0e3)),
            penalty_invalid_domain=float(ow.get("penalty_invalid_domain", 1.0e7)),
            penalty_simulation_failure=float(ow.get("penalty_simulation_failure", 1.0e6)),
        )

    @classmethod
    def default_config(cls) -> "PSOConfig":
        return cls()

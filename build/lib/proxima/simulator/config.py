"""Configuracion del simulador relativista (Fase 2).

Parametros de integracion, guardas, malla de energia y umbrales.
Todo en unidades SI salvo que se indique lo contrario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

def _resolve_default_config() -> Path:
    pkg_cfg = Path(__file__).resolve().parent.parent / "configs" / "phase2" / "simulator_defaults.yaml"
    if pkg_cfg.exists():
        return pkg_cfg
    repo_cfg = Path(__file__).resolve().parents[3] / "configs" / "phase2" / "simulator_defaults.yaml"
    return repo_cfg


_DEFAULT_CONFIG = _resolve_default_config()


@dataclass(frozen=True)
class SimulatorConfig:
    # --- Integracion adaptativa (DP5) ---
    h0: float          # paso inicial en tau (s)
    h_min: float       # paso minimo (s)
    h_max: float       # paso maximo (s)
    atol: float        # tolerancia absoluta
    rtol: float        # tolerancia relativa
    norm_tol: float    # tolerancia de deriva |u.u + c^2|/c^2

    # --- Guardas fisicas ---
    tidal_limit_per_s2: float   # umbral de marea (1/s^2)
    tau_final_s: float          # tiempo propio objetivo (s)

    # --- Malla de energia Euleriana ---
    energy_n: int               # resolucion de la malla 3D (puntos por eje)
    energy_extent_rel: float    # extension en unidades de radio de burbuja

    # --- Backend de condiciones de energia ---
    energy_backend: str         # "own" o "warpax"

    # --- Umbrales de energia (convenciones Fase 0, seccion 5) ---
    wec_tolerance: float
    nec_tolerance: float
    dec_tolerance: float
    sec_tolerance: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SimulatorConfig":
        raw = yaml.safe_load(Path(path or _DEFAULT_CONFIG).read_text(encoding="utf-8"))
        sec = raw["phase2"]
        return cls(
            h0=float(sec["integration"]["h0_s"]),
            h_min=float(sec["integration"]["h_min_s"]),
            h_max=float(sec["integration"]["h_max_s"]),
            atol=float(sec["integration"]["atol"]),
            rtol=float(sec["integration"]["rtol"]),
            norm_tol=float(sec["integration"]["norm_tol"]),
            tidal_limit_per_s2=float(sec["guards"]["tidal_limit_per_s2"]),
            tau_final_s=float(sec["guards"]["tau_final_s"]),
            energy_n=int(sec["energy_integral"]["n_grid"]),
            energy_extent_rel=float(sec["energy_integral"]["extent_rel"]),
            energy_backend=str(sec["energy_backend"]),
            wec_tolerance=float(sec["energy_thresholds"]["wec"]),
            nec_tolerance=float(sec["energy_thresholds"]["nec"]),
            dec_tolerance=float(sec["energy_thresholds"]["dec"]),
            sec_tolerance=float(sec["energy_thresholds"]["sec"]),
        )

    @classmethod
    def default_config(cls) -> "SimulatorConfig":
        """Retorna la configuracion por defecto sin leer YAML."""
        return default_config()


def default_config() -> SimulatorConfig:
    """Configuracion por defecto sin leer YAML (para tests y uso directo)."""
    return SimulatorConfig(
        h0=1e-3,
        h_min=1e-12,
        h_max=1e-1,
        atol=1e-10,
        rtol=1e-8,
        norm_tol=1e-6,
        tidal_limit_per_s2=9.81,
        tau_final_s=1.0,
        energy_n=32,
        energy_extent_rel=2.0,
        energy_backend="own",
        wec_tolerance=1e-10,
        nec_tolerance=1e-10,
        dec_tolerance=1e-10,
        sec_tolerance=1e-10,
    )
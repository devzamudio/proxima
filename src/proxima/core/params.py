"""Parametrizacion BM: orden canonico, limites de dominio y validacion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import yaml

from proxima.contracts.exceptions import ContractValidationError, DomainError
from proxima.contracts.types import WarpParameterVector

PARAM_ORDER: tuple[str, ...] = (
    "bubble_radius_m",
    "shell_thickness_m",
    "velocity_fraction_c",
    "shift_amplitude",
    "lapse_floor",
)

# Indices en el vector canonico.
I_R, I_W, I_V, I_A, I_LF = range(5)

def _resolve_default_config() -> Path:
    pkg_cfg = Path(__file__).resolve().parent.parent / "configs" / "phase1" / "core_defaults.yaml"
    if pkg_cfg.exists():
        return pkg_cfg
    repo_cfg = Path(__file__).resolve().parents[3] / "configs" / "phase1" / "core_defaults.yaml"
    return repo_cfg


_DEFAULT_CONFIG = _resolve_default_config()


@dataclass(frozen=True)
class CoreConfig:
    version: str
    bounds: tuple[tuple[float, float], ...]  # en PARAM_ORDER
    superluminal_velocity_max: float
    radius_regularization_rel: float

    def bounds_array(self, allow_superluminal: bool = False) -> jnp.ndarray:
        b = jnp.asarray(self.bounds, dtype=jnp.float64)
        if allow_superluminal:
            b = b.at[I_V, 1].set(self.superluminal_velocity_max)
        return b


def load_core_config(path: str | Path | None = None) -> CoreConfig:
    raw = yaml.safe_load(Path(path or _DEFAULT_CONFIG).read_text(encoding="utf-8"))
    order = tuple(raw["parameter_order"])
    if order != PARAM_ORDER:
        raise ContractValidationError(f"parameter_order inesperado: {order}")
    bounds = tuple(tuple(float(x) for x in raw["parameter_bounds"][k]) for k in PARAM_ORDER)
    return CoreConfig(
        version=str(raw["phase1"]["version"]),
        bounds=bounds,
        superluminal_velocity_max=float(raw["superluminal_velocity_max"]),
        radius_regularization_rel=float(raw["safety"]["radius_regularization_rel"]),
    )


def allows_superluminal(params: WarpParameterVector) -> bool:
    return bool(params.metadata.get("allow_superluminal", False))


def params_to_array(params: WarpParameterVector) -> jnp.ndarray:
    """Vector canonico float64 en el orden PARAM_ORDER."""
    return jnp.asarray([float(getattr(params, k)) for k in PARAM_ORDER], dtype=jnp.float64)


def domain_mask(p: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """Mascara booleana pura JAX (compatible con jit/vmap)."""
    finite = jnp.all(jnp.isfinite(p))
    inside = jnp.all((p >= bounds[:, 0]) & (p <= bounds[:, 1]))
    return finite & inside


def validate_params(params: WarpParameterVector, config: CoreConfig) -> None:
    """Validacion Python con errores tipados (fuera de jit)."""
    bounds = config.bounds_array(allows_superluminal(params))
    for i, name in enumerate(PARAM_ORDER):
        value = float(getattr(params, name))
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        if not math.isfinite(value):
            raise DomainError(f"{name} no es finito: {value}")
        if not (lo <= value <= hi):
            hint = ""
            if name == "velocity_fraction_c" and value > hi:
                hint = " (use metadata['allow_superluminal']=True para escenario superluminal)"
            raise DomainError(f"{name}={value} fuera de dominio [{lo}, {hi}]{hint}")

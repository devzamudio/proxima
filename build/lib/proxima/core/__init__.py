"""Nucleo metrico Bobrick-Martire en formulacion ADM (Fase 1), basado en JAX.

Al importar este paquete se activa float64 en JAX (convenciones Fase 0, seccion 6).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from .batch import evaluate_batch, evaluate_point  # noqa: E402
from .metric_core import BobrickMartireMetricCore  # noqa: E402
from .params import (  # noqa: E402
    PARAM_ORDER,
    CoreConfig,
    load_core_config,
    params_to_array,
    validate_params,
)

__all__ = [
    "BobrickMartireMetricCore",
    "CoreConfig",
    "PARAM_ORDER",
    "evaluate_batch",
    "evaluate_point",
    "load_core_config",
    "params_to_array",
    "validate_params",
]

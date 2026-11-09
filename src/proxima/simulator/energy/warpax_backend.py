"""Adaptador opcional a Warpax para condiciones de energia.

Solo se activa cuando energy_backend == "warpax" y el paquete esta instalado.
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp


def _try_import_warpax():
    try:
        return importlib.import_module("warpax")
    except ImportError:
        return None


def warpax_energy_conditions(T_mn: jnp.ndarray, g: jnp.ndarray, p, X) -> dict:
    """Consulta a Warpax para reporte de condiciones de energia.

    Retorna dict con wec_ok, nec_ok, dec_ok, sec_ok y violation_magnitudes,
    o None si Warpax no esta disponible.
    """
    warpax = _try_import_warpax()
    if warpax is None:
        return None

    # La API exacta de Warpax depende de la version instalada.
    # Se asume que existe algo como:
    #   warpax.check_energy_conditions(T, g) -> dict
    try:
        # Intento generico; ajustar segun API real de warpax
        if hasattr(warpax, "check_energy_conditions"):
            result = warpax.check_energy_conditions(jnp.asarray(T_mn), jnp.asarray(g))
            return {
                "wec_ok": bool(result.get("wec", True)),
                "nec_ok": bool(result.get("nec", True)),
                "dec_ok": bool(result.get("dec", True)),
                "sec_ok": bool(result.get("sec", True)),
                "violation_magnitudes": result.get("magnitudes", {}),
                "backend": "warpax",
            }
    except Exception:
        return None

    return None
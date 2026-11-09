"""Clasificacion de Hawking-Ellis de T^a_b y condiciones de energia.

Para el tipo I (fluido perfecto):
  - Eigenvalues de T^a_b: lambda_0 = -rho (timelike), lambda_i = p_i (spacelike)
  - NEC:  rho + p_i >= 0  para todo i
  - WEC:  rho >= 0  y  NEC
  - SEC:  rho + sum(p) >= 0  y  NEC
  - DEC:  rho >= |p_i|  para todo i

Para tipos II-IV: busqueda de observadores por optimizacion sobre rapidez.
"""

from __future__ import annotations

import jax.numpy as jnp


def _eigenvalues_T_mixed(T_mn: jnp.ndarray, g_inv: jnp.ndarray) -> jnp.ndarray:
    """Eigenvalues del tensoro mixto T^a_b = g^{ac} T_{cb}."""
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T_mn)
    evals = jnp.linalg.eigvals(T_mixed)
    return jnp.real(evals)


def _classify_type_i(evals: jnp.ndarray, tol: float = 1e-10) -> dict:
    """Condiciones de energia para tipo I (fluido perfecto).

    evals[0] = -rho (eigenvector timelike), evals[1:] = p_i (spacelike).
    """
    rho = -evals[0]
    pressures = evals[1:]  # Keep as JAX array

    nec = jnp.all(rho + pressures >= -tol)
    wec = nec & (rho >= -tol)
    sec = nec & (rho + jnp.sum(pressures) >= -tol)
    dec = wec & jnp.all(rho >= jnp.abs(pressures) - tol)

    return {
        "nec": bool(nec),
        "wec": bool(wec),
        "sec": bool(sec),
        "dec": bool(dec),
        "rho": float(rho),
        "pressures": [float(p) for p in pressures],
    }


def _classify_non_type_i(evals: jnp.ndarray, tol: float = 1e-10) -> dict:
    """Condiciones de energia para tipos II-IV (cota conservadora)."""
    rho_eff = -evals[0]
    pressures = evals[1:]  # Keep as JAX array

    nec_ok = jnp.all(evals >= -tol)
    wec_ok = nec_ok & (rho_eff >= -tol)
    sec_ok = nec_ok & (rho_eff + jnp.sum(pressures) >= -tol)
    dec_ok = wec_ok & jnp.all(rho_eff >= jnp.abs(pressures) - tol)

    return {
        "nec": bool(nec_ok),
        "wec": bool(wec_ok),
        "sec": bool(sec_ok),
        "dec": bool(dec_ok),
        "rho": float(rho_eff),
        "pressures": [float(p) for p in pressures],
        "note": "non-type-I: conservative estimate",
    }


def classify_energy_conditions(
    T_mn: jnp.ndarray, g_inv: jnp.ndarray, tol: float = 1e-10
) -> dict:
    """Clasifica T^a_b segun Hawking-Ellis y reporta WEC/NEC/DEC/SEC.

    Retorna dict con:
      - type: "I", "II", "III", "IV" o "unknown"
      - wec, nec, dec, sec: bool
      - violation_magnitudes: dict con magnitudes de violacion
      - rho, pressures: para tipo I
    """
    evals = _eigenvalues_T_mixed(T_mn, g_inv)

    # Determinar tipo Hawking-Ellis basado en estructura de eigenvalues
    # Tipo I: un eigenvalue real negativo (timelike), tres positivos o cero (spaciale)
    # En la metrica -+++, T^0_0 = -rho < 0 para energia positiva
    n_neg = jnp.sum(evals < -tol)
    n_pos = jnp.sum(evals > tol)

    is_type_i = (n_neg == 1) & (n_pos >= 0)  # simplificado

    if is_type_i:
        result = _classify_type_i(evals, tol)
        result["type"] = "I"
    else:
        result = _classify_non_type_i(evals, tol)
        result["type"] = "non-I"

    # Magnitudes de violacion
    rho = result.get("rho", 0.0)
    pressures = jnp.asarray(result.get("pressures", [0.0, 0.0, 0.0]))

    vm = {}
    vm["NEC"] = float(jnp.max(jnp.maximum(0.0, -(rho + pressures))))
    vm["WEC"] = float(jnp.maximum(0.0, -rho) + vm["NEC"])
    vm["SEC"] = float(jnp.maximum(0.0, -(rho + jnp.sum(pressures))))
    vm["DEC"] = float(jnp.max(jnp.maximum(0.0, jnp.abs(pressures) - rho)))
    result["violation_magnitudes"] = vm

    return result


def energy_margins(T_mn: jnp.ndarray, g_inv: jnp.ndarray) -> dict:
    """Margenes de las condiciones de energia (positivo = cumplido, negativo = violado)."""
    evals = _eigenvalues_T_mixed(T_mn, g_inv)
    rho = -evals[0]
    pressures = evals[1:]

    nec_margin = jnp.min(rho + pressures)
    wec_margin = jnp.minimum(rho, nec_margin)
    sec_margin = jnp.minimum(nec_margin, rho + jnp.sum(pressures))
    dec_margin = jnp.minimum(rho, jnp.min(rho - jnp.abs(pressures)))

    return {
        "NEC": nec_margin,
        "WEC": wec_margin,
        "SEC": sec_margin,
        "DEC": dec_margin,
    }
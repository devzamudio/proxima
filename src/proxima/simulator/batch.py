"""Simulacion por lote (jit + vmap) para uso por la PSO (Fase 4).

simulate_batch(P, X0, cfg) ejecuta N simulaciones en paralelo y retorna
observables, codigos de terminacion y margenes de energia como arrays JAX.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import SimulatorConfig
from .state import pack_state, unpack_state, normalize_4velocity
from .geodesic import geodesic_rhs
from .integrator import integrate_geodesic, TERM_SUCCESS, TERM_TIDAL_LIMIT
from .tidal import tidal_tensor, max_tidal_stress
from .energy_integral import _build_grid, total_energy
from .energy import classify_energy_conditions, energy_margins


@partial(jax.jit, static_argnames=("eps_rel", "c", "config", "G_newton"))
def _simulate_single(
    y0: jnp.ndarray,
    p: jnp.ndarray,
    eps_rel: float,
    c: float,
    config: SimulatorConfig,
    G_newton: float,
):
    """Ejecuta una simulacion individual (compatible con vmap en simulate_batch)."""
    from proxima.core.kernels import adm, geometry as geom
    from .integrator import integrate_geodesic as _ig

    X0 = y0[:4]
    u0_raw = y0[4:]
    u0_fallback = jnp.where(
        jnp.all(u0_raw == 0.0),
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=y0.dtype),
        u0_raw,
    )
    g4_0 = adm.metric4(p, X0, eps_rel)
    u0_norm = normalize_4velocity(u0_fallback, g4_0, c)
    y0_norm = pack_state(X0, u0_norm)

    result = _ig(y0_norm, p, eps_rel, c, config, f=geodesic_rhs, max_steps=1000)

    X_final = result.y_final[:4]
    u_final = result.y_final[4:]

    delta_tau = result.tau_final
    delta_t = (X_final[0] - X0[0]) / c
    term_code = result.termination_code

    # Energia en el punto final (proxy rapido para batch)
    g4 = adm.metric4(p, X_final, eps_rel)
    g_inv = jnp.linalg.inv(g4)
    Gmn, _ = geom.einstein_tensor(p, X_final, eps_rel)
    T_mn = (c**4 / (8.0 * jnp.pi * G_newton)) * Gmn
    margins = energy_margins(T_mn, g_inv)

    # Marea en el punto final
    u_norm = normalize_4velocity(u_final, g4, c)
    E_tidal = tidal_tensor(u_norm, X_final, p, eps_rel)
    tidal_max = max_tidal_stress(E_tidal)

    term_code = jnp.where(
        (term_code == TERM_SUCCESS) & (tidal_max > config.tidal_limit_per_s2),
        TERM_TIDAL_LIMIT,
        term_code,
    )

    return {
        "delta_tau": delta_tau,
        "delta_t": delta_t,
        "total_energy": jnp.asarray(0.0, dtype=y0.dtype),
        "tidal_max": tidal_max,
        "term_code": term_code,
        "nec_margin": margins["NEC"],
        "wec_margin": margins["WEC"],
        "sec_margin": margins["SEC"],
        "dec_margin": margins["DEC"],
    }


def simulate_batch(
    P: jnp.ndarray,
    X0_list: jnp.ndarray,
    cfg: SimulatorConfig,
    c: float = 299792458.0,
    G_newton: float = 6.67430e-11,
    eps_rel: float = 1.0e-9,
) -> dict:
    """Ejecuta un lote de simulaciones.

    P: (n_particles, 5) vector de parametros canonicos
    X0_list: (n_particles, 8) estados iniciales empaquetados (X+u)
    cfg: SimulatorConfig

    Retorna dict con arrays de observables y codigos.
    """
    n_particles = P.shape[0]

    # Construir estados iniciales para cada particula
    # X0_list ya contiene estados empaquetados (8,)

    # vmapped simulate (integrator usa while_loop, compatible con vmap)
    vm_sim = jax.vmap(
        lambda y0, p: _simulate_single(y0, p, eps_rel, c, cfg, G_newton),
        in_axes=(0, 0),
    )

    results = vm_sim(X0_list, P)

    return {
        "delta_tau": results["delta_tau"],
        "delta_t": results["delta_t"],
        "total_energy": results["total_energy"],
        "tidal_max": results["tidal_max"],
        "term_code": results["term_code"],
        "nec_margin": results["nec_margin"],
        "wec_margin": results["wec_margin"],
        "sec_margin": results["sec_margin"],
        "dec_margin": results["dec_margin"],
        "success": results["term_code"] == TERM_SUCCESS,
    }
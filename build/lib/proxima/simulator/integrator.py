"""Integrador Dormand-Prince 5(4) adaptativo sobre jax.lax.while_loop."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

TERM_SUCCESS = 0
TERM_MAX_STEPS = 1
TERM_STEP_UNDERFLOW = 2
TERM_NONFINITE = 3
TERM_NORM_DRIFT = 4
TERM_TIDAL_LIMIT = 5
TERM_DOMAIN = 6

TERMINATION_REASONS = {
    TERM_SUCCESS: "success",
    TERM_MAX_STEPS: "MAX_STEPS: limite computacional alcanzado",
    TERM_STEP_UNDERFLOW: "STEP_UNDERFLOW: h < h_min, error fuera de tolerancia",
    TERM_NONFINITE: "NONFINITE: divergencia numerica no recuperable",
    TERM_NORM_DRIFT: "NORM_DRIFT: |u.u + c^2|/c^2 supera tolerancia",
    TERM_TIDAL_LIMIT: "TIDAL_LIMIT: umbral de marea superado",
    TERM_DOMAIN: "DOMAIN: parametros invalidos",
}


@dataclass(frozen=True)
class IntegrationResult:
    y_final: jnp.ndarray
    lambda_final: float
    tau_final: float
    t_final: float
    termination_code: int
    steps_taken: int
    trajectory: jnp.ndarray
    valid_mask: jnp.ndarray
    h_history: jnp.ndarray
    error_history: jnp.ndarray


_DP5_C = jnp.asarray([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=jnp.float64)
_DP5_A = jnp.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ],
    dtype=jnp.float64,
)
_DP5_B = jnp.asarray(
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    dtype=jnp.float64,
)
_DP5_B_ERR = jnp.asarray(
    [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
    dtype=jnp.float64,
)


def _dp5_step(y, h, f, p, eps_rel, c):
    k = jnp.zeros((7, y.shape[0]), dtype=y.dtype)
    for i in range(7):
        y_stage = y + h * jnp.sum(_DP5_A[i, :i, None] * k[:i], axis=0)
        k = k.at[i].set(f(y_stage, p, eps_rel, c))
    y_next = y + h * jnp.sum(_DP5_B[:, None] * k, axis=0)
    y_error = h * jnp.sum(_DP5_B_ERR[:, None] * k, axis=0)
    return y_next, y_error, h


def integrate_geodesic(y0, p, eps_rel, c, config, f=None, max_steps=10000):
    if f is None:
        from .geodesic import geodesic_rhs

        f = geodesic_rhs

    h0_lam = c * config.h0
    h_min_lam = c * config.h_min
    h_max_lam = c * config.h_max
    atol = config.atol
    rtol = config.rtol
    norm_tol = config.norm_tol
    lambda_final = c * config.tau_final_s

    h_init = jnp.clip(jnp.asarray(h0_lam, dtype=y0.dtype), h_min_lam, h_max_lam)
    traj_buf = jnp.zeros((max_steps, 8), dtype=y0.dtype)
    valid_buf = jnp.zeros((max_steps,), dtype=jnp.bool_)
    h_buf = jnp.zeros((max_steps,), dtype=y0.dtype)
    err_buf = jnp.zeros((max_steps,), dtype=y0.dtype)

    def cond_fn(state):
        idx, rejections, y, lam, h, term_code, _, _, _, _ = state
        return (
            (term_code == TERM_SUCCESS)
            & (lam < lambda_final * (1.0 - 1e-12))
            & (idx < max_steps)
            & (rejections < 64)
        )

    def body_fn(state):
        idx, rejections, y, lam, h, term_code, traj, mask, h_hist, err_hist = state

        X = y[:4]
        u = y[4:]

        # Finidad
        state_finite = jnp.all(jnp.isfinite(y))

        # Norma 4-velocidad: g_{mu nu} u^mu u^nu = -1
        from proxima.core.kernels import adm

        g4 = adm.metric4(p, X, eps_rel)
        norm_sq = jnp.einsum("ij,i,j->", g4, u, u)
        norm_ok = jnp.abs(norm_sq + 1.0) < norm_tol

        hard_stop = (~state_finite) | (~norm_ok)

        new_term = jnp.where(~state_finite, TERM_NONFINITE, term_code)
        new_term = jnp.where((~norm_ok) & (new_term == TERM_SUCCESS), TERM_NORM_DRIFT, new_term)

        should_step = (new_term == TERM_SUCCESS) & (~hard_stop)

        remaining = jnp.maximum(0.0, lambda_final - lam)
        h_try = jnp.minimum(h, remaining)

        y_next, y_err, h_used = _dp5_step(y, h_try, f, p, eps_rel, c)

        next_finite = jnp.all(jnp.isfinite(y_next)) & jnp.all(jnp.isfinite(y_err))
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_next))
        err_norm = jnp.where(
            next_finite,
            jnp.sqrt(jnp.mean((y_err / scale) ** 2)),
            jnp.inf,
        )

        accept = (err_norm <= 1.0) & should_step & next_finite

        grow_factor = jnp.minimum(
            5.0, jnp.maximum(0.2, 0.9 * (1.0 / jnp.maximum(err_norm, 1e-16)) ** 0.2)
        )
        shrink_factor = jnp.where(
            next_finite,
            jnp.maximum(0.1, 0.9 * (1.0 / jnp.maximum(err_norm, 1.0)) ** 0.25),
            0.2,
        )
        h_candidate_rej = h_try * shrink_factor

        h_underflow = (
            ((h_candidate_rej < h_min_lam) | (rejections + 1 >= 64))
            & (~accept)
            & should_step
        )
        new_term = jnp.where(
            h_underflow & (~next_finite),
            TERM_NONFINITE,
            jnp.where(h_underflow, TERM_STEP_UNDERFLOW, new_term),
        )

        h_new = jnp.where(
            accept,
            jnp.clip(h * grow_factor, h_min_lam, h_max_lam),
            jnp.clip(h_candidate_rej, h_min_lam, h_max_lam),
        )

        y_adv = jnp.where(accept, y_next, y)
        lam_adv = lam + jnp.where(accept, h_used, 0.0)
        h_adv = h_new
        idx_adv = idx + jnp.where(accept, 1, 0)
        rej_adv = jnp.where(accept, 0, rejections + 1)

        write_idx = jnp.clip(idx, 0, max_steps - 1)

        traj = traj.at[write_idx].set(jnp.where(accept, y_adv, traj[write_idx]))
        mask = mask.at[write_idx].set(jnp.where(accept, True, mask[write_idx]))
        h_hist = h_hist.at[write_idx].set(jnp.where(accept, h_used / c, h_hist[write_idx]))
        err_hist = err_hist.at[write_idx].set(jnp.where(accept, err_norm, err_hist[write_idx]))

        return (idx_adv, rej_adv, y_adv, lam_adv, h_adv, new_term, traj, mask, h_hist, err_hist)

    init_state = (0, 0, y0, jnp.asarray(0.0, dtype=y0.dtype), h_init, TERM_SUCCESS, traj_buf, valid_buf, h_buf, err_buf)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    idx_f, rej_f, y_f, lam_f, h_f, term_f, traj_f, mask_f, h_hist_f, err_hist_f = final_state

    # Verificar también la norma del estado final si terminó con éxito
    from proxima.core.kernels import adm as _adm
    g4_f = _adm.metric4(p, y_f[:4], eps_rel)
    norm_sq_f = jnp.einsum("ij,i,j->", g4_f, y_f[4:], y_f[4:])
    norm_ok_f = jnp.abs(norm_sq_f + 1.0) < norm_tol
    finite_f = jnp.all(jnp.isfinite(y_f))

    term_f = jnp.where((term_f == TERM_SUCCESS) & (~finite_f), TERM_NONFINITE, term_f)
    term_f = jnp.where((term_f == TERM_SUCCESS) & (~norm_ok_f), TERM_NORM_DRIFT, term_f)

    X_f = y_f[:4]
    tau_f = lam_f / c
    t_f = X_f[0] / c

    reached_end = lam_f >= lambda_final * (1.0 - 1e-12)
    final_term = jnp.where(
        (term_f == TERM_SUCCESS) & (~reached_end),
        TERM_MAX_STEPS,
        term_f,
    )

    is_traced = isinstance(tau_f, jax.core.Tracer)
    return IntegrationResult(
        y_final=y_f,
        lambda_final=lam_f if is_traced else float(lam_f),
        tau_final=tau_f if is_traced else float(tau_f),
        t_final=t_f if is_traced else float(t_f),
        termination_code=final_term if is_traced else int(final_term),
        steps_taken=idx_f if is_traced else int(idx_f),
        trajectory=traj_f,
        valid_mask=mask_f,
        h_history=h_hist_f,
        error_history=err_hist_f,
    )
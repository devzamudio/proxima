"""Motor tensorial por diferenciacion automatica.

Todas las magnitudes en coordenadas X = (ct, x, y, z) [m]:
  Gamma^mu_{nu rho}  [1/m],  Ricci, G_mu_nu  [1/m^2],  K_ij  [1/m].
T_mu_nu = c^4/(8 pi G) G_mu_nu  [J/m^3]  (componentes en base coordenada adimensional).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import adm


def christoffel(p, X, eps_rel):
    g = adm.metric4(p, X, eps_rel)
    g_inv = jnp.linalg.inv(g)
    dg = jax.jacfwd(adm.metric4, argnums=1)(p, X, eps_rel)  # dg[a,b,c] = d_c g_ab
    # Gamma^m_{ab} = 1/2 g^{mk} (d_a g_kb + d_b g_ka - d_k g_ab)
    # term[k,a,b] = dg[k,b,a] + dg[k,a,b] - dg[a,b,k]
    term = jnp.einsum("kba->kab", dg) + dg - jnp.einsum("abk->kab", dg)
    return 0.5 * jnp.einsum("mk,kab->mab", g_inv, term)


def ricci_tensor(p, X, eps_rel):
    G = christoffel(p, X, eps_rel)
    dG = jax.jacfwd(christoffel, argnums=1)(p, X, eps_rel)  # dG[r,m,n,s] = d_s Gamma^r_{mn}
    # R_mn = d_r G^r_mn - d_n G^r_mr + G^r_rl G^l_mn - G^r_nl G^l_mr
    return (
        jnp.einsum("rmnr->mn", dG)
        - jnp.einsum("rmrn->mn", dG)
        + jnp.einsum("rrl,lmn->mn", G, G)
        - jnp.einsum("rnl,lmr->mn", G, G)
    )


def einstein_tensor(p, X, eps_rel):
    g = adm.metric4(p, X, eps_rel)
    ric = ricci_tensor(p, X, eps_rel)
    R = jnp.einsum("mn,mn->", jnp.linalg.inv(g), ric)
    return ric - 0.5 * g * R, R


def stress_energy(p, X, eps_rel, c, G_newton):
    Gmn, _ = einstein_tensor(p, X, eps_rel)
    return (c**4 / (8.0 * jnp.pi * G_newton)) * Gmn


def eulerian_normal(p, X, eps_rel):
    """n^mu = (1/alpha, -beta^i/alpha)."""
    a = adm.lapse(p, X, eps_rel)
    b = adm.shift(p, X, eps_rel)
    return jnp.concatenate([jnp.ones(1) / a, -b / a])


def riemann(p, X, eps_rel):
    """R^rho_{sigma mu nu} por diferenciacion automatica de Christoffel.

    dG[a,b,c,d] = d_d Gamma^a_{bc}  (shape 4,4,4,4)
    R[rho,sigma,mu,nu] = d_d Gamma^rho_{sigma nu}|_{d=mu}
                         - d_d Gamma^rho_{sigma mu}|_{d=nu}
                         + Gamma^rho_{mu lam} Gamma^lam_{sigma nu}
                         - Gamma^rho_{nu lam} Gamma^lam_{sigma mu}
    """
    G = christoffel(p, X, eps_rel)                      # (4,4,4) G[a,b,c] = Gamma^a_{bc}
    dG = jax.jacfwd(christoffel, argnums=1)(p, X, eps_rel)  # dG[a,b,c,d] = d_d Gamma^a_{bc}

    # term1[rho,sigma,mu,nu] = d_mu Gamma^rho_{sigma nu} = dG[rho,sigma,nu,mu]
    term1 = jnp.transpose(dG, (0, 1, 3, 2))
    # term2[rho,sigma,mu,nu] = d_nu Gamma^rho_{sigma mu} = dG[rho,sigma,mu,nu]
    term2 = dG
    # term3[rho,sigma,mu,nu] = sum_lam Gamma^rho_{mu lam} Gamma^lam_{sigma nu}
    term3 = jnp.einsum("rml,lsv->rsmv", G, G)
    # term4[rho,sigma,mu,nu] = sum_lam Gamma^rho_{nu lam} Gamma^lam_{sigma mu}
    term4 = jnp.einsum("rnl,lsm->rsnm", G, G)
    term4 = jnp.swapaxes(term4, 2, 3)

    return term1 - term2 + term3 - term4


def extrinsic_curvature(p, X, eps_rel):
    """K_ij = 1/(2 alpha) (D_i beta_j + D_j beta_i - d_0 gamma_ij)."""
    a = adm.lapse(p, X, eps_rel)
    gam = adm.spatial_metric(p, X, eps_rel)
    gam_inv = jnp.linalg.inv(gam)

    def beta_dn(Y):
        return adm.spatial_metric(p, Y, eps_rel) @ adm.shift(p, Y, eps_rel)

    db = jax.jacfwd(beta_dn)(X)  # db[j, mu]
    dgam = jax.jacfwd(adm.spatial_metric, argnums=1)(p, X, eps_rel)  # dgam[i,j,mu]
    d_sp = dgam[:, :, 1:]
    # Christoffel espacial Gamma^k_ij
    # term[k,i,j] = d_i g_kj + d_j g_ki - d_k g_ij
    term = jnp.einsum("kji->kij", d_sp) + d_sp - jnp.einsum("ijk->kij", d_sp)
    gam3 = 0.5 * jnp.einsum("lk,kij->lij", gam_inv, term)
    b_dn = beta_dn(X)
    Db = db[:, 1:].T - jnp.einsum("kij,k->ij", gam3, b_dn)  # Db[i,j] = D_i beta_j
    K = (Db + Db.T - dgam[:, :, 0]) / (2.0 * a)
    trK = jnp.einsum("ij,ij->", gam_inv, K)
    K_up = gam_inv @ K @ gam_inv
    KK = jnp.einsum("ij,ij->", K, K_up)
    return K, trK, KK

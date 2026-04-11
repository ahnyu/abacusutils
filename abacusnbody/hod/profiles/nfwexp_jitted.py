from numba import njit
import numpy as np
import math

@njit(fastmath=True)
def _A_of_c(c):
    return math.log1p(c) - c / (1.0 + c)

@njit(fastmath=True)
def nfwexp_reset_halo(r200, c):
    c_eff   = c    if c    > 1e-10 else 1e-10
    r200eff = r200 if r200 > 1e-12 else 1e-12
    rs = r200eff / c_eff
    return rs, c_eff

@njit(fastmath=True)
def _nfw_sample_radius(u, rs, c, tol_radius):
    if u <= 0.0: return 0.0
    if u >= 1.0: return c * rs
    amp_c = _A_of_c(c)
    if amp_c <= 1e-18: amp_c = 1e-18
    target = u * amp_c
    low = 0.0; high = c
    tol_x = tol_radius / rs
    if tol_x <= 1e-14: tol_x = 1e-14
    span = high - low
    niter = int(math.ceil(math.log2(span / tol_x))) if span > tol_x else 1
    if niter > 64: niter = 64
    for _ in range(niter):
        mid = 0.5*(low+high)
        A = math.log1p(mid) - mid/(1.0+mid)
        if A < target: low = mid
        else:          high = mid
        if (high-low) <= tol_x: break
    return 0.5*(low+high)*rs

@njit(fastmath=True)
def _sample_exp(u, lam):
    # r ~ Exp(scale=lam) on [0, ∞)
    if lam <= 0.0:
        return 0.0
    # clamp u to avoid log(0)
    if u <= 0.0:
        u = 1e-16
    elif u >= 1.0:
        u = 1.0 - 1e-16
    return -lam * math.log1p(-u)

@njit(fastmath=True)
def nfwexp_sample_radius(u_switch, u_r,
                         rs_base, c_base, tol_radius,
                         exp_frac, exp_scale, nfw_rescale):
    # clamp inputs
    if exp_frac < 0.0: exp_frac = 0.0
    if exp_frac > 1.0: exp_frac = 1.0
    tau   = exp_scale if exp_scale > 0.0 else 0.0
    scale = nfw_rescale if nfw_rescale > 0.0 else 1.0

    if u_switch < exp_frac:
        # untruncated exponential: r ~ Exp(λ) with λ = τ * rs_base
        lam = tau * rs_base
        return _sample_exp(u_r, lam)
    else:
        return _nfw_sample_radius(u_r, rs_base, c_base, tol_radius) * scale

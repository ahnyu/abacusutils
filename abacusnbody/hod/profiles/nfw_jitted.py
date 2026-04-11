from numba import njit
import math

@njit(fastmath=True)
def nfw_reset_halo(r200, c):
    c_eff = c if c > 1e-10 else 1e-10
    r200e = r200 if r200 > 1e-12 else 1e-12
    rs = r200e / c_eff
    amp_c = math.log1p(c_eff) - c_eff / (1.0 + c_eff)
    if amp_c <= 1e-18:
        amp_c = 1e-18
    return rs, c_eff, amp_c 

@njit(fastmath=True)
def nfw_sample_radius(u, rs, c, amp_c, tol_radius):
    if u <= 0.0: return 0.0
    if u >= 1.0: return c * rs
    target = u * amp_c
    low = 0.0; high = c
    tol_x = tol_radius / rs
    if tol_x <= 1e-14: tol_x = 1e-14
    span = high - low
    niter = int(math.ceil(math.log2(span / tol_x))) if span > tol_x else 1
    if niter > 64: niter = 64
    for _ in range(niter):
        mid = 0.5*(low + high)
        A = math.log1p(mid) - mid / (1.0 + mid)
        if A < target: low = mid
        else:          high = mid
        if (high - low) <= tol_x:
            break
    return 0.5*(low + high) * rs
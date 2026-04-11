from numba import njit
import numpy as np, math

EMPTY_F64 = np.empty(0, dtype=np.float64)

def load_inv_cdf_or_empty(want_dv, tracers, tracer_key, npy_path):
    if want_dv and (tracer_key in tracers):
        if npy_path is None:
            raise RuntimeError(f"want_dv is True and {tracer_key} is in tracers, but dv_draw_{tracer_key} is None")
        print(f"Loading dv error inv CDF from {npy_path} for {tracer_key}")
        d = np.load(npy_path)
        u_grid, x_grid = build_inv_cdf_table(d["vbin"], d["cdf"])
        return u_grid, x_grid
    else:
        return EMPTY_F64, EMPTY_F64

def build_inv_cdf_table(vbin, cdf):
    import numpy as np
    cdf_u, ind = np.unique(cdf, return_index=True)
    u_grid = (cdf_u / cdf_u[-1]).astype(np.float64)
    x_grid = vbin[ind].astype(np.float64)  # exponent: log10|Δv|
    return np.ascontiguousarray(u_grid), np.ascontiguousarray(x_grid)

@njit(fastmath=True)
def inv_cdf_eval_linear(u, u_grid, x_grid):
    if u <= u_grid[0]:  return x_grid[0]
    if u >= u_grid[-1]: return x_grid[-1]
    i = np.searchsorted(u_grid, u)
    u0=u_grid[i-1]; u1=u_grid[i]; x0=x_grid[i-1]; x1=x_grid[i]
    t = (u - u0) / (u1 - u0)
    return x0 + t * (x1 - x0)   # exponent = log10|Δv|
    
@njit(fastmath=True)
def redshift_error_draw(u_mag, u_sign, u_grid, x_grid):
    expo = inv_cdf_eval_linear(u_mag, u_grid, x_grid)  # log10|Δv|
    dv   = 10.0 ** expo
    return dv if u_sign < 0.5 else -dv
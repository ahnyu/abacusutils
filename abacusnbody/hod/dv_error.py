from __future__ import annotations

from numba import njit
import numpy as np

EMPTY_F64 = np.empty(0, dtype=np.float64)


def load_inv_cdf_or_empty(want_dv, tracers, tracer_key, npz_path):
    if not (want_dv and (tracer_key in tracers)):
        return EMPTY_F64, EMPTY_F64

    if npz_path is None:
        raise RuntimeError(
            f"want_dv is True and {tracer_key} is in tracers, "
            f"but dv_draw_{tracer_key} is None"
        )

    print(f"Loading dv error inv CDF from {npz_path} for {tracer_key}")

    with np.load(npz_path, allow_pickle=False) as d:
        if "grid" in d.files:
            grid = d["grid"]
        elif "vbin" in d.files:   # optional legacy support
            grid = d["vbin"]
        else:
            raise KeyError(f"Missing `grid` (or legacy `vbin`) in {npz_path}")

        if "cdf" not in d.files:
            raise KeyError(f"Missing `cdf` in {npz_path}")

        cdf = d["cdf"]
        pdf = d["pdf"] if "pdf" in d.files else None

    return build_inv_cdf_table(grid, cdf, pdf=pdf)


def build_inv_cdf_table(grid, cdf, pdf=None):
    """
    Build inverse-CDF lookup arrays (u_grid, x_grid) from raw grid/cdf/pdf.

    Here x_grid is the native sampling grid itself, not assumed to be log-spaced
    or linearly spaced.
    """
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    cdf = np.asarray(cdf, dtype=np.float64).reshape(-1)

    if grid.size != cdf.size:
        raise ValueError("`grid` and `cdf` must have the same length")
    if grid.size < 2:
        raise ValueError("`grid`/`cdf` must contain at least 2 points")

    order = np.argsort(grid)
    grid = grid[order]
    cdf = cdf[order]

    if pdf is not None:
        pdf = np.asarray(pdf, dtype=np.float64).reshape(-1)
        if pdf.size == grid.size:
            pdf = pdf[order]
            cdf = np.zeros_like(grid, dtype=np.float64)
            dx = np.diff(grid)
            cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)

    cdf = np.maximum.accumulate(cdf)
    cdf = np.clip(cdf, 0.0, None)

    cdf0 = float(cdf[0])
    cdf1 = float(cdf[-1])
    if not np.isfinite(cdf0) or not np.isfinite(cdf1) or cdf1 <= cdf0:
        raise ValueError("Invalid CDF values: cannot normalize for sampling")

    cdf = (cdf - cdf0) / (cdf1 - cdf0)

    keep = np.r_[True, np.diff(cdf) > 0.0]
    u_grid = np.ascontiguousarray(cdf[keep], dtype=np.float64)
    x_grid = np.ascontiguousarray(grid[keep], dtype=np.float64)

    if u_grid.size < 2:
        raise ValueError("CDF has insufficient dynamic range for sampling")

    return u_grid, x_grid


@njit(fastmath=True, cache=True)
def inv_cdf_eval_linear(u, u_grid, x_grid):
    if u <= u_grid[0]:
        return x_grid[0]
    if u >= u_grid[-1]:
        return x_grid[-1]

    i = np.searchsorted(u_grid, u)
    u0 = u_grid[i - 1]
    u1 = u_grid[i]
    x0 = x_grid[i - 1]
    x1 = x_grid[i]

    t = (u - u0) / (u1 - u0)
    return x0 + t * (x1 - x0)


@njit(fastmath=True, cache=True)
def redshift_error_draw(u_mag, u_sign, u_grid, x_grid):
    """
    Draw one signed redshift error.

    x_grid is the native |Δv| grid itself.
    """
    dv_abs = inv_cdf_eval_linear(u_mag, u_grid, x_grid)
    return dv_abs if u_sign < 0.5 else -dv_abs


@njit(fastmath=True, cache=True)
def redshift_error_draw_vec(u_mag, u_sign, u_grid, x_grid):
    n = u_mag.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        dv_abs = inv_cdf_eval_linear(u_mag[i], u_grid, x_grid)
        out[i] = dv_abs if u_sign[i] < 0.5 else -dv_abs
    return out
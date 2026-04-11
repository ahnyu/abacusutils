import math
import os
import time
import warnings

import numba
import numba as nb
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from numba import njit, types
from numba.typed import Dict
from .profiles.pop_sats_profiles import gen_sats_profiles
from .dv_error import redshift_error_draw

# import yaml
# config = yaml.safe_load(open('config/abacus_hod.yaml'))
# numba.set_num_threads(16)
float_array = types.float64[:]
int_array = types.int64[:]
G = 4.302e-6  # in kpc/Msol (km.s)^2


@njit(fastmath=True)
def n_sat_LRG_modified(M_h, logM_cut, M_cut, M_1, sigma, alpha, kappa):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    if M_h - kappa * M_cut < 0:
        return 0
    return (
        ((M_h - kappa * M_cut) / M_1) ** alpha
        * 0.5
        * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))
    )


@njit(fastmath=True)
def n_cen_LRG(M_h, logM_cut, sigma):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    return 0.5 * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))


@njit(fastmath=True)
def N_sat_generic(M_h, M_cut, kappa, M_1, alpha, A_s=1.0):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    if M_h - kappa * M_cut < 0:
        return 0
    return A_s * ((M_h - kappa * M_cut) / M_1) ** alpha


@njit(fastmath=True)
def N_sat_elg(M_h, M_cut, kappa, M_1, alpha, A_s=1.0, alpha1=0.0, beta=0.0):
    """
    Standard power law modulated by an exponential fall off at small M
    """
    # return (M_h/M_1)**alpha/(1+np.exp(-A_s*(np.log10(M_h)-np.log10(kappa*M_cut)))) + beta*(M_h/M_1)**(-alpha1)/100
    if M_h - kappa * M_cut < 0:
        return 0
    return (
        A_s * ((M_h - kappa * M_cut) / M_1) ** alpha
    )  # + beta*(M_h/M_1)**(-alpha1)/100


@njit(fastmath=True)
def N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma, Anorm=1):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log10(M_h)
    phi = phi_fun(logM_h, logM_cut, sigma)
    Phi = Phi_fun(logM_h, logM_cut, sigma, gamma)
    return (
        2.0 * (p_max - 1.0 / Q) * phi * Phi / Anorm
    )  # + 0.5/Q*(1 + math.erf((logM_h-logM_cut-0.8)*3))


@njit(fastmath=True)
def N_cen_ELG_v2(M_h, p_max, logM_cut, sigma, gamma):
    """
    HOD function for ELG centrals taken from arXiv:2007.09012.
    """
    logM_h = np.log10(M_h)
    if logM_h <= logM_cut:
        return p_max * Gaussian_fun(logM_h, logM_cut, sigma)
    else:
        return p_max * (M_h / 10**logM_cut) ** gamma / (2.5066283 * sigma)


@njit(fastmath=True)
def N_cen_QSO(M_h, logM_cut, sigma):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    return 0.5 * (1 + math.erf((np.log10(M_h) - logM_cut) / 1.41421356 / sigma))


@njit(fastmath=True)
def phi_fun(logM_h, logM_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(logM_h, logM_cut, sigma)
    return phi


@njit(fastmath=True)
def Phi_fun(logM_h, logM_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma * (logM_h - logM_cut) / sigma
    Phi = 0.5 * (1 + math.erf(x / np.sqrt(2)))
    return Phi


@njit(fastmath=True)
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327 / sigma * np.exp(-((x - mean) ** 2) / 2 / sigma**2)


#@njit(fastmath=True)
#def wrap(x, L):
#    """Fast scalar mod implementation"""
#    L2 = L / 2
#    if x >= L2:
#        return x - L
#    elif x < -L2:
#        return x + L
#    return x

@njit(fastmath=True)
def wrap(x, L):
    """Wrap x into [-L/2, L/2). """
    half = 0.5 * L
    if (-half <= x) and (x < half):
        return x
    k = math.floor((x + half) / L)
    return x - L * k


@njit(parallel=True, fastmath=True)
def gen_cent(
    pos,
    vel,
    mass,
    ids,
    multis,
    randoms,
    vdev,
    deltac,
    fenv,
    shear,
    u_cent_mag,
    u_cent_sign,
    LRG_hod_dict,
    ELG_hod_dict,
    QSO_hod_dict,
    rsd,
    want_dv,
    u_grid_L, 
    x_grid_L,
    u_grid_E, 
    x_grid_E,
    u_grid_Q, 
    x_grid_Q,
    inv_velz2kms,
    lbox,
    want_LRG,
    want_ELG,
    want_QSO,
    Nthread,
    origin,
):
    """
    Generate central galaxies in place in memory with a two pass numba parallel implementation.
    """

    if want_LRG:
        # parse out the hod parameters
        logM_cut_L, sigma_L = LRG_hod_dict['logM_cut'], LRG_hod_dict['sigma']
        ic_L, alpha_c_L, Ac_L, Bc_L = (
            LRG_hod_dict['ic'],
            LRG_hod_dict['alpha_c'],
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Bcent'],
        )

    if want_ELG:
        pmax_E, Q_E, logM_cut_E, sigma_E, gamma_E = (
            ELG_hod_dict['p_max'],
            ELG_hod_dict['Q'],
            ELG_hod_dict['logM_cut'],
            ELG_hod_dict['sigma'],
            ELG_hod_dict['gamma'],
        )
        alpha_c_E, Ac_E, Bc_E, Cc_E, ic_E = (
            ELG_hod_dict['alpha_c'],
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Ccent'],
            ELG_hod_dict['ic'],
        )

    if want_QSO:
        logM_cut_Q, sigma_Q = QSO_hod_dict['logM_cut'], QSO_hod_dict['sigma']
        alpha_c_Q, Ac_Q, Bc_Q, ic_Q = (
            QSO_hod_dict['alpha_c'],
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['ic'],
        )

    H = len(mass)

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # first create the markers between 0 and 1 for different tracers
            LRG_marker = 0
            if want_LRG:
                # do assembly bias and secondary bias
                logM_cut_L_temp = logM_cut_L + Ac_L * deltac[i] + Bc_L * fenv[i]
                LRG_marker += (
                    n_cen_LRG(mass[i], logM_cut_L_temp, sigma_L) * ic_L * multis[i]
                )
            ELG_marker = LRG_marker
            if want_ELG:
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * deltac[i] + Bc_E * fenv[i] + Cc_E * shear[i]
                )
                ELG_marker += (
                    N_cen_ELG_v1(
                        mass[i], pmax_E, Q_E, logM_cut_E_temp, sigma_E, gamma_E
                    )
                    * ic_E
                    * multis[i]
                )
            QSO_marker = ELG_marker
            if want_QSO:
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * deltac[i] + Bc_Q * fenv[i]
                QSO_marker += (
                    N_cen_QSO(mass[i], logM_cut_Q_temp, sigma_Q) * ic_Q * multis[i]
                )

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1  # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1  # counting
                keep[i] = 3
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype=mass.dtype)
    lrg_y = np.empty(N_lrg, dtype=mass.dtype)
    lrg_z = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vx = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vy = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vz = np.empty(N_lrg, dtype=mass.dtype)
    lrg_mass = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vsmear = np.empty(N_lrg, dtype=mass.dtype)
    lrg_id = np.empty(N_lrg, dtype=ids.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype=mass.dtype)
    elg_y = np.empty(N_elg, dtype=mass.dtype)
    elg_z = np.empty(N_elg, dtype=mass.dtype)
    elg_vx = np.empty(N_elg, dtype=mass.dtype)
    elg_vy = np.empty(N_elg, dtype=mass.dtype)
    elg_vz = np.empty(N_elg, dtype=mass.dtype)
    elg_mass = np.empty(N_elg, dtype=mass.dtype)
    elg_vsmear = np.empty(N_elg, dtype=mass.dtype)
    elg_id = np.empty(N_elg, dtype=ids.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype=mass.dtype)
    qso_y = np.empty(N_qso, dtype=mass.dtype)
    qso_z = np.empty(N_qso, dtype=mass.dtype)
    qso_vx = np.empty(N_qso, dtype=mass.dtype)
    qso_vy = np.empty(N_qso, dtype=mass.dtype)
    qso_vz = np.empty(N_qso, dtype=mass.dtype)
    qso_mass = np.empty(N_qso, dtype=mass.dtype)
    qso_vsmear = np.empty(N_qso, dtype=mass.dtype)
    qso_id = np.empty(N_qso, dtype=ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                # loop thru three directions to assign galaxy velocities and positions
                lrg_x[j1] = pos[i, 0]
                lrg_vx[j1] = vel[i, 0] + alpha_c_L * vdev[i, 0]  # velocity bias
                lrg_y[j1] = pos[i, 1]
                lrg_vy[j1] = vel[i, 1] + alpha_c_L * vdev[i, 1]  # velocity bias
                lrg_z[j1] = pos[i, 2]
                lrg_vz[j1] = vel[i, 2] + alpha_c_L * vdev[i, 2]  # velocity bias
                # rsd only applies to the z direction, dv treatment
                if want_dv and u_grid_L.size > 0:
                    tmp_dv_los_L = redshift_error_draw(u_cent_mag[i], u_cent_sign[i], u_grid_L, x_grid_L)
                else:
                    tmp_dv_los_L = 0.0
                lrg_vsmear[j1] = tmp_dv_los_L
                if rsd and origin is not None:
                    nx = lrg_x[j1] - origin[0]
                    ny = lrg_y[j1] - origin[1]
                    nz = lrg_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (lrg_vx[j1] * nx + lrg_vy[j1] * ny + lrg_vz[j1] * nz) + tmp_dv_los_L
                    )
                    lrg_x[j1] = lrg_x[j1] + proj * nx
                    lrg_y[j1] = lrg_y[j1] + proj * ny
                    lrg_z[j1] = lrg_z[j1] + proj * nz
                elif rsd:
                    lrg_z[j1] = wrap(pos[i, 2] + (lrg_vz[j1] + tmp_dv_los_L) * inv_velz2kms, lbox)
                lrg_mass[j1] = mass[i]
                lrg_id[j1] = ids[i]
                j1 += 1
            elif keep[i] == 2:
                # loop thru three directions to assign galaxy velocities and positions
                elg_x[j2] = pos[i, 0]
                elg_vx[j2] = vel[i, 0] + alpha_c_E * vdev[i, 0]  # velocity bias
                elg_y[j2] = pos[i, 1]
                elg_vy[j2] = vel[i, 1] + alpha_c_E * vdev[i, 1]  # velocity bias
                elg_z[j2] = pos[i, 2]
                elg_vz[j2] = vel[i, 2] + alpha_c_E * vdev[i, 2]  # velocity bias
                if want_dv and u_grid_E.size > 0:
                    tmp_dv_los_E = redshift_error_draw(u_cent_mag[i], u_cent_sign[i], u_grid_E, x_grid_E)
                else:
                    tmp_dv_los_E = 0.0
                elg_vsmear[j2] = tmp_dv_los_E
                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = elg_x[j2] - origin[0]
                    ny = elg_y[j2] - origin[1]
                    nz = elg_z[j2] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (elg_vx[j2] * nx + elg_vy[j2] * ny + elg_vz[j2] * nz)+ tmp_dv_los_E
                    )
                    elg_x[j2] = elg_x[j2] + proj * nx
                    elg_y[j2] = elg_y[j2] + proj * ny
                    elg_z[j2] = elg_z[j2] + proj * nz
                elif rsd:
                    elg_z[j2] = wrap(pos[i, 2] + (elg_vz[j2] + tmp_dv_los_E) * inv_velz2kms, lbox)
                elg_mass[j2] = mass[i]
                elg_id[j2] = ids[i]
                j2 += 1
            elif keep[i] == 3:
                # loop thru three directions to assign galaxy velocities and positions
                qso_x[j3] = pos[i, 0]
                qso_vx[j3] = vel[i, 0] + alpha_c_Q * vdev[i, 0]  # velocity bias
                qso_y[j3] = pos[i, 1]
                qso_vy[j3] = vel[i, 1] + alpha_c_Q * vdev[i, 1]  # velocity bias
                qso_z[j3] = pos[i, 2]
                qso_vz[j3] = vel[i, 2] + alpha_c_Q * vdev[i, 2]  # velocity bias
                if want_dv and u_grid_Q.size > 0:
                    tmp_dv_los_Q = redshift_error_draw(u_cent_mag[i], u_cent_sign[i], u_grid_Q, x_grid_Q)
                else:
                    tmp_dv_los_Q = 0.0
                qso_vsmear[j3] = tmp_dv_los_Q
                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = qso_x[j3] - origin[0]
                    ny = qso_y[j3] - origin[1]
                    nz = qso_z[j3] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (qso_vx[j3] * nx + qso_vy[j3] * ny + qso_vz[j3] * nz) + tmp_dv_los_Q
                    )
                    qso_x[j3] = qso_x[j3] + proj * nx
                    qso_y[j3] = qso_y[j3] + proj * ny
                    qso_z[j3] = qso_z[j3] + proj * nz
                elif rsd:
                    qso_z[j3] = wrap(pos[i, 2] + (qso_vz[j3]+tmp_dv_los_Q) * inv_velz2kms, lbox)
                qso_mass[j3] = mass[i]
                qso_id[j3] = ids[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ELG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    QSO_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)
    LRG_dict['x'] = lrg_x
    LRG_dict['y'] = lrg_y
    LRG_dict['z'] = lrg_z
    LRG_dict['vx'] = lrg_vx
    LRG_dict['vy'] = lrg_vy
    LRG_dict['vz'] = lrg_vz
    LRG_dict['mass'] = lrg_mass
    LRG_dict['vsmear'] = lrg_vsmear
    ID_dict['LRG'] = lrg_id

    ELG_dict['x'] = elg_x
    ELG_dict['y'] = elg_y
    ELG_dict['z'] = elg_z
    ELG_dict['vx'] = elg_vx
    ELG_dict['vy'] = elg_vy
    ELG_dict['vz'] = elg_vz
    ELG_dict['mass'] = elg_mass
    ELG_dict['vsmear'] = elg_vsmear
    ID_dict['ELG'] = elg_id

    QSO_dict['x'] = qso_x
    QSO_dict['y'] = qso_y
    QSO_dict['z'] = qso_z
    QSO_dict['vx'] = qso_vx
    QSO_dict['vy'] = qso_vy
    QSO_dict['vz'] = qso_vz
    QSO_dict['mass'] = qso_mass
    QSO_dict['vsmear'] = qso_vsmear
    ID_dict['QSO'] = qso_id
    return LRG_dict, ELG_dict, QSO_dict, ID_dict, keep

@njit(parallel=True, fastmath=True)
def gen_sats_particles(
    ppos,
    pvel,
    hvel,
    hmass,
    hid,
    weights,
    randoms,
    hdeltac,
    hfenv,
    hshear,
    enable_ranks,
    ranks,
    ranksv,
    ranksp,
    ranksr,
    ranksc,
    u_sat_mag,
    u_sat_sign,
    LRG_hod_dict,
    ELG_hod_dict,
    QSO_hod_dict,
    rsd,
    want_dv,
    u_grid_L, 
    x_grid_L,
    u_grid_E, 
    x_grid_E,
    u_grid_Q, 
    x_grid_Q,
    inv_velz2kms,
    lbox,
    Mpart,
    want_LRG,
    want_ELG,
    want_QSO,
    Nthread,
    origin,
    keep_cent,
):
    """
    Generate satellite galaxies in place in memory with a two pass numba parallel implementation.
    """

    if want_LRG:
        logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = (
            LRG_hod_dict['logM_cut'],
            LRG_hod_dict['logM1'],
            LRG_hod_dict['sigma'],
            LRG_hod_dict['alpha'],
            LRG_hod_dict['kappa'],
        )
        alpha_s_L, s_L, s_v_L, s_p_L, s_r_L, Ac_L, As_L, Bc_L, Bs_L, ic_L = (
            LRG_hod_dict['alpha_s'],
            LRG_hod_dict['s'],
            LRG_hod_dict['s_v'],
            LRG_hod_dict['s_p'],
            LRG_hod_dict['s_r'],
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Asat'],
            LRG_hod_dict['Bcent'],
            LRG_hod_dict['Bsat'],
            LRG_hod_dict['ic'],
        )

    if want_ELG:
        logM_cut_E, kappa_E, logM1_E, alpha_E, A_E = (
            ELG_hod_dict['logM_cut'],
            ELG_hod_dict['kappa'],
            ELG_hod_dict['logM1'],
            ELG_hod_dict['alpha'],
            ELG_hod_dict['A_s'],
        )
        (
            alpha_s_E,
            s_E,
            s_v_E,
            s_p_E,
            s_r_E,
            Ac_E,
            As_E,
            Bc_E,
            Bs_E,
            Cc_E,
            Cs_E,
            ic_E,
            logM1_EE,
            alpha_EE,
            logM1_EL,
            alpha_EL,
        ) = (
            ELG_hod_dict['alpha_s'],
            ELG_hod_dict['s'],
            ELG_hod_dict['s_v'],
            ELG_hod_dict['s_p'],
            ELG_hod_dict['s_r'],
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Asat'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Bsat'],
            ELG_hod_dict['Ccent'],
            ELG_hod_dict['Csat'],
            ELG_hod_dict['ic'],
            ELG_hod_dict['logM1_EE'],
            ELG_hod_dict['alpha_EE'],
            ELG_hod_dict['logM1_EL'],
            ELG_hod_dict['alpha_EL'],
        )

    if want_QSO:
        logM_cut_Q, kappa_Q, logM1_Q, alpha_Q = (
            QSO_hod_dict['logM_cut'],
            QSO_hod_dict['kappa'],
            QSO_hod_dict['logM1'],
            QSO_hod_dict['alpha'],
        )
        alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q, ic_Q = (
            QSO_hod_dict['alpha_s'],
            QSO_hod_dict['s'],
            QSO_hod_dict['s_v'],
            QSO_hod_dict['s_p'],
            QSO_hod_dict['s_r'],
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Asat'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['Bsat'],
            QSO_hod_dict['ic'],
        )

    H = len(hmass)  # num of particles

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread):  # numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            LRG_marker = 0
            if want_LRG:
                M1_L_temp = 10 ** (logM1_L + As_L * hdeltac[i] + Bs_L * hfenv[i])
                logM_cut_L_temp = logM_cut_L + Ac_L * hdeltac[i] + Bc_L * hfenv[i]
                base_p_L = (
                    n_sat_LRG_modified(
                        hmass[i],
                        logM_cut_L_temp,
                        10**logM_cut_L_temp,
                        M1_L_temp,
                        sigma_L,
                        alpha_L,
                        kappa_L,
                    )
                    * weights[i]
                    * ic_L
                )
                if enable_ranks:
                    decorator_L = (
                        1
                        + s_L * ranks[i]
                        + s_v_L * ranksv[i]
                        + s_p_L * ranksp[i]
                        + s_r_L * ranksr[i]
                    )
                    exp_sat = base_p_L * decorator_L
                else:
                    exp_sat = base_p_L
                LRG_marker += exp_sat

            ELG_marker = LRG_marker
            if want_ELG:
                M1_E_temp = 10 ** (
                    logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i] + Cs_E * hshear[i]
                )
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i] + Cc_E * hshear[i]
                )
                base_p_E = (
                    N_sat_elg(
                        hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E, A_E
                    )
                    * weights[i]
                    * ic_E
                )
                # elg conformity
                if keep_cent[i] == 1:
                    M1_E_temp = 10 ** (logM1_EL + As_E * hdeltac[i] + Bs_E * hfenv[i])
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EL,
                            A_E,
                        )
                        * weights[i]
                        * ic_E
                    )
                elif keep_cent[i] == 2:
                    M1_E_temp = 10 ** (
                        logM1_EE + As_E * hdeltac[i] + Bs_E * hfenv[i]
                    )  # M1_E_temp*10**delta_M1
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EE,
                            A_E,
                        )
                        * weights[i]
                        * ic_E
                    )

                    # if base_p_E > 1:
                    #     print("ExE new p", base_p_E, np.log10(hmass[i]), N_sat_elg(
                    #     hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E_temp, A_E, alpha1, beta), weights[i], ic_E)

                # rank mods
                if enable_ranks:
                    decorator_E = (
                        1
                        + s_E * ranks[i]
                        + s_v_E * ranksv[i]
                        + s_p_E * ranksp[i]
                        + s_r_E * ranksr[i]
                    )
                    base_p_E = base_p_E * decorator_E

                ELG_marker += base_p_E

            QSO_marker = ELG_marker
            if want_QSO:
                M1_Q_temp = 10 ** (logM1_Q + As_Q * hdeltac[i] + Bs_Q * hfenv[i])
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * hdeltac[i] + Bc_Q * hfenv[i]
                base_p_Q = (
                    N_sat_generic(
                        hmass[i], 10**logM_cut_Q_temp, kappa_Q, M1_Q_temp, alpha_Q
                    )
                    * weights[i]
                    * ic_Q
                )
                if enable_ranks:
                    decorator_Q = (
                        1
                        + s_Q * ranks[i]
                        + s_v_Q * ranksv[i]
                        + s_p_Q * ranksp[i]
                        + s_r_Q * ranksr[i]
                    )
                    exp_sat = base_p_Q * decorator_Q
                else:
                    exp_sat = base_p_Q
                QSO_marker += exp_sat

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1  # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1  # counting
                keep[i] = 3
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_y = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_z = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vx = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vy = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vz = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_mass = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vsmear = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_id = np.empty(N_lrg, dtype=hid.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype=hmass.dtype)
    elg_y = np.empty(N_elg, dtype=hmass.dtype)
    elg_z = np.empty(N_elg, dtype=hmass.dtype)
    elg_vx = np.empty(N_elg, dtype=hmass.dtype)
    elg_vy = np.empty(N_elg, dtype=hmass.dtype)
    elg_vz = np.empty(N_elg, dtype=hmass.dtype)
    elg_mass = np.empty(N_elg, dtype=hmass.dtype)
    elg_vsmear = np.empty(N_elg, dtype=hmass.dtype)
    elg_id = np.empty(N_elg, dtype=hid.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype=hmass.dtype)
    qso_y = np.empty(N_qso, dtype=hmass.dtype)
    qso_z = np.empty(N_qso, dtype=hmass.dtype)
    qso_vx = np.empty(N_qso, dtype=hmass.dtype)
    qso_vy = np.empty(N_qso, dtype=hmass.dtype)
    qso_vz = np.empty(N_qso, dtype=hmass.dtype)
    qso_mass = np.empty(N_qso, dtype=hmass.dtype)
    qso_vsmear = np.empty(N_qso, dtype=hmass.dtype)
    qso_id = np.empty(N_qso, dtype=hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                lrg_x[j1] = ppos[i, 0]
                lrg_vx[j1] = hvel[i, 0] + alpha_s_L * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                lrg_y[j1] = ppos[i, 1]
                lrg_vy[j1] = hvel[i, 1] + alpha_s_L * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                lrg_z[j1] = ppos[i, 2]
                lrg_vz[j1] = hvel[i, 2] + alpha_s_L * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if want_dv and u_grid_L.size > 0:
                    tmp_dv_los_L = redshift_error_draw(u_sat_mag[i], u_sat_sign[i], u_grid_L, x_grid_L)
                else:
                    tmp_dv_los_L = 0.0
                lrg_vsmear[j1] = tmp_dv_los_L
                
                if rsd and origin is not None:
                    nx = lrg_x[j1] - origin[0]
                    ny = lrg_y[j1] - origin[1]
                    nz = lrg_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (lrg_vx[j1] * nx + lrg_vy[j1] * ny + lrg_vz[j1] * nz) + tmp_dv_los_L
                    )
                    lrg_x[j1] = lrg_x[j1] + proj * nx
                    lrg_y[j1] = lrg_y[j1] + proj * ny
                    lrg_z[j1] = lrg_z[j1] + proj * nz
                elif rsd:
                    lrg_z[j1] = wrap(lrg_z[j1] + (lrg_vz[j1] + tmp_dv_los_L) * inv_velz2kms, lbox)
                lrg_mass[j1] = hmass[i]
                lrg_id[j1] = hid[i]
                j1 += 1
            elif keep[i] == 2:
                elg_x[j2] = ppos[i, 0]
                elg_vx[j2] = hvel[i, 0] + alpha_s_E * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                elg_y[j2] = ppos[i, 1]
                elg_vy[j2] = hvel[i, 1] + alpha_s_E * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                elg_z[j2] = ppos[i, 2]
                elg_vz[j2] = hvel[i, 2] + alpha_s_E * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias

                if want_dv and u_grid_E.size > 0:
                    tmp_dv_los_E = redshift_error_draw(u_sat_mag[i], u_sat_sign[i], u_grid_E, x_grid_E)
                else:
                    tmp_dv_los_E = 0.0
                elg_vsmear[j2] = tmp_dv_los_E
                
                if rsd and origin is not None:
                    nx = elg_x[j2] - origin[0]
                    ny = elg_y[j2] - origin[1]
                    nz = elg_z[j2] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (elg_vx[j2] * nx + elg_vy[j2] * ny + elg_vz[j2] * nz) + tmp_dv_los_E
                    )
                    elg_x[j2] = elg_x[j2] + proj * nx
                    elg_y[j2] = elg_y[j2] + proj * ny
                    elg_z[j2] = elg_z[j2] + proj * nz
                elif rsd:
                    elg_z[j2] = wrap(elg_z[j2] + (elg_vz[j2]+tmp_dv_los_E) * inv_velz2kms, lbox)
                elg_mass[j2] = hmass[i]
                elg_id[j2] = hid[i]
                j2 += 1
            elif keep[i] == 3:
                qso_x[j3] = ppos[i, 0]
                qso_vx[j3] = hvel[i, 0] + alpha_s_Q * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                qso_y[j3] = ppos[i, 1]
                qso_vy[j3] = hvel[i, 1] + alpha_s_Q * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                qso_z[j3] = ppos[i, 2]
                qso_vz[j3] = hvel[i, 2] + alpha_s_Q * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if want_dv and u_grid_Q.size > 0:
                    tmp_dv_los_Q = redshift_error_draw(u_sat_mag[i], u_sat_sign[i], u_grid_Q, x_grid_Q)
                else:
                    tmp_dv_los_Q = 0.0

                qso_vsmear[j3] = tmp_dv_los_Q
                
                if rsd and origin is not None:
                    nx = qso_x[j3] - origin[0]
                    ny = qso_y[j3] - origin[1]
                    nz = qso_z[j3] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        (qso_vx[j3] * nx + qso_vy[j3] * ny + qso_vz[j3] * nz)+tmp_dv_los_Q
                    )
                    qso_x[j3] = qso_x[j3] + proj * nx
                    qso_y[j3] = qso_y[j3] + proj * ny
                    qso_z[j3] = qso_z[j3] + proj * nz
                elif rsd:
                    qso_z[j3] = wrap(qso_z[j3] + (qso_vz[j3]+tmp_dv_los_Q) * inv_velz2kms, lbox)
                qso_mass[j3] = hmass[i]
                qso_id[j3] = hid[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ELG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    QSO_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)
    LRG_dict['x'] = lrg_x
    LRG_dict['y'] = lrg_y
    LRG_dict['z'] = lrg_z
    LRG_dict['vx'] = lrg_vx
    LRG_dict['vy'] = lrg_vy
    LRG_dict['vz'] = lrg_vz
    LRG_dict['mass'] = lrg_mass
    LRG_dict['vsmear'] = lrg_vsmear
    ID_dict['LRG'] = lrg_id

    ELG_dict['x'] = elg_x
    ELG_dict['y'] = elg_y
    ELG_dict['z'] = elg_z
    ELG_dict['vx'] = elg_vx
    ELG_dict['vy'] = elg_vy
    ELG_dict['vz'] = elg_vz
    ELG_dict['mass'] = elg_mass
    ELG_dict['vsmear'] = elg_vsmear
    ID_dict['ELG'] = elg_id

    QSO_dict['x'] = qso_x
    QSO_dict['y'] = qso_y
    QSO_dict['z'] = qso_z
    QSO_dict['vx'] = qso_vx
    QSO_dict['vy'] = qso_vy
    QSO_dict['vz'] = qso_vz
    QSO_dict['mass'] = qso_mass
    QSO_dict['vsmear'] = qso_vsmear
    ID_dict['QSO'] = qso_id
    return LRG_dict, ELG_dict, QSO_dict, ID_dict


@njit(parallel=True, fastmath=True)
def fast_concatenate(array1, array2, Nthread):
    """Fast concatenate with numba parallel"""

    N1 = len(array1)
    N2 = len(array2)
    if N1 == 0:
        return array2
    elif N2 == 0:
        return array1

    final_array = np.empty(N1 + N2, dtype=array1.dtype)
    # if one thread, then no need to parallel
    if Nthread == 1:
        for i in range(N1):
            final_array[i] = array1[i]
        for j in range(N2):
            final_array[j + N1] = array2[j]
        return final_array

    numba.set_num_threads(Nthread)
    Nthread1 = max(1, int(np.floor(Nthread * N1 / (N1 + N2))))
    Nthread2 = Nthread - Nthread1
    hstart1 = np.rint(np.linspace(0, N1, Nthread1 + 1)).astype(np.int64)
    hstart2 = np.rint(np.linspace(0, N2, Nthread2 + 1)).astype(np.int64) + N1

    for tid in numba.prange(Nthread):  # numba.prange(Nthread):
        if tid < Nthread1:
            for i in range(hstart1[tid], hstart1[tid + 1]):
                final_array[i] = array1[i]
        else:
            for i in range(hstart2[tid - Nthread1], hstart2[tid + 1 - Nthread1]):
                final_array[i] = array2[i - N1]
    # final_array = np.concatenate((array1, array2))
    return final_array


def gen_gals(
    halos_array,
    subsample,
    tracers,
    params,
    Nthread,
    enable_ranks,
    rsd,
    use_particles,
    use_profiles,
    want_dv,
    u_grid_L, 
    x_grid_L,
    u_grid_E, 
    x_grid_E,
    u_grid_Q, 
    x_grid_Q,
    verbose,
):
    """
    parse hod parameters, pass them on to central and satellite generators
    and then format the results

    Parameters
    ----------

    halos_array : dictionary of arrays
        a dictionary of halo properties (pos, vel, mass, id, randoms, ...)

    subsample : dictionary of arrays
        a dictionary of particle propoerties (pos, vel, hmass, hid, Np, subsampling, randoms, ...)

    tracers : dictionary of dictionaries
        Dictionary of multi-tracer HODs

    enable_ranks : boolean
        Flag of whether to implement particle ranks.

    rsd : boolean
        Flag of whether to implement RSD.

    params : dict
        Dictionary of various simulation parameters.

    """

    # B.H. TODO: pass as dictionary; make what's below more succinct
    for tracer in tracers.keys():
        if tracer == 'LRG':
            LRG_HOD = tracers[tracer]
        if tracer == 'ELG':
            ELG_HOD = tracers[tracer]
        if tracer == 'QSO':
            QSO_HOD = tracers[tracer]

    if 'LRG' in tracers.keys():
        want_LRG = True

        LRG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in LRG_HOD.items():
            LRG_hod_dict[key] = value

        # LRG design and decorations
        logM_cut_L, logM1_L = map(LRG_HOD.get, ('logM_cut', 'logM1'))
        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + LRG_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = LRG_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = LRG_HOD.get('logM1_pr', 0.0)
        logM_cut_L = logM_cut_L + logM_cut_pr * Delta_a
        logM1_L = logM1_L + logM1_pr * Delta_a

        LRG_hod_dict['logM_cut'] = logM_cut_L
        LRG_hod_dict['logM1'] = logM1_L

        LRG_hod_dict['Acent'] = LRG_HOD.get('Acent', 0.0)
        LRG_hod_dict['Asat'] = LRG_HOD.get('Asat', 0.0)
        LRG_hod_dict['Bcent'] = LRG_HOD.get('Bcent', 0.0)
        LRG_hod_dict['Bsat'] = LRG_HOD.get('Bsat', 0.0)

        if use_particles:
            LRG_hod_dict['s'] = LRG_HOD.get('s', 0.0)
            LRG_hod_dict['s_v'] = LRG_HOD.get('s_v', 0.0)
            LRG_hod_dict['s_p'] = LRG_HOD.get('s_p', 0.0)
            LRG_hod_dict['s_r'] = LRG_HOD.get('s_r', 0.0)
                    
        LRG_hod_dict['ic'] = LRG_HOD.get('ic', 1.0)

        if use_profiles:
            LRG_hod_dict['profile_code'] = LRG_HOD.get('profile_code', 1)
            if LRG_hod_dict['profile_code'] == 2:
                LRG_hod_dict['theta_ej'] = LRG_HOD.get('theta_ej', 4.0)
                LRG_hod_dict['logM_gas'] = LRG_HOD.get('logM_gas', 14.0)
                LRG_hod_dict['mu'] = LRG_HOD.get('mu', 0.3)
                LRG_hod_dict['eta_star'] = LRG_HOD.get('eta_star', 0.3)
                LRG_hod_dict['eta_cga'] = LRG_HOD.get('eta_cga', 0.6)
                LRG_hod_dict['max_grid_size'] = LRG_HOD.get('max_grid_size', 64)
                LRG_hod_dict['zeta_grid_size'] = LRG_HOD.get('zeta_grid_size', 10)
    else:
        want_LRG = False
        LRG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    if 'ELG' in tracers.keys():
        # ELG design
        want_ELG = True

        ELG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in ELG_HOD.items():
            ELG_hod_dict[key] = value

        logM_cut_E, logM1_E = map(ELG_HOD.get, ('logM_cut', 'logM1'))

        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + ELG_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = ELG_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = ELG_HOD.get('logM1_pr', 0.0)
        logM_cut_E = logM_cut_E + logM_cut_pr * Delta_a
        logM1_E = logM1_E + logM1_pr * Delta_a

        ELG_hod_dict['logM_cut'] = logM_cut_E
        ELG_hod_dict['logM1'] = logM1_E

        ELG_hod_dict['Acent'] = ELG_HOD.get('Acent', 0.0)
        ELG_hod_dict['Asat'] = ELG_HOD.get('Asat', 0.0)
        ELG_hod_dict['Bcent'] = ELG_HOD.get('Bcent', 0.0)
        ELG_hod_dict['Bsat'] = ELG_HOD.get('Bsat', 0.0)
        ELG_hod_dict['Ccent'] = ELG_HOD.get('Ccent', 0.0)
        ELG_hod_dict['Csat'] = ELG_HOD.get('Csat', 0.0)
        ELG_hod_dict['ic'] = ELG_HOD.get('ic', 1.0)
        ELG_hod_dict['logM1_EE'] = ELG_HOD.get('logM1_EE', ELG_hod_dict['logM1'])
        ELG_hod_dict['alpha_EE'] = ELG_HOD.get('alpha_EE', ELG_hod_dict['alpha'])
        ELG_hod_dict['logM1_EL'] = ELG_HOD.get('logM1_EL', ELG_hod_dict['logM1'])
        ELG_hod_dict['alpha_EL'] = ELG_HOD.get('alpha_EL', ELG_hod_dict['alpha'])


        if use_particles:
            ELG_hod_dict['s'] = ELG_HOD.get('s', 0.0)
            ELG_hod_dict['s_v'] = ELG_HOD.get('s_v', 0.0)
            ELG_hod_dict['s_p'] = ELG_HOD.get('s_p', 0.0)
            ELG_hod_dict['s_r'] = ELG_HOD.get('s_r', 0.0)
        
        if use_profiles:
            ELG_hod_dict['profile_code'] = ELG_HOD.get('profile_code', 1)
            if ELG_hod_dict['profile_code'] == 2:
                ELG_hod_dict['theta_ej'] = ELG_HOD.get('theta_ej', 4.0)
                ELG_hod_dict['logM_gas'] = ELG_HOD.get('logM_gas', 14.0)
                ELG_hod_dict['mu'] = ELG_HOD.get('mu', 0.3)
                ELG_hod_dict['eta_star'] = ELG_HOD.get('eta_star', 0.3)
                ELG_hod_dict['eta_cga'] = ELG_HOD.get('eta_cga', 0.6)
                ELG_hod_dict['max_grid_size'] = ELG_HOD.get('max_grid_size', 64)
                ELG_hod_dict['zeta_grid_size'] = ELG_HOD.get('zeta_grid_size', 10)
            elif ELG_hod_dict['profile_code'] == 3:
                ELG_hod_dict['exp_frac'] = ELG_HOD.get('exp_frac', 0)
                ELG_hod_dict['exp_scale'] = ELG_HOD.get('exp_scale', 1)
                ELG_hod_dict['nfw_rescale'] = ELG_HOD.get('nfw_rescale', 1)
    else:
        want_ELG = False
        ELG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    if 'QSO' in tracers.keys():
        # QSO design
        want_QSO = True
        QSO_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in QSO_HOD.items():
            QSO_hod_dict[key] = value

        logM_cut_Q, logM1_Q = map(QSO_HOD.get, ('logM_cut', 'logM1'))
        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + QSO_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = QSO_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = QSO_HOD.get('logM1_pr', 0.0)
        logM_cut_Q = logM_cut_Q + logM_cut_pr * Delta_a
        logM1_Q = logM1_Q + logM1_pr * Delta_a

        QSO_hod_dict['logM_cut'] = logM_cut_Q
        QSO_hod_dict['logM1'] = logM1_Q

        QSO_hod_dict['Acent'] = QSO_HOD.get('Acent', 0.0)
        QSO_hod_dict['Asat'] = QSO_HOD.get('Asat', 0.0)
        QSO_hod_dict['Bcent'] = QSO_HOD.get('Bcent', 0.0)
        QSO_hod_dict['Bsat'] = QSO_HOD.get('Bsat', 0.0)
        QSO_hod_dict['ic'] = QSO_HOD.get('ic', 1.0)

        if use_particles:
            QSO_hod_dict['s'] = QSO_HOD.get('s', 0.0)
            QSO_hod_dict['s_v'] = QSO_HOD.get('s_v', 0.0)
            QSO_hod_dict['s_p'] = QSO_HOD.get('s_p', 0.0)
            QSO_hod_dict['s_r'] = QSO_HOD.get('s_r', 0.0)
        
        if use_profiles:
            QSO_hod_dict['profile_code'] = QSO_HOD.get('profile_code', 1)
            if QSO_hod_dict['profile_code'] == 2:
                QSO_hod_dict['theta_ej'] = QSO_HOD.get('theta_ej', 4.0)
                QSO_hod_dict['logM_gas'] = QSO_HOD.get('logM_gas', 14.0)
                QSO_hod_dict['mu'] = QSO_HOD.get('mu', 0.3)
                QSO_hod_dict['eta_star'] = QSO_HOD.get('eta_star', 0.3)
                QSO_hod_dict['eta_cga'] = QSO_HOD.get('eta_cga', 0.6)
                QSO_hod_dict['max_grid_size'] = QSO_HOD.get('max_grid_size', 64)
                QSO_hod_dict['zeta_grid_size'] = QSO_HOD.get('zeta_grid_size', 10)
        
    else:
        want_QSO = False
        QSO_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    start = time.time()

    velz2kms = params['velz2kms']
    inv_velz2kms = 1 / velz2kms
    lbox = params['Lbox']
    origin = params['origin']

    LRG_dict_cent, ELG_dict_cent, QSO_dict_cent, ID_dict_cent, keep_cent = gen_cent(
        halos_array['hpos'],
        halos_array['hvel'],
        halos_array['hmass'],
        halos_array['hid'],
        halos_array['hmultis'],
        halos_array['hrandoms'],
        halos_array['hveldev'],
        halos_array.get('hdeltac', np.zeros(len(halos_array['hmass']))),
        halos_array.get('hfenv', np.zeros(len(halos_array['hmass']))),
        halos_array.get('hshear', np.zeros(len(halos_array['hmass']))),
        halos_array['u_cent_mag'],
        halos_array['u_cent_sign'],        
        LRG_hod_dict,
        ELG_hod_dict,
        QSO_hod_dict,
        rsd,
        want_dv,
        u_grid_L, 
        x_grid_L,
        u_grid_E, 
        x_grid_E,
        u_grid_Q, 
        x_grid_Q,
        inv_velz2kms,
        lbox,
        want_LRG,
        want_ELG,
        want_QSO,
        Nthread,
        origin,
    )
    if verbose:
        print('generating centrals took ', time.time() - start)

    start = time.time()
    if use_profiles:
        LRG_dict_sat, ELG_dict_sat, QSO_dict_sat, ID_dict_sat = gen_sats_profiles(
            subsample['ppos'], 
            subsample['phpos'], 
            subsample['phvel'], 
            subsample['phconc'], 
            subsample['phrvir'], 
            subsample['phmass'], 
            subsample['phveldev'], 
            subsample['phid'], 
            subsample['pweights'], 
            subsample['prandoms'], 
            subsample['prandoms_sate'],
            subsample.get('pdeltac', np.zeros(len(subsample['phid']))),
            subsample.get('pfenv', np.zeros(len(subsample['phid']))),
            subsample['extra_randoms'],
            subsample['u_sat_mag'],
            subsample['u_sat_sign'],
            LRG_hod_dict, 
            ELG_hod_dict, 
            QSO_hod_dict,
            rsd,
            want_dv,
            u_grid_L, 
            x_grid_L,
            u_grid_E, 
            x_grid_E,
            u_grid_Q, 
            x_grid_Q,
            inv_velz2kms, 
            lbox, 
            want_LRG, 
            want_ELG, 
            want_QSO, 
            Nthread, 
            origin,
            keep_cent[subsample['pinds']]
        )
    elif use_particles:
        LRG_dict_sat, ELG_dict_sat, QSO_dict_sat, ID_dict_sat = gen_sats_particles(
            subsample['ppos'],
            subsample['pvel'],
            subsample['phvel'],
            subsample['phmass'],
            subsample['phid'],
            subsample['pweights'],
            subsample['prandoms'],
            subsample.get('pdeltac', np.zeros(len(subsample['phid']))),
            subsample.get('pfenv', np.zeros(len(subsample['phid']))),
            subsample.get('pshear', np.zeros(len(subsample['phid']))),
            enable_ranks,
            subsample['pranks'],
            subsample['pranksv'],
            subsample['pranksp'],
            subsample['pranksr'],
            subsample['pranksc'],
            subsample['u_sat_mag'],
            subsample['u_sat_sign'],
            LRG_hod_dict,
            ELG_hod_dict,
            QSO_hod_dict,
            rsd,
            want_dv,
            u_grid_L, 
            x_grid_L,
            u_grid_E, 
            x_grid_E,
            u_grid_Q, 
            x_grid_Q,
            inv_velz2kms,
            lbox,
            params['Mpart'],
            want_LRG,
            want_ELG,
            want_QSO,
            Nthread,
            origin,
            keep_cent[subsample['pinds']],
        )
    else:
        raise RuntimeError(
            'must chose a satellite proxy'
        )
    if verbose:
        print('generating satellites took ', time.time() - start)

    # B.H. TODO: need a for loop above so we don't need to do this by hand
    HOD_dict_sat = {'LRG': LRG_dict_sat, 'ELG': ELG_dict_sat, 'QSO': QSO_dict_sat}
    HOD_dict_cent = {'LRG': LRG_dict_cent, 'ELG': ELG_dict_cent, 'QSO': QSO_dict_cent}

    # do a concatenate in numba parallel
    start = time.time()
    HOD_dict = {}
    for tracer in tracers:
        tracer_dict = {'Ncent': len(HOD_dict_cent[tracer]['x'])}
        for k in HOD_dict_cent[tracer]:
            tracer_dict[k] = fast_concatenate(
                HOD_dict_cent[tracer][k], HOD_dict_sat[tracer][k], Nthread
            )
        tracer_dict['id'] = fast_concatenate(
            ID_dict_cent[tracer], ID_dict_sat[tracer], Nthread
        )
        if verbose:
            print(tracer, 'number of galaxies ', len(tracer_dict['x']))
            print(
                'satellite fraction ',
                len(HOD_dict_sat[tracer]['x']) / len(tracer_dict['x']),
            )
        HOD_dict[tracer] = tracer_dict
    if verbose:
        print('organizing outputs took ', time.time() - start)
    return HOD_dict


def gen_gal_cat(
    halo_data,
    particle_data,
    tracers,
    params,
    Nthread=16,
    enable_ranks=False,
    rsd=True,
    use_particles=False,# H.Z. proxy choice
    use_profiles=True,
    want_dv=False,
    u_grid_L=None, 
    x_grid_L=None,
    u_grid_E=None, 
    x_grid_E=None,
    u_grid_Q=None, 
    x_grid_Q=None,
    write_to_disk=False,
    savedir='./',
    verbose=False,
    fn_ext=None,
):
    """
    pass on inputs to the gen_gals function and takes care of I/O

    Parameters
    ----------

    halos_data : dictionary of arrays
        a dictionary of halo properties (pos, vel, mass, id, randoms, ...)

    particle_data : dictionary of arrays
        a dictionary of particle propoerties (pos, vel, hmass, hid, Np, subsampling, randoms, ...)

    tracers : dictionary of dictionaries
        Dictionary of multi-tracer HODs

    enable_ranks : boolean
        Flag of whether to implement particle ranks.

    rsd : boolean
        Flag of whether to implement RSD.

    use_particles : boolean
        Flag of whether to generate satellites using particles.

    use_profiles : boolean
        Flag of whether to generate satellites from profiles.

    write_to_disk : boolean
        Flag of whether to output to disk.

    verbose : boolean
        Whether to output detailed outputs.

    savedir : str
        where to save the output if write_to_disk == True.

    params : dict
        Dictionary of various simulation parameters.

    fn_ext: str
        filename extension for saved files. Only relevant when ``write_to_disk = True``.

    Output
    ------

    HOD_dict : dictionary of dictionaries
        Dictionary of the format: {tracer1_dict, tracer2_dict, ...},
        where tracer1_dict = {x, y, z, vx, vy, vz, mass, id}

    """

    if not isinstance(rsd, bool):
        raise ValueError('Error: rsd has to be a boolean')
    # find the halos, populate them with galaxies and write them to files
    HOD_dict = gen_gals(
        halo_data,
        particle_data,
        tracers,
        params,
        Nthread,
        enable_ranks,
        rsd,
        use_particles,# H.Z. proxy choice
        use_profiles,
        want_dv,
        u_grid_L, 
        x_grid_L,
        u_grid_E, 
        x_grid_E,
        u_grid_Q, 
        x_grid_Q,
        verbose,
    )
    # how many galaxies were generated and write them to disk
    for tracer in tracers.keys():
        Ncent = HOD_dict[tracer]['Ncent']
        if verbose:
            print(
                'generated %ss:' % tracer,
                len(HOD_dict[tracer]['x']),
                'satellite fraction ',
                1 - Ncent / len(HOD_dict[tracer]['x']),
            )

        if write_to_disk:
            if verbose:
                print('outputting galaxies to disk')

            if rsd:
                rsd_string = '_rsd'
            else:
                rsd_string = ''

            if fn_ext is None:
                outdir = (savedir) / ('galaxies' + rsd_string)
            else:
                outdir = (savedir) / ('galaxies' + rsd_string + fn_ext)

            # create directories if not existing
            os.makedirs(outdir, exist_ok=True)

            # save to file
            # outdict =
            HOD_dict[tracer].pop('Ncent', None)
            table = Table(
                HOD_dict[tracer],
                meta={'Ncent': Ncent, 'Gal_type': tracer, **tracers[tracer]},
            )
            if params['chunk'] == -1:
                ascii.write(
                    table, outdir / (f'{tracer}s.dat'), overwrite=True, format='ecsv'
                )
            else:
                ascii.write(
                    table,
                    outdir / (f'{tracer}s_chunk{params["chunk"]:d}.dat'),
                    overwrite=True,
                    format='ecsv',
                )

    return HOD_dict

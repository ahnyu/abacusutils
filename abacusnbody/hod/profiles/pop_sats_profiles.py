import os
import math
import time
import warnings

import numba
import numba as nb
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from numba import njit, types,jit
from numba.typed import Dict
from .nfw_jitted import nfw_reset_halo,nfw_sample_radius
from .nfwexp_jitted import nfwexp_reset_halo, nfwexp_sample_radius
from ..dv_error import redshift_error_draw

MODEL_NONE = 0
MODEL_NFW  = 1
MODEL_BCM  = 2
MODEL_NFWEXP = 3

float_array = types.float64[:]
int_array = types.int64[:]

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
    """Wrap x into [-L/2, L/2)"""
    half = 0.5 * L
    if (-half <= x) and (x < half):
        return x
    k = math.floor((x + half) / L) 
    return x - L * k

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


@njit(parallel=True, fastmath=True)
def gen_sats_profiles(
    ppos,
    hpos,
    hvel,
    hconc,
    hrvir,
    hmass,
    hveldev,
    hid,
    weights,
    randoms,
    randoms_sate,
    hdeltac,
    hfenv,
    extra_randoms,
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
        alpha_s_L, Ac_L, As_L, Bc_L, Bs_L, ic_L = (
            LRG_hod_dict['alpha_s'],
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Asat'],
            LRG_hod_dict['Bcent'],
            LRG_hod_dict['Bsat'],
            LRG_hod_dict['ic'],
        )
        profile_L = int(LRG_hod_dict['profile_code'])
        tol_L = 1e-4
                        
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
            Ac_E,
            As_E,
            Bc_E,
            Bs_E,
            ic_E,
            logM1_EE,
            alpha_EE,
            logM1_EL,
            alpha_EL,
        ) = (
            ELG_hod_dict['alpha_s'],
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Asat'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Bsat'],
            ELG_hod_dict['ic'],
            ELG_hod_dict['logM1_EE'],
            ELG_hod_dict['alpha_EE'],
            ELG_hod_dict['logM1_EL'],
            ELG_hod_dict['alpha_EL'],
        )
        profile_E = int(ELG_hod_dict['profile_code'])
        tol_E = 1e-4
        if profile_E==MODEL_NFWEXP:
            exp_frac_E = ELG_hod_dict['exp_frac']
            exp_scale_E = ELG_hod_dict['exp_scale']
            nfw_rescale_E = ELG_hod_dict['nfw_rescale']            
            
    if want_QSO:
        logM_cut_Q, kappa_Q, logM1_Q, alpha_Q = (
            QSO_hod_dict['logM_cut'],
            QSO_hod_dict['kappa'],
            QSO_hod_dict['logM1'],
            QSO_hod_dict['alpha'],
        )
        alpha_s_Q, Ac_Q, As_Q, Bc_Q, Bs_Q, ic_Q = (
            QSO_hod_dict['alpha_s'],
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Asat'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['Bsat'],
            QSO_hod_dict['ic'],
        )
        profile_Q = int(QSO_hod_dict['profile_code'])
        tol_Q = 1e-4

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
                LRG_marker += base_p_L
                

            ELG_marker = LRG_marker
            if want_ELG:
                M1_E_temp = 10 ** (
                    logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i]# + Cs_E * hshear[i]
                )
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i]# + Cc_E * hshear[i]
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

                QSO_marker += base_p_Q

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
        prev_hid_L = -1
        prev_hid_E = -1
        prev_hid_Q = -1

        model_L = MODEL_NONE
        if want_LRG:
            if profile_L == MODEL_NFW:
                model_L = MODEL_NFW

        model_E = MODEL_NONE
        if want_ELG:
            if profile_E == MODEL_NFW: 
                model_E = MODEL_NFW
            elif profile_E == MODEL_NFWEXP:
                model_E = MODEL_NFWEXP
                
        model_Q = MODEL_NONE
        if want_QSO:
            if profile_Q == MODEL_NFW:
                model_Q = MODEL_NFW
            
        j1 = gstart[tid, 0]; j2 = gstart[tid, 1]; j3 = gstart[tid, 2]
        for i in range(hstart[tid], hstart[tid + 1]):
            this_hid = hid[i]
            if keep[i] == 1 and model_L != MODEL_NONE:
                if this_hid != prev_hid_L:
                    if model_L == MODEL_NFW:
                        tmp_rs_L, tmp_conc_L, tmp_amp_L = nfw_reset_halo(hrvir[i], hconc[i])
                    prev_hid_L = this_hid
                if model_L == MODEL_NFW:
                    tmp_r_L = nfw_sample_radius(randoms_sate[i], 
                                                tmp_rs_L, 
                                                tmp_conc_L, 
                                                tmp_amp_L,
                                                tol_L
                                               )               
                lrg_x[j1] =  wrap(hpos[i, 0] + tmp_r_L*ppos[i,0], lbox)
                lrg_vx[j1] = hvel[i, 0] + alpha_s_L * hveldev[i,0]
                lrg_y[j1] =  wrap(hpos[i, 1] + tmp_r_L*ppos[i,1], lbox)
                lrg_vy[j1] = hvel[i, 1] + alpha_s_L * hveldev[i,1]
                lrg_z[j1] =  wrap(hpos[i, 2] + tmp_r_L*ppos[i,2], lbox)
                lrg_vz[j1] = hvel[i, 2] + alpha_s_L * hveldev[i,2]
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
            elif keep[i] == 2 and model_E != MODEL_NONE:
                if this_hid != prev_hid_E:
                    if model_E == MODEL_NFW:
                        tmp_rs_E, tmp_conc_E, tmp_amp_E = nfw_reset_halo(hrvir[i], hconc[i])
                    elif model_E == MODEL_NFWEXP:
                        tmp_rs_E, tmp_conc_E = nfwexp_reset_halo(hrvir[i], hconc[i])
                    prev_hid_E = this_hid
                if model_E == MODEL_NFW:
                    tmp_r_E = nfw_sample_radius(randoms_sate[i],
                                                tmp_rs_E, 
                                                tmp_conc_E, 
                                                tmp_amp_E,
                                                tol_E
                                               )
                elif model_E == MODEL_NFWEXP:

                    tmp_r_E = nfwexp_sample_radius(extra_randoms[i], randoms_sate[i],
                                                   tmp_rs_E, tmp_conc_E, tol_E,
                                                   exp_frac_E, exp_scale_E, nfw_rescale_E
                                                  )                    
                    
                elg_x[j2] =  wrap(hpos[i, 0] + tmp_r_E*ppos[i,0], lbox)
                elg_vx[j2] = hvel[i, 0]+ alpha_s_E * hveldev[i,0]
                elg_y[j2] =  wrap(hpos[i, 1] + tmp_r_E*ppos[i,1], lbox)
                elg_vy[j2] = hvel[i, 1]+ alpha_s_E * hveldev[i,1]
                elg_z[j2] =  wrap(hpos[i, 2] + tmp_r_E*ppos[i,2], lbox)
                elg_vz[j2] = hvel[i, 2]+ alpha_s_E * hveldev[i,2]
                
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
            elif keep[i] == 3 and model_Q != MODEL_NONE:
                if this_hid != prev_hid_Q:
                    if model_Q == MODEL_NFW:
                        tmp_rs_Q, tmp_conc_Q, tmp_amp_Q = nfw_reset_halo(hrvir[i], hconc[i])
                    prev_hid_Q = this_hid
                if model_Q == MODEL_NFW:
                    tmp_r_Q = nfw_sample_radius(randoms_sate[i],
                                                tmp_rs_Q, 
                                                tmp_conc_Q, 
                                                tmp_amp_Q,
                                                tol_Q
                                               )               
                qso_x[j3] =  wrap(hpos[i, 0] + tmp_r_Q*ppos[i,0], lbox)
                qso_vx[j3] = hvel[i, 0]+ alpha_s_Q * hveldev[i,0]
                qso_y[j3] =  wrap(hpos[i, 1] + tmp_r_Q*ppos[i,1], lbox)
                qso_vy[j3] = hvel[i, 1]+ alpha_s_Q * hveldev[i,1]
                qso_z[j3] =  wrap(hpos[i, 2] + tmp_r_Q*ppos[i,2], lbox)
                qso_vz[j3] = hvel[i, 2]+ alpha_s_Q * hveldev[i,2]
                
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



"""
This is a script for loading simulation data and generating subsamples.

Usage
-----
$ python -m abacusnbody.hod.AbacusHOD.prepare_sim --path2config /path/to/config.yaml

-----
modified version by H.Z., do not support want_ranks, but you can code arbitary satellite profile.
"""

import argparse
import concurrent.futures
import gc
import glob
import multiprocessing
import os
import time
from pathlib import Path

import h5py
import numba
import numpy as np
import yaml
from numba import njit
from scipy.interpolate import interpn
from scipy.spatial import cKDTree

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf

from ..analysis.shear import get_shear, smooth_density
from ..analysis.tsc import tsc_parallel
from .menv import do_Menv_from_tree

from astropy.table import Table ###H.Z. for NFW###

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

###H.Z. for NFW###
def generate_random_points_on_sphere(N):
    u1, u2 = np.random.uniform(0, 1, (2, N))
    ra = 2 * np.pi * u1
    dec = np.arccos(1 - 2 * u2)

    x = np.sin(dec) * np.cos(ra)
    y = np.sin(dec) * np.sin(ra)
    z = np.cos(dec)

    return np.vstack((x, y, z)).T
###end H.Z.###

###subsample_halos and submask_particles got removed for better accuracy with as scarification of speed###

def get_vertices_cube(units=0.5, N=3):
    vertices = 2 * ((np.arange(2**N)[:, None] & (1 << np.arange(N))) > 0) - 1
    return vertices * units


def is_in_cube(x_pos, y_pos, z_pos, verts):
    x_min = np.min(verts[:, 0])
    x_max = np.max(verts[:, 0])
    y_min = np.min(verts[:, 1])
    y_max = np.max(verts[:, 1])
    z_min = np.min(verts[:, 2])
    z_max = np.max(verts[:, 2])

    mask = (
        (x_pos > x_min)
        & (x_pos <= x_max)
        & (y_pos > y_min)
        & (y_pos <= y_max)
        & (z_pos > z_min)
        & (z_pos <= z_max)
    )
    return mask


def gen_rand(N, chi_min, chi_max, fac, Lbox, offset, origins, rng):
    # number of randoms to generate
    N_rands = fac * N

    # location of observer
    origin = origins[0]

    # generate randoms on the unit sphere
    if (
        origins.shape[0] > 1
    ):  # not true of only the huge box where the origin is at the center
        assert origins.shape[0] == 3
        assert np.all(origins[1] + np.array([0.0, 0.0, Lbox]) == origins[0])
        assert np.all(origins[2] + np.array([0.0, Lbox, 0.0]) == origins[0])
        costheta = rng.random(N_rands)  # between zero and one
        phi = rng.random(N_rands) * np.pi / 2.0
    else:
        costheta = rng.random(N_rands) * 2.0 - 1.0
        phi = rng.random(N_rands) * 2.0 * np.pi
    theta = np.arccos(costheta)
    x_cart = np.sin(theta) * np.cos(phi)
    y_cart = np.sin(theta) * np.sin(phi)
    z_cart = np.cos(theta)
    rands_chis = rng.random(N_rands) * (chi_max - chi_min) + chi_min

    # multiply the unit vectors by that
    x_cart *= rands_chis
    y_cart *= rands_chis
    z_cart *= rands_chis

    # vector between centers of the cubes and origin in Mpc/h (i.e. placing observer at 0, 0, 0)
    box0 = np.array([0.0, 0.0, 0.0]) - origin
    if (
        origins.shape[0] > 1
    ):  # not true of only the huge box where the origin is at the center
        assert origins.shape[0] == 3
        assert np.all(origins[1] + np.array([0.0, 0.0, Lbox]) == origins[0])
        assert np.all(origins[2] + np.array([0.0, Lbox, 0.0]) == origins[0])
        box1 = np.array([0.0, 0.0, Lbox]) - origin
        box2 = np.array([0.0, Lbox, 0.0]) - origin

    # vertices of a cube centered at 0, 0, 0
    vert = get_vertices_cube(units=Lbox / 2.0)

    # remove edges because this is inherent to the light cone catalogs
    x_vert = vert[:, 0]
    y_vert = vert[:, 1]
    z_vert = vert[:, 2]
    vert[x_vert < 0, 0] += offset
    vert[x_vert > 0, 0] -= offset
    vert[y_vert < 0, 1] += offset
    vert[z_vert < 0, 2] += offset
    if origins.shape[0] == 1:  # true of the huge box where the origin is at the center
        vert[y_vert > 0, 1] -= offset
        vert[z_vert > 0, 2] -= offset

    # vertices for all three boxes
    vert0 = box0 + vert
    if origins.shape[0] > 1 and chi_max >= (
        Lbox - offset
    ):  # not true of only the huge boxes and at low zs for base
        vert1 = box1 + vert
        vert2 = box2 + vert

    # mask for whether or not the coordinates are within the vertices
    mask0 = is_in_cube(x_cart, y_cart, z_cart, vert0)
    if origins.shape[0] > 1 and chi_max >= (Lbox - offset):
        mask1 = is_in_cube(x_cart, y_cart, z_cart, vert1)
        mask2 = is_in_cube(x_cart, y_cart, z_cart, vert2)
        mask = mask0 | mask1 | mask2
    else:
        mask = mask0
    print('masked randoms = ', np.sum(mask) * 100.0 / len(mask))

    rands_pos = np.vstack((x_cart[mask], y_cart[mask], z_cart[mask])).T
    rands_chis = rands_chis[mask]
    rands_pos += origin

    return rands_pos, rands_chis


@njit(parallel=True)
def calc_fenv_opt(Menv, mbins, halosM):
    fenv_rank = np.zeros(len(Menv))
    for ibin in numba.prange(len(mbins) - 1):
        mmask = (halosM > mbins[ibin]) & (halosM < mbins[ibin + 1])
        Nmask = np.sum(mmask)
        if Nmask > 1:
            new_fenv_rank = Menv[mmask].argsort().argsort()
            fenv_rank[mmask] = (
                new_fenv_rank / (Nmask - 1) - 0.5
            )  # max rank is always Nmask - 1
    return fenv_rank

def prepare_slab(
    i,
    savedir,
    simdir,
    simname,
    z_mock,
    want_AB,
    cleaning,
    newseed,
    halo_lc=False,
    nthread=1,
    overwrite=1,
    mcut=10**10.8,
    rad_outer=10,
):
    outfilename_halos = (
        savedir
        + '/halos_xcom_'
        + str(i)
        + '_seed'
        + str(newseed)
        + '_abacushod_profiles'
    )
    outfilename_particles = (
        savedir
        + '/particles_xcom_'
        + str(i)
        + '_seed'
        + str(newseed)
        + '_abacushod_profiles'
    )
    print('processing slab ', i)

    outfilename_particles += '.h5'
    outfilename_halos += '.h5'

    seeder = np.random.default_rng(newseed + i)
    np.random.seed(seeder.integers(0, 2**32 - 1))
    halo_lc_randoms_seed = seeder.integers(0, 2**32 - 1)
    # if file already exists, just skip
    overwrite = int(overwrite)
    if (
        (not overwrite)
        and (os.path.exists(outfilename_halos))
        and (os.path.exists(outfilename_particles))
    ):
        print('files exists, skipping ', i)
        return 0

    # load the halo catalog slab
    print('loading halo catalog ')
    if halo_lc:
        slabname = (
            simdir
            + '/'
            + simname
            + '/z'
            + str(z_mock).ljust(5, '0')
            + '/lc_halo_info.asdf'
        )
        id_key = 'index_halo'
        pos_key = 'pos_interp'
        vel_key = 'vel_interp'
        N_key = 'N_interp'
    else:
        slabname = (
            simdir
            + '/'
            + simname
            + '/halos/z'
            + str(z_mock).ljust(5, '0')
            + '/halo_info/halo_info_'
            + str(i).zfill(3)
            + '.asdf'
        )
        id_key = 'id'
        pos_key = 'x_L2com'
        vel_key = 'v_L2com'
        N_key = 'N'

        
    ###H.Z. for NFW, read only needed fields###

    cat = CompaSOHaloCatalog(
        slabname,
        fields=[
            N_key,
            pos_key,
            vel_key,
#                'r90_L2com',
            'r25_L2com',
            'r98_L2com',
#                'npstartA',
#                'npoutA',
            id_key,
            'sigmav3d_L2com',
        ],
        cleaned=cleaning,
    )
        
    ###end H.Z.###
    assert halo_lc == cat.halo_lc
    
    halos = cat.halos

    if halo_lc:
        halos['id'] = halos[id_key]
        halos['x_L2com'] = halos[pos_key]
        halos['v_L2com'] = halos[vel_key]
        halos['N'] = halos[N_key]
    if cleaning:
        halos = halos[halos['N'] > 0]
    ###H.Z. for NFW###
    halos_conc=halos['r98_L2com']/halos['r25_L2com']
    parts = Table()
    ###end H.Z.###

    ###H.Z. for NFW, halos_pnum define number of NFW particles to generate for different type of boxes###
    halos_pnum = np.zeros(len(halos),dtype=int)
    halos_pstart = np.zeros(len(halos),dtype=int)
    
    if "hugebase" in simname.lower().replace(" ", ""):
        halos_pnum = np.int64(27.0 * 27.0 * 3e-7 * halos['N'] * halos['N'] + 1.5)  # 27^2 for hugebase
    elif "base" in simname.lower().replace(" ", ""):
        halos_pnum = np.int64(3e-7 * halos['N'] * halos['N'] + 1.5)  
    elif "small" in simname.lower().replace(" ", ""):
        halos_pnum = np.int64(3e-7 * halos['N'] * halos['N'] + 1.5)         
    halos_pstart[1:] = np.cumsum(halos_pnum)[:-1]
    ###end H.Z.###

    header = cat.header
    Lbox = cat.header['BoxSizeHMpc']
    Mpart = header['ParticleMassHMsun']  # msun / h
    H0 = header['H0']
    h = H0 / 100.0

    # # form a halo table of the columns i care about
    # creating a mask of which halos to keep, which halos to drop
    ###H.Z. for NFW, keep all halos###
    p_halos = np.ones(len(halos['N']))
    mask_halos = np.ones(len(halos['N']),dtype=bool)
    ###end H.Z.###
    print('total number of halos, ', len(halos), 'keeping ', np.sum(mask_halos))

    halos['mask_subsample'] = mask_halos
    halos['multi_halos'] = 1.0 / p_halos

    # only generate fenv ranks and c ranks if the user wants to enable secondary biases
    if want_AB:
        nbins = 100
        mbins = np.logspace(np.log10(mcut), 15.5, nbins + 1)

        allpos = halos['x_L2com']
        allmasses = halos['N'] * Mpart

        if halo_lc:
            # origin dependent and simulation dependent
            origins = np.array(header['LightConeOrigins']).reshape(-1, 3)
            alldist = np.sqrt(np.sum((allpos - origins[0]) ** 2.0, axis=1))
            offset = 10.0  # offset intrinsic to light cones catalogs (removing edges +/- 10 Mpc/h from the sides of the box)

            r_min = alldist.min()
            r_max = alldist.max()
            x_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            y_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            z_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            x_max_edge = Lbox / 2.0 - offset - rad_outer
            r_min_edge = alldist.min() + rad_outer
            r_max_edge = alldist.max() - rad_outer
            if (
                origins.shape[0] == 1
            ):  # true only of the huge box where the origin is at the center
                y_max_edge = Lbox / 2.0 - offset - rad_outer
                z_max_edge = Lbox / 2.0 - offset - rad_outer
            else:
                y_max_edge = 3.0 / 2 * Lbox - rad_outer
                z_max_edge = 3.0 / 2 * Lbox - rad_outer

            bounds_edge = (
                (x_min_edge <= allpos[:, 0])
                & (x_max_edge >= allpos[:, 0])
                & (y_min_edge <= allpos[:, 1])
                & (y_max_edge >= allpos[:, 1])
                & (z_min_edge <= allpos[:, 2])
                & (z_max_edge >= allpos[:, 2])
                & (r_min_edge <= alldist)
                & (r_max_edge >= alldist)
            )
            index_bounds = np.arange(allpos.shape[0], dtype=int)[~bounds_edge]
            del bounds_edge, alldist

            if len(index_bounds) > 0:
                # the randoms should be within 2 times rad_outer so as to match the true halo distn
                x_min_edge = -(Lbox / 2.0 - offset - 2.0 * rad_outer)
                y_min_edge = -(Lbox / 2.0 - offset - 2.0 * rad_outer)
                z_min_edge = -(Lbox / 2.0 - offset - 2.0 * rad_outer)
                x_max_edge = Lbox / 2.0 - offset - 2.0 * rad_outer
                r_min_edge = r_min + 2.0 * rad_outer
                r_max_edge = r_max - 2.0 * rad_outer
                if (
                    origins.shape[0] == 1
                ):  # true only of the huge box where the origin is at the center
                    y_max_edge = Lbox / 2.0 - offset - 2.0 * rad_outer
                    z_max_edge = Lbox / 2.0 - offset - 2.0 * rad_outer
                else:
                    y_max_edge = 3.0 / 2 * Lbox - 2.0 * rad_outer
                    z_max_edge = 3.0 / 2 * Lbox - 2.0 * rad_outer

                # factor of rands over all halos to generate in each iteration (can be less than 1)
                rand = 1
                rand_N = int(allpos.shape[0] * rand)

                # this is the number density (note that it depends on how the randoms are generated)
                if origins.shape[0] == 1:
                    rand_n = rand_N / (4.0 / 3.0 * np.pi * (r_max**3 - r_min**3))
                else:
                    rand_n = rand_N / (4.0 / 3.0 / 8.0 * np.pi * (r_max**3 - r_min**3))

                # aim to have rand_final times more randoms than halos at the edges at the end
                rand_final = 10
                count = 0
                repeats = 0
                rand_norm = np.zeros(len(index_bounds))
                rng = np.random.default_rng(halo_lc_randoms_seed)

                # repeat until condition satisfied
                while count < len(index_bounds) * rand_final:
                    # generate randoms in L shape
                    randpos, randdist = gen_rand(
                        allpos.shape[0], r_min, r_max, rand, Lbox, offset, origins, rng
                    )

                    # boundaries of the random particles for cutting
                    randbounds_edge = (
                        (x_min_edge <= randpos[:, 0])
                        & (x_max_edge >= randpos[:, 0])
                        & (y_min_edge <= randpos[:, 1])
                        & (y_max_edge >= randpos[:, 1])
                        & (z_min_edge <= randpos[:, 2])
                        & (z_max_edge >= randpos[:, 2])
                        & (r_min_edge <= randdist)
                        & (r_max_edge >= randdist)
                    )
                    randpos = randpos[~randbounds_edge]
                    del randbounds_edge, randdist

                    if randpos.shape[0] > 0:
                        # random points on the edges
                        rand_N = randpos.shape[0]
                        randpos_tree = cKDTree(randpos)
                        randinds_inner = randpos_tree.query_ball_point(
                            allpos[index_bounds],
                            r=halos['r98_L2com'][index_bounds],
                            workers=nthread,
                        )
                        randinds_outer = randpos_tree.query_ball_point(
                            allpos[index_bounds], r=rad_outer, workers=nthread
                        )
                        # this is the true number of randoms
                        for ind in range(len(index_bounds)):
                            rand_norm[ind] += len(randinds_outer[ind]) - len(
                                randinds_inner[ind]
                            )
                    repeats += 1
                    count += randpos.shape[0]
                    del randpos
                    gc.collect()

                # every iteration, you generate rand_n density of halos, so need to account for it
                rand_n *= repeats

                # number of randoms divided by expected number of
                # randoms (should be ~1 away from boundaries and < 1 near them)
                rand_norm /= (
                    (rad_outer**3.0 - halos['r98_L2com'][index_bounds] ** 3.0)
                    * 4.0
                    / 3.0
                    * np.pi
                    * rand_n
                )

        Menv = do_Menv_from_tree(
            allpos,
            allmasses,
            r_inner=halos['r98_L2com'],
            r_outer=rad_outer,
            halo_lc=halo_lc,
            Lbox=Lbox,
            nthread=nthread,
            mcut=mcut,
        )
        gc.collect()

        if halo_lc and len(index_bounds) > 0:
            mask = rand_norm == 0.0
            rand_norm[mask] = 1.0
            tmp = Menv[index_bounds]
            tmp /= rand_norm  # fixed (pull request #142)
            tmp[mask] = 0.0
            Menv[index_bounds] = tmp
            del mask
            gc.collect()
        halos['fenv_rank'] = calc_fenv_opt(Menv, mbins, allmasses)

        # compute delta concentration
        print('computing c rank')
        #halos_c = halos['r98_L2com'] / halos['r25_L2com']
        deltac_rank = np.zeros(len(halos))
        for ibin in range(nbins):
            mmask = (allmasses > mbins[ibin]) & (allmasses < mbins[ibin + 1])
            if np.sum(mmask) > 0:
                if np.sum(mmask) == 1:
                    deltac_rank[mmask] = 0
                else:
                    new_deltac = halos_conc[mmask] - np.median(halos_conc[mmask])
                    new_deltac_rank = new_deltac.argsort().argsort()
                    deltac_rank[mmask] = new_deltac_rank / np.max(new_deltac_rank) - 0.5
        halos['deltac_rank'] = deltac_rank

    else:
        halos['fenv_rank'] = np.zeros(len(halos))
        halos['deltac_rank'] = np.zeros(len(halos))

    halos_pstart_new = np.zeros(len(halos))
    halos_pnum_new = np.zeros(len(halos))

    # particle arrays for ranks and mask
    len_old = len_old = sum(halos_pnum) ###H.Z. for NFW###
    mask_parts = np.zeros(len_old) ###H.Z. for NFW###
    hvel_parts = np.full((len_old, 3), -1.0)
    Mh_parts = np.full(len_old, -1.0)
    Np_parts = np.full(len_old, -1.0)
    downsample_parts = np.full(len_old, -1.0)
    idh_parts = np.full(len_old, -1)
    deltach_parts = np.full(len_old, -1.0)
    fenvh_parts = np.full(len_old, -1.0)

    ###H.Z. for NFW###
    hpos_parts = np.full((len_old, 3), -1.0)
    hconc_parts = np.full(len_old, -1.0)
    hrvir_parts = np.full(len_old, -1.0)
    hsigmav_parts = np.full((len_old, 3), -1.0)
    ###end H.Z.###
    
    print('compiling particle subsamples')
    start_tracker = 0
    for j in range(len(halos)):
        if j % 10000 == 0:
            print('halo id', j, end='\r')
        if mask_halos[j] and halos_pnum[j] > 0:

            submask = np.ones(halos_pnum[j],dtype=int) ###H.Z. for NFW###
            
            # updating the particles' masks, downsample factors, halo mass
            mask_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = submask
            # print(j, halos_pstart, halos_pnum, p_halos, downsample_parts)
            downsample_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                p_halos[j]
            )
            hvel_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                'v_L2com'
            ][j]
            Mh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                halos['N'][j] * Mpart
            )  # in msun / h
            Np_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = np.sum(
                submask
            )
            idh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                'id'
            ][j]
            deltach_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                halos['deltac_rank'][j]
            )
            fenvh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                'fenv_rank'
            ][j]
            
            ###H.Z. for NFW###
            hpos_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['x_L2com'][j]
            hrvir_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['r98_L2com'][j]
            hconc_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos_conc[j]
            hsigmav_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = np.random.normal(loc = 0, scale=np.repeat(halos['sigmav3d_L2com'][j], 3).reshape((-1, 3)) / np.sqrt(3), size = (halos_pnum[j],3))
            ###end H.Z.###                

            # updating the pstart, pnum, for the halos
            halos_pstart_new[j] = start_tracker
            halos_pnum_new[j] = np.sum(submask)
            start_tracker += np.sum(submask)
            
        else:
            halos_pstart_new[j] = -1
            halos_pnum_new[j] = -1

    halos['npstartA'] = halos_pstart_new
    halos['npoutA'] = halos_pnum_new
    halos['randoms'] = np.random.random(len(halos))  # attaching random numbers
    halos['randoms_exp'] = (
        np.random.randint(0, 2, size=(len(halos), 3)) * 2 - 1
    ) * np.random.exponential(
        scale=np.repeat(halos['sigmav3d_L2com'], 3).reshape((-1, 3)) / np.sqrt(3),
        size=(len(halos), 3),
    )  # attaching random numbers
    halos['randoms_gaus_vrms'] = np.random.normal(
        loc=0,
        scale=np.repeat(halos['sigmav3d_L2com'], 3).reshape((-1, 3)) / np.sqrt(3),
        size=(len(halos), 3),
    )  # attaching random numbers

    # output halo file
    print('outputting new halo file ')
    # output_dir = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT_new.h5'
    if os.path.exists(outfilename_halos):
        os.remove(outfilename_halos)
    newfile = h5py.File(outfilename_halos, 'w')
    newfile.create_dataset('halos', data=halos[mask_halos])
    newfile.close()

    print('adding rank fields to particle data ')
    mask_parts = mask_parts.astype(bool)
    parts = parts[mask_parts]
    print(
        'pre process particle number ',
        len_old,
        ' post process particle number ',
        len(parts),
    )

    parts['downsample_halo'] = downsample_parts[mask_parts]
    parts['halo_vel'] = hvel_parts[mask_parts]
    parts['halo_mass'] = Mh_parts[mask_parts]
    parts['Np'] = Np_parts[mask_parts]
    parts['halo_id'] = idh_parts[mask_parts]
    parts['randoms'] = np.random.random(len(parts))
    parts['halo_deltac'] = deltach_parts[mask_parts]
    parts['halo_fenv'] = fenvh_parts[mask_parts]

    ###H.Z. for NFW###
    parts['halo_pos'] = hpos_parts[mask_parts]
    parts['halo_conc'] = hconc_parts[mask_parts]
    parts['halo_rvir'] = hrvir_parts[mask_parts]
    parts['randoms_sate'] = np.random.random(len(parts))
    parts['halo_randoms_gaus_vrms'] = hsigmav_parts[mask_parts]
    parts['pos'] = generate_random_points_on_sphere(len(parts))  
    ###end H.Z.###
    print(
        'are there any negative particle values? ',
        np.sum(parts['downsample_halo'] < 0),
        np.sum(parts['halo_mass'] < 0),
    )
    print('outputting new particle file ')

    if os.path.exists(outfilename_particles):
        os.remove(outfilename_particles)
    newfile = h5py.File(outfilename_particles, 'w')
    newfile.create_dataset('particles', data=parts)
    newfile.close()

    print(
        'pre process particle number ',
        len_old,
        ' post process particle number ',
        len(parts),
    )

def main(
    path2config,
    params=None,
    alt_simname=None,
    alt_z=None,
    newseed=600,
    halo_lc=False,
    overwrite=1,
):
    print('compiling compaso halo catalogs into subsampled catalogs')

    config = yaml.safe_load(open(path2config))
    # update params if needed
    if params:
        config.update(params)
    if alt_simname:
        config['sim_params']['sim_name'] = alt_simname
    if alt_z:
        config['sim_params']['z_mock'] = alt_z

    simname = config['sim_params']['sim_name']  # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = float(config['sim_params']['z_mock'])
    savedir = (
        config['sim_params']['subsample_dir']
        + simname
        + '/z'
        + str(z_mock).ljust(5, '0')
    )
    cleaning = config['sim_params']['cleaned_halos']
    if 'halo_lc' in config['sim_params'].keys():
        halo_lc = config['sim_params']['halo_lc']

    # build in some redshift checks
    ztype = None
    if halo_lc:
        raise Exception('lightcone not work for NFW')###H.Z. not tested for lightcone###
        ztype = 'lightcone'
    elif z_mock in [
        0.0,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.575,
        0.65,
        0.725,
        0.8,
        0.875,
        0.95,
        1.025,
        1.1,
        1.175,
        1.25,
        1.325,
        1.4,
        1.475,
        1.55,
        1.625,
        1.7,
        1.85,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        5.0,
        8.0,
    ]:
        ztype = 'primary'
    else:
        raise Exception('illegal redshift')

    if halo_lc:
        halo_info_fns = [
            str(
                Path(simdir) / Path(simname) / ('z%4.3f' % z_mock) / 'lc_halo_info.asdf'
            )
        ]
    else:
        halo_info_fns = list(
            sorted(
                (
                    Path(simdir)
                    / Path(simname)
                    / 'halos'
                    / ('z%4.3f' % z_mock)
                    / 'halo_info'
                ).glob('*.asdf')
            )
        )
    numslabs = len(halo_info_fns)

    os.makedirs(savedir, exist_ok=True)

    if numslabs == 0:
        raise ValueError('prepare_sim could not find any slabs!')

    want_AB = config['HOD_params'].get('want_AB', True)
    nthread = config['prepare_sim'].get('Nthread_per_load', 'auto')
    if nthread == 'auto':
        nthread = (
            len(os.sched_getaffinity(0)) // config['prepare_sim']['Nparallel_load']
        )
        print(f'prepare_sim inferred Nthread_per_load = {nthread}')
    else:
        nthread = int(nthread)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config['prepare_sim']['Nparallel_load'],
        mp_context=multiprocessing.get_context('spawn'),
    ) as pool:
        futures = [
            pool.submit(
                prepare_slab,
                i,
                savedir=savedir,
                simdir=simdir,
                simname=simname,
                z_mock=z_mock,
                want_AB=want_AB,
                cleaning=cleaning,
                newseed=newseed,
                halo_lc=halo_lc,
                nthread=nthread,
                overwrite=overwrite,
            )
            for i in range(numslabs)
        ]

    # check that all futures succeeded
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except concurrent.futures.process.BrokenProcessPool as bpp:
            raise RuntimeError(
                'A subprocess died in prepare_sim. Did prepare_slab() run out of memory?'
            ) from bpp
    # print("done, took time ", time.time() - start)


class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument(
        '--path2config', help='Path to the config file', default=DEFAULTS['path2config']
    )
    parser.add_argument(
        '--alt_simname',
        help='alternative simname to process, like "AbacusSummit_base_c000_ph003"',
    )
    parser.add_argument(
        '--alt_z',
        help='alternative z to process, like "0.8"',
        type=float,
    )
    parser.add_argument(
        '--newseed',
        help='alternative random number seed, positive integer',
        default=600,
        type=int,
    )
    parser.add_argument(
        '--overwrite', help='overwrite existing subsamples', default=1, type=int
    )
    args = vars(parser.parse_args())
    main(**args)

    print('done')
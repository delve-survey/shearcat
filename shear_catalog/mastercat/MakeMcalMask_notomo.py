
import sys
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

import numpy as np
import h5py
import healpy as hp

tag = '20240209'
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_'+tag+'.hdf'
master_mask = project_dir+'metacal_gold_combined_mask_'+tag+'.hdf'

def get_mcal_pz_mask(fr, fi, fz):

    mr = -2.5*np.log10(fr)+30
    mi = -2.5*np.log10(fi)+30
    mz = -2.5*np.log10(fz)+30

    # PZ mask
    mcal_pz_mask = ((mi < 23.5) & (mi > 18) &
                    (mr < 26)   & (mr > 15) &
                    (mz < 26)   & (mz > 15) &
                    (mr - mi < 4)   & (mr - mi > -1.5) &
                    (mi - mz < 4)   & (mi - mz > -1.5))

    del mi, mz
    return mcal_pz_mask, mr

def get_mcal_mask(sr, sn, t, mf, s1, s2, mr):

    SNR_Mask   = (sn > 10) & (sn < 1000)
    Tratio_Mask= sr > 0.5
    T_Mask = t < 10
    Flag_Mask = (mf == 0)
    Other_Mask = np.invert((t > 2) & (sn < 30)) & np.invert((np.log10(t) < (22.25 - mr)/3.5) & (s1**2 + s2**2 > 0.8**2))

    mcal_mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask
    del SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, Other_Mask

    return mcal_mask

def get_sgfgftbc_mask(sgf, fgf, ftf, bcf):

    # SG mask not used here!
    FG_Mask = (fgf==0)
    FT_Mask = (ftf==0)
    BC_Mask = (bcf==0)

    sgfgftbc_mask = FG_Mask & FT_Mask & BC_Mask
    del FG_Mask, FT_Mask, BC_Mask

    return sgfgftbc_mask

def get_baseline_mcal_mask():

    Mask = {}

    for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
        print(shear_type)

        with h5py.File(master_cat, 'r') as h5r:
            flux_r, flux_i, flux_z = h5r['mcal_flux_'+shear_type+'_dered_sfd98'][:].T

        mcal_pz_mask, mag_r = get_mcal_pz_mask(flux_r, flux_i, flux_z)
        del flux_r, flux_i, flux_z

        with h5py.File(master_cat, 'r') as h5r:
            T = h5r['mcal_T_'+shear_type][:]
            s2n = h5r['mcal_s2n_'+shear_type][:]
            size_ratio = h5r['mcal_T_ratio_'+shear_type][:]
            mcal_flags = h5r['mcal_flags'][:]
            g1, g2  = h5r['mcal_g_'+shear_type][:].T

        # Metacal cuts based on DES Y3 ones 
        # (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)

        mcal_mask = get_mcal_mask(size_ratio, s2n, T, mcal_flags, g1, g2, mag_r)

        del size_ratio, s2n, T, mcal_flags, g1, g2, mag_r

        with h5py.File(master_cat, 'r') as h5r:

            #sg = h5r['FLAGS_FOREGROUND'][:]*0+4
            # this is a hack because we don't have fitvd
            sg = h5r['FLAGS_SG_BDF'][:]
            fg = h5r['FLAGS_FOREGROUND'][:]
            ft = h5r['FLAGS_FOOTPRINT'][:]
            bc = h5r['FLAGS_BAD_COLOR'][:]

        # s/g and foreground mask
        sgfgftbc_mask = get_sgfgftbc_mask(sg, fg, ft, bc)
        del sg, fg, ft, bc

        mask_total_X = mcal_pz_mask & mcal_mask & sgfgftbc_mask
        del mcal_pz_mask, mcal_mask, sgfgftbc_mask

        print('got mask for '+shear_type)
        Mask[shear_type] = mask_total_X.astype('int')

        del mask_total_X

    return Mask


### get all the masks and store them ###

print("get the total baseline mask with selection...")
mask_total_X = get_baseline_mcal_mask()
print("total number of galaxies", len(mask_total_X['noshear'])) 
print("after cut", np.sum(mask_total_X['noshear']), np.sum(mask_total_X['1p']), np.sum(mask_total_X['1m']), np.sum(mask_total_X['2p']), np.sum(mask_total_X['2m']))

with h5py.File(master_mask, 'w') as h5r:
    h5r.create_dataset('baseline_mcal_mask_noshear', data = mask_total_X['noshear'])
    h5r.create_dataset('baseline_mcal_mask_1p', data = mask_total_X['1p'])
    h5r.create_dataset('baseline_mcal_mask_1m', data = mask_total_X['1m'])
    h5r.create_dataset('baseline_mcal_mask_2p', data = mask_total_X['2p'])
    h5r.create_dataset('baseline_mcal_mask_2m', data = mask_total_X['2m'])



import treecorr
import numpy as np
from astropy.io import fits
from astropy import coordinates
from astropy import units as u
import os, sys
import pandas as pd
import re
import h5py

import argparse

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--bin_slop',  action='store', type = float, default = 0.001)
    my_parser.add_argument('--min_angle', action='store', type = float, default = 0.1) #in arcmin
    my_parser.add_argument('--max_angle', action='store', type = float, default = 250) #in arcmin
    my_parser.add_argument('--nbins',     action='store', type = int,   default = 25)

    my_parser.add_argument('--cat_path',  action='store', type = str,   required = True)

    my_parser.add_argument('--Name', action='store', type = str, default  = None)

    args = vars(my_parser.parse_args())

    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')

    with h5py.File(args['cat_path'], 'r') as f:


        ra  = np.array(f['ra'])
        dec = np.array(f['dec'])
        g1, g2  = np.array(f['mcal_g_noshear']).T
        mag_r = 30 - 2.5*np.log10(np.array(f['mcal_flux_noshear'])[:, 0])

        SNR     = np.array(f['mcal_s2n_noshear'])
        T_ratio = np.array(f['mcal_T_ratio_noshear'])
        T       = np.array(f['mcal_T_noshear'])
        flags   = np.array(f['mcal_flags'])

        print(mag_r, SNR, T_ratio, T, flags)
        #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
        SNR_Mask   = (SNR > 10) & (SNR < 1000)
        Tratio_Mask= T_ratio > 0.5
        T_Mask     = T < 10
        Flag_Mask  = flags == 0

        Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

        print(np.sum(SNR_Mask), np.sum(Tratio_Mask), np.sum(T_Mask), np.sum(Flag_Mask), np.sum(Other_Mask))
        Mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask

        print("TOTAL NUM", np.sum(Mask))
        g1 = g1[Mask]
        g2 = g2[Mask]
        ra  = ra[Mask]
        dec = dec[Mask]

    print("LOADED EVERYTHING")

    #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
    for a in [g1, g2]:
        a -= np.mean(a)

    name = '' if args['Name'] is None else '_%s' % args['Name']
    output_path = os.environ['ROWE_STATS_DIR']
    center_path = os.environ['TMPDIR'] + './Patch_centers_TreeCorr_tmp'
    tmp_dir = os.environ['TMPDIR']

    Nth    = int(len(g1)/5_000_000) #Select every Nth object such that we end up using 5million to define patches
    Npatch = 50
    small_cat = treecorr.Catalog(g1=g1[::Nth], g2=g2[::Nth], ra=ra[::Nth], dec=dec[::Nth], ra_units='deg',dec_units='deg', npatch = Npatch)
    small_cat.write_patch_centers(center_path)
    del small_cat


    #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
    cat = treecorr.Catalog(g1=g1, g2=g2, ra=ra, dec=dec, ra_units='deg',dec_units='deg', patch_centers=center_path)

    #Compute the rowe stats
    GG = treecorr.GGCorrelation(nbins = args['nbins'], min_sep = args['min_angle'], max_sep = args['max_angle'],
                                sep_units = 'arcmin',verbose = 0,bin_slop = args['bin_slop'], var_method='jackknife')
    GG.process(cat, low_mem=True)
    print('rho0', GG.xim)
    GG.write(os.path.join(output_path, 'DELVE_2pt%s_trecorr.txt' % name))
    print('Done rho0',flush=True)

    cov_jk = GG.estimate_cov('jackknife')
    np.savetxt(os.path.join(output_path, 'DELVE_2pt_cov%s_trecorr.txt' % name), cov_jk)

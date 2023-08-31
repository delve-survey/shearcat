import treecorr
import numpy as np
from astropy.io import fits
from astropy import coordinates
from astropy import units as u
import os, sys
import pandas as pd
import re
import healpy as hp
import h5py

import argparse

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--bin_slop',  action='store', type = float, default = 0.001)
    my_parser.add_argument('--min_angle', action='store', type = float, default = 0.1) #in arcmin
    my_parser.add_argument('--max_angle', action='store', type = float, default = 250) #in arcmin
    my_parser.add_argument('--nbins',     action='store', type = int,   default = 25)

    my_parser.add_argument('--gal_cat_path',  action='store', type = str,   required = True)
    my_parser.add_argument('--psf_cat_path',  action='store', type = str,   required = True)

    my_parser.add_argument('--Name', action='store', type = str, default  = None)

    my_parser.add_argument('--band',      action='store', type = str,   default  = 'ALL')
    my_parser.add_argument('--SNRCut',    action='store', type = float, default  = 40)
    my_parser.add_argument('--Keepgband', action='store_true', default = False, dest = 'Keepgband')

    args = vars(my_parser.parse_args())

    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')

    with h5py.File(args['gal_cat_path'], 'r') as f:


        gal_ra  = np.array(f['RA'])
        gal_dec = np.array(f['DEC'])
        gal_w   = np.array(f['mcal_g_w'])
        #gal_w   = np.ones_like(gal_w)
        gal_g1, gal_g2  = np.array(f['mcal_g_noshear']).T
        mag_r = 30 -2.5*np.log10(np.array(f['mcal_flux_noshear'])[:, 0])

        SNR     = np.array(f['mcal_s2n_noshear'])
        T_ratio = np.array(f['mcal_T_ratio_noshear'])
        T       = np.array(f['mcal_T_noshear'])
        flags   = np.array(f['mcal_flags'])
        foreg   = np.array(f['FLAGS_FOREGROUND'])
        stargal = np.array(f['sg_bdf'])

        #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
        SNR_Mask   = (SNR > 10) & (SNR < 1000)
        Tratio_Mask= T_ratio > 0.5
        T_Mask     = T < 10
        Flag_Mask  = flags == 0
        FG_Mask    = foreg == 0
        sg_Mask    = stargal >= 4
        Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (gal_g1**2 + gal_g2**2 > 0.8**2))

        print(np.sum(SNR_Mask), np.sum(Tratio_Mask), np.sum(T_Mask), np.sum(Flag_Mask), np.sum(FG_Mask), np.sum(sg_Mask), np.sum(Other_Mask))
        Mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & FG_Mask & sg_Mask & Other_Mask

        print("TOTAL NUM", np.sum(Mask))
        gal_g1  = gal_g1[Mask]
        gal_g2  = gal_g2[Mask]
        gal_ra  = gal_ra[Mask]
        gal_dec = gal_dec[Mask]
        gal_w   = gal_w[Mask]

        del mag_r, SNR, T_ratio, T, flags, foreg, stargal, SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, FG_Mask, sg_Mask, Other_Mask, Mask

    
    with h5py.File(args['psf_cat_path'], 'r') as f:


        psf_ra   = np.array(f['ra'])
        psf_dec  = np.array(f['dec'])
        g1_star  = np.array(f['g1_star_hsm'])
        g2_star  = np.array(f['g2_star_hsm'])
        g1_model = np.array(f['g1_model_hsm'])
        g2_model = np.array(f['g2_model_hsm'])

        w1 = g1_star * (np.array(f['T_star_hsm']) - np.array(f['T_model_hsm']))/np.array(f['T_star_hsm'])
        w2 = g2_star * (np.array(f['T_star_hsm']) - np.array(f['T_model_hsm']))/np.array(f['T_star_hsm'])

        q1 = g1_star - g1_model
        q2 = g2_star - g2_model

        del g1_star, g2_star

        band = np.array(f['BAND']).astype('U1')
        SNR  = np.array(f['FLUX_APER_8'])/np.array(f['FLUXERR_APER_8'])
        
        Band_Mask = band == args['band'].lower() if args['band'].upper() != 'ALL' else True
        No_Gband  = band != 'g' if not args['Keepgband'] else True #We don't use g-band in shear
        SNR_Mask  = SNR > args['SNRCut']

        print(np.sum(Band_Mask), np.sum(No_Gband), np.sum(SNR_Mask))
        Mask = Band_Mask & SNR_Mask & No_Gband

        print("TOTAL NUM", np.sum(Mask))
        psf_ra   = psf_ra[Mask]
        psf_dec  = psf_dec[Mask]
        g1_model = g1_model[Mask]
        g2_model = g2_model[Mask]
        q1  = q1[Mask]
        q2  = q2[Mask]
        w1  = w1[Mask]
        w2  = w2[Mask]

        del Mask, Band_Mask, SNR_Mask, No_Gband, band, SNR

    print("LOADED EVERYTHING")

    #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
    for a in [gal_g1, gal_g2, g1_model, g2_model, q1, q2, w1, w2]:
        a -= np.mean(a)

    name = '' if args['Name'] is None else '_%s' % args['Name']
    output_path = '/project/chihway/dhayaa/DECADE/rowe_stats/TreeCorr/'
    center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
    tmp_dir = os.environ['TMPDIR']

    Nth    = int(len(gal_g1)/5_000_000) #Select every Nth object such that we end up using 5million to define patches
    Npatch = 100
    small_cat = treecorr.Catalog(g1=gal_g1[::Nth], g2=gal_g1[::Nth], ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg',dec_units='deg', npatch = Npatch)
    small_cat.write_patch_centers(center_path)
    del small_cat
    
    
    
    ########################################################################################################################
    #NOW COMPUTE STAR WEIGHTS
    ########################################################################################################################
    
    star   = np.zeros(hp.nside2npix(256))
    galaxy = np.zeros(hp.nside2npix(256))

    pix = hp.ang2pix(256, gal_ra, gal_dec, lonlat = True)
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    galaxy[unique_pix] += np.bincount(idx_rep, weights = gal_w)
    
    
    pix = hp.ang2pix(256, psf_ra, psf_dec, lonlat = True)        
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    star[unique_pix] += np.bincount(idx_rep)

    weight_map = np.zeros_like(star)
    weight_map[star != 0] = galaxy[star != 0]/star[star != 0]

    psf_w = weight_map[pix] #Assign individual stars weights from the map
    #psf_w = np.ones_like(psf_w)

    #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
    cat_g = treecorr.Catalog(g1=gal_g1,   g2=gal_g2,   ra=gal_ra, dec=gal_dec, w = gal_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_e = treecorr.Catalog(g1=g1_model, g2=g2_model, ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_q = treecorr.Catalog(g1=q1,       g2=q2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_w = treecorr.Catalog(g1=w1,       g2=w2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)

    ########################################################################################################################
    #Compute the shear 2pt
    ########################################################################################################################

    GG = treecorr.GGCorrelation(nbins = args['nbins'], min_sep = args['min_angle'], max_sep = args['max_angle'],
                                sep_units = 'arcmin',verbose = 0,bin_slop = args['bin_slop'], var_method='jackknife')
    GG.process(cat_g, low_mem=True)
    GG.write(os.path.join(output_path, 'taustats_shear_2pt%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_shear_2pt%s_cov_trecorr.txt' % name), cov_jk)


    ########################################################################################################################
    #Compute the rowe stats
    ########################################################################################################################

    EE = GG.copy()
    EE.process(cat_e, low_mem=True)
    EE.write(os.path.join(output_path, 'taustats_rho0%s_trecorr.txt' % name))
    # cov_jk = EE.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho0%s_cov_trecorr.txt' % name), cov_jk)


    QQ = GG.copy()
    QQ.process(cat_q, low_mem=True)
    QQ.write(os.path.join(output_path, 'taustats_rho1%s_trecorr.txt' % name))
    # cov_jk = QQ.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho1%s_cov_trecorr.txt' % name), cov_jk)

    EQ = GG.copy()
    EQ.process(cat_e, cat_q, low_mem=True)
    EQ.write(os.path.join(output_path, 'taustats_rho2%s_trecorr.txt' % name))
    # cov_jk = EQ.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho2%s_cov_trecorr.txt' % name), cov_jk)

    WW = GG.copy()
    WW.process(cat_w, low_mem=True)
    WW.write(os.path.join(output_path, 'taustats_rho3%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho3%s_cov_trecorr.txt' % name), cov_jk)

    QW = GG.copy()
    QW.process(cat_q, cat_w, low_mem=True)
    QW.write(os.path.join(output_path, 'taustats_rho4%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho4%s_cov_trecorr.txt' % name), cov_jk)

    EW = GG.copy()
    EW.process(cat_e, cat_w, low_mem=True)
    EW.write(os.path.join(output_path, 'taustats_rho5%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_rho5%s_cov_trecorr.txt' % name), cov_jk)

    ########################################################################################################################
    #Compute the tau stats
    ########################################################################################################################

    GE = GG.copy()
    GE.process(cat_g, cat_e, low_mem=True)
    GE.write(os.path.join(output_path, 'taustats_tau0%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_tau0%s_cov_trecorr.txt' % name), cov_jk)

    GQ = GG.copy()
    GQ.process(cat_g, cat_q, low_mem=True)
    GQ.write(os.path.join(output_path, 'taustats_tau1%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_tau1%s_cov_trecorr.txt' % name), cov_jk)

    GW = GG.copy()
    GW.process(cat_g, cat_w, low_mem=True)
    GW.write(os.path.join(output_path, 'taustats_tau2%s_trecorr.txt' % name))
    # cov_jk = GG.estimate_cov('jackknife')
    # np.savetxt(os.path.join(output_path, 'taustats_tau2%s_cov_trecorr.txt' % name), cov_jk)


    cov_jk = treecorr.estimate_multi_cov([GG, EE, QQ, EQ, WW, QW, EW, GE, GQ, GW], 'jackknife')
    np.savetxt(os.path.join(output_path, 'taustats_All%s_cov_trecorr.txt' % name), cov_jk)

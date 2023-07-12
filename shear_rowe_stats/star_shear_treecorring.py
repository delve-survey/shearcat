import treecorr
import numpy as np
from astropy.io import fits
from astropy import coordinates
from astropy import units as u
import os, sys
import pandas as pd
import re
import h5py
import healpy as hp

import argparse

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--bin_slop',  action='store', type = float, default = 0.001)
    my_parser.add_argument('--min_angle', action='store', type = float, default = 0.1) #in arcmin
    my_parser.add_argument('--max_angle', action='store', type = float, default = 250) #in arcmin
    my_parser.add_argument('--nbins',     action='store', type = int,   default = 25)

    my_parser.add_argument('--gal_cat_path',  action='store', type = str,   required = True)
    my_parser.add_argument('--psf_cat_path',  action='store', type = str,   required = True)
    
    my_parser.add_argument('--band',      action='store', type = str,   default  = 'ALL')
    my_parser.add_argument('--SNRCut',    action='store', type = float, default  = 40)
    
    
    my_parser.add_argument('--Keepgband', action='store_true', default = False, dest = 'Keepgband')
    
    my_parser.add_argument('--m_min',     action='store', type = float, default  = -40)
    my_parser.add_argument('--m_max',     action='store', type = float, default  = 40)
    
    my_parser.add_argument('--Name', action='store', type = str, default  = None)

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
        gal_w   = np.ones_like(gal_w)
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

    #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
    for a in [gal_g1, gal_g2]:
        a -= np.mean(a)
        
    with h5py.File(args['psf_cat_path'], 'r') as f:
        
        psf_ra   = np.array(f['ra'])
        psf_dec  = np.array(f['dec'])
        
        band = np.array(f['BAND']).astype('U1')
        mag  = np.array(f['MAGZP']) - 2.5*np.log10(f['FLUX_AUTO']) #Use this instead of MAG_AUTO so we use the better zeropoints
        SNR  = np.array(f['FLUX_APER_8'])/np.array(f['FLUXERR_APER_8'])
        
        
        Band_Mask = band == args['band'].lower() if args['band'].upper() != 'ALL' else True
        No_Gband  = band != 'g' if not args['Keepgband'] else True #We don't use g-band in shear
        SNR_Mask  = SNR > args['SNRCut']
        Mag_Mask  = (mag > args['m_min']) & (mag < args['m_max'])

        print(np.sum(Band_Mask), np.sum(No_Gband), np.sum(SNR_Mask), np.sum(Mag_Mask))
        Mask = Band_Mask & SNR_Mask & No_Gband & Mag_Mask
        
        print("TOTAL NUM", np.sum(Mask))
        psf_ra   = psf_ra[Mask]
        psf_dec  = psf_dec[Mask]
        
        del Mask, Band_Mask, SNR_Mask, No_Gband, Mag_Mask, band, mag, SNR
    
    print("LOADED EVERYTHING")

    name = '' if args['Name'] is None else '_%s' % args['Name']
    output_path = '/project/chihway/dhayaa/DECADE/rowe_stats/TreeCorr/'
    center_path = os.environ['ROWE_STATS_DIR'] + '/Patch_centers_TreeCorr_tmp'
    tmp_dir = os.environ['TMPDIR']

    Nth    = int(len(gal_ra)/5_000_000) #Select every Nth object such that we end up using 5million to define patches
    Npatch = 50
    small_cat = treecorr.Catalog(ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg',dec_units='deg', npatch = Npatch)
    small_cat.write_patch_centers(center_path)
    del small_cat 
    
    
    #NOW COMPUTE STAR WEIGHTS
    NSIDE  = 512
    star   = np.zeros(hp.nside2npix(NSIDE))
    galaxy = np.zeros(hp.nside2npix(NSIDE))

    pix = hp.ang2pix(NSIDE, gal_ra, gal_dec, lonlat = True)
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    galaxy[unique_pix] += np.bincount(idx_rep, weights = gal_w)
    
    
    pix = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)        
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    star[unique_pix] += np.bincount(idx_rep)

    weight_map = np.zeros_like(star)
    weight_map[star != 0] = galaxy[star != 0]/star[star != 0]

    psf_w = weight_map[pix] #Assign individual stars weights from the map
    
    #Remove stars that are not in the galaxy sample's footprint
    Mask     = psf_w > 0
    psf_ra   = psf_ra[Mask]
    psf_dec  = psf_dec[Mask]
    psf_w    = psf_w[Mask]
    
    del pix, star, galaxy, idx_rep, idx
    
    
    #NOW MAKE A RANDOMS CATALOG
    N_randoms = 1_000_000_000 #Doing rejection sampling so start with many more points than needed
    phi   = np.random.uniform(0, 2*np.pi, N_randoms)
    theta = np.arccos(1 - 2*np.random.uniform(0, 1, N_randoms))

    # Remove points that aren't within the galaxy Mask
    hpix = hp.ang2pix(NSIDE, theta, phi)
    pix_mask   = weight_map[hpix] > 0
    phi, theta = phi[pix_mask], theta[pix_mask]

    #convert to RA and DEC
    rand_ra  = phi*180/np.pi
    rand_dec = 90 - theta*180/np.pi
    
    del phi, theta, hpix, pix_mask, weight_map
    
    #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
    cat_g = treecorr.Catalog(g1 = gal_g1, g2 = gal_g2, ra = gal_ra, dec = gal_dec, w = gal_w, ra_units='deg', dec_units='deg', patch_centers=center_path)
    cat_s = treecorr.Catalog(ra = psf_ra,  dec = psf_dec,  w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_r = treecorr.Catalog(ra = rand_ra, dec = rand_dec, ra_units='deg',dec_units='deg', patch_centers=center_path)
    
    del gal_g1, gal_g2, gal_ra, gal_dec, gal_w
    del psf_ra, psf_dec, psf_w
    del rand_ra, rand_dec
    
    #Compute the rowe stats
    NG = treecorr.NGCorrelation(nbins = args['nbins'], min_sep = args['min_angle'], max_sep = args['max_angle'],
                                sep_units = 'arcmin',verbose = 0,bin_slop = args['bin_slop'], var_method='jackknife')
    
    NG.process(cat_s, cat_g, low_mem=True)
    NG.write(os.path.join(output_path, 'starshears%s_trecorr.txt' % name))
    cov_jk = NG.estimate_cov('jackknife')
    np.savetxt(os.path.join(output_path, 'starshears%s_cov_trecorr.txt' % name), cov_jk)

    NG.process(cat_r, cat_g, low_mem=True)
    NG.write(os.path.join(output_path, 'starshears_rands%s_trecorr.txt' % name))
    

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


if __name__ == "__main__":
    
    
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--gal_cat_path',  action='store', type = str,   required = True)
    my_parser.add_argument('--psf_cat_path',  action='store', type = str,   required = True)
    my_parser.add_argument('--psf_mag_path',  action='store', type = str,   required = True)

    my_parser.add_argument('--Name', action='store', type = str, default  = None)

    my_parser.add_argument('--SNRCut',    action='store', type = float, default  = 40)

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

        mcal_m_r, mcal_m_i, mcal_m_z = 30 - 2.5*np.log10(f['mcal_flux_noshear'][:][Mask]).T
        
        print("GAL r-i [95% bounds]", np.round(np.nanquantile(mcal_m_r - mcal_m_z, [0.025, 0.5, 0.975]), 3))

        del mag_r, SNR, T_ratio, T, flags, foreg, stargal, SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, FG_Mask, sg_Mask, Other_Mask, Mask
        del gal_g1, gal_g2, mcal_m_r, mcal_m_i, mcal_m_z
        
                
        
    
    #'/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5'
    with h5py.File(args['psf_cat_path'], 'r') as f:

        print(list(f.keys()))

        Ngrid = 1
        psf_ra, psf_dec = f['ra'][::Ngrid], f['dec'][::Ngrid]
        ZE1 = f['g1_model_hsm'][::Ngrid]
        ZE2 = f['g2_model_hsm'][::Ngrid]
        ONE = f['T_star_hsm'][::Ngrid]
        TWO = 1 - f['T_model_hsm'][::Ngrid]/f['T_star_hsm'][::Ngrid]
        TH1 = f['g1_star_hsm'][::Ngrid] - f['g1_model_hsm'][::Ngrid]
        TH2 = f['g2_star_hsm'][::Ngrid] - f['g2_model_hsm'][::Ngrid]

        SNR = f['FLUX_AUTO'][::Ngrid]/f['FLUXERR_AUTO'][::Ngrid]
        
        
        #'/project/chihway/dhayaa/DECADE/matched_star_psf_shapecatalog_20230630.npy'
        m_r, m_i, m_z = np.load(args['psf_mag_path'], mmap_mode = 'r')[:, ::Ngrid]
        color = m_r - m_z
        
        del m_r, m_i, m_z
        
        
        
        ########################################################################################################################
        #NOW COMPUTE STAR WEIGHTS
        ########################################################################################################################

        Nside  = 256
        star   = np.zeros(hp.nside2npix(Nside))
        galaxy = np.zeros(hp.nside2npix(Nside))

        pix = hp.ang2pix(Nside, gal_ra, gal_dec, lonlat = True)
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        galaxy[unique_pix] += np.bincount(idx_rep, weights = gal_w)


        pix = hp.ang2pix(Nside, psf_ra, psf_dec, lonlat = True)        
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        star[unique_pix] += np.bincount(idx_rep)

        weight_map = np.zeros_like(star)
        weight_map[star != 0] = galaxy[star != 0]/star[star != 0]

        psf_w = weight_map[pix] #Assign individual stars weights from the map
        
        Mask = (psf_w > 0) & (SNR > args['SNRCut'])
        
        del psf_ra, psf_dec, SNR, weight_map, gal_ra, gal_dec
        
        ZE1 = ZE1[Mask]
        ZE2 = ZE2[Mask]
        ONE = ONE[Mask]
        TWO = TWO[Mask]
        TH1 = TH1[Mask]
        TH2 = TH2[Mask]
        
        color = color[Mask]
        psf_w = psf_w[Mask] 
        
        
        ########################################################################################################################
        #NOW AVERAGE ACROSS FOOTPRINTS
        ########################################################################################################################
        
        print("<p> e1_psf:", np.average(ZE1, weights = psf_w))
        print("<p> e2_psf:", np.average(ZE2, weights = psf_w))
        
        
        print("<q> e1_err:", np.average(TH1, weights = psf_w))
        print("<q> e2_err:", np.average(TH2, weights = psf_w))
        
        
        print("<T> T_err:", np.average(TWO, weights = psf_w))
        print("sig(T) T_err:", np.sqrt(np.average(TWO**2, weights = psf_w) - np.average(TWO, weights = psf_w)**2))
        
        ########################################################################################################################
        #NOW COMPUTE COLOR DEPENDENT QUANTITIES
        ########################################################################################################################


        bins = np.linspace(-0.6, 2, 20 + 1)
        cen  = (bins[1:] + bins[:-1])/2

        counts = np.histogram(color, bins = bins)[0]

        print("HERE")
        #FOR FIRST ONE

        avg1 = np.histogram(color, bins = bins, weights = ONE)[0]/counts

        avg2 = np.histogram(color, bins = bins, weights = TWO * 1e2)[0]/counts
        
        avg3 = np.histogram(color, bins = bins, weights = TH1 * 1e3)[0]/counts

        avg4 = np.histogram(color, bins = bins, weights = TH2 * 1e3)[0]/counts

        
        avg = np.concatenate([avg1, avg2, avg3, avg4], axis = 0)
        
        
        Name = '' if args['Name'] == None else '_%s'%args['Name']
        np.save(os.path.dirname(args['psf_cat_path']) + '/PSFColor%s.npy'%Name, avg)
        np.save(os.path.dirname(args['psf_cat_path']) + '/PSFColorBins%s.npy'%Name, cen)
        
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
    my_parser.add_argument('--band',      action='store', type = str,   default  = 'ALL')
    my_parser.add_argument('--SNRCut',    action='store', type = float, default  = 40)
    
    
    my_parser.add_argument('--Keepgband', action='store_true', default = False, dest = 'Keepgband')
    
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
        g1_star  = np.array(f['g1_star_hsm'])
        g2_star  = np.array(f['g2_star_hsm'])
        g1_model = np.array(f['g1_model_hsm'])
        g2_model = np.array(f['g2_model_hsm'])
         
        w1 = g1_star * (np.array(f['T_star_hsm']) - np.array(f['T_model_hsm']))/np.array(f['T_star_hsm'])
        w2 = g2_star * (np.array(f['T_star_hsm']) - np.array(f['T_model_hsm']))/np.array(f['T_star_hsm'])

        q1 = g1_star - g1_model
        q2 = g2_star - g2_model

        del g1_star, g2_star
        
        Band_Mask = np.array(f['BAND']).astype('U1') == args['band'].lower() if args['band'].upper() != 'ALL' else True
        No_Gband  = np.array(f['BAND']).astype('U1') != 'g' if not args['Keepgband'] else True #We don't use g-band in shear
        SNR_Mask  = np.array(f['FLUX_APER_8'])/np.array(f['FLUXERR_APER_8']) > args['SNRCut']

        print(np.sum(Band_Mask), np.sum(No_Gband), np.sum(SNR_Mask))
        Mask = Band_Mask & SNR_Mask & No_Gband
        
        print("TOTAL NUM", np.sum(Mask))
        g1_model = g1_model[Mask]
        g2_model = g2_model[Mask]
        q1  = q1[Mask]
        q2  = q2[Mask]
        w1  = w1[Mask]
        w2  = w2[Mask]
        ra  = ra[Mask]
        dec = dec[Mask]
    
    print("LOADED EVERYTHING")

    #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
    for a in [g1_model, g2_model, q1, q2, w1, w2]:
        a -= np.mean(a)

    name = '' if args['Name'] is None else '_%s' % args['Name']
    output_path = os.path.dirname(args['cat_path'])
    center_path = os.environ['ROWE_STATS_DIR'] + './Patch_centers_TreeCorr_tmp'
    tmp_dir = os.environ['TMPDIR']

    Nth    = int(len(g1_model)/5_000_000) #Select every Nth object such that we end up using 5million to define patches
    Npatch = 50
    small_cat = treecorr.Catalog(g1=g1_model[::Nth], g2=g2_model[::Nth], ra=ra[::Nth], dec=dec[::Nth], ra_units='deg',dec_units='deg', npatch = Npatch)
    small_cat.write_patch_centers(center_path)
    del small_cat 


    #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
    cat_e = treecorr.Catalog(g1=g1_model, g2=g2_model, ra=ra, dec=dec, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_q = treecorr.Catalog(g1=q1,       g2=q2,       ra=ra, dec=dec, ra_units='deg',dec_units='deg', patch_centers=center_path)
    cat_w = treecorr.Catalog(g1=w1,       g2=w2,       ra=ra, dec=dec, ra_units='deg',dec_units='deg', patch_centers=center_path)
    
    #Compute the rowe stats
    GG = treecorr.GGCorrelation(nbins = args['nbins'], min_sep = args['min_angle'], max_sep = args['max_angle'],
                                sep_units = 'arcmin',verbose = 0,bin_slop = args['bin_slop'], var_method='jackknife')
    GG.process(cat_e, low_mem=True)
    print('rho0', GG.xim)
    GG.write(os.path.join(output_path, 'rho0%s_trecorr.txt' % name))
    print('Done rho0',flush=True)

    GG.process(cat_q, low_mem=True)
    print('rho1', GG.xim)
    GG.write(os.path.join(output_path, 'rho1%s_trecorr.txt' % name))
    print('Done rho1',flush=True)

    GG.process(cat_e, cat_q, low_mem=True)
    GG.write(os.path.join(output_path, 'rho2%s_trecorr.txt' % name))
    print('Done rho2',flush=True)

    GG.process(cat_w, low_mem=True)
    GG.write(os.path.join(output_path, 'rho3%s_trecorr.txt' % name))
    print('Done rho3',flush=True)

    GG.process(cat_q, cat_w, low_mem=True)
    GG.write(os.path.join(output_path, 'rho4%s_trecorr.txt' % name))
    print('Done rho4',flush=True)

    GG.process(cat_e, cat_w, low_mem=True)
    GG.write(os.path.join(output_path, 'rho5%s_trecorr.txt' % name))
    print('Done rho5',flush=True)
         
         

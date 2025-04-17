import numpy as np, pandas as pd
import h5py, healpy as hp

NSIDE = 128
Npix  = hp.nside2npix(NSIDE)

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20240209.hdf', 'r') as f:
    hpix = hp.ang2pix(NSIDE, f['RA'][:], f['DEC'][:], lonlat = True)

    #The mask for the four selections. We don't need "noshear" for this
    m1p  = f['baseline_mcal_mask_1p'][:] > 0
    m1m  = f['baseline_mcal_mask_1m'][:] > 0
    m2p  = f['baseline_mcal_mask_2p'][:] > 0
    m2m  = f['baseline_mcal_mask_2m'][:] > 0

    #Keep track of pixels for each of the selected objects
    h1p  = hpix[m1p]
    h1m  = hpix[m1m]
    h2p  = hpix[m2p]
    h2m  = hpix[m2m]

    #Weighting is part of the selection function so need to account for that
    w1p  = f['mcal_g_w_1p'][:][m1p]
    w1m  = f['mcal_g_w_1m'][:][m1m]
    w2p  = f['mcal_g_w_2p'][:][m2p]
    w2m  = f['mcal_g_w_2m'][:][m2m]

    #Finally have the shear
    g1p  = f['mcal_g_1p'][:][w1p]
    g1m  = f['mcal_g_1m'][:][w1m]
    g2p  = f['mcal_g_2p'][:][w2p]
    g2m  = f['mcal_g_2m'][:][w2m]

    #Compute response as R_11 = (<e_1+> - <e_1->) / delta_gamma
    avg_1p = np.bincount(h1p, weights = g1p[:, 0] * w1p, minlength = Npix) / np.bincount(h1p, weights = w1p, minlength = Npix)
    avg_1m = np.bincount(h1m, weights = g1m[:, 0] * w1m, minlength = Npix) / np.bincount(h1m, weights = w1m, minlength = Npix)
    avg_2p = np.bincount(h2p, weights = g2p[:, 1] * w2p, minlength = Npix) / np.bincount(h2p, weights = w2p, minlength = Npix)
    avg_2m = np.bincount(h2m, weights = g2m[:, 1] * w2m, minlength = Npix) / np.bincount(h2m, weights = w2m, minlength = Npix) 

    #Need to clean afterwards to account for divide-by-zero errors above
    R11    = (avg_1p - avg_1m) / (2 * 0.01)
    R22    = (avg_2p - avg_2m) / (2 * 0.01)





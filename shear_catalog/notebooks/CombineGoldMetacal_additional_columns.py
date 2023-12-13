# this is to add weights, s/g flag, extinction and foreground to the final combined catalog

import sys
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

import numpy as np
import astropy.io.fits as pf
import h5py
import healpy as hp
import scipy
from scipy import interpolate
from scipy.interpolate import griddata

import time
print(time.ctime())

operation = sys.argv[1]

#### Settings ####

tag = '1212'
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_2023'+tag+'.hdf'

foreground_map = hp.read_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5.fits', dtype = int)
nside_fg = 4096

response_dir = '/project/chihway/chihway/shearcat/shear_catalog/response_s2n_size/'

extinction_map = hp.read_map('/project/chihway/dhayaa/DECADE/Imsim_Inputs/ebv_sfd98_fullres_nside_4096_ring_equatorial.fits')
nside_ext = 4096

##################


def extProduction(BDF_T, BDF_S2N):
    
    x = [-3.       ,  0.79891862,  0.90845217,  0.98558583,  1.05791208,
         1.13603715,  1.22479487,  1.33572223,  1.48983602,  1.74124395,
         2.43187589,  6.        ] 
    y_1 = [0.028, 0.028, 0.008, 0.   , 0.004, 0.012, 0.012, 0.004, 0.012,
           0.024, 0.04 , 0.04 ]
    y_2 = [-0.028, -0.028, -0.04 , -0.032, -0.036, -0.032, -0.028, -0.016,
           -0.012,  0.008,  0.016,  0.016]
    y_3 = [-0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  , -0.1  ,
           -0.012,  0.008,  0.016,  0.016]
    y_4 = [0.252, 0.252, 0.188, 0.14 , 0.096, 0.104, 0.052, 0.048, 0.04 ,
           0.052, 0.088, 0.088]

    f_array = [scipy.interpolate.interp1d(x, y_1, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_2, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_3, fill_value=-99, bounds_error=False),
               scipy.interpolate.interp1d(x, y_4, fill_value=-99, bounds_error=False)]

    x_data = np.log10(BDF_S2N)
    x_data = np.where(np.isfinite(x_data), x_data, x[0])
    y_data = BDF_T.copy()
    ext = np.tile(0, len(x_data))
    for f in f_array:
        selection = (y_data > f(x_data))
        ext += selection.astype(int)
    
    # Sentinel values
    selection = np.isclose(BDF_T, -9.999e+09) | np.isclose(BDF_S2N, -9.999e+09) | (BDF_S2N <= 0.)
    ext[selection] = -9

    return np.where(np.isfinite(ext), ext, -9)


#### star-galaxy flag #############################################

if operation == 'sg':

    print('Adding star-galaxy flag...')

    with h5py.File(master_cat, 'r') as h5r:
        bdf_s2n = h5r['BDF_S2N'][:]
        bdf_t = h5r['BDF_T'][:]
        size_ratio = h5r['mcal_T_ratio_noshear'][:]
        s2n = h5r['mcal_s2n_noshear'][:]

    sg_bdf = extProduction(bdf_t, bdf_s2n)

    with h5py.File(master_cat, 'a') as h5r:
        h5r.create_dataset('FLAGS_SG_BDF', data = sg_bdf)

    del bdf_s2n, bdf_t, size_ratio, s2n, sg_bdf

#### foreground flag #############################################

if operation == 'foreground':

    print('Adding foreground flag...')

    with h5py.File(master_cat, 'r') as h5r:
        ra = h5r['RA'][:]
        dec = h5r['DEC'][:]

    phi = ra/180*np.pi
    theta = (90.-dec)/180*np.pi
    pix = hp.ang2pix(nside_fg, theta, phi)
    foreground = foreground_map[pix]

    with h5py.File(master_cat, 'a') as h5r:
        h5r.create_dataset('FLAGS_FOREGROUND', data = foreground)

    del ra, dec, theta, phi, pix, foreground

#### shear weights #############################################
# (already pre-calculated grid of response and shape noise) 

if operation == 'weights':

    print('Adding weights...')

    # this is the grid where the responses were calculated, can change if needed
    size_ratio_grid = np.logspace(np.log10(0.5), np.log10(6), 21)
    size_ratio_grid_center = np.logspace(np.log10(0.5), np.log10(6), 41)[1:][::2]

    s2n_grid = np.logspace(np.log10(10), np.log10(400), 21)
    s2n_grid_center = np.logspace(np.log10(10), np.log10(400), 41)[1:][::2]

    # center of all the 2D grid points
    xx, yy = np.meshgrid(size_ratio_grid_center, s2n_grid_center) 

    # read the pre-calculated values
    counts = np.zeros((20,20))
    sigma_e2 = np.zeros((20,20))
    sigma_e2_m = np.zeros((20,20))
    R11 = np.zeros((20,20))
    R11s = np.zeros((20,20))
    R22 = np.zeros((20,20))
    R22s = np.zeros((20,20))

    # i stands for size, j stands for s/n
    for i in range(20):
        for j in range(20):
            X = np.loadtxt(response_dir+'response_'+str(i)+'_'+str(j)+'.txt')
            counts[i][j] = X[0]
            R11[i][j] = X[1]
            R11s[i][j] = X[2]
            R22[i][j] = X[3]
            R22s[i][j] = X[4]
            sigma_e2[i][j] = X[5]
            sigma_e2_m[i][j] = X[6]

    R_tot = (R11+R11s+R22+R22s)/2 
    # in Y3, Rs was not included here
    w = (1./(sigma_e2*R_tot**(-2))).T

    with h5py.File(master_cat, 'r') as h5r:
        size_ratio = h5r['mcal_T_ratio_noshear'][:]
        s2n = h5r['mcal_s2n_noshear'][:]

    size_ratio = np.nan_to_num(size_ratio)
    s2n = np.nan_to_num(s2n)
    weights = griddata(np.array([xx.flatten(),yy.flatten()]).T, ww.flatten(), (size_ratio, s2n), method='nearest')
    # this might run slow, split it up?

    with h5py.File(master_cat, 'a') as h5r:
        h5r.create_dataset('mcal_g_w', data = weights)

    del size_ratio, s2n, weights


#### dered photometry  #############################################

if operation == 'dered':

    print('Adding deredded fluxes...')

    with h5py.File(master_cat, 'r') as h5r:

        ra = h5r['RA'][:]
        dec = h5r['DEC'][:]

    R_SFD98    = extinction_map[hp.ang2pix(nside_ext, ra, dec, lonlat = True)]
    Ag, Ar, Ai, Az = R_SFD98*3.186, R_SFD98*2.140, R_SFD98*1.569, R_SFD98*1.196

    del ra, dec

    #Metacal first
    for c in ['mcal_flux_1m', 'mcal_flux_1p', 'mcal_flux_2m', 'mcal_flux_2p', 'mcal_flux_err_1m', 'mcal_flux_err_1p',
              'mcal_flux_err_2m', 'mcal_flux_err_2p', 'mcal_flux_err_noshear', 'mcal_flux_noshear']:

        print(c + '_dered')
        with h5py.File(master_cat, 'r') as h5r:

            arr = h5r[c][:]

            arr[:, 0] *= 10**(Ar/2.5)
            arr[:, 1] *= 10**(Ai/2.5)
            arr[:, 2] *= 10**(Az/2.5)

            h5r.create_dataset(c + '_dered', data = arr)

        del arr
   

    for c in ['FLUX_AUTO_G', 'FLUX_AUTO_R', 'FLUX_AUTO_I', 'FLUX_AUTO_Z',
            'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R', 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z']: 
    #    ,   'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',
    #              'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z']:

        print(c + '_DERED')
        with h5py.File(master_cat, 'r') as h5r:

            arr = h5r[c][:]

            if c[-1] == 'G': arr *= 10**(Ag/2.5)
            elif c[-1] == 'R': arr *= 10**(Ar/2.5)
            elif c[-1] == 'I': arr *= 10**(Ai/2.5)
            elif c[-1] == 'Z': arr *= 10**(Az/2.5)

            h5r.create_dataset(c + '_DERED', data = arr)
        del arr

    with h5py.File(master_cat, 'r') as h5r:
        h5r.create_dataset('Ag', data = Ag)
        h5r.create_dataset('Ar', data = Ar)
        h5r.create_dataset('Ai', data = Ai)
        h5r.create_dataset('Az', data = Az)

    del Ag, Ar, Ai, Az

#####################################################################

        
print(time.ctime())



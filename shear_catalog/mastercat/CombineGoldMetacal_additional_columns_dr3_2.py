# this is to add weights, s/g flag, extinction and foreground to the final combined catalog

import sys
#sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

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

tag = '20241003'
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_'+tag+'.hdf'

foreground_map = hp.read_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5_MCs2.fits', dtype = int)
nside_fg = 4096

badcolor_map = hp.read_map('/project2/kadrlica/chinyi/DELVE_DR3_1_bad_colour_mask.fits', dtype = int)
nside_badcolor = 4096

response_dir = '/project/chihway/chihway/shearcat/shear_catalog/response_s2n_size/'

ext_sfd = '/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_sfd98_nside_4096_ring_equatorial.fits'
ext_planck = '/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_planck13_nside_4096_ring_equatorial.fits'
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
        del h5r['FLAGS_SG_BDF']
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
        del h5r['FLAGS_FOREGROUND']
        h5r.create_dataset('FLAGS_FOREGROUND', data = foreground)

    del ra, dec, theta, phi, pix, foreground

#### footprint flag #############################################

if operation == 'footprint':

    print('Adding footprint flag...')

    with h5py.File(master_cat, 'r') as h5r:
        ra = h5r['RA'][:]
        dec = h5r['DEC'][:]

    island = (dec > np.where((310 < ra) & (ra < 350),
                                            3.5,
                                            np.where(ra > 350,
                                                     (ra - 350) * (18 - 3.5)/(20) + 3.5,
                                                     (ra + 10)  * (18 - 3.5)/(20) + 3.5)
                                     )
                           )

    with h5py.File(master_cat, 'a') as h5r:
        del h5r['FLAGS_FOOTPRINT']
        h5r.create_dataset('FLAGS_FOOTPRINT', data = island)

    del ra, dec

#### bad color flag ##########################################

if operation == 'bad_color':

    print('Adding bad color flag...')

    with h5py.File(master_cat, 'r') as h5r:
        ra = h5r['RA'][:]
        dec = h5r['DEC'][:]

    phi = ra/180*np.pi
    theta = (90.-dec)/180*np.pi
    pix = hp.ang2pix(nside_badcolor, theta, phi)
    badcolor_map2 = badcolor_map.astype('bool')
    badcolor = badcolor_map2[pix].astype('int')

    with h5py.File(master_cat, 'a') as h5r:
        del h5r['FLAGS_BAD_COLOR']
        h5r.create_dataset('FLAGS_BAD_COLOR', data = badcolor)

    del ra, dec, theta, phi, pix, badcolor


#### shear weights new #############################################
# (already pre-calculated grid of response and shape noise) 

if operation == 'weights-new':


    # DR3_2
    results = np.load('/project/chihway/chihway/shearcat/shear_catalog/response_s2n_size/grid_quantities_20241219_DR3_2.npy')
    

    for tag in ['noshear', '1p', '1m', '2p', '2m']:
        with h5py.File(master_cat, 'r') as f:
        
            snr    = f['mcal_s2n_'+tag][:]
            Tr     = f['mcal_T_ratio_'+tag][:]

        snr = np.nan_to_num(snr)
        Tr  = np.nan_to_num(Tr)
        snr_line = results['SNR']
        Tr_line  = results['T_ratio']
        w_line   = results['weight']
        I   = interpolate.NearestNDInterpolator(np.vstack([snr_line, Tr_line]).T, w_line)
        w   = I(snr, Tr)

        with h5py.File(master_cat, 'a') as h5r:
            #del h5r['mcal_g_w']
            h5r.create_dataset('mcal_g_w_'+tag, data = w)

        del snr, Tr




#### dered photometry  #############################################

if operation == 'dered':

    print('Adding deredded fluxes...')

    for name in ['SFD98', 'Planck13']:
        
        with h5py.File(master_cat, 'r') as h5r:

            ra = h5r['RA'][:]
            dec = h5r['DEC'][:]

        if name == 'SFD98':
            EXTINCTION = hp.read_map(ext_sfd)
            R_SFD98    = EXTINCTION[hp.ang2pix(nside_ext, ra, dec, lonlat = True)]
            Ag, Ar, Ai, Az = R_SFD98*3.186, R_SFD98*2.140, R_SFD98*1.569, R_SFD98*1.196

            
        elif name == 'Planck13':
            EXTINCTION = hp.read_map(ext_planck)
            R_PLK13    = EXTINCTION[hp.ang2pix(nside_ext, ra, dec, lonlat = True)]
            Ag, Ar, Ai, Az = R_PLK13*4.085, R_PLK13*2.744, R_PLK13*2.012, R_PLK13*1.533
       
        del ra, dec

        #Metacal first
        for c in ['mcal_flux_1m', 'mcal_flux_1p', 'mcal_flux_2m', 'mcal_flux_2p', 'mcal_flux_err_1m', 'mcal_flux_err_1p',
              'mcal_flux_err_2m', 'mcal_flux_err_2p', 'mcal_flux_err_noshear', 'mcal_flux_noshear']:

            print(c + '_dered_'+ name.lower())
            with h5py.File(master_cat, 'r') as h5r:

                arr = h5r[c][:]

            arr[:, 0] *= 10**(Ar/2.5)
            arr[:, 1] *= 10**(Ai/2.5)
            arr[:, 2] *= 10**(Az/2.5)

            with h5py.File(master_cat, 'a') as h5r:
                del h5r[c + '_dered_' + name.lower()]
                h5r.create_dataset(c + '_dered_' + name.lower(), data = arr)

            del arr
   

        for c in ['FLUX_AUTO_G', 'FLUX_AUTO_R', 'FLUX_AUTO_I', 'FLUX_AUTO_Z',
                'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R', 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z',
                'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',
               'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z']:

            print(c + '_DERED_'+ name.upper())
            with h5py.File(master_cat, 'r') as h5r:

                arr = h5r[c][:]

            if c[-1] == 'G': arr *= 10**(Ag/2.5)
            elif c[-1] == 'R': arr *= 10**(Ar/2.5)
            elif c[-1] == 'I': arr *= 10**(Ai/2.5)
            elif c[-1] == 'Z': arr *= 10**(Az/2.5)

            with h5py.File(master_cat, 'a') as h5r:

                del h5r[c + '_DERED_' + name.upper()]
                h5r.create_dataset(c + '_DERED_' + name.upper(), data = arr)

            del arr

        with h5py.File(master_cat, 'a') as h5r:
            del h5r['Ag_' + name.lower()]
            del h5r['Ar_' + name.lower()]
            del h5r['Ai_' + name.lower()]
            del h5r['Az_' + name.lower()]
            h5r.create_dataset('Ag_' + name.lower(), data = Ag)
            h5r.create_dataset('Ar_'+ name.lower(), data = Ar)
            h5r.create_dataset('Ai_'+ name.lower(), data = Ai)
            h5r.create_dataset('Az_'+ name.lower(), data = Az)

        del Ag, Ar, Ai, Az

#####################################################################

        
print(time.ctime())



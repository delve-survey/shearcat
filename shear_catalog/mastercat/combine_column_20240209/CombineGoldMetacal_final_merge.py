# this is the final script 

import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

import time

print(time.ctime())

tag = '20240209'
path = '/project/chihway/data/decade/' 
out_file = '/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf'

columns_mcal = ['badfrac', 'id',
           'mcal_T_1m', 'mcal_T_1p', 'mcal_T_2m', 'mcal_T_2p',
           'mcal_T_noshear', 'mcal_T_ratio_1m', 'mcal_T_ratio_1p', 'mcal_T_ratio_2m',
           'mcal_T_ratio_2p', 'mcal_T_ratio_noshear', 'mcal_flags', 'mcal_flux_1m', 'mcal_flux_1p',
           'mcal_flux_2m', 'mcal_flux_2p', 'mcal_flux_err_1m', 'mcal_flux_err_1p',
           'mcal_flux_err_2m', 'mcal_flux_err_2p', 'mcal_flux_err_noshear', 'mcal_flux_noshear',
           'mcal_g_1m', 'mcal_g_1p', 'mcal_g_2m', 'mcal_g_2p', 'mcal_g_cov_1m', 'mcal_g_cov_1p',
           'mcal_g_cov_2m', 'mcal_g_cov_2p', 'mcal_g_cov_noshear', 'mcal_g_noshear',
           'mcal_psf_T_noshear', 'mcal_psf_g_noshear', 'mcal_s2n_1m', 'mcal_s2n_1p', 'mcal_s2n_2m',
           'mcal_s2n_2p', 'mcal_s2n_noshear', 'x', 'y',
           'Ncutouts_raw']
            #, 'ccdnum'] 
            #, 'x_exp', 'y_exp']

columns_gold = ['COADD_OBJECT_ID', 'RA', 'DEC',
           'FLUX_AUTO_G', 'FLUX_AUTO_R', 'FLUX_AUTO_I', 'FLUX_AUTO_Z', 
           'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R', 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z', 
           'FLUX_RADIUS_G', 'FLUX_RADIUS_R', 'FLUX_RADIUS_I', 'FLUX_RADIUS_Z', 
           'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',  
           'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_T', 'BDF_S2N'] 

#columns_gold = ['RA', 'DEC',
#           'FLUX_AUTO_G', 'FLUX_AUTO_R', 'FLUX_AUTO_I', 'FLUX_AUTO_Z',
#           'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R', 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z',
#           'FLUX_RADIUS_G', 'FLUX_RADIUS_R', 'FLUX_RADIUS_I', 'FLUX_RADIUS_Z']


print(len(columns_mcal))
print(len(columns_gold))

with h5py.File(out_file, "w") as f:

    for col_name in columns_mcal:
        with h5py.File(path+'metacal_gold_columns_'+tag+'/metacal_gold_combined_'+tag+'_'+col_name+'.hdf', 'r') as ff:
            f.create_dataset(col_name, data = ff[col_name][:])

    for col_name in columns_gold:
        with h5py.File(path+'metacal_gold_columns_'+tag+'/metacal_gold_combined_'+tag+'_'+col_name+'.hdf', 'r') as ff:
            f.create_dataset(col_name, data = ff[col_name][:])

print(time.ctime())



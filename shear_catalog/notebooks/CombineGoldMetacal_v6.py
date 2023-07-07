import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

metadata1 = np.genfromtxt('../Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:]
metadata2 = np.genfromtxt('../Tilelist_DR3_1_2_withASTROFIX.csv', dtype='str', delimiter=",")[1:]


import time

print(time.ctime())

columns_mcal = ['badfrac', 'id', 'ra', 'dec',
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
 
shear_dir1 = '/project2/chihway/data/decade/shearcat_v2/'
shear_dir2 = '/project/chihway/data/decade/shearcat_v3/'
path = '/project2/chihway/data/decade/metacal_combined_20230606.hdf'

Ntile1 = 6537
Ntile2 = 6508


def get_column_mcal(column):

    output = []

    for i in tqdm(range(Ntile1), desc = column): 

        tile = metadata1[i][0]

        if os.path.exists(shear_dir1+'metacal_output_'+tile+'.fits'):
            
 #           fits = fitsio.FITS(shear_dir1+'metacal_output_'+tile+'.fits')
 #           shear_cat = fits[1].read(vstorage='object')
 
            shear_cat = pf.open(shear_dir1+'metacal_output_'+tile+'.fits')
            arr = shear_cat[1].data[column][:].copy()
            shear_cat.close()
            del shear_cat
       
 #           arr = shear_cat[column][:]
            if column!= 'id':
                arr.astype(np.float32)

            output.append(arr)
            del arr

    for i in tqdm(range(Ntile2), desc = column):

        tile = metadata2[i][0]

        if os.path.exists(shear_dir2+'metacal_output_'+tile+'.fits'):

            #fits = fitsio.FITS(shear_dir2+'metacal_output_'+tile+'.fits')
            #shear_cat = fits[1].read(vstorage='object')

            shear_cat = pf.open(shear_dir2+'metacal_output_'+tile+'.fits')
            arr = shear_cat[1].data[column][:].copy()
            shear_cat.close()
            del shear_cat            

            if column!= 'id':
                arr.astype(np.float32)

            output.append(arr)
            del arr

    return np.concatenate(output, axis = 0)



with h5py.File(path, "w") as f:

    for c in columns_mcal:
        f.create_dataset(c, data = get_column_mcal(c))


print(time.ctime())




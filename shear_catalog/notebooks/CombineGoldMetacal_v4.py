import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

metadata = np.genfromtxt('../Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:]

import time

print(time.ctime())

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
           'Ncutouts_raw', 'ccdnum', 'x_exp', 'y_exp']

columns_gold = ['COADD_OBJECT_ID', 'RA', 'DEC', 'SG'] 
#, 
#           'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 
#           'MAGERR_AUTO_G', 'MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 
#           'FLUX_RADIUS_G', 'FLUX_RADIUS_R', 'FLUX_RADIUS_I', 'FLUX_RADIUS_Z', 
#           'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z', 'BDF_FLUX_Y', 
#           'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_FLUX_ERR_Y', 'Tilename1', 'Tilename2', 'SG'] #SG
 

shear_dir = '/project2/chihway/data/decade/shearcat_v2/'
gold_cat = '/project2/chihway/data/decade/gold_20230522_v2.hdf'

path = '/project2/chihway/data/decade/metacal_gold_combined_20230522_v2.hdf'

Ntile = 6537

GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}

#6537
for i in tqdm(range(Ntile), desc = 'Build GoldMask, McalMask'): 

    tile = metadata[i][0]
    tilename1 = int(tile[3:7])
    tilename2 = int(tile[-5:])
    print(tilename1, tilename2)

    if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits'):

        fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
        shear_id = fits[1].read(vstorage='object')['id'] 
        # mcal coadd object ids

        with h5py.File(gold_cat, 'r') as f:
            
            tile1 = f['Tilename1'][:]
            tile2 = f['Tilename2'][:]
            sg = f['SG'][:]
            mask_tile = (tile1==tilename1)*(tile2==tilename2) #*(sg==4)
            # note that this s/g cut still needs to be tested maybe
            print('objects in gold', len(f['RA'][mask_tile]))
            ids_gold = f['COADD_OBJECT_ID'][mask_tile]

        # ask which metacal galaxies are in this id list
        mask_joint_on_mcal = np.in1d(shear_id, ids_gold)
        ids_to_keep_mcal = shear_id[mask_joint_on_mcal]

        # inversely, ask which of the galaxies in gold are in this ids list above
        mask_joint_on_gold = np.in1d(ids_gold, ids_to_keep_mcal)
        ids_to_keep_gold = ids_gold[mask_joint_on_gold]
 #       print(ids_to_keep_gold)
 #       print(ids_to_keep_mcal)

        GOLD_Mask[tile] = mask_joint_on_gold
        GOLD_Sort[tile] = np.argsort(ids_to_keep_gold)
        MCAL_Mask[tile] = mask_joint_on_mcal
        MCAL_Sort[tile] = np.argsort(ids_to_keep_mcal)

 #       print(ids_to_keep_gold[GOLD_Sort[tile]])
 #       print(ids_to_keep_mcal[MCAL_Sort[tile]])


def get_column_mcal(column):

    output = []
    for i in tqdm(range(Ntile), desc = column): #6537

        tile = metadata[i][0]

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits'): 
            
            fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
            shear_cat = fits[1].read(vstorage='object')
        
            arr = shear_cat[column][MCAL_Mask[tile]][MCAL_Sort[tile]]

            #Hardcoding this in because ccdnum and x_exp/y_exp has too large a datatype
            #Wont have enough memory to create the final array (more than 50GB).
            if column == 'ccdnum':
                arr = arr.astype(np.int16)
            elif (column == 'x_exp') | (column == 'y_exp'):
                arr = arr.astype(np.float32)
            
            output.append(arr)

    return np.concatenate(output, axis = 0)


def get_column_gold(column):

    output = []
    for i in tqdm(range(Ntile), desc = column): #6537

        tile = metadata[i][0]
        tilename1 = int(tile[3:7])
        tilename2 = int(tile[-5:])

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits'):

            with h5py.File(gold_cat, 'r') as f:

                tile1 = f['Tilename1'][:]
                tile2 = f['Tilename2'][:]
                #sg = f['SG'][:]
                mask_tile = (tile1==tilename1)*(tile2==tilename2) #*(sg==4)
                arr = f[column][mask_tile][GOLD_Mask[tile]][GOLD_Sort[tile]]
                
                #if column=='COADD_OBJECT_ID':
                #    print(arr[50:60])
                #if column=='RA':
                #    print(arr[50:60])


                output.append(arr)

    return np.concatenate(output, axis = 0)


with h5py.File(path, "w") as f:

#    for c in columns_mcal:
#        f.create_dataset(c, data = get_column_mcal(c))

    for c in columns_gold:
        f.create_dataset(c, data = get_column_gold(c))


print(time.ctime())



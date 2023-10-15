# this is the final script for DR3_1

import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

metadata1 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]
metadata2 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_2.txt', dtype='str', delimiter=",")[1:]

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
           'Ncutouts_raw']
            #, 'ccdnum'] 
            #, 'x_exp', 'y_exp']

columns_gold = ['COADD_OBJECT_ID', 'RA', 'DEC',
           'FLUX_AUTO_G', 'FLUX_AUTO_R', 'FLUX_AUTO_I', 'FLUX_AUTO_Z', 
           'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R', 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z', 
           'FLUX_RADIUS_G', 'FLUX_RADIUS_R', 'FLUX_RADIUS_I', 'FLUX_RADIUS_Z', 
           'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',  
           'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_T', 'BDF_S2N'] #SG
 

shear_dir1 = '/project/chihway/data/decade/shearcat_v2/'
gold_dir1 = '/project/chihway/data/decade/coaddcat_v6/'

shear_dir2 = '/project/chihway/data/decade/shearcat_v3/'
gold_dir2 = '/project/chihway/data/decade/coaddcat_v7/'


path = '/project/chihway/data/decade/metacal_gold_combined_20230919.hdf'

Ntile1 = len(metadata1)
Ntile2 = len(metadata2)

GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}

# DR3_1_1
for i in tqdm(range(Ntile1), desc = 'Build GoldMask, McalMask'): 

    tile = metadata1[i][0]
    print(i, tile)

    if os.path.exists(shear_dir1+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir1+'gold_'+tile+'.fits'):

        fits = fitsio.FITS(shear_dir1+'metacal_output_'+tile+'.fits')
        shear_id = fits[1].read(vstorage='object')['id'] 
        # mcal coadd object ids

        fits_gold = fitsio.FITS(gold_dir1+'gold_'+tile+'.fits')
        gold_id = fits_gold[1].read(vstorage='object')['COADD_OBJECT_ID']

        # ask which metacal galaxies are in this id list
        mask_joint_on_mcal = np.in1d(shear_id, gold_id)
        ids_to_keep_mcal = shear_id[mask_joint_on_mcal]

        # inversely, ask which of the galaxies in gold are in this ids list above
        mask_joint_on_gold = np.in1d(gold_id, ids_to_keep_mcal)
        ids_to_keep_gold = gold_id[mask_joint_on_gold]

        GOLD_Mask[tile] = mask_joint_on_gold
        GOLD_Sort[tile] = np.argsort(ids_to_keep_gold)
        MCAL_Mask[tile] = mask_joint_on_mcal
        MCAL_Sort[tile] = np.argsort(ids_to_keep_mcal)

        print(ids_to_keep_gold[GOLD_Sort[tile]])
        print(ids_to_keep_mcal[MCAL_Sort[tile]])


# DR3_1_2
for i in tqdm(range(Ntile2), desc = 'Build GoldMask, McalMask'):

    tile = metadata2[i][0]
    print(i, tile)

    if os.path.exists(shear_dir2+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir2+'gold_'+tile+'.fits'):

        fits = fitsio.FITS(shear_dir2+'metacal_output_'+tile+'.fits')
        shear_id = fits[1].read(vstorage='object')['id']
        # mcal coadd object ids

        fits_gold = fitsio.FITS(gold_dir2+'gold_'+tile+'.fits')
        gold_id = fits_gold[1].read(vstorage='object')['COADD_OBJECT_ID']

        # ask which metacal galaxies are in this id list
        mask_joint_on_mcal = np.in1d(shear_id, gold_id)
        ids_to_keep_mcal = shear_id[mask_joint_on_mcal]

        # inversely, ask which of the galaxies in gold are in this ids list above
        mask_joint_on_gold = np.in1d(gold_id, ids_to_keep_mcal)
        ids_to_keep_gold = gold_id[mask_joint_on_gold]

        GOLD_Mask[tile] = mask_joint_on_gold
        GOLD_Sort[tile] = np.argsort(ids_to_keep_gold)
        MCAL_Mask[tile] = mask_joint_on_mcal
        MCAL_Sort[tile] = np.argsort(ids_to_keep_mcal)

        print(ids_to_keep_gold[GOLD_Sort[tile]])
        print(ids_to_keep_mcal[MCAL_Sort[tile]])



def get_column_mcal(column):

    output = []

    #DR3_1_1
    for i in tqdm(range(Ntile1), desc = column):

        tile = metadata1[i][0]

        if os.path.exists(shear_dir1+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir1+'gold_'+tile+'.fits'):
            
            fits = fitsio.FITS(shear_dir1+'metacal_output_'+tile+'.fits')
            shear_cat = fits[1].read(vstorage='object')
        
            arr = shear_cat[column][MCAL_Mask[tile]][MCAL_Sort[tile]]

            #Hardcoding this in because ccdnum and x_exp/y_exp has too large a datatype
            #Wont have enough memory to create the final array (more than 50GB).
            #if column == 'ccdnum':
            #    arr = arr.astype(np.int16)
            #elif (column == 'x_exp') | (column == 'y_exp'):
            #    arr = arr.astype(np.float32)
            
            output.append(arr)

    #DR3_1_2
    for i in tqdm(range(Ntile2), desc = column):

        tile = metadata2[i][0]

        if os.path.exists(shear_dir2+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir2+'gold_'+tile+'.fits'):

            fits = fitsio.FITS(shear_dir2+'metacal_output_'+tile+'.fits')
            shear_cat = fits[1].read(vstorage='object')

            arr = shear_cat[column][MCAL_Mask[tile]][MCAL_Sort[tile]]

            output.append(arr)


    return np.concatenate(output, axis = 0)


def get_column_gold(column):

    output = []

    #DR3_1_1
    for i in tqdm(range(Ntile1), desc = column): #6537

        tile = metadata1[i][0]

        if os.path.exists(shear_dir1+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir1+'gold_'+tile+'.fits'):

            fits = fitsio.FITS(gold_dir1+'gold_'+tile+'.fits')
            gold_cat = fits[1].read(vstorage='object')

            arr = gold_cat[column][GOLD_Mask[tile]][GOLD_Sort[tile]]

            output.append(arr)

    #DR3_1_2
    for i in tqdm(range(Ntile2), desc = column): #6537

        tile = metadata2[i][0]

        if os.path.exists(shear_dir2+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir2+'gold_'+tile+'.fits'):

            fits = fitsio.FITS(gold_dir2+'gold_'+tile+'.fits')
            gold_cat = fits[1].read(vstorage='object')

            arr = gold_cat[column][GOLD_Mask[tile]][GOLD_Sort[tile]]

            output.append(arr)


    return np.concatenate(output, axis = 0)


with h5py.File(path, "w") as f:

    for c in columns_mcal:
        f.create_dataset(c, data = get_column_mcal(c))

    for c in columns_gold:
        f.create_dataset(c, data = get_column_gold(c))


print(time.ctime())



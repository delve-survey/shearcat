# this is the final script for DR3_1 to make masks

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

shear_dir1 = '/project/chihway/data/decade/shearcat_v2/'
gold_dir1 = '/project/chihway/data/decade/coaddcat_v6/'

shear_dir2 = '/project/chihway/data/decade/shearcat_v3/'
gold_dir2 = '/project/chihway/data/decade/coaddcat_v7/'

path = '/project/chihway/data/decade/'

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

import pickle

with open(path+"GOLD_Mask.pk", "wb") as file_pk:
    pickle.dump(GOLD_Mask, file_pk)

with open(path+"MCAL_Mask.pk", "wb") as file_pk:
    pickle.dump(MCAL_Mask, file_pk)

with open(path+"GOLD_Sort.pk", "wb") as file_pk:
    pickle.dump(GOLD_Sort, file_pk)

with open(path+"MCAL_Sort.pk", "wb") as file_pk:
    pickle.dump(MCAL_Sort, file_pk)


print(time.ctime())



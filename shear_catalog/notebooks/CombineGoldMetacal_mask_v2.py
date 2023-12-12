# this is a temporary script to make combined file for rerun tiles with part of them not having fitvd run

import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

# combine 3 lists
metadata1 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]
metadata2 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist//07112023/new_final_list_DR3_1_2_without_rerun.txt', dtype='str', delimiter=",")[1:]

metadata3 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist//07112023/Tilelist_Reprocess_20231124.csv', dtype='str', delimiter=",")[1:]

import time

print(time.ctime())

Metadata = [metadata1, metadata2, metadata3]
Ntile = [len(metadata1), len(metadata2), len(metadata3)]
Shear_dir = ['/project/chihway/data/decade/shearcat_v2/','/project/chihway/data/decade/shearcat_v3/', '/project/chihway/data/decade/shearcat_v5/']
Gold_dir = ['/project/chihway/data/decade/coaddcat_v6/', '/project/chihway/data/decade/coaddcat_v7/', '/project/chihway/data/decade/coaddcat_v8/']

path = '/project/chihway/data/decade/'

GOLD_Mask0 = {}
GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}


for X in range(2):
    # range(Ntile[X])
    for i in tqdm(range(Ntile[X]), desc = 'Build GoldMask, McalMask'): 

        tile = Metadata[X][i][0]
        print(i, tile)

        if os.path.exists(Shear_dir[X]+'metacal_output_'+tile+'.fits') and os.path.exists(Gold_dir[X]+'gold_'+tile+'.fits'):

            fits = fitsio.FITS(Shear_dir[X]+'metacal_output_'+tile+'.fits')
            shear_id = fits[1].read(vstorage='object')['id'] 
            # mcal coadd object ids

            fits_gold = fitsio.FITS(Gold_dir[X]+'gold_'+tile+'.fits')
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


X = 2

#range(Ntile[X])
for i in tqdm(range(Ntile[X]), desc = 'Build GoldMask, McalMask'):

    tile = Metadata[X][i][0]
    print(i, tile)

    if os.path.exists(Shear_dir[X]+'metacal_output_'+tile+'.fits') and os.path.exists(Gold_dir[X]+'gold_'+tile+'.hdf5'):

        fits = fitsio.FITS(Shear_dir[X]+'metacal_output_'+tile+'.fits')
        shear_id = fits[1].read(vstorage='object')['id']
        # mcal coadd object ids

        with h5py.File(Gold_dir[X]+'gold_'+tile+'.hdf5') as hdf5_gold:
            gold_mask = hdf5_gold['GOLD_MASK'][:]
        gold_id = np.load(Shear_dir[X]+'ids_match_'+tile+'.npz')['ids'][gold_mask]

        # ask which metacal galaxies are in this id list
        mask_joint_on_mcal = np.in1d(shear_id, gold_id)
        ids_to_keep_mcal = shear_id[mask_joint_on_mcal]

        # inversely, ask which of the galaxies in gold are in this ids list above
        mask_joint_on_gold = np.in1d(gold_id, ids_to_keep_mcal)
        ids_to_keep_gold = gold_id[mask_joint_on_gold]

        GOLD_Mask0[tile] = gold_mask
        GOLD_Mask[tile] = mask_joint_on_gold
        GOLD_Sort[tile] = np.argsort(ids_to_keep_gold)
        MCAL_Mask[tile] = mask_joint_on_mcal
        MCAL_Sort[tile] = np.argsort(ids_to_keep_mcal)

        print(ids_to_keep_gold[GOLD_Sort[tile]])
        print(ids_to_keep_mcal[MCAL_Sort[tile]])


import pickle

with open(path+"GOLD_Mask0_20231212.pk", "wb") as file_pk:
    pickle.dump(GOLD_Mask0, file_pk)

with open(path+"GOLD_Mask_20231212.pk", "wb") as file_pk:
    pickle.dump(GOLD_Mask, file_pk)

with open(path+"MCAL_Mask_20231212.pk", "wb") as file_pk:
    pickle.dump(MCAL_Mask, file_pk)

with open(path+"GOLD_Sort_20231212.pk", "wb") as file_pk:
    pickle.dump(GOLD_Sort, file_pk)

with open(path+"MCAL_Sort_20231212.pk", "wb") as file_pk:
    pickle.dump(MCAL_Sort, file_pk)


print(time.ctime())



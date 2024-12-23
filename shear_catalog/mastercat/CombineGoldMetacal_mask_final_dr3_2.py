
# this is a script to make combined mask for DR3_1

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


tag = '20241003'
metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/06042024/dr3_2_final_20241003.csv', dtype='str', delimiter=",")[1:]
ntile = len(metadata)
shear_dir = '/project/chihway/data/decade/shearcat_dr3_2/'
gold_dir = '/project/chihway/data/decade/coaddcat_final_dr3_2/'
out_dir = '/project/chihway/data/decade/'

GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}

for i in tqdm(range(ntile), desc = 'Build GoldMask, McalMask'): 
    tile = metadata[i][0]
    print(i, tile)

    if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir+'gold_'+tile+'.fits'):
        fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
        shear_id = fits[1].read(vstorage='object')['id'] 
        # mcal coadd object ids

        fits_gold = fitsio.FITS(gold_dir+'gold_'+tile+'.fits')
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

with open(out_dir+"GOLD_Mask_"+tag+".pk", "wb") as file_pk:
    pickle.dump(GOLD_Mask, file_pk)

with open(out_dir+"MCAL_Mask_"+tag+".pk", "wb") as file_pk:
    pickle.dump(MCAL_Mask, file_pk)

with open(out_dir+"GOLD_Sort_"+tag+".pk", "wb") as file_pk:
    pickle.dump(GOLD_Sort, file_pk)

with open(out_dir+"MCAL_Sort_"+tag+".pk", "wb") as file_pk:
    pickle.dump(MCAL_Sort, file_pk)


print(time.ctime())



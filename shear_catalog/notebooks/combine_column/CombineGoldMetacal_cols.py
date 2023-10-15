# this is the final script for DR3_1

import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

cat_type = sys.argv[1]
col_name = sys.argv[2]

metadata1 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]
metadata2 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_2.txt', dtype='str', delimiter=",")[1:]

import time

print(time.ctime())

shear_dir1 = '/project/chihway/data/decade/shearcat_v2/'
gold_dir1 = '/project/chihway/data/decade/coaddcat_v6/'

shear_dir2 = '/project/chihway/data/decade/shearcat_v3/'
gold_dir2 = '/project/chihway/data/decade/coaddcat_v7/'

path = '/project/chihway/data/decade/' 
out_file = '/project/chihway/data/decade/metacal_gold_columns/metacal_gold_combined_20230919_'+str(col_name)+'.hdf'

Ntile1 = len(metadata1)
Ntile2 = len(metadata2)

GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}

# read in pre-saved masks
import pickle

with open(path+"GOLD_Mask.pk", "rb") as file_pk:
    GOLD_Mask = pickle.load(file_pk)

with open(path+"MCAL_Mask.pk", "rb") as file_pk:
    MCAL_Mask = pickle.load(file_pk)

with open(path+"GOLD_Sort.pk", "rb") as file_pk:
    GOLD_Sort = pickle.load(file_pk)

with open(path+"MCAL_Sort.pk", "rb") as file_pk:
    MCAL_Sort = pickle.load(file_pk)

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


with h5py.File(out_file, "w") as f:

    if cat_type=='mcal':
        f.create_dataset(col_name, data = get_column_mcal(col_name))

    if cat_type=='gold':
        f.create_dataset(col_name, data = get_column_gold(col_name))


print(time.ctime())



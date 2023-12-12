# this is a temporary script for DR3_1, where part of DE3_1_2 does not have fitvd

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

out_file = '/project/chihway/data/decade/metacal_gold_columns_20231212/metacal_gold_combined_20231212_'+str(col_name)+'.hdf'

# read in pre-saved masks
import pickle

with open(path+"GOLD_Mask0_20231212.pk", "rb") as file_pk:
    GOLD_Mask0 = pickle.load(file_pk)

with open(path+"GOLD_Mask_20231212.pk", "rb") as file_pk:
    GOLD_Mask = pickle.load(file_pk)

with open(path+"MCAL_Mask_20231212.pk", "rb") as file_pk:
    MCAL_Mask = pickle.load(file_pk)

with open(path+"GOLD_Sort_20231212.pk", "rb") as file_pk:
    GOLD_Sort = pickle.load(file_pk)

with open(path+"MCAL_Sort_20231212.pk", "rb") as file_pk:
    MCAL_Sort = pickle.load(file_pk)

def get_column_mcal(column):

    output = []

    for X in range(3):

        # range(Ntile[X])
        for i in tqdm(range(10), desc = column):

            tile = Metadata[X][i][0]

            if os.path.exists(Shear_dir[X]+'metacal_output_'+tile+'.fits') and os.path.exists(Gold_dir[X]+'gold_'+tile+'.fits'):
            
                fits = fitsio.FITS(Shear_dir[X]+'metacal_output_'+tile+'.fits')
                shear_cat = fits[1].read(vstorage='object')
        
                arr = shear_cat[column][MCAL_Mask[tile]][MCAL_Sort[tile]]
                output.append(arr)

    return np.concatenate(output, axis = 0)


def get_column_gold(column):

    output = []

    for X in range(2):
        #range(Ntile[X])
        for i in tqdm(range(10), desc = column): #6537

            tile = Metadata[X][i][0]

            if os.path.exists(Shear_dir[X]+'metacal_output_'+tile+'.fits') and os.path.exists(Gold_dir[X]+'gold_'+tile+'.fits'):

                fits = fitsio.FITS(Gold_dir[X]+'gold_'+tile+'.fits')
                gold_cat = fits[1].read(vstorage='object')

                arr = gold_cat[column][GOLD_Mask[tile]][GOLD_Sort[tile]]

                output.append(arr)

    X = 2
    #range(Ntile[X])
    for i in tqdm(range(10), desc = column): #6537

        tile = Metadata[X][i][0]

        if os.path.exists(Shear_dir[X]+'metacal_output_'+tile+'.fits') and os.path.exists(Gold_dir[X]+'gold_'+tile+'.hdf5'):

            with h5py.File(Gold_dir[X]+'gold_'+tile+'.hdf5', 'r') as gold_cat:
                gold_col = gold_cat[column][:]

            arr = gold_col[GOLD_Mask0[tile]][GOLD_Mask[tile]][GOLD_Sort[tile]]

            output.append(arr)


    return np.concatenate(output, axis = 0)


with h5py.File(out_file, "w") as f:

    if cat_type=='mcal':
        f.create_dataset(col_name, data = get_column_mcal(col_name))

    if cat_type=='gold':
        f.create_dataset(col_name, data = get_column_gold(col_name))


print(time.ctime())



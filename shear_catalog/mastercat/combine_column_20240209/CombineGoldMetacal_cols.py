# this is to combine all tiles for one column

import numpy as np
import sys
import os
#sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py
from tqdm import tqdm

cat_type = sys.argv[1]
col_name = sys.argv[2]

import time
print(time.ctime())

tag = '20240209'
metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/final_'+tag+'/Tilelist_final_DR3_1.csv', dtype='str', delimiter=",")[1:]
ntile = len(metadata)
shear_dir = '/project/chihway/data/decade/shearcat_final/'
gold_dir = '/project/chihway/data/decade/coaddcat_final/'
out_dir = '/project/chihway/data/decade/'

GOLD_Mask = {}
MCAL_Mask = {}
GOLD_Sort = {}
MCAL_Sort = {}

out_file = '/project/chihway/data/decade/metacal_gold_columns_'+tag+'/metacal_gold_combined_'+tag+'_'+str(col_name)+'.hdf'

# read in pre-saved masks
import pickle

with open(out_dir+"GOLD_Mask_"+tag+".pk", "rb") as file_pk:
    GOLD_Mask = pickle.load(file_pk)

with open(out_dir+"MCAL_Mask_"+tag+".pk", "rb") as file_pk:
    MCAL_Mask = pickle.load(file_pk)

with open(out_dir+"GOLD_Sort_"+tag+".pk", "rb") as file_pk:
    GOLD_Sort = pickle.load(file_pk)

with open(out_dir+"MCAL_Sort_"+tag+".pk", "rb") as file_pk:
    MCAL_Sort = pickle.load(file_pk)

def get_column_mcal(column):

    output = []

    for i in tqdm(range(ntile), desc = column):

        tile = metadata[i][0]

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir+'gold_'+tile+'.fits'):
            
            fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
            shear_cat = fits[1].read(vstorage='object')
        
            arr = shear_cat[column][MCAL_Mask[tile]][MCAL_Sort[tile]]
            output.append(arr)

    return np.concatenate(output, axis = 0)


def get_column_gold(column):

    output = []

    for i in tqdm(range(ntile), desc = column): #6537

        tile = metadata[i][0]

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(gold_dir+'gold_'+tile+'.fits'):

            fits = fitsio.FITS(gold_dir+'gold_'+tile+'.fits')
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



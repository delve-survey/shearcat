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

columns = [#'dec',  'ra', #Not using ra dec here as we'll add it manually later (since it comes from SrcExtractor cat)
           'badfrac', 'id',
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


shear_dir = '/project2/chihway/data/decade/shearcat_v2/'
coadd_dir = '/project2/chihway/data/decade/coaddcat_v2/'

path = '/project2/chihway/data/decade/metacal_test_20230427_v3.hdf'


GOLD_Mask = {}
ra  = []
dec = []

for i in tqdm(range(6537), desc = 'Build GoldMask & RADEC'): #6357

    tile = metadata[i][0]

    if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(coadd_dir+'gold_mask_'+tile+'.npz'):

        fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')

        shear_cat = fits[1].read(vstorage='object')
        # metacal output files, it has an ID that is the correct coadd IDs, but already
        # with a cut that only includes objects that have a metacal measurement

        shear_id = np.load(shear_dir+'ids_match_'+tile+'.npz', allow_pickle=True)
        # ad hoc file we save when running metacal, the full list of coadd IDs for all objects in the MEDS file
        # these are row-by-row matched to the catalog files that contain e.g. photometry

        gold_mask = np.load(coadd_dir+'gold_mask_'+tile+'.npz', allow_pickle=True)
        gold_mask_full = gold_mask['maskSE'] & gold_mask['maskIMA'] & (gold_mask['maskSG']>=2)
        # from the catalog files, we make the gold cut
        # still using rough star/galaxy separation

        shear_id_masked_gold = shear_id['ids'][gold_mask_full]
        # this is IDs of all the objects that pass the gold cut

        mask_joint = np.in1d(shear_cat['id'], shear_id_masked_gold)
        # mask to apply on the metacal catalog, to remove objects that don't pass the gold cut

        mask_joint_invert = np.in1d(shear_id_masked_gold, shear_cat['id'][mask_joint])
        # invert mask to apply to the original tile catalog, to selecout the final objects

        ra_tmp  = gold_mask['ra'][gold_mask_full][mask_joint_invert]
        dec_tmp = gold_mask['dec'][gold_mask_full][mask_joint_invert]

        ra.append(ra_tmp)
        dec.append(dec_tmp)
        
        GOLD_Mask[tile] = mask_joint

        out2 = open("good_tiles.txt", "a")
        out2.write(str(i)+'\t'+tile+'\n')
        out2.close()

    else:
        print('tile missing '+tile)
        out = open("bad_tiles.txt", "a")
        out.write(str(i)+'\t'+tile+'\n')
        out.close()

def get_column(column):

    output = []
    for i in tqdm(range(6537), desc = column): #6357

        tile = metadata[i][0]

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(coadd_dir+'gold_mask_'+tile+'.npz'): 
            
            fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
            shear_cat = fits[1].read(vstorage='object')
        
            arr = shear_cat[column][GOLD_Mask[tile]]
            
            #Hardcoding this in because ccdnum and x_exp/y_exp has too large a datatype
            #Wont have enough memory to create the final array (more than 50GB).
            if column == 'ccdnum':
                arr = arr.astype(np.int16)
            elif (column == 'x_exp') | (column == 'y_exp'):
                arr = arr.astype(np.float32)
            
            output.append(arr)

    return np.concatenate(output, axis = 0)


with h5py.File(path, "w") as f:

    for c in columns:
        f.create_dataset(c, data = get_column(c))

    #Now add ra_dec
    f.create_dataset('ra',  data = np.concatenate(ra))
    f.create_dataset('dec', data = np.concatenate(dec))


print(time.ctime())

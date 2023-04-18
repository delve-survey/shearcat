
import numpy as np
import sys
import os
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages/')
import astropy.io.fits as pf
import fitsio
import h5py

metadata = np.genfromtxt('../Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:]

import time

print(time.ctime())

Columns = ['dec', 'badfrac', 'id', 
           'mcal_T_1m', 'mcal_T_1p', 'mcal_T_2m', 'mcal_T_2p', 
           'mcal_T_noshear', 'mcal_T_ratio_1m', 'mcal_T_ratio_1p', 'mcal_T_ratio_2m',
           'mcal_T_ratio_2p', 'mcal_T_ratio_noshear', 'mcal_flags', 'mcal_flux_1m', 'mcal_flux_1p',
           'mcal_flux_2m', 'mcal_flux_2p', 'mcal_flux_err_1m', 'mcal_flux_err_1p',
           'mcal_flux_err_2m', 'mcal_flux_err_2p', 'mcal_flux_err_noshear', 'mcal_flux_noshear',
           'mcal_g_1m', 'mcal_g_1p', 'mcal_g_2m', 'mcal_g_2p', 'mcal_g_cov_1m', 'mcal_g_cov_1p',
           'mcal_g_cov_2m', 'mcal_g_cov_2p', 'mcal_g_cov_noshear', 'mcal_g_noshear', 
           'mcal_psf_T_noshear', 'mcal_psf_g_noshear', 'mcal_s2n_1m', 'mcal_s2n_1p', 'mcal_s2n_2m',
           'mcal_s2n_2p', 'mcal_s2n_noshear', 'ra', 'x', 'y', 
           'Ncutouts_raw', 'ccdnum', 'x_exp', 'y_exp']


shear_dir = '/project2/chihway/data/decade/shearcat_v2/'
coadd_dir = '/project2/chihway/data/decade/coaddcat_v2/'

path = '/project2/chihway/data/decade/metacal_test_20230323.hdf'

with h5py.File(path, "w") as f:

    #Create all columns you need
    test_tile = metadata[1][0]
    test_fits = fitsio.FITS(shear_dir+'metacal_output_'+test_tile+'.fits')
    
    for col in Columns:
        if col!='ra' and col!='dec':
            Ndim = len(test_fits[1].read(vstorage='object')[col].shape)
            if Ndim==1:
                type_col = str(type(test_fits[1].read(vstorage='object')[col][0])).replace('class', '').replace("< '", "").replace("'>", "").replace("numpy.", "")
                f.create_dataset(col, data = [], chunks=(10**4,), maxshape = (None,), dtype=type_col)
            if Ndim==2:
                type_col = str(type(test_fits[1].read(vstorage='object')[col][0][0])).replace('class', '').replace("< '", "").replace("'>", "").replace("numpy.", "")
                f.create_dataset(col, data = [[]], chunks=((10**4,20)), maxshape = (None,20), dtype=type_col)
            if Ndim==3:
                type_col = str(type(test_fits[1].read(vstorage='object')[col][0][0][0])).replace('class', '').replace("< '", "").replace("'>", "").replace("numpy.", "")
                f.create_dataset(col, data = [[[]]], chunks=((10**4,20,20)), maxshape = (None,20,20), dtype=type_col)

    f.create_dataset('ra', data = [], chunks=(10**4,), maxshape = (None,))            
    f.create_dataset('dec', data = [], chunks=(10**4,), maxshape = (None,))            

    # Appends new_data array into existing dataset
    def add_data(dataset, new_data):

        if len(new_data.shape)==1:
            dataset.resize(dataset.shape[0] + len(new_data), axis=0)
            dataset[-len(new_data):] = new_data

        if len(new_data.shape)==2:
            if dataset.shape[0]==1:
                dataset.resize((dataset.shape[0] + len(new_data)-1,new_data.shape[1]))
            else:
                dataset.resize((dataset.shape[0] + len(new_data),new_data.shape[1]))
            dataset[-len(new_data):] = new_data

        if len(new_data.shape)==3:
            if dataset.shape[0]==1:
                dataset.resize((dataset.shape[0] + len(new_data)-1,new_data.shape[1],new_data.shape[2]))
            else:
                dataset.resize((dataset.shape[0] + len(new_data),new_data.shape[1],new_data.shape[2]))
            dataset[-len(new_data):] = new_data


    for i in range(6357): #6357
        tile = metadata[i][0]
        print(tile)

        if os.path.exists(shear_dir+'metacal_output_'+tile+'.fits') and os.path.exists(coadd_dir+'gold_mask_'+tile+'.npz'):
            fits = fitsio.FITS(shear_dir+'metacal_output_'+tile+'.fits')
 
            shear_cat = fits[1].read(vstorage='object')
            # metacal output files, it has an ID that is the correct coadd IDs, but already 
            # with a cut that only includes objects that have a metacal measurement

            shear_id = np.load(shear_dir+'ids_match_'+tile+'.npz', allow_pickle=True)
            # ad hoc file we save when running metacal, the full list of coadd IDs for all objects in the MEDS file
            # these are row-by-row matched to the catalog files that contain e.g. photometry

            gold_mask = np.load(coadd_dir+'gold_mask_'+tile+'.npz', allow_pickle=True)
            gold_mask_full = gold_mask['maskSE']*gold_mask['maskIMA']*(gold_mask['maskSG']>=2) 
            # from the catalog files, we make the gold cut
            # still using rough star/galaxy separation

            shear_id_masked_gold = shear_id['ids'][gold_mask_full]
            # this is IDs of all the objects that pass the gold cut

            mask_joint = np.in1d(shear_cat['id'], shear_id_masked_gold)
            # mask to apply on the metacal catalog, to remove objects that don't pass the gold cut

            mask_joint_invert = np.in1d(shear_id_masked_gold, shear_cat['id'][mask_joint])
            # invert mask to apply to the original tile catalog, to selecout the final objects

            ra = gold_mask['ra'][gold_mask_full][mask_joint_invert]
            dec = gold_mask['dec'][gold_mask_full][mask_joint_invert]
            add_data(f['ra'], ra)
            add_data(f['dec'], dec)
            
            for col2 in Columns:
                if col2!='ra' and col2!='dec':
                    data2 = shear_cat[col2][mask_joint]
                    add_data(f[col2], data2)
                    if col2=='id':
                        print(i, len(f[col2]))

        else:
            print('tile missing '+tile)

print(time.ctime())





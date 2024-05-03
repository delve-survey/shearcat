
import sys
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

import numpy as np
import h5py
import healpy as hp

tag = '20240209'
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_'+tag+'.hdf'
master_mask = project_dir+'metacal_gold_combined_mask_'+tag+'.hdf'


mask_total_X = {}
with h5py.File(master_mask, 'r') as h5r:
    mask_total_X['noshear'] = h5r['baseline_mcal_mask_noshear'][:]
    mask_total_X['1p'] = h5r['baseline_mcal_mask_1p'][:]
    mask_total_X['1m'] = h5r['baseline_mcal_mask_1m'][:]
    mask_total_X['2p'] = h5r['baseline_mcal_mask_2p'][:]
    mask_total_X['2m'] = h5r['baseline_mcal_mask_2m'][:]

with h5py.File(master_cat, 'r') as h5r:
    ids_mcal = h5r['id'][:]

# read in tomography catalogs
#RES = np.load('/project/chihway/raulteixeira/data/ID_MATCHED_DR3_1_20240123.npy')
#ids = RES[:,0]
#tomo = RES[:, 1] #Tomobins from Raul.
#tomo = tomo[ids>-1] + 1.
#on_mask = (mask_total_X['noshear']==1)
#mask_total_X['noshear'][on_mask] = tomo.copy()

# read in tomography catalogs
#X = pd.read_hdf('/project/chihway/dhayaa/DECADE/SOMPZ/For_Chihway/BinAssignments_20240209.hdf5', key = 'data')

#with h5py.File('/project/chihway/dhayaa/DECADE/SOMPZ/For_Chihway/BinAssignments_20240209.hdf5','r') as f:
#    ids = f['data/block0_values'][:,0]
#    tomo = f['data/block0_values'][:,2]
#tomo = tomo[ids>-1] + 1.
#on_mask = (mask_total_X['noshear']==1)
#mask_total_X['noshear'][on_mask] = tomo.copy()


for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
    print(shear_type)
#    RES = np.load('/project/chihway/raulteixeira/data/ID_MATCHED_'+str(shear_type)+'_DR3_1_20240123.npy')
#    ids = RES[:,0]
#    tomo = RES[:, 1] #Tomobins from Raul.
#    tomo = tomo[ids>-1] + 1.
#    on_mask = (mask_total_X[shear_type]==1)
#    mask_total_X[shear_type][on_mask] = tomo.copy()
#    print(mask_total_X[shear_type][on_mask])
#    print(tomo)

    with h5py.File('/project/chihway/dhayaa/DECADE/SOMPZ/For_Chihway/BinAssignments_'+str(shear_type)+'_20240209.hdf5','r') as f:
        ids = f['data/block0_values'][:,0]
        tomo = f['data/block0_values'][:,2]
    tomo = tomo[ids>-1] + 1.
    on_mask = (mask_total_X[shear_type]==1)
    mask_total_X[shear_type][on_mask] = tomo.copy()
    print(ids, ids_mcal[on_mask])
    print("IDs matched?", np.allclose(ids, ids_mcal[on_mask]))


with h5py.File(master_cat, 'a') as h5r:
    for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
        del h5r['baseline_mcal_mask_'+shear_type]    
        h5r.create_dataset('baseline_mcal_mask_'+shear_type, data = mask_total_X[shear_type])



import pandas as pd, h5py

Raul = pd.read_hdf('/project/chihway/raulteixeira/data/DR3_1_ID+TomoBin.hdf5')

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20231212.hdf') as f:

    ID = f['id'][:]

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_mask_20231212.hdf') as f:
    
    mask = f['baseline_mcal_mask_noshear'][:]
    
match, ind_r, ind_c = np.intersect1d(Raul.id.values, ID, return_indices = True)


tomo_bin = np.zeros(ID.size) + -1
ID_bin   = np.zeros(ID.size) + -1


tomo_bin[ind_c] = Raul.TomoBin.values[ind_r]
ID_bin[ind_c]   = Raul.id.values[ind_r]


np.save('/project/chihway/raulteixeira/data/ID_MATCHED_DR3_1_20240123.npy', np.vstack([ID_bin, tomo_bin]).T)

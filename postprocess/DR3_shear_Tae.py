import numpy as np, pandas as pd
import h5py

DR3_1 = pd.DataFrame()
DR3_2 = pd.DataFrame()

TARGET_RA, TARGET_DEC = 37.86501659859067, 6.982204815599694

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20241003.hdf', 'r') as f:
    MASK = f['baseline_mcal_mask_noshear'][:] > 0
    BINS = f['baseline_mcal_mask_noshear'][:][MASK] 

    Rt   = np.array([0.8445, 0.7789, 0.748, 0.627]) #From my Fiducial/ShearSummary.txt files

    DR3_2['RA']   = f['RA'][:][MASK]
    DR3_2['DEC']  = f['DEC'][:][MASK]
    DR3_2['g1']   = f['mcal_g_noshear'][:, 0][MASK] #/ Rt[BINS - 1]
    DR3_2['g2']   = f['mcal_g_noshear'][:, 1][MASK] #/ Rt[BINS - 1]
    DR3_2['BIN']  = BINS
    DR3_2['mcal_g_w'] = f['mcal_g_w_noshear'][:][MASK]

    flux = f['mcal_flux_noshear'][:][MASK]
    DR3_2['mcal_flux_r'] = flux[:, 0]
    DR3_2['mcal_flux_i'] = flux[:, 1]
    DR3_2['mcal_flux_z'] = flux[:, 2]

    msk = (np.abs(DR3_2['RA'].values - TARGET_RA) < 5) & (np.abs(DR3_2['DEC'].values - TARGET_DEC) < 5)

    DR3_2 = DR3_2.iloc[msk].reset_index(drop = True)

    print(DR3_2)

with h5py.File('/project/chihway/dhayaa/DECADE/MassMaps/ShearCatalogDR3_A360_NoResponse.hdf5', 'w') as f:

    for k in DR3_2.keys(): 
        f.create_dataset(name = k, data = DR3_2[k])
        print("WRITTEN", k)





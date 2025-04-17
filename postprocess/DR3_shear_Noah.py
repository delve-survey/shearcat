import numpy as np, pandas as pd
import h5py

DR3_1 = pd.DataFrame()
DR3_2 = pd.DataFrame()

rng = np.random.default_rng(seed = 42)

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20240209.hdf', 'r') as f:
    MASK = f['baseline_mcal_mask_noshear'][:] > 0
    MASK = rng.choice(np.where(MASK)[0], replace = False, size = 10_000_000)
    BINS = f['baseline_mcal_mask_noshear'][:][MASK] 

    Rt   = np.array([0.836, 0.770, 0.740, 0.620]) #From my Fiducial/ShearSummary.txt files

    DR3_1['RA']    = f['RA'][:][MASK]
    DR3_1['DEC']   = f['DEC'][:][MASK]
    DR3_1['BDF_T'] = f['BDF_T'][:][MASK]
    
    for b in 'GRIZ':
        DR3_1[f'BDF_FLUX_{b}'] = f[f'BDF_FLUX_{b}'][:][MASK]


print("READ DR3_1")

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20241003.hdf', 'r') as f:
    MASK = f['baseline_mcal_mask_noshear'][:] > 0
    MASK = rng.choice(np.where(MASK)[0], replace = False, size = 10_000_000)
    BINS = f['baseline_mcal_mask_noshear'][:][MASK] 

    Rt   = np.array([0.8445, 0.7789, 0.748, 0.627]) #From my Fiducial/ShearSummary.txt files

    DR3_2['RA']    = f['RA'][:][MASK]
    DR3_2['DEC']   = f['DEC'][:][MASK]
    DR3_2['BDF_T'] = f['BDF_T'][:][MASK]
    
    for b in 'GRIZ':
        DR3_2[f'BDF_FLUX_{b}'] = f[f'BDF_FLUX_{b}'][:][MASK]

print("NGC:", len(DR3_1))
print("SGC:", len(DR3_2))

with h5py.File('/project/chihway/dhayaa/DECADE/MassMaps/Noah_Catalog.hdf5', 'w') as f:

    fNGC = f.create_group(name = 'NGC')
    fSGC = f.create_group(name = 'SGC')
    
    for k in DR3_1.keys(): 
        fNGC.create_dataset(name = k, data = DR3_1[k])
        fSGC.create_dataset(name = k, data = DR3_2[k])

        print("WRITTEN", k)





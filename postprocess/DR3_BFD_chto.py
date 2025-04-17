import numpy as np, pandas as pd
import h5py

DR3_1 = pd.DataFrame()
DR3_2 = pd.DataFrame()

BANDS = 'GRIZ'
keys  = []
keys += [f'BDF_FLUX_{B}_DERED_SFD98' for B in BANDS]
keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20240209.hdf', 'r') as f:
    GLD = ( (f['FLAGS_SG_BDF'][:] >= 2) & 
            (f['FLAGS_FOREGROUND'][:] == 0) &
            (f['FLAGS_FOOTPRINT'][:] == 0) &
            (f['FLAGS_BAD_COLOR'][:] == 0)
            )
    for k in keys: DR3_1[k] = f[k][:][GLD]; print("READ", k)

print("READ DR3_1")

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20241003.hdf', 'r') as f:
    GLD = ( (f['FLAGS_SG_BDF'][:] >= 2) & 
            (f['FLAGS_FOREGROUND'][:] == 0) &
            (f['FLAGS_FOOTPRINT'][:] == 1) &
            (f['FLAGS_BAD_COLOR'][:] == 0)
            )
    for k in keys: DR3_2[k] = f[k][:][GLD]; print("READ", k)

print("READ DR3_2")

DR3 = pd.concat([DR3_1, DR3_2]); del DR3_1, DR3_2
print("SIZE :", len(DR3))

with h5py.File('/project/chto/dhayaa/decade/DR3_combined_20250313.hdf', 'w') as f:

    for k in DR3.keys():
        f.create_dataset(name = k, data = DR3[k])


        print("WRITTEN", k)




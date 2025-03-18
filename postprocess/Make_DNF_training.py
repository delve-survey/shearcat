import numpy as np
import pandas as pd
import fitsio
import glob
from sklearn.neighbors import BallTree
from tqdm import tqdm
import joblib, os

Julia = pd.read_csv('/project/chihway/dhayaa/DECADE/BRPORTAL_E_6315_18670.csv', low_memory = False)

DESI  = []
Names = ['BGS_ANY', 'ELGnotqso', 'LRG', 'QSO']
for n in Names:
    paths = (glob.glob(f'/project2/chihway/dhayaa/DESI/{n}_N_clustering.dat.fits') + 
             glob.glob(f'/project2/chihway/dhayaa/DESI/{n}_S_clustering.dat.fits'))
    for p in tqdm(paths, desc = n):
        X = fitsio.read(p)
        DESI.append(pd.DataFrame({'RA' : X['RA'], 'DEC' : X['DEC'], 'SOURCE' : 'DESI_' + n, 'Z' : X['Z']}))
    
DESI = pd.concat(DESI, ignore_index = True)

Tree = BallTree(np.vstack([Julia['DEC'], Julia['RA']]).T * np.pi/180, leaf_size = 2, metric = "haversine")

d = []
j = []

Nsize  = len(DESI)
N_per_batch = 500_000
Nbatches = Nsize // N_per_batch + 1

def one_step(i):

    s = slice(i*N_per_batch, (i+1)*N_per_batch)

    r = Tree.query(np.vstack([DESI['DEC'].values[s], DESI['RA'].values[s]]).T * np.pi/180, k = 1)
    
    return r

res = joblib.Parallel(n_jobs = os.cpu_count(), verbose = 10)(joblib.delayed(one_step)(i) for i in range(Nbatches))

for r in res:
    d.append(r[0][:, 0])
    j.append(r[1][:, 0])

d = np.concatenate(d) * 180/np.pi * 60*60 #convert to arcsec
j = np.concatenate(j)

print(f"FOUND {np.sum(d < 0.5)} MATCHES")

print(Julia['SOURCE'][j[d < 0.5]].value_counts())
print(Julia['SOURCE'].value_counts())

Julia = Julia.drop(index = np.unique(j[d < 0.5]).astype(int))
Julia = pd.concat([Julia, DESI])

Julia.to_csv('/project2/chihway/dhayaa/DNF/Spec_z_sample.csv', index = False)


Julia = Julia[['RA', 'DEC', 'Z']]
Julia = np.array(list(Julia.to_records(index=False)), dtype=[(col, Julia[col].dtype) for col in Julia.columns])
Julia = Julia[Julia['DEC'] < 40]
np.save('/project2/chihway/dhayaa/DNF/DECCUT_Spec_z_sample.npy', Julia)

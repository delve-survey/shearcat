import numpy as np, pandas as pd
import h5py, fitsio, healpy as hp
from sklearn.neighbors import BallTree
import joblib, time, sys, os
from mpi4py import MPI
sys.path.insert(0, os.path.dirname(__file__) + '/dnf/')
import juan_dnf, time
from tqdm import tqdm
import argparse

my_parser = argparse.ArgumentParser()

#Cosmology params
my_parser.add_argument('--FITVD',   action='store_true', default = False, help = 'Run on fitvd fluxes')
my_parser.add_argument('--METACAL', action='store_true', default = False, help = 'Run on metacal fluxes')

args = vars(my_parser.parse_args())

assert args['FITVD'] + args['METACAL'] == 1, "Can only use one of FITVD or METACAL flags"

if args['FITVD']:
    name = 'fitvd'
elif args['METACAL']:
    name = 'metacal'

print("USING FLUXES FROM", name.upper())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of processes

############################################
# STEP 1: POSITION MATCH FOR TRAINING
############################################
if rank < size//2:
    file = '/project/chihway/data/decade/metacal_gold_combined_20240209.hdf'
else:
    file = '/project/chihway/data/decade/metacal_gold_combined_20241003.hdf'

#Virtual hdf5 file to use for randomized indices (to help load balancing)
vds_file = file.replace('/project/chihway/data/decade/metacal_', '/scratch/midway3/dhayaa/random_metacal_')

DATACOLUMNS  = ['COADD_OBJECT_ID', 'FLAGS_SG_BDF', 'FLAGS_FOREGROUND', 'FLAGS_FOOTPRINT', 'FLAGS_BAD_COLOR', 'RA', 'DEC']
BANDS = 'GRIZ'
DATACOLUMNS += [f'BDF_FLUX_{b}_DERED_SFD98' for b in BANDS]
DATACOLUMNS += [f'BDF_FLUX_ERR_{b}_DERED_SFD98' for b in BANDS]
# DATACOLUMNS += [f'mcal_flux_noshear_dered_sfd98', 'mcal_flux_err_noshear_dered_sfd98']
DATACOLUMNS += ['BDF_S2N']

p = f'/scratch/midway3/dhayaa/DNF_{name}_training_data.npy'
if os.path.isfile(p):
    training_data = np.load(p)
else:

    # SPECZ_sample = fitsio.read('/project/chihway/data/decade/BOSS_eBOSS.fits')
    SPECZ_sample = np.load('/project2/chihway/dhayaa/DNF/DECCUT_Spec_z_sample.npy')[:]
    SOURCE       = pd.read_csv('/project2/chihway/dhayaa/DNF/Spec_z_sample.csv', usecols = ['SOURCE', 'DEC'])
    SOURCE       = SOURCE[SOURCE['DEC'] < 40]
    SOURCE       = SOURCE['SOURCE'].values

    print(SOURCE)
    print(f"STARTING WITH {len(SPECZ_sample)} SAMPLES")

    MSK = np.invert(np.isin(SOURCE, ['DESI_BGS_BRIGHT-21.5', 'DESI_ELG_LOPnotqso', 'DESI_LRG', 'DESI_QSO']))
    MSK = np.isin(SOURCE, ['DESI_BGS_BRIGHT-21.5', 'DESI_ELG_LOPnotqso', 'DESI_LRG', 'DESI_QSO'])
    MSK = np.ones_like(MSK, dtype = bool)
    SPECZ_sample = SPECZ_sample[MSK]

    print(f"NOW HAVE WITH {len(SPECZ_sample)} SAMPLES")
    tree = BallTree(np.vstack([SPECZ_sample['DEC'], SPECZ_sample['RA']]).T * np.pi/180, leaf_size = 1, metric = "haversine")

    print("Load spec-z sample")


    if (rank == 0) or (rank == size - 1):

        if os.path.isfile(vds_file):
            pass
        else:
            with h5py.File(file, 'r') as f:
                total_size = f['RA'].shape[0]
                indices    = np.random.default_rng(seed = 42).choice(total_size, total_size, replace = False)
                
                with h5py.File(vds_file, 'w') as vds_f:
                    for dataset_name in DATACOLUMNS:
                        
                        vds_f.create_dataset(dataset_name, data = f[dataset_name][:][indices])

                        print(dataset_name, vds_file)
                        print("ORIG", vds_f[dataset_name][:10])
                        print("RAND", f[dataset_name][:10])

                del indices

    comm.Barrier()
    with h5py.File(vds_file, 'r') as f:
        
        Nbatch = size//2
        i_here = rank % (size//2) 
        N_per_batch = f['RA'].size // Nbatch + 1
        s   = slice(i_here * N_per_batch, (i_here + 1) * N_per_batch)
        
        GLD = ( (f['FLAGS_SG_BDF'][s] >= 2) & 
                (f['FLAGS_FOREGROUND'][s] == 0) & 
                (f['FLAGS_FOOTPRINT'][s] == (0 if '0209' in file else 1)) & 
                (f['FLAGS_BAD_COLOR'][s] == 0) & 
                (f['BDF_S2N'][s] > 5)
                )
        
        ra  = f['RA'][s][GLD]
        dec = f['DEC'][s][GLD]

        if args['FITVD']:
            m   = 30 - 2.5 * np.log10([f[f'BDF_FLUX_{b}_DERED_SFD98'][s][GLD] for b in BANDS]).T
            dm  = 2.5/np.log(10) * np.array([f[f'BDF_FLUX_ERR_{b}_DERED_SFD98'][s][GLD] / f[f'BDF_FLUX_{b}_DERED_SFD98'][s][GLD] for b in BANDS]).T
        
        elif args['METACAL']:
            m   = 30 - 2.5 * np.log10(f['mcal_flux_noshear_dered_sfd98'][s][GLD])
            dm  = 2.5/np.log(10) * np.array(f['mcal_flux_err_noshear_dered_sfd98'][s][GLD] / f['mcal_flux_noshear_dered_sfd98'][s][GLD])
            
        ts = time.time()

        if comm.Iprobe(source=MPI.ANY_SOURCE): comm.Barrier()

        batch_size = 10_000
        Nbatch     = dec.size // batch_size + 1

        d = []
        j = []
        for i in tqdm(range(Nbatch), desc = f'RANK {rank}'):
            s = slice(i*batch_size, (i+1)*batch_size)
            r = tree.query(np.vstack([dec[s], ra[s]]).T * np.pi/180)
            d.append(r[0][:, 0])
            j.append(r[1][:, 0])

        del tree

        d = np.concatenate(d)
        j = np.concatenate(j)
        d = d * 180/np.pi * 60*60 #convert to arcsec

        #Keep only ids below 0.5 arcsec
        Mask = d < 0.5
        j    = j[Mask]

        m    = m[Mask]
        dm   = dm[Mask]
        z    = SPECZ_sample['Z'][j]

        #Find duplicates and keep only non-duplicated ones
        j, inds, counts = np.unique(j, return_counts = True, return_index = True)
        Mask = counts == 1

        m    = m[inds][Mask]
        dm   = dm[inds][Mask]
        z    = z[inds][Mask]
        j    = j[Mask]

        del ra, dec

    if comm.Iprobe(source=MPI.ANY_SOURCE): comm.Barrier()

    # Step 1: Gather the sizes of each local array
    local_size = np.array(len(z), dtype=np.int32)
    all_sizes  = np.zeros(size, dtype=np.int32) 
    comm.Gather(local_size, all_sizes, root = 0)
    comm.Bcast(all_sizes, root = 0)  # Ensure all ranks know sizes

    if comm.Iprobe(source=MPI.ANY_SOURCE): comm.Barrier()

    # Step 2: Compute displacements (only needed on root)
    displacements     = np.zeros(size, dtype=np.int32)
    displacements[1:] = np.cumsum(all_sizes)[:-1]  # Compute starting indices

    # Prepare buffer for full array
    total_size    = np.sum(all_sizes)
    Ncols         = m.shape[1] + dm.shape[1] + 1
    training_data = np.empty(total_size * Ncols, dtype=np.float64)

    # Step 3: Gather variable-length arrays using Gatherv
    tmp = np.concatenate([m, dm, z[:, None]], axis = 1).flatten()
    comm.Gatherv(tmp, (training_data, all_sizes * Ncols, displacements * Ncols, MPI.DOUBLE), root = 0)

    inds = np.empty(total_size, dtype=np.int64)
    comm.Gatherv(j, (inds, all_sizes, displacements, MPI.SIGNED_LONG_LONG), root = 0)
    comm.Bcast(inds, root = 0)

    if comm.Iprobe(source=MPI.ANY_SOURCE): comm.Barrier()

    # Step 4: Broadcast back the training data
    comm.Bcast(training_data, root = 0)
    training_data = np.concatenate([x.reshape(-1, Ncols) for x in np.split(training_data, displacements[1:] * Ncols)], axis = 0)
    if comm.Iprobe(source=MPI.ANY_SOURCE): comm.Barrier()

    # Step 5: Make sure to only keep objects with no ambigous matches
    inds, unique_inds, counts = np.unique(inds, return_index = True, return_counts = True)
    training_data = training_data[unique_inds[counts == 1]]

    if rank == 0: 
        print(f"HAVE {counts.size} MATCHES AFTER REMOVING DUPLICATES")
        print(f"HAVE {np.sum(counts == 1)} MATCHES AFTER REMOVING DUPLICATES")
        np.save(f'/scratch/midway3/dhayaa/DNF_{name}_training_data.npy', training_data)
        np.save(f'/scratch/midway3/dhayaa/DNF_{name}_training_data_specz.npy', SPECZ_sample[inds])

    del SPECZ_sample

############################################
# STEP 2: LOAD THE MAGS
############################################

with h5py.File(file, 'r') as f:
    
    Nbatch = size//2
    i_here = rank % (size//2) 
    N_per_batch = f['RA'].size // Nbatch + 1
    s   = slice(i_here * N_per_batch, (i_here + 1) * N_per_batch)
    
    id  = f['COADD_OBJECT_ID'][s]
    if args['FITVD']:
        m   = 30 - 2.5 * np.log10([f[f'BDF_FLUX_{b}_DERED_SFD98'][s] for b in BANDS]).T
        dm  = 2.5/np.log(10) * np.array([f[f'BDF_FLUX_ERR_{b}_DERED_SFD98'][s] / f[f'BDF_FLUX_{b}_DERED_SFD98'][s] for b in BANDS]).T
    elif args['METACAL']:
        m   = 30 - 2.5 * np.log10(f['mcal_flux_noshear_dered_sfd98'][s])
        dm  = 2.5/np.log(10) * np.array(f['mcal_flux_err_noshear_dered_sfd98'][s] / f['mcal_flux_noshear_dered_sfd98'][s])

train_mask    = np.all(np.isfinite(training_data[:, :m.shape[1]+dm.shape[1]]), axis = 1)
target_mask   = np.all(np.isfinite(m), axis = 1) & np.all(np.isfinite(dm), axis = 1)


############################################
# STEP 3: Test DNF
############################################
zbins = np.arange(0, 1.6, 0.01)
inds  = np.random.default_rng(seed = rank).choice(train_mask.sum(), train_mask.sum(), replace = False)

Nsize      = int(0.5*inds.size)
inds_train = inds[:Nsize]
inds_test  = inds[Nsize:]
Ntest      = len(inds_test)

chunksize = 5_000
Totalsize = inds_test.size
nchunk = Totalsize // chunksize + 1

tmp_z      = np.zeros(Ntest) - 99
tmp_z_err  = np.zeros(Ntest) - 99

# for i in range(nchunk):
#     ts  = time.time()
#     s   = slice(i*chunksize, (i+1)*chunksize)
#     RES = juan_dnf.dnf( training_data[train_mask, :m.shape[1]][inds_train], training_data[train_mask, -1][inds_train], 
#                         training_data[train_mask, :m.shape[1]][inds_test][s],  
#                         training_data[train_mask, m.shape[1]:(m.shape[1] + dm.shape[1])][inds_test][s], zbins,
#                         pdf = True, Nneighbors = 80, bound = False, radius = 2.0, magflux = 'mag', metric = "DNF",coeff = True)
    
#     tmp_z[s]      = RES[0]
#     tmp_z_err[s]  = RES[1]

#     print("RANK", rank, f"Finished chunk {i} of {nchunk} using {time.time()-ts} s", flush=True)

# print("I AM DONE", rank)

# z      = np.empty(Ntest * size, dtype = np.float64) if rank == 0 else None
# z_err  = np.empty(Ntest * size, dtype = np.float64) if rank == 0 else None
# z_true = np.empty(Ntest * size, dtype = np.float64) if rank == 0 else None
# ainds  = np.empty(Ntest * size, dtype = np.int64) if rank == 0 else None

# comm.Gather(tmp_z,     z,      root = 0)
# comm.Gather(tmp_z_err, z_err,  root = 0)
# comm.Gather(inds_test, ainds,  root = 0)
# comm.Gather(training_data[train_mask, -1][inds_test], z_true, root = 0)

# if rank == 0:
#     with h5py.File(f'/scratch/midway3/dhayaa/DECADE_DR3_DNF_{name}_validation_20250331_All.hdf', 'w') as f:
#         f.create_dataset('z',     data = z.reshape(Ntest, size))
#         f.create_dataset('zerr',  data = z_err.reshape(Ntest, size))
#         f.create_dataset('ztrue', data = z_true.reshape(Ntest, size))

#         for i, b in enumerate(BANDS):
#             f.create_dataset(f'MAG_{b}',     data = training_data[train_mask, i][ainds].reshape(Ntest, size))
#             f.create_dataset(f'MAG_ERR_{b}', data = training_data[train_mask, i+m.shape[1]][ainds].reshape(Ntest, size))
        
#     print("FINISHED OUTPUTTING VALIDATION")

############################################
# STEP 4: Run DNF
############################################

zbins = np.arange(0, 1.6, 0.01)

chunksize = 10_000
Totalsize = target_mask.size
nchunk = Totalsize // chunksize + 1

z_photo = np.zeros(m.shape[0]) - 99
zerr_e  = np.zeros(m.shape[0]) - 99
z1      = np.zeros(m.shape[0]) - 99

for i in range(nchunk):
    print("RANK", rank, "working on chunk {0} out of {1}".format(i, nchunk), flush=True)
    ts  = time.time()
    s   = slice(i*chunksize, (i+1)*chunksize)
    msk = target_mask[s]
    RES = juan_dnf.dnf(training_data[train_mask, :m.shape[1]], training_data[train_mask, -1], m[s][msk], dm[s][msk], zbins,
                       pdf = True, Nneighbors = 80, bound = False, radius = 2.0, magflux = 'mag', metric = "DNF", coeff = True)
    
    z_photo[s][msk] = RES[0]
    zerr_e[s][msk]  = RES[1]
    z1[s][msk]      = RES[5]

    print("RANK", rank, "Finished, using {0} s", time.time()-ts, flush=True)

del m, dm

# Step 1: Gather the sizes of each local array
local_size = np.array(len(z_photo), dtype=np.int32)
all_sizes  = np.zeros(size, dtype=np.int32) 
comm.Gather(local_size, all_sizes, root = 0)

displacements     = np.zeros(size, dtype=np.int32)
displacements[1:] = np.cumsum(all_sizes)[:-1]  # Compute starting indices


z_photo_all = np.zeros(np.sum(all_sizes)) if rank == 0 else None
z1_all      = np.zeros(np.sum(all_sizes)) if rank == 0 else None
zerr_e_all  = np.zeros(np.sum(all_sizes)) if rank == 0 else None
id_all      = np.zeros(np.sum(all_sizes), dtype = np.int64) if rank == 0 else None

print(rank, z1.size, id.size, z1.dtype, id.dtype)

comm.Gatherv(z_photo, (z_photo_all, all_sizes, displacements, MPI.DOUBLE), root = 0)
comm.Gatherv(zerr_e,  (zerr_e_all,  all_sizes, displacements, MPI.DOUBLE), root = 0)
comm.Gatherv(z1,      (z1_all,      all_sizes, displacements, MPI.DOUBLE), root = 0)
comm.Gatherv(id,      (id_all,      all_sizes, displacements, MPI.SIGNED_LONG_LONG), root = 0)

if rank == 0:
    print(rank, z1_all[:100], z1_all.size)
    print(rank, z_photo[:100], z1_all.size)
    print(rank, zerr_e[:100], z1_all.size)
    print(rank, id_all[:100], id_all.size)

if rank == 0:
    with h5py.File(f'/project/chihway/data/decade/dnf/DECADE_DR3_DNF_{name}_20250331_All.hdf', 'w') as f:
        f.create_dataset('z_photo', data = z_photo_all)
        f.create_dataset('zerr_e',  data = zerr_e_all)
        f.create_dataset('z1', data = z1_all)
        f.create_dataset('id', data = id_all)


exit()
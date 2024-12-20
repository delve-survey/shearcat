import numpy as np, h5py
import sys, os, gc
import joblib
from scipy import interpolate
import healpy as hp

prefix = np.random.default_rng(seed = 42).integers(2**30)
TMPDIR = os.environ['TMPDIR']
base   = TMPDIR + f'/{prefix}'
dgamma = 2*0.01


def create_reader(file_suffix):
    def reader(mask = None):
        file_path = f'{base}_{file_suffix}.npy'
        data = np.load(file_path, mmap_mode='r')
        
        if mask is None:
            return data
        else:
            return data[mask]
        
    return reader

func = {}


def de_island_ify(ra, dec):
    
    mask = np.invert(dec > np.where((310 < ra) & (ra < 350), 
                                            3.5, 
                                            np.where(ra > 350, 
                                                     (ra - 350) * (18 - 3.5)/(20) + 3.5,
                                                     (ra + 10)  * (18 - 3.5)/(20) + 3.5)
                                          )
                           )
    return mask

#STEP ONE: build a subset of the quantities to use in defining the weights
with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20241003.hdf', 'r') as f:
        
    
    #Get a mask of all objects that would be used
    mask = False
    GOLD = hp.read_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5_MCs2.fits')
    GOLD = GOLD[hp.ang2pix(4096, f['RA'][:], f['DEC'][:], lonlat = True)] == 0
    isld = de_island_ify(f['RA'][:], f['DEC'][:])
    Flgs = f['mcal_flags'][:]
    for m in ['noshear', '1p', '1m', '2p', '2m']:
        
        msk_here = (
        (Flgs == 0) & 
        (f[f'mcal_s2n_{m}'][:] > 10) & (f[f'mcal_s2n_{m}'][:] < 1000) & 
        (f[f'mcal_T_{m}'][:] < 10) & 
        (f[f'mcal_T_ratio_{m}'][:] > 0.5) & 
        np.invert((f[f'mcal_T_{m}'][:] > 2) & (f[f'mcal_s2n_{m}'][:] < 30)) & 
        np.invert((np.log10(f[f'mcal_T_{m}'][:]) < (22.25 - (30 - 2.5*np.log10(f[f'mcal_flux_{m}'][:, 0])))/3.5) & 
                  (f[f'mcal_g_{m}'][:, 0]**2 + f[f'mcal_g_{m}'][:, 1]**2 > 0.8**2)
                  )
                  )
        
        msk_here = msk_here & GOLD & isld
        
        mask = mask | msk_here

    print(mask.sum(), flush = True)

    for m in ['noshear', '1p', '1m', '2p', '2m']:

        msk_here = (
            (Flgs == 0) & 
            (f[f'mcal_s2n_{m}'][:] > 10) & (f[f'mcal_s2n_{m}'][:] < 1000) & 
            (f[f'mcal_T_{m}'][:] < 10) & 
            (f[f'mcal_T_ratio_{m}'][:] > 0.5) & 
            np.invert((f[f'mcal_T_{m}'][:] > 2) & (f[f'mcal_s2n_{m}'][:] < 30)) & 
            np.invert((np.log10(f[f'mcal_T_{m}'][:]) < (22.25 - (30 - 2.5*np.log10(f[f'mcal_flux_{m}'][:, 0])))/3.5) & 
                    (f[f'mcal_g_{m}'][:, 0]**2 + f[f'mcal_g_{m}'][:, 1]**2 > 0.8**2)
                    )
                )
        
        msk_here = msk_here & GOLD & isld

        np.save(f'{base}_mask_{m}.npy', msk_here[mask])

        print(np.load(f'{base}_mask_{m}.npy').sum(), flush = True)

        for q in ['mcal_g', 'mcal_s2n', 'mcal_T_ratio']:    
            np.save(f'{base}_{q}_{m}.npy', f[f'{q}_{m}'][:][mask])

            func[f'{q}_{m}'] = create_reader(f'{q}_{m}')


    #Ran this just to determine the limits of the S2N and T_ratio of Grid
    m = np.load(f'{base}_mask_noshear.npy') > 0
    print("S2N [95%] :", np.percentile(np.load(f'{base}_mcal_s2n_noshear.npy')[m],     [2.5, 97.5]))
    print("Tr  [95%] :", np.percentile(np.load(f'{base}_mcal_T_ratio_noshear.npy')[m], [2.5, 97.5]))

def one_step(SNR_range, Tratio_range, bin = None):
    
    mask_h = {}

    for m in ['noshear', '1p', '1m', '2p', '2m']:
        snr         = func[f'mcal_s2n_{m}']()
        Tr          = func[f'mcal_T_ratio_{m}']()
        mask_h[m]   = (snr > SNR_range[0]) & (snr < SNR_range[1]) & (Tr > Tratio_range[0]) & (Tr < Tratio_range[1])
    
    del Tr, snr; gc.collect()

    if bin is None:
        mask = {m : (np.load(f'{base}_mask_{m}.npy', mmap_mode = 'r') > 0) & mask_h[m] for m in mask_h.keys()}
    else:
        mask = {m : (np.load(f'{base}_mask_{m}.npy', mmap_mode = 'r') == bin) & mask_h[m] for m in mask_h.keys()}

    del mask_h; gc.collect()

    #Get sigma_e in this cell
    g1, g2   = func['mcal_g_noshear'](mask['noshear']).T
    sigma_e  = np.sqrt(np.average(g1**2 + g2**2)/2)
    Ncounts  = np.sum(mask['noshear'])
    del g1, g2; gc.collect()
    
    #Now get the responses
    R11      = (np.average(func['mcal_g_1p'](mask['noshear'])[:, 0]) - np.average(func['mcal_g_1m'](mask['noshear'])[:, 0])) / dgamma    
    R11s     = (np.average(func['mcal_g_noshear'](mask['1p'])[:, 0]) - np.average(func['mcal_g_noshear'](mask['1m'])[:, 0])) / dgamma    
    R22      = (np.average(func['mcal_g_2p'](mask['noshear'])[:, 1]) - np.average(func['mcal_g_2m'](mask['noshear'])[:, 1])) / dgamma    
    R22s     = (np.average(func['mcal_g_noshear'](mask['2p'])[:, 1]) - np.average(func['mcal_g_noshear'](mask['2m'])[:, 1])) / dgamma   

    del mask; gc.collect()

    R11_tot  = R11 + R11s
    R22_tot  = R22 + R22s    
    Rg       = (R11 + R22)/2
    weight   = (Rg/sigma_e)**2

    SNR = np.average(SNR_range)
    Tr  = np.average(Tratio_range)

    dtype = ['weight', 'sigma_e', 'Rgamma', 'Ncounts', 'R11_tot', 'R22_tot', 'R11', 'R11s', 'R22', 'R22s', 'SNR', 'T_ratio']
    dtype = [(k, float) for k in dtype]
    out   = np.array([tuple([weight, sigma_e, Rg, Ncounts, R11_tot, R22_tot, R11, R11s, R22, R22s, SNR, Tr])], dtype = dtype)
    return out


N    = 20
SNR  = np.geomspace(10,  330, N + 1)
Tr   = np.geomspace(0.5, 4.5, N + 1)
size = N * N

jobs    = [joblib.delayed(one_step)([SNR[i % N], SNR[i % N +1]], [Tr[i // N], Tr[i // N + 1]], bin = None) for i in range(size)]
results = joblib.Parallel(n_jobs = -1, verbose = 10)(jobs)
results = np.concatenate(results)

np.save('./grid_quantities_20241219_DR3_2.npy', results)
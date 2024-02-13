
import sys
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

import numpy as np
import astropy.io.fits as pf
import h5py
import healpy as hp
import scipy

tag = '20240209'

size_ratio_grid = np.logspace(np.log10(0.5), np.log10(6), 21)
s2n_grid = np.logspace(np.log10(10), np.log10(400), 21)

def mcal_mask(size_min, size_max, s2n_min, s2n_max):

    Mask = {}
    for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
        print(shear_type)
        
        with h5py.File('/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
            flux_r, flux_i, flux_z = h5r['mcal_flux_'+shear_type+'_dered_sfd98'][:].T

        mag_r = -2.5*np.log10(flux_r)+30
        mag_i = -2.5*np.log10(flux_i)+30
        mag_z = -2.5*np.log10(flux_z)+30

        # PZ mask
        mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) &
                            (mag_r < 26)   & (mag_r > 15) &
                            (mag_z < 26)   & (mag_z > 15) &
                            (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) &
                            (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

        del mag_i, mag_z, flux_r, flux_i, flux_z

        with h5py.File('/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
            T = h5r['mcal_T_'+shear_type][:]
            s2n = h5r['mcal_s2n_'+shear_type][:]
            size_ratio = h5r['mcal_T_ratio_'+shear_type][:]
            mcal_flags = h5r['mcal_flags'][:]
            g1, g2  = h5r['mcal_g_'+shear_type][:].T

        # Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
        SNR_Mask   = (s2n > 10) & (s2n < 1000)
        Tratio_Mask= size_ratio > 0.5
        T_Mask = T < 10
        Flag_Mask = (mcal_flags == 0)
        Other_Mask = np.invert((T > 2) & (s2n < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))
        bin_Mask = (size_ratio>=size_min)*(size_ratio<size_max)*(s2n>=s2n_min)*(s2n<s2n_max)

        del T, s2n, size_ratio, mag_r, mcal_flags

        with h5py.File('/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
            
            sg = h5r['FLAGS_SG_BDF'][:]
            fg = h5r['FLAGS_FOREGROUND'][:]

        SG_Mask = (sg>=4)
        FG_Mask = (fg==0)

        mask_total_X = bin_Mask & mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & SG_Mask & FG_Mask

        Mask[shear_type] = mask_total_X
        
        del sg, fg
        del mask_total_X, bin_Mask, mcal_pz_mask, SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, Other_Mask, SG_Mask, FG_Mask

    return Mask


dgamma = 2*0.01

i = int(sys.argv[1])
j = int(sys.argv[2])

print(i,j)
Mask_mcal = mcal_mask(size_ratio_grid[i], size_ratio_grid[i+1], s2n_grid[j], s2n_grid[j+1])

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
    R11 =  (np.mean(h5r['mcal_g_1p'][:,0][Mask_mcal['noshear']]) - np.mean(h5r['mcal_g_1m'][:,0][Mask_mcal['noshear']]))/dgamma
    R11s = (np.mean(h5r['mcal_g_noshear'][:,0][Mask_mcal['1p']]) - np.mean(h5r['mcal_g_noshear'][:,0][Mask_mcal['1m']]))/dgamma
    R22 =  (np.mean(h5r['mcal_g_2p'][:,1][Mask_mcal['noshear']]) - np.mean(h5r['mcal_g_2m'][:,1][Mask_mcal['noshear']]))/dgamma
    R22s = (np.mean(h5r['mcal_g_noshear'][:,1][Mask_mcal['2p']]) - np.mean(h5r['mcal_g_noshear'][:,1][Mask_mcal['2m']]))/dgamma
    print(R11, R11s, R22, R22s)
    
with h5py.File('/project/chihway/data/decade/metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
    print(Mask_mcal['noshear'])

    g1, g2  = h5r['mcal_g_noshear'][:][Mask_mcal['noshear']].T
    mcal_g_cov = h5r['mcal_g_cov_noshear'][:][Mask_mcal['noshear']]
    count = len(g1)
    sig_e_squared = 0.5*(np.mean(g1**2) + np.mean(g2**2))
    mcal_g_var = 0.5*(np.mean(mcal_g_cov[:,0,0])+np.mean(mcal_g_cov[:,1,1]))

string = str(count)+'\t'+str(R11)+'\t'+str(R11s)+'\t'+str(R22)+'\t'+str(R22s)+'\t'+str(sig_e_squared)+'\t'+str(mcal_g_var)+'\n'

with open('response_'+str(i)+'_'+str(j)+'.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(string)




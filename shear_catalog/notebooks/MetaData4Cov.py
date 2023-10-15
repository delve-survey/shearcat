

import sys
import numpy as np
import astropy.io.fits as pf
from astropy.table import Table, vstack
import yaml
import h5py
import healpy as hp
import scipy
from scipy import interpolate

tag = '0613'
project_dir = '/project2/chihway/data/decade/'

with h5py.File(project_dir+'metacal_gold_combined_2023'+tag+'.hdf', 'r') as h5r:

        print(h5r.keys())
        ids = h5r['COADD_OBJECT_ID'][:]

tomo = np.load('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_26x26_ids+cells+fluxes_TomoBins.npz')


# first get mask total

with h5py.File(project_dir+'metacal_gold_combined_2023'+tag+'.hdf', 'r') as h5r:
    size_ratio = h5r['mcal_T_ratio_noshear'][:]
    s2n = h5r['mcal_s2n_noshear'][:]
    sg = h5r['sg_bdf'][:] 
    fg = h5r['FLAGS_FOREGROUND'][:] 
    T = h5r['mcal_T_noshear'][:]
    mcal_flags = h5r['mcal_flags'][:]
    g1, g2  = h5r['mcal_g_noshear'][:].T
    flux_r, flux_i, flux_z = h5r['mcal_flux_noshear'][:].T

mag_r = -2.5*np.log10(flux_r)+30
mag_i = -2.5*np.log10(flux_i)+30
mag_z = -2.5*np.log10(flux_z)+30

# PZ mask
mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                        (mag_r < 26)   & (mag_r > 15) & 
                        (mag_z < 26)   & (mag_z > 15) & 
                        (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                        (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

# Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
SNR_Mask   = (s2n > 10) & (s2n < 1000)
Tratio_Mask= size_ratio > 0.5
T_Mask = T < 10
Flag_Mask = (mcal_flags == 0)
Other_Mask = np.invert((T > 2) & (s2n < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))
SG_Mask = (sg>=4)
FG_Mask = (fg==0)

mask_total = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & SG_Mask & FG_Mask

del size_ratio, s2n, sg, fg, T, mcal_flags, g1, g2, flux_r, flux_i, flux_z, mag_r, mag_i, mag_z
del mcal_pz_mask, SNR_Mask, Tratio_Mask, T_Mask, Flag_Mask, Other_Mask, SG_Mask

with h5py.File(project_dir+'metacal_gold_combined_2023'+tag+'.hdf', 'r') as h5r:

    print(h5r.keys())
    ra = h5r['RA'][mask_total]
    dec = h5r['DEC'][mask_total]
    w = h5r['mcal_g_w_v2'][mask_total]   
    g1, g2  = h5r['mcal_g_noshear'][:][mask_total].T


ids_tomo = tomo['arr_0']['COADD_OBJECT_ID']
tomo_tomo = tomo['arr_0']['TomoBin']
z_samp_tomo = tomo['arr_0']['Z_SAMP']


# match tomo and metacal (there's a small number of missing galaxies in the tomo file)

mask_tomo = np.in1d(ids[mask_total], ids_tomo)
ids = ids[mask_total][mask_tomo]
ra = ra[mask_tomo]
dec = dec[mask_tomo]
w = w[mask_tomo]
g1 = g1[mask_tomo]
g2 = g2[mask_tomo]

print(len(ids))

# reorder by ids

X = np.argsort(ids)
ids = ids[X]
ra = ra[X]
dec = dec[X]

Y = np.argsort(ids_tomo)
ids_tomo = ids_tomo[Y]
tomo_tomo = tomo_tomo[Y]
z_samp_tomo = z_samp_tomo[Y]


nside = 4096
map_counts = np.zeros(hp.nside2npix(nside))

phi = ra/180*np.pi
theta = (90.-dec)/180*np.pi

pix = hp.ang2pix(nside, theta, phi)

for i in range(len(pix)):
        map_counts[pix[i]] += 1
            
area = len(map_counts[map_counts>0])/len(map_counts)*4*np.pi*(180./np.pi)**2*60*60
print('area', area/60/60, 'deg^2')

print('total', len(ra)/area, '/arcmin^2')



def neff_H12(w, A):
    return 1./A * (np.sum(w)**2) / (np.sum(w**2))

def sigmae2_H12(w, g1, R1, g2, R2, neff, A):
    return 0.5*((np.sum(w**2*(g1/R1)**2)/(np.sum(w))**2)
                                            +(np.sum(w**2*(g2/R2)**2)/(np.sum(w))**2))*(A*neff)


R1 = 0.7915714444599093
R2 = 0.7932463366088794


print("Bin     total ngal    raw ngal(/arcmin^2)  neff(/arcmin^2)     sigmae ")
print("======================================================================")
for i in range(4):
    mask_bin = (tomo_tomo == i+1)
    w_bin = w[mask_bin]
    g1_bin = g1[mask_bin]
    g2_bin = g2[mask_bin]
                        
    neff = neff_H12(w_bin, area)
    sigmae2 = sigmae2_H12(w_bin, g1_bin, R1, g2_bin, R2, neff, area)

    print(str(i+1)+"\t"+ str(len(ids[mask_bin]))+"\t \t"+str(len(ids[mask_bin])/area)[:7]+"\t\t"+str(neff)[:7]+"\t\t"+str(sigmae2**0.5)[:7])


for i in range(4):
    mask_bin = (tomo_tomo == i+1)
    z_bin = z_samp_tomo[mask_bin]
    output = mplot.hist(z_bin, range=(0,2.5), bins=50, histtype='step', label='bin'+str(i))
    np.savetxt('/project/chihway/data/decade/forecast/v1/zbin_'+str(i)+'.txt', np.vstack([(output[1][:-1]+output[1][1:])/2, output[0]/np.sum(output[0])]).T)
                

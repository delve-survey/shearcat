
import sys
sys.path.append('.conda/envs/shear/lib/python3.9/site-packages/')

import numpy as np
import astropy.io.fits as pf
import pylab as mplot
import yaml
import h5py
import healpy as hp

tag = '20240209'
nside = 4096
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_'+tag+'.hdf'
master_mask = project_dir+'metacal_gold_combined_mask_'+tag+'.hdf'
tomo=1

# read mask ########################
if tomo==0:
    with h5py.File(master_mask, 'r') as f:
        mask_noshear = f['baseline_mcal_mask_noshear'][:]
        mask_1p = f['baseline_mcal_mask_1p'][:]
        mask_1m = f['baseline_mcal_mask_1m'][:]
        mask_2p = f['baseline_mcal_mask_2p'][:]
        mask_2m = f['baseline_mcal_mask_2m'][:]

if tomo==1:
    with h5py.File(master_cat, 'r') as f:
        mask_noshear = f['baseline_mcal_mask_noshear'][:]
        mask_1p = f['baseline_mcal_mask_1p'][:]
        mask_1m = f['baseline_mcal_mask_1m'][:]
        mask_2p = f['baseline_mcal_mask_2p'][:]
        mask_2m = f['baseline_mcal_mask_2m'][:]
    
print('read mask')

def weight_average(values, weights):
    return np.sum(values*weights)/np.sum(weights)

# get response #####################

# take this out of the with
with h5py.File(master_cat, 'r') as h5r:
    g_noshear = h5r['mcal_g_noshear'][:]
    g_1p = h5r['mcal_g_1p'][:]
    g_1m = h5r['mcal_g_1m'][:]
    g_2p = h5r['mcal_g_2p'][:]
    g_2m = h5r['mcal_g_2m'][:]
    w = h5r['mcal_g_w'][:]
print('read shear')

dgamma = 2*0.01

R_11 = []
R_11s = []
R_22 = []
R_22s = []
Ngal = []

if tomo==1:
    for i in range(4):
        print('bin'+str(i))
                  
        mask_noshear_bin = (mask_noshear==i+1)
        mask_1p_bin = (mask_1p==i+1)
        mask_1m_bin = (mask_1m==i+1)
        mask_2p_bin = (mask_2p==i+1)
        mask_2m_bin = (mask_2m==i+1)
                                          
        R11 =  (weight_average(g_1p[:,0][mask_noshear_bin],w[mask_noshear_bin]) - weight_average(g_1m[:,0][mask_noshear_bin], w[mask_noshear_bin]))/dgamma
        R11s = (weight_average(g_noshear[:,0][mask_1p_bin], w[mask_1p_bin]) - weight_average(g_noshear[:,0][mask_1m_bin], w[mask_1m_bin]))/dgamma
        R22 =  (weight_average(g_2p[:,1][mask_noshear_bin], w[mask_noshear_bin]) - weight_average(g_2m[:,1][mask_noshear_bin], w[mask_noshear_bin]))/dgamma
        R22s = (weight_average(g_noshear[:,1][mask_2p_bin], w[mask_2p_bin]) - weight_average(g_noshear[:,1][mask_2m_bin], w[mask_2m_bin]))/dgamma
        R11tot = R11+R11s
        R22tot = R22+R22s
                                                                      
        print('R11', R11, 'R11s', R11s)
        print('R22', R22, 'R22s', R22s)
        print('R11tot', R11tot, 'R22tot', R22tot)
    
        R_11.append(R11)
        R_11s.append(R11s)
        R_22.append(R22)
        R_22s.append(R22s)
        Ngal.append([len(w[mask_noshear_bin]), len(w[mask_1p_bin]), len(w[mask_1m_bin]), len(w[mask_2p_bin]), len(w[mask_2m_bin])])
    
print("non-tomographic")
mask_noshear_bin = (mask_noshear>0)
mask_1p_bin = (mask_1p>0)
mask_1m_bin = (mask_1m>0)
mask_2p_bin = (mask_2p>0)
mask_2m_bin = (mask_2m>0)

R11 =  (weight_average(g_1p[:,0][mask_noshear_bin],w[mask_noshear_bin]) - weight_average(g_1m[:,0][mask_noshear_bin], w[mask_noshear_bin]))/dgamma
R11s = (weight_average(g_noshear[:,0][mask_1p_bin], w[mask_1p_bin]) - weight_average(g_noshear[:,0][mask_1m_bin], w[mask_1m_bin]))/dgamma
R22 =  (weight_average(g_2p[:,1][mask_noshear_bin], w[mask_noshear_bin]) - weight_average(g_2m[:,1][mask_noshear_bin], w[mask_noshear_bin]))/dgamma
R22s = (weight_average(g_noshear[:,1][mask_2p_bin], w[mask_2p_bin]) - weight_average(g_noshear[:,1][mask_2m_bin], w[mask_2m_bin]))/dgamma
R11tot = R11+R11s
R22tot = R22+R22s

print('R11', R11, 'R11s', R11s)
print('R22', R22, 'R22s', R22s)
print('R11tot', R11tot, 'R22tot', R22tot)

R_11.append(R11)
R_11s.append(R11s)
R_22.append(R22)
R_22s.append(R22s)
Ngal.append([len(w[mask_noshear_bin]), len(w[mask_1p_bin]), len(w[mask_1m_bin]), len(w[mask_2p_bin]), len(w[mask_2m_bin])])

del g_noshear, g_1p, g_1m, g_2p, g_2m

# get area ###########################

with h5py.File(master_cat, 'r') as h5r:
    ra = h5r['RA'][:]
    dec = h5r['DEC'][:]
    g1, g2  = h5r['mcal_g_noshear'][:].T
    w = h5r['mcal_g_w'][:]

mask_noshear_nontomo = (mask_noshear>0)
ra = ra[mask_noshear_nontomo]
dec = dec[mask_noshear_nontomo]
g1 = g1[mask_noshear_nontomo]
g2 = g2[mask_noshear_nontomo]
w = w[mask_noshear_nontomo]

nside = 4096
map_counts = np.zeros(hp.nside2npix(nside))

phi = ra/180*np.pi
theta = (90.-dec)/180*np.pi

pix = hp.ang2pix(nside, theta, phi)

unique_pix, idx_rep = np.unique(pix, return_inverse=True)
map_counts[unique_pix] += np.bincount(idx_rep)

area = len(map_counts[map_counts>0])/len(map_counts)*4*np.pi*(180./np.pi)**2*60*60
print('area', area/60/60, 'deg^2')
print('total number', len(ra))

# raw number
print(len(ra)/area)
n = len(ra)/area

del ra, dec, g1, g2, w, pix, theta, phi

mask_all = map_counts.copy()
mask_all[mask_all>0] = 1
hp.write_map(project_dir+'footprint_mask_delve_cs_'+tag+'.fits', mask_all, dtype=int, overwrite=True)

# get Neff and Sigmae #######################

def neff_H12(w, A):
    return 1./A * (np.sum(w)**2) / (np.sum(w**2))

def sigmae2_H12(w, g1, R1, g2, R2, neff, A):
    return np.sqrt(0.5*((np.sum(w**2*(g1/R1)**2)/(np.sum(w))**2)+(np.sum(w**2*(g2/R2)**2)/(np.sum(w))**2))*(A*neff))

# should double check this...
def sigmae2_C13(w, g1, R1, g2, R2, sigma2_e1_m, sigma2_e2_m):
    return np.sqrt(0.5* np.sum(w**2*((g1/R1)**2+(g2/R2)**2 
                                                 - sigma2_e1_m/R1**2 - sigma2_e2_m/R2**2)) / np.sum(w**2))

def neff_C13(w, g1, R1, g2, R2, sigma2_e1_m, sigma2_e2_m, A):
    return (1./A * (np.sum(w)**2) / (np.sum(w**2))
                        *(np.sum(w**2*((g1/R1)**2+(g2/R2)**2 - sigma2_e1_m/R1**2 - sigma2_e2_m/R2**2)))
                        / np.sum(w**2*((g1/R1)**2+(g2/R2)**2)))

with h5py.File(project_dir+'metacal_gold_combined_'+tag+'.hdf', 'r') as h5r:
    mcal_g_cov = h5r['mcal_g_cov_noshear'][:]
    g1, g2  = h5r['mcal_g_noshear'][:].T
    w = h5r['mcal_g_w'][:]

# tomographic

Neff_H12 = []
Sigmae_H12 = []
Neff_C13 = []
Sigmae_C13 = []
N = []

if tomo==1:
    for i in range(4):
        mask_noshear_bin = (mask_noshear==i+1)
           
        mcal_g_cov_bin = mcal_g_cov[mask_noshear_bin]
        g1_bin = g1[mask_noshear_bin]
        g2_bin = g2[mask_noshear_bin]
        w_bin = w[mask_noshear_bin]
                                    
        sigma2_e1_m = mcal_g_cov_bin[:,0,0] 
        sigma2_e2_m = mcal_g_cov_bin[:,1,1]
    
        R11tot = R_11[i]+R_11s[i]
        R22tot = R_22[i]+R_22s[i]
                                   
        neff_H12_bin = neff_H12(w_bin, area)
        sigmae_H12_bin = sigmae2_H12(w_bin, g1_bin, R11tot, g2_bin, R22tot, neff_H12_bin, area)
                                                                    
        sigmae_C13_bin = sigmae2_C13(w_bin, g1_bin, R11tot, g2_bin, R22tot, sigma2_e1_m, sigma2_e2_m)
        neff_C13_bin = neff_C13(w_bin, g1_bin, R11tot, g2_bin, R22tot, sigma2_e1_m, sigma2_e2_m, area)
                                                                                
        print("bin"+str(i), neff_H12_bin, sigmae_H12_bin, neff_C13_bin, sigmae_C13_bin)
    
        Neff_H12.append(neff_H12_bin)
        Neff_C13.append(neff_C13_bin)
        Sigmae_H12.append(sigmae_H12_bin)
        Sigmae_C13.append(sigmae_C13_bin)
        N.append(len(w_bin)/area/60/60)


# non-tomographic

mask_noshear_bin = (mask_noshear>0)

mcal_g_cov_bin = mcal_g_cov[mask_noshear_bin]
g1_bin = g1[mask_noshear_bin]
g2_bin = g2[mask_noshear_bin]
w_bin = w[mask_noshear_bin]

sigma2_e1_m = mcal_g_cov_bin[:,0,0] 
sigma2_e2_m = mcal_g_cov_bin[:,1,1]

if tomo==1:
    R11tot = R_11[4]+R_11s[4]
    R22tot = R_22[4]+R_22s[4]
else:
    R11tot = R_11[0]+R_11s[0]
    R22tot = R_22[0]+R_22s[0]


neff_H12_bin = neff_H12(w_bin, area)
sigmae_H12_bin = sigmae2_H12(w_bin, g1_bin, R11tot, g2_bin, R22tot, neff_H12_bin, area)

sigmae_C13_bin = sigmae2_C13(w_bin, g1_bin, R11tot, g2_bin, R22tot, sigma2_e1_m, sigma2_e2_m)
neff_C13_bin = neff_C13(w_bin, g1_bin, R11tot, g2_bin, R22tot, sigma2_e1_m, sigma2_e2_m, area)

print("non-tomo", neff_H12_bin, sigmae_H12_bin, neff_C13_bin, sigmae_C13_bin)

Neff_H12.append(neff_H12_bin)
Neff_C13.append(neff_C13_bin)
Sigmae_H12.append(sigmae_H12_bin)
Sigmae_C13.append(sigmae_C13_bin)
N.append(len(w_bin)/area/60/60)

if tomo==1:
    NN=5
else: 
    NN=1
    
for i in range(NN):
    print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f " 
    % (N[i]*60*60, R_11[i], R_11s[i], R_11[i]+R_11s[i], R_22[i], R_22s[i], R_22[i]+R_22s[i], Neff_C13[i], Sigmae_C13[i], Neff_H12[i], Sigmae_H12[i]))

np.savez('metadata_'+str(tag)+'.npz', R11=R_11, R11s=R_11s, R22=R_22, R22s=R_22s, area=area, N=N, Neff_C13=Neff_C13, Sigmae_C13=Sigmae_C13, Neff_H12=Neff_H12, Sigmae_H12=Sigmae_H12, Ngal=Ngal)


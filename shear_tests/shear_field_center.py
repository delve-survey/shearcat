import sys
import numpy as np
import astropy.io.fits as pf
from astropy.table import Table, vstack
import pylab as mplot
import yaml
import h5py
import scipy.optimize
import tqdm
from pylab import cm
#import healpy as hp
import os
import fitsio
import treecorr
import pandas
import matplotlib.pyplot as plt
import healpy as hp

savepath = '/home/zhuoqizhang/delve_shear/results/shear_field_center'

def get_Mask(label):
    
    with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230613.hdf', 'r') as f:

        ra      = np.array(f['RA'])
        dec     = np.array(f['DEC'])
        e1, e2  = np.array(f[f'mcal_g_{label}']).T

        mag_r   = 30 - 2.5*np.log10(np.array(f[f'mcal_flux_{label}'])[:, 0])
        mag_i   = 30 - 2.5*np.log10(np.array(f[f'mcal_flux_{label}'])[:, 1])
        mag_z   = 30 - 2.5*np.log10(np.array(f[f'mcal_flux_{label}'])[:, 2])

        SNR     = np.array(f[f'mcal_s2n_{label}'])
        T_ratio = np.array(f[f'mcal_T_ratio_{label}'])
        T       = np.array(f[f'mcal_T_{label}'])
        flags   = np.array(f['mcal_flags'])

        GOLD_Foreground  = hp.read_map('/project/chihway/dhayaa/DECADE/Gold_Foreground_20230607.fits', dtype = int)
        FLAGS_Foreground = GOLD_Foreground[hp.ang2pix(hp.npix2nside(GOLD_Foreground.size), ra, dec, lonlat = True)]

        #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)

        SNR_Mask   = (SNR > 10) & (SNR < 1000)
        Tratio_Mask= T_ratio > 0.5
        T_Mask     = T < 10
        Flag_Mask  = flags == 0
        Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (e1**2 + e2**2 > 0.8**2))
        GOLD_Mask  = FLAGS_Foreground == 0 #From gold catalog
        SG_Mask    = np.array(f['sg_bdf']) >= 4 #Star-galaxy separator
        Color_Mask = ((18 < mag_i) & (mag_i < 23.5) & 
                      (15 < mag_r) & (mag_r < 26) & 
                      (15 < mag_z) & (mag_z < 26) & 
                      (-1.5 < mag_r - mag_i) & (mag_r - mag_i < 4) & 
                      (-1.5 < mag_i - mag_z) & (mag_i - mag_z < 4)
                     )

        print(np.sum(SNR_Mask), np.sum(Tratio_Mask), np.sum(T_Mask), np.sum(Flag_Mask), np.sum(Other_Mask))

        Mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Color_Mask & Other_Mask & GOLD_Mask & SG_Mask
        print("TOTAL NUM", np.sum(Mask))

    return Mask[:]


def get_healpixel_footprint(ra, dec, nside, nest=False, count_threshold=None):
    pixels = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True)
    if count_threshold and count_threshold > 1:
        pixels, counts = np.unique(pixels, return_counts=True)
        return pixels[counts >= count_threshold]
    return np.unique(pixels)

def generate_uniform_random_ra_dec_healpixel(n, pix, nside, nest=False):

    ra, dec = hp.vec2ang(hp.boundaries(nside, pix, 1, nest=nest).T, lonlat=True)
    ra_dec_min_max = ra.min(), ra.max(), dec.min(), dec.max()

    ra = np.empty(n)
    dec = np.empty_like(ra)
    n_needed = n

    while n_needed > 0:
        ra_this, dec_this = generate_uniform_random_ra_dec_min_max(n_needed*2, *ra_dec_min_max)
        mask = np.where(hp.ang2pix(nside, ra_this, dec_this, nest=nest, lonlat=True) == pix)[0]
        count_this = mask.size
        if n_needed - count_this < 0:
            count_this = n_needed
            mask = mask[:n_needed]

        s = slice(-n_needed, -n_needed+count_this if -n_needed+count_this < 0 else None)
        ra[s] = ra_this[mask]
        dec[s] = dec_this[mask]
        n_needed -= count_this

    return ra, dec

def generate_uniform_random_ra_dec_min_max(n, ra_min, ra_max, dec_min, dec_max):

    ra = np.random.uniform(ra_min, ra_max, size=n)
    dec = np.random.uniform(np.sin(np.deg2rad(dec_min)), np.sin(np.deg2rad(dec_max)), size=n)
    dec = np.arcsin(dec, out=dec)
    dec = np.rad2deg(dec, out=dec)
    return ra, dec

def generate_uniform_random_ra_dec_footprint(n, footprint=None, nside=None, nest=False):

    if footprint is None or hp.nside2npix(nside) == len(footprint):
        return generate_uniform_random_ra_dec(n)

    n_per_pix_all = np.histogram(np.random.rand(n), np.linspace(0, 1, len(footprint)+1))[0]

    ra = np.empty(n)
    dec = np.empty_like(ra)
    count = 0

    for n_per_pix, pix in zip(n_per_pix_all, footprint):
        ra_this, dec_this = generate_uniform_random_ra_dec_healpixel(n_per_pix, pix, nside, nest)
        s = slice(count, count+n_per_pix)
        ra[s] = ra_this
        dec[s] = dec_this
        count += n_per_pix

    assert count == n

    return ra, dec



Mask0 = get_Mask('noshear')

with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230613.hdf', 'r') as f:
    e1 = np.array(f['mcal_g_noshear'][:,0][Mask0])
    e2 = np.array(f['mcal_g_noshear'][:,1][Mask0])
    ra = np.array(f['RA'][:][Mask0])
    dec = np.array(f['DEC'][:][Mask0])
FP_NSIDE = int(sys.argv[1])
print(FP_NSIDE)
inds = np.unique(hp.ang2pix(FP_NSIDE,ra,dec,lonlat=True))


fcenters = pandas.read_csv('/project/chihway/dhayaa/DECADE/FieldCenters_DR3_1_1.csv')
fc_ra = np.array(fcenters['RADEG'])
fc_dec = np.array(fcenters['DECDEG'])

# mask field centers:
fc_pix = hp.ang2pix(FP_NSIDE,fc_ra,fc_dec,lonlat=True)
mask = np.in1d(fc_pix, inds)
print(np.sum(mask)/np.size(mask))
fc_ra = fc_ra[mask]
fc_dec = fc_dec[mask]

RND_NSIDE=1024
healpix_id = get_healpixel_footprint(ra,dec,RND_NSIDE)
rnd_ra, rnd_dec = generate_uniform_random_ra_dec_footprint(1_000_000, footprint=healpix_id, nside=RND_NSIDE, nest=False)
rnd_pix = hp.ang2pix(FP_NSIDE, rnd_ra, rnd_dec, lonlat=True)
mask = np.in1d(rnd_pix,inds)
rnd_ra = rnd_ra[mask]
rnd_dec = rnd_dec[mask]
print(np.size(rnd_ra))

# make patch centers
shearcat = treecorr.Catalog(g1=e1, g2=e2, ra=ra, dec=dec, ra_units='deg',dec_units='deg',npatch=100)
shearcat.write_patch_centers(f'{savepath}/centers')

print('should work')

shearcat = treecorr.Catalog(g1=e1, g2=e2, ra=ra, dec=dec, ra_units='deg',dec_units='deg', patch_centers=f'{savepath}/centers')
fcentercat = treecorr.Catalog(ra=fc_ra, dec=fc_dec, ra_units='deg',dec_units='deg', patch_centers=f'{savepath}/centers')
NG = treecorr.NGCorrelation(nbins = 25, min_sep = 2.5, max_sep = 250,
                                sep_units = 'arcmin',verbose = 0,bin_slop = 0, var_method='jackknife')
NG.process(fcentercat, shearcat, low_mem=True)

cov = NG.cov
xi = NG.xi
rsep = np.exp(NG.logr)


rndcat = treecorr.Catalog(ra=rnd_ra, dec=rnd_dec, ra_units='deg',dec_units='deg', patch_centers=f'{savepath}/centers')

NGrnd = treecorr.NGCorrelation(nbins = 25, min_sep = 2.5, max_sep = 250,
                                sep_units = 'arcmin',verbose = 0,bin_slop = 0, var_method='jackknife')
NGrnd.process(rndcat, shearcat, low_mem=True)

xi, xi_im, var = NG.calculateXi(NGrnd)

cov = NG.cov
chi2 = np.round(np.dot(np.dot(xi,np.linalg.inv(cov)),xi),1)
print(chi2)

np.save(f'{savepath}/rsep_{FP_NSIDE}.npy',rsep)
np.save(f'{savepath}/cov_{FP_NSIDE}.npy',cov)
np.save(f'{savepath}/xi_{FP_NSIDE}.npy',xi)

import sys
import numpy as np
import astropy.io.fits as pf
import treecorr
import h5py

tag = '20240209'
project_dir = '/project/chihway/data/decade/'
master_cat = project_dir+'metacal_gold_combined_'+tag+'.hdf'

# read response
response = np.load('/project/chihway/chihway/shearcat/paper_plots/metadata_'+str(tag)+'.npz')

R11 = response['R11']
R11s = response['R11s']
R22 = response['R22']
R22s = response['R22s']

# read from catalog     
with h5py.File(master_cat, 'r') as h5r:
    mask_noshear = h5r['baseline_mcal_mask_noshear'][:]
    ra = h5r['RA'][:]
    dec = h5r['DEC'][:]
    g1, g2  = h5r['mcal_g_noshear'][:].T
    w = h5r['mcal_g_w'][:]

for i in range(4):
    for j in range(4):
        if j>=i:
            print(i, j)

            R_1 = (R11[i]+R11s[i]+R22[i]+R22s[i])/2
            R_2 = (R11[j]+R11s[j]+R22[j]+R22s[j])/2

            mask_noshear_1 = (mask_noshear==i+1)
            mask_noshear_2 = (mask_noshear==j+1)

            ra_1 = ra[mask_noshear_1]
            dec_1 = dec[mask_noshear_1]
            g1_1 = g1[mask_noshear_1]
            g2_1 = g2[mask_noshear_1]
            w_1 = w[mask_noshear_1]

            ra_2 = ra[mask_noshear_2]
            dec_2 = dec[mask_noshear_2]
            g1_2 = g1[mask_noshear_2]
            g2_2 = g2[mask_noshear_2]
            w_2 = w[mask_noshear_2]

            gg = treecorr.GGCorrelation(nbins=20, min_sep=0.25, max_sep=250, sep_units='arcmin', bin_slop=0.00)

            print('Doing jackknife...')
            cat_g_1 = treecorr.Catalog(ra=ra_1, dec=dec_1, g1=(g1_1-np.average(g1_1, weights=w_1))/R_1,
                                 g2=(g2_1-np.average(g2_1, weights=w_1))/R_1, w=w_1,
                                 ra_units='deg', dec_units='deg', npatch=100)

            cat_g_1.write_patch_centers('jk_centers')

            cat_g_2 = treecorr.Catalog(ra=ra_2, dec=dec_2, g1=(g1_2-np.average(g1_2, weights=w_2))/R_2,
                                g2=(g2_2-np.average(g2_2, weights=w_2))/R_2, w=w_2,
                                ra_units='deg', dec_units='deg', patch_centers='jk_centers')

            
            gg.process(cat_g_1, cat_g_2)

            # compute cov for both gammax and gammt parts
            #cov_jkx = treecorr.estimate_multi_cov([gg], 'jackknife', func=lambda corrs: corrs[0].xim)
            cov_jk  = treecorr.estimate_multi_cov([gg], 'jackknife')

            theta = np.exp(gg.meanlogr)
            mean = gg.xip
            mean_x = gg.xim
            weights = gg.weight
            npairs = gg.npairs
 
            np.savetxt('gg_'+str(i+1)+'_'+str(j+1)+'/mean_gg', list(zip(theta, mean, mean_x, weights, npairs)),
                         header='th, xip, xim, w_gk, npairs_gk')

            np.savetxt('gg_'+str(i+1)+'_'+str(j+1)+'/cov_jk', cov_jk)



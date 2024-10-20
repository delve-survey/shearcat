'''
Code that runs all tests for DELVE shear
'''


import numpy as np
import pandas as pd
from scipy.interpolate import interpolate
import healpy as hp
import joblib
import treecorr, pymaster as nmt
import h5py
from tqdm import tqdm
import sys, os
from datetime import datetime as dt
import time
from functools import lru_cache

#For keeping track of how long steps take
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.5} seconds to run.")
        return result
    return wrapper


class AllTests(object):



    def __init__(self, psf_cat, galaxy_cat, psf_cat_inds, galaxy_cat_inds, output_path, sim_Cls_path, 
                 Npatch = 100, Star_SNR_min = 80, MapNSIDE_weightrands = 256):
        
        self.psf_cat    = psf_cat
        self.galaxy_cat = galaxy_cat

        self.psf_cat_inds = np.load(psf_cat_inds)
        self.galaxy_cat_inds = np.load(galaxy_cat_inds)

        self.Npatch = Npatch
        self.Star_SNR_min = Star_SNR_min
        self.MapNSIDE_weightrands = MapNSIDE_weightrands

        self.rng = np.random.default_rng(seed = 42)
        
        self.psf_inds, self.gal_inds = self.define_patches()

        self.output_path = output_path

        self.sim_Cls = np.loadtxt(sim_Cls_path) #Cls to use in making Gaussian mocks for covariance 
        
        self.dered = '_dered_sfd98'
        #self.dered = ''



    @timeit
    def define_patches(self):

        
        with h5py.File(self.galaxy_cat, 'r') as f:
            RA, DEC = f['RA'][:][self.galaxy_cat_inds], f['DEC'][:][self.galaxy_cat_inds]

        S = self.rng.choice(RA.size, size = 5_000_000, replace = False)
        centers  = treecorr.Catalog(ra = RA[S], dec = DEC[S], ra_units='deg',dec_units='deg', npatch = self.Npatch)._centers
        gal_inds = treecorr.Catalog(ra = RA, dec = DEC, ra_units='deg',dec_units='deg', patch_centers = centers)._patch
        
        
        with h5py.File(self.psf_cat, 'r') as f:
            RA, DEC = f['ra'][:][self.psf_cat_inds], f['dec'][:][self.psf_cat_inds]

        psf_inds = treecorr.Catalog(ra = RA, dec = DEC, ra_units='deg',dec_units='deg', patch_centers = centers)._patch
        
        return psf_inds, gal_inds


    @timeit
    def mean_shear(self):
        
        dgamma = 0.02
        Mask0  = self.get_mcal_Mask('noshear')
        Mask1p = self.get_mcal_Mask('1p')
        Mask2p = self.get_mcal_Mask('2p')
        Mask1m = self.get_mcal_Mask('1m')
        Mask2m = self.get_mcal_Mask('2m')
        
        with h5py.File(self.galaxy_cat, 'r') as f:

            mcal_g_w       = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds]
            mcal_g_w_1p    = f['mcal_g_w_1p'][:][self.galaxy_cat_inds]
            mcal_g_w_1m    = f['mcal_g_w_1m'][:][self.galaxy_cat_inds]
            mcal_g_w_2p    = f['mcal_g_w_2p'][:][self.galaxy_cat_inds]
            mcal_g_w_2m    = f['mcal_g_w_2m'][:][self.galaxy_cat_inds]

            mcal_g_noshear = f['mcal_g_noshear'][:][self.galaxy_cat_inds]
            mcal_g_1p      = f['mcal_g_1p'][:][self.galaxy_cat_inds]
            mcal_g_2p      = f['mcal_g_2p'][:][self.galaxy_cat_inds]
            mcal_g_1m      = f['mcal_g_1m'][:][self.galaxy_cat_inds]
            mcal_g_2m      = f['mcal_g_2m'][:][self.galaxy_cat_inds]
        
        
        R11_p  = np.sum( (mcal_g_1p[Mask0, 0]       *  mcal_g_w[Mask0])  )
        R11_m  = np.sum( (mcal_g_1m[Mask0, 0]       *  mcal_g_w[Mask0])  )
        R11s_p = np.sum( (mcal_g_noshear[Mask1p, 0] *  mcal_g_w_1p[Mask1p]) )
        R11s_m = np.sum( (mcal_g_noshear[Mask1m, 0] *  mcal_g_w_1m[Mask1m]) )
        
        
        R22_p  = np.sum( (mcal_g_2p[Mask0, 1]       *  mcal_g_w[Mask0]) )
        R22_m  = np.sum( (mcal_g_2m[Mask0, 1]       *  mcal_g_w[Mask0]) )
        R22s_p = np.sum( (mcal_g_noshear[Mask2p, 1] *  mcal_g_w_2p[Mask2p]) )
        R22s_m = np.sum( (mcal_g_noshear[Mask2m, 1] *  mcal_g_w_2m[Mask2m]) )

        
        R_counts     = np.sum(mcal_g_w[Mask0])
        Rs_1p_counts = np.sum(mcal_g_w_1p[Mask1p])
        Rs_1m_counts = np.sum(mcal_g_w_1m[Mask1m])
        Rs_2p_counts = np.sum(mcal_g_w_2p[Mask2p])
        Rs_2m_counts = np.sum(mcal_g_w_2m[Mask2m])

        
        e1 = np.sum( (mcal_g_noshear[:, 0] * mcal_g_w)[Mask0] )
        e2 = np.sum( (mcal_g_noshear[:, 1] * mcal_g_w)[Mask0] )

        output = np.zeros([4, self.Npatch])
        #Remove individual patches now
        for j in tqdm(range(self.Npatch), desc = 'Mean shear'):

            mask = self.gal_inds == j

            R11_p_here  = R11_p  - np.sum( (mcal_g_1p[Mask0 & mask, 0]       * mcal_g_w[Mask0 & mask]) )
            R11_m_here  = R11_m  - np.sum( (mcal_g_1m[Mask0 & mask, 0]       * mcal_g_w[Mask0 & mask]) )
            R11s_p_here = R11s_p - np.sum( (mcal_g_noshear[Mask1p & mask, 0] * mcal_g_w_1p[Mask1p & mask]) )
            R11s_m_here = R11s_m - np.sum( (mcal_g_noshear[Mask1m & mask, 0] * mcal_g_w_1m[Mask1m & mask]) )


            R22_p_here  = R22_p  - np.sum( (mcal_g_2p[Mask0 & mask, 1]       * mcal_g_w[Mask0 & mask]) )
            R22_m_here  = R22_m  - np.sum( (mcal_g_2m[Mask0 & mask, 1]       * mcal_g_w[Mask0 & mask]) )
            R22s_p_here = R22s_p - np.sum( (mcal_g_noshear[Mask2p & mask, 1] * mcal_g_w_2p[Mask2p & mask]) )
            R22s_m_here = R22s_m - np.sum( (mcal_g_noshear[Mask2m & mask, 1] * mcal_g_w_2m[Mask2m & mask]) )


            R_counts_here     = R_counts     - np.sum( mcal_g_w[Mask0 & mask] )
            Rs_1p_counts_here = Rs_1p_counts - np.sum( mcal_g_w_1p[Mask1p & mask] )
            Rs_1m_counts_here = Rs_1m_counts - np.sum( mcal_g_w_1m[Mask1m & mask] )
            Rs_2p_counts_here = Rs_2p_counts - np.sum( mcal_g_w_2p[Mask2p & mask] )
            Rs_2m_counts_here = Rs_2m_counts - np.sum( mcal_g_w_2m[Mask2m & mask] )

            e1_here = e1 - np.sum( (mcal_g_noshear[Mask0 & mask, 0] * mcal_g_w[Mask0 & mask]) )
            e2_here = e2 - np.sum( (mcal_g_noshear[Mask0 & mask, 1] * mcal_g_w[Mask0 & mask]) )


            R11  = (R11_p_here/R_counts_here - R11_m_here/R_counts_here)/dgamma
            R11s = (R11s_p_here/Rs_1p_counts_here - R11s_m_here/Rs_1m_counts_here)/dgamma
            R11_tot = R11 + R11s

            R22  = (R22_p_here/R_counts_here - R22_m_here/R_counts_here)/dgamma
            R22s = (R22s_p_here/Rs_2p_counts_here - R22s_m_here/Rs_2m_counts_here)/dgamma
            R22_tot = R22 + R22s

            output[0, j] = e1_here/R_counts_here
            output[1, j] = e2_here/R_counts_here
            
            output[2, j] = e1_here/R_counts_here / R11_tot
            output[3, j] = e2_here/R_counts_here / R22_tot
            
            print(output[:, j])
            
        savepath = self.output_path + '/mean_shear.npy'
        np.save(savepath, output)
        
        
        
    @timeit
    def brighter_fatter_effect(self):

        N_bin = 30
        bins  = np.linspace(15, 21, N_bin + 1)
        bincenter = 0.5*(bins[1:]+bins[:-1])
        
        with h5py.File(self.psf_cat, 'r') as f:

            flux     = f['FLUX_AUTO_DERED_SFD98'][:][self.psf_cat_inds]
            T_model  = f['T_model_hsm'][:][self.psf_cat_inds]
            T_star   = f['T_star_hsm'][:][self.psf_cat_inds]
            mag_zp   = f['MAGZP'][:][self.psf_cat_inds]
            e1_model = f['g1_model_hsm'][:][self.psf_cat_inds]
            e2_model = f['g2_model_hsm'][:][self.psf_cat_inds]
            e1_star  = f['g1_star_hsm'][:][self.psf_cat_inds]
            e2_star  = f['g2_star_hsm'][:][self.psf_cat_inds]
            s2n      = (f['FLUX_AUTO'][:]/f['FLUXERR_AUTO'][:])[self.psf_cat_inds]          
            band     = f['BAND'][:][self.psf_cat_inds]
            

            mask = (s2n > self.Star_SNR_min) & (band != b'g')

        dT      = (T_star-T_model)[mask]
        dT_frac = ((T_star-T_model)/T_star)[mask]
        de1     = (e1_star-e1_model)[mask]
        de2     = (e2_star-e2_model)[mask]
        mag     = mag_zp[mask] - 2.5*np.log10(flux[mask])

        
        output  = np.zeros([5, self.Npatch, bins.size - 1])

        #Counts for the total (all patches)
        output[0] = np.histogram(mag, bins = bins, weights = dT)[0]
        output[1] = np.histogram(mag, bins = bins, weights = dT_frac)[0]
        output[2] = np.histogram(mag, bins = bins, weights = de1)[0]
        output[3] = np.histogram(mag, bins = bins, weights = de2)[0]
        output[4] = np.histogram(mag, bins = bins)[0]

        #Remove individual patches now
        for j in tqdm(range(self.Npatch), desc = 'Brighter_Fatter_effect'):

            patch_mask = self.psf_inds[mask] == j

            output[0, j] -= np.histogram(mag[patch_mask], bins = bins, weights = dT[patch_mask])[0]
            output[1, j] -= np.histogram(mag[patch_mask], bins = bins, weights = dT_frac[patch_mask])[0]
            output[2, j] -= np.histogram(mag[patch_mask], bins = bins, weights = de1[patch_mask])[0]
            output[3, j] -= np.histogram(mag[patch_mask], bins = bins, weights = de2[patch_mask])[0]
            output[4, j] -= np.histogram(mag[patch_mask], bins = bins)[0]


        #Normalize by the counts per patch version
        for i in range(4):
            output[i] /= output[4]


        np.save(self.output_path + '/BrighterFatter.npy', output)
        np.save(self.output_path + '/BrighterFatter_bins.npy', bincenter)

    @timeit
    def get_mcal_Mask(self, label):
    
        dered = self.dered
        if os.path.isfile(os.environ['TMPDIR'] + '/MASK_%s.npy' % label):
            Mask = np.load(os.environ['TMPDIR'] + '/MASK_%s.npy' % label)
            
        else:
            with h5py.File(self.galaxy_cat, 'r') as f:

            
                #Normally needed for GOLD foreground cut but
                #we don't do that here, so it's fine.
                # ra      = f['RA'][:][self.galaxy_cat_inds]
                # dec     = f['DEC'][:][self.galaxy_cat_inds]

                e1, e2  = f[f'mcal_g_{label}'][:][self.galaxy_cat_inds].T 

                with np.errstate(invalid = 'ignore', divide = 'ignore'):
                    mag_r   = 30 - 2.5*np.log10(f[f'mcal_flux_{label}{dered}'][:, 0][self.galaxy_cat_inds])
                    mag_i   = 30 - 2.5*np.log10(f[f'mcal_flux_{label}{dered}'][:, 1][self.galaxy_cat_inds])
                    mag_z   = 30 - 2.5*np.log10(f[f'mcal_flux_{label}{dered}'][:, 2][self.galaxy_cat_inds])

                SNR     = f[f'mcal_s2n_{label}'][:][self.galaxy_cat_inds]
                T_ratio = f[f'mcal_T_ratio_{label}'][:][self.galaxy_cat_inds]
                T       = f[f'mcal_T_{label}'][:][self.galaxy_cat_inds]
                flags   = f['mcal_flags'][:][self.galaxy_cat_inds]
                
            # We don't use any of the area-based cuts, as the expectations is to include them
            # in the indices that are passed in.

            #GOLD_Foreground  = hp.read_map(fgpath, dtype = int)
            #FLAGS_Foreground = GOLD_Foreground[hp.ang2pix(hp.npix2nside(GOLD_Foreground.size), ra, dec, lonlat = True)]

            #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
                SNR_Mask   = (SNR > 10) & (SNR < 1000)
                Tratio_Mask= T_ratio > 0.5
                T_Mask     = T < 10
                Flag_Mask  = flags == 0
                Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (e1**2 + e2**2 > 0.8**2))
                Color_Mask = ((18 < mag_i) & (mag_i < 23.5) & 
                              (15 < mag_r) & (mag_r < 26) & 
                              (15 < mag_z) & (mag_z < 26) & 
                              (-1.5 < mag_r - mag_i) & (mag_r - mag_i < 4) & 
                              (-1.5 < mag_i - mag_z) & (mag_i - mag_z < 4)
                             )

            Mask = SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Color_Mask & Other_Mask
            
            np.save(os.environ['TMPDIR'] + '/MASK_%s.npy' % label, Mask)

            print("Loaded Mask")
        
        return Mask
    
    
    @timeit
    def get_star_Mask(self):
    
        dered = self.dered
        label = 'noshear'
        with h5py.File(self.galaxy_cat, 'r') as f:

            #Normally needed for GOLD foreground cut but
            #we don't do that here, so it's fine.
            # ra      = f['RA'][:][self.galaxy_cat_inds]
            # dec     = f['DEC'][:][self.galaxy_cat_inds]

            SNR     = f[f'mcal_s2n_{label}'][:]
            flags   = f['mcal_flags'][:]
            sg_bdf  = f['FLAGS_SG_BDF'][:]

        # We don't use the gold cuts, as the expectations is to include them
        # in the indices that are passed in.

        #GOLD_Foreground  = hp.read_map(fgpath, dtype = int)
        #FLAGS_Foreground = GOLD_Foreground[hp.ang2pix(hp.npix2nside(GOLD_Foreground.size), ra, dec, lonlat = True)]

        #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            SNR_Mask   = (SNR > 0) & (SNR < 1000)
            Flag_Mask  = flags == 0
            SG_Mask    = sg_bdf == 0 #Star-galaxy separator
            
        Mask = SNR_Mask & Flag_Mask & SG_Mask
        
        print("Loaded Mask")
        
        return Mask
    
    
    
    @timeit
    def compute_response(self, mask):

        Mask0  = self.get_mcal_Mask('noshear')
        Mask1p = self.get_mcal_Mask('1p')
        Mask2p = self.get_mcal_Mask('2p')
        Mask1m = self.get_mcal_Mask('1m')
        Mask2m = self.get_mcal_Mask('2m')

        dgamma = 0.01*2

        with h5py.File(self.galaxy_cat, 'r') as f:
            R11    = (np.average(f['mcal_g_1p'][:][self.galaxy_cat_inds, 0][Mask0 & mask], weights = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask0 & mask])
                    - np.average(f['mcal_g_1m'][:][self.galaxy_cat_inds, 0][Mask0 & mask], weights = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask0 & mask]))/dgamma
            R11s   = (np.average(f['mcal_g_noshear'][:][self.galaxy_cat_inds, 0][Mask1p & mask], weights = f['mcal_g_w_1p'][:][self.galaxy_cat_inds][Mask1p & mask])
                    - np.average(f['mcal_g_noshear'][:][self.galaxy_cat_inds, 0][Mask1m & mask], weights = f['mcal_g_w_1m'][:][self.galaxy_cat_inds][Mask1m & mask]))/dgamma
            R11tot = R11 + R11s
            
            R22    = (np.average(f['mcal_g_2p'][:][self.galaxy_cat_inds, 1][Mask0 & mask], weights = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask0 & mask])
                    - np.average(f['mcal_g_2m'][:][self.galaxy_cat_inds, 1][Mask0 & mask], weights = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask0 & mask]))/dgamma
            R22s   = (np.average(f['mcal_g_noshear'][:][self.galaxy_cat_inds, 1][Mask2p & mask], weights = f['mcal_g_w_2p'][:][self.galaxy_cat_inds][Mask2p & mask])
                    - np.average(f['mcal_g_noshear'][:][self.galaxy_cat_inds, 1][Mask2m & mask], weights = f['mcal_g_w_2m'][:][self.galaxy_cat_inds][Mask2m & mask]))/dgamma
            R22tot = R22 + R22s

        return R11tot, R22tot

    
    @timeit
    def shear_vs_X(self):


        Quantities = ['SNR', 'Tratio', 'Tpsf', 'e1psf', 'e2psf', 'r_minus_i', 'r_minus_z', 'i_minus_z']

        Mask0  = self.get_mcal_Mask('noshear')
        Mask1p = self.get_mcal_Mask('1p')
        Mask2p = self.get_mcal_Mask('2p')
        Mask1m = self.get_mcal_Mask('1m')
        Mask2m = self.get_mcal_Mask('2m')

        with h5py.File(self.galaxy_cat, 'r') as f:

            mcal_g_w       = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds]
            mcal_g_w_1p    = f['mcal_g_w_1p'][:][self.galaxy_cat_inds]
            mcal_g_w_1m    = f['mcal_g_w_1m'][:][self.galaxy_cat_inds]
            mcal_g_w_2p    = f['mcal_g_w_2p'][:][self.galaxy_cat_inds]
            mcal_g_w_2m    = f['mcal_g_w_2m'][:][self.galaxy_cat_inds]
            no_wgts = np.ones_like(mcal_g_w) #Need this for doing unweighted counts of objects later

            mcal_g_noshear = f['mcal_g_noshear'][:][self.galaxy_cat_inds]
            mcal_g_1p      = f['mcal_g_1p'][:][self.galaxy_cat_inds]
            mcal_g_2p      = f['mcal_g_2p'][:][self.galaxy_cat_inds]
            mcal_g_1m      = f['mcal_g_1m'][:][self.galaxy_cat_inds]
            mcal_g_2m      = f['mcal_g_2m'][:][self.galaxy_cat_inds]


        for q in Quantities:


            N_bins = 20 + 1
            with h5py.File(self.galaxy_cat, 'r') as f:

                if q  == 'SNR':
                    
                    mcal_s2n_noshear = f['mcal_s2n_noshear'][:][self.galaxy_cat_inds]
                    mcal_s2n_1p = f['mcal_s2n_1p'][:][self.galaxy_cat_inds]
                    mcal_s2n_2p = f['mcal_s2n_2p'][:][self.galaxy_cat_inds]
                    mcal_s2n_1m = f['mcal_s2n_1m'][:][self.galaxy_cat_inds]
                    mcal_s2n_2m = f['mcal_s2n_2m'][:][self.galaxy_cat_inds]
                    
                    bin_edge = np.nanpercentile(mcal_s2n_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    A = mcal_s2n_noshear
                    B, C = mcal_s2n_1p, mcal_s2n_1m
                    D, E = mcal_s2n_2p, mcal_s2n_2m
                                        

                elif q == 'Tratio':

                    mcal_T_ratio_noshear = f['mcal_T_ratio_noshear'][:][self.galaxy_cat_inds]
                    mcal_T_ratio_1p = f['mcal_T_ratio_1p'][:][self.galaxy_cat_inds]
                    mcal_T_ratio_2p = f['mcal_T_ratio_2p'][:][self.galaxy_cat_inds]
                    mcal_T_ratio_1m = f['mcal_T_ratio_1m'][:][self.galaxy_cat_inds]
                    mcal_T_ratio_2m = f['mcal_T_ratio_2m'][:][self.galaxy_cat_inds]
                    
                    bin_edge = np.nanpercentile(mcal_T_ratio_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    A = mcal_T_ratio_noshear
                    B, C = mcal_T_ratio_1p, mcal_T_ratio_1m
                    D, E = mcal_T_ratio_2p, mcal_T_ratio_2m


                elif q == 'Tpsf':

                    mcal_psf_T_noshear = f['mcal_psf_T_noshear'][:][self.galaxy_cat_inds]
                    
                    bin_edge = np.nanpercentile(mcal_psf_T_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    A = B = C = D = E = mcal_psf_T_noshear


                elif q == 'e1psf':

                    mcal_psf_g_noshear = f['mcal_psf_g_noshear'][:, 0][self.galaxy_cat_inds]
                    
                    bin_edge = np.nanpercentile(mcal_psf_g_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    A = B = C = D = E = mcal_psf_g_noshear

                elif q == 'e2psf':

                    mcal_psf_g_noshear = f['mcal_psf_g_noshear'][:, 1][self.galaxy_cat_inds]
                    
                    bin_edge = np.nanpercentile(mcal_psf_g_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    A = B = C = D = E = mcal_psf_g_noshear

                
                elif q in ['r_minus_i', 'r_minus_z', 'i_minus_z']:

                    keymatch = {'r' : 0, 'i': 1, 'z': 2} #Hardcoded as this is true for metacal runs
                    
                    f_1 = keymatch[q.split('_')[0]]
                    f_2 = keymatch[q.split('_')[2]]

                    def get_color(label):
                        dered = self.dered
                        c1 = 30 - 2.5*np.log10(f[f'mcal_flux_{label}{dered}'][:][self.galaxy_cat_inds, f_1])
                        c2 = 30 - 2.5*np.log10(f[f'mcal_flux_{label}{dered}'][:][self.galaxy_cat_inds, f_2])

                        return c1 - c2

                    color_noshear = get_color('noshear')
                    color_1p, color_1m = get_color('1p'), get_color('1m')
                    color_2p, color_2m = get_color('2p'), get_color('2m')
                    
                    bin_edge = np.nanpercentile(color_noshear[Mask0], np.linspace(0, 100, N_bins))
                    bin_edge[0], bin_edge[-1] = -99999, 99999
                    
                    
                    A = color_noshear
                    B, C = color_1p, color_1m
                    D, E = color_2p, color_2m


                dgamma = 2*0.01                

                output  = np.zeros([3, self.Npatch, bin_edge.size - 1])
                
                
                def hist(x, y, bin_edges, w, masks):
            
                    x = x[masks]
                    y = y[masks]
                    w = w[masks]

                    return np.histogram(x, bin_edges, weights=y*w)[0]

                mean_noshear = np.average(mcal_g_noshear[Mask0], weights = mcal_g_w[Mask0], axis = 0)
                    
                R11_p  = hist(A, mcal_g_1p[:, 0],      bin_edge, mcal_g_w,    Mask0)
                R11_m  = hist(A, mcal_g_1m[:, 0],      bin_edge, mcal_g_w,    Mask0)
                R11s_p = hist(B, mcal_g_noshear[:, 0], bin_edge, mcal_g_w_1p, Mask1p)
                R11s_m = hist(C, mcal_g_noshear[:, 0], bin_edge, mcal_g_w_1m, Mask1m)
                
                
                R22_p  = hist(A, mcal_g_2p[:, 1],      bin_edge, mcal_g_w,    Mask0)
                R22_m  = hist(A, mcal_g_2m[:, 1],      bin_edge, mcal_g_w,    Mask0)
                R22s_p = hist(D, mcal_g_noshear[:, 1], bin_edge, mcal_g_w_2p, Mask2p)
                R22s_m = hist(E, mcal_g_noshear[:, 1], bin_edge, mcal_g_w_2m, Mask2m)
                
                
                R_counts     = hist(A, no_wgts, bin_edge, mcal_g_w,    Mask0)
                Rs_1p_counts = hist(B, no_wgts, bin_edge, mcal_g_w_1p, Mask1p)
                Rs_1m_counts = hist(C, no_wgts, bin_edge, mcal_g_w_1m, Mask1m)
                Rs_2p_counts = hist(D, no_wgts, bin_edge, mcal_g_w_2p, Mask2p)
                Rs_2m_counts = hist(E, no_wgts, bin_edge, mcal_g_w_2m, Mask2m)
                
                e1 = hist(A, mcal_g_noshear[:, 0] - mean_noshear[0], bin_edge, mcal_g_w, Mask0)
                e2 = hist(A, mcal_g_noshear[:, 1] - mean_noshear[1], bin_edge, mcal_g_w, Mask0)
                X  = hist(A, A, bin_edge, mcal_g_w, Mask0)
                
                #Remove individual patches now
                for j in tqdm(range(self.Npatch), desc = 'shear vs %s' % q):

                    mask = self.gal_inds == j
                
                
                    R11_p_here  = R11_p  - hist(A, mcal_g_1p[:, 0],      bin_edge, mcal_g_w,    Mask0  & mask)
                    R11_m_here  = R11_m  - hist(A, mcal_g_1m[:, 0],      bin_edge, mcal_g_w,    Mask0  & mask)
                    R11s_p_here = R11s_p - hist(B, mcal_g_noshear[:, 0], bin_edge, mcal_g_w_1p, Mask1p & mask)
                    R11s_m_here = R11s_m - hist(C, mcal_g_noshear[:, 0], bin_edge, mcal_g_w_1m, Mask1m & mask)


                    R22_p_here  = R22_p  - hist(A, mcal_g_2p[:, 1],      bin_edge, mcal_g_w,    Mask0  & mask)
                    R22_m_here  = R22_m  - hist(A, mcal_g_2m[:, 1],      bin_edge, mcal_g_w,    Mask0  & mask)
                    R22s_p_here = R22s_p - hist(D, mcal_g_noshear[:, 1], bin_edge, mcal_g_w_2p, Mask2p & mask)
                    R22s_m_here = R22s_m - hist(E, mcal_g_noshear[:, 1], bin_edge, mcal_g_w_2m, Mask2m & mask)


                    R_counts_here     = R_counts     - hist(A, no_wgts, bin_edge, mcal_g_w,    Mask0  & mask)
                    Rs_1p_counts_here = Rs_1p_counts - hist(B, no_wgts, bin_edge, mcal_g_w_1p, Mask1p & mask)
                    Rs_1m_counts_here = Rs_1m_counts - hist(C, no_wgts, bin_edge, mcal_g_w_1m, Mask1m & mask)
                    Rs_2p_counts_here = Rs_2p_counts - hist(D, no_wgts, bin_edge, mcal_g_w_2p, Mask2p & mask)
                    Rs_2m_counts_here = Rs_2m_counts - hist(E, no_wgts, bin_edge, mcal_g_w_2m, Mask2m & mask)


                    e1_here = e1 - hist(A, mcal_g_noshear[:, 0] - mean_noshear[0], bin_edge, mcal_g_w, Mask0 & mask)
                    e2_here = e2 - hist(A, mcal_g_noshear[:, 1] - mean_noshear[1], bin_edge, mcal_g_w, Mask0 & mask)
                    X_here  = X  - hist(A, A, bin_edge, mcal_g_w, Mask0 & mask)
                    
                    
                    R11  = (R11_p_here/R_counts_here - R11_m_here/R_counts_here)/dgamma
                    R11s = (R11s_p_here/Rs_1p_counts_here - R11s_m_here/Rs_1m_counts_here)/dgamma
                    R11_tot = R11 + R11s
                    
                    R22  = (R22_p_here/R_counts_here - R22_m_here/R_counts_here)/dgamma
                    R22s = (R22s_p_here/Rs_2p_counts_here - R22s_m_here/Rs_2m_counts_here)/dgamma
                    R22_tot = R22 + R22s
                    
                    output[0, j] = X_here/R_counts_here
                    output[1, j] = e1_here/R_counts_here / R11_tot
                    output[2, j] = e2_here/R_counts_here / R22_tot
                    
                
            savepath = self.output_path + '/e_vs_%s.npy' % q
            np.save(savepath, output)
    

    @timeit
    def tangential_shear_field_centers(self):


        #First load the field centers
        fcenters = pd.read_csv('/project/chihway/dhayaa/DECADE/FieldCenters_DR3_1_20240305.csv')
        fcenters = fcenters.drop_duplicates()
        fc_ra  = np.array(fcenters['RADEG'])
        fc_dec = np.array(fcenters['DECDEG'])
        
        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

        # #####################################################
        # #Use only DR3_1_1 region to replicate Jackie
        # DR3_1_Mask = np.invert((fc_ra < 180) & (fc_dec > -30))
        # fc_ra  = fc_ra[DR3_1_Mask]
        # fc_dec = fc_dec[DR3_1_Mask]
        
        # DR3_1_Mask = np.invert((gal_ra < 180) & (gal_dec > -30))
        # gal_ra  = gal_ra[DR3_1_Mask]
        # gal_dec = gal_dec[DR3_1_Mask]
        
        # gal_w = gal_w[DR3_1_Mask]
        # gal_g1 = gal_g1[DR3_1_Mask]
        # gal_g2 = gal_g2[DR3_1_Mask]
        
        # print(DR3_1_Mask.size, np.average(DR3_1_Mask), "FINISHED DR3_1_1 maker")
        # #####################################################
        
        #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
        for a in [gal_g1, gal_g2]:
            a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
        
        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'

        Nth    = int(len(gal_ra)/10_000_000) #Select every Nth object such that we end up using 10 million to define patches
        if Nth < 1: Nth = 1
        small_cat = treecorr.Catalog(ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg',dec_units='deg', npatch = self.Npatch)
        small_cat.write_patch_centers(center_path)
        del small_cat 
        
        #NOW MAKE A RANDOMS CATALOG
        N_randoms = 100_000_000 #Doing rejection sampling so start with many more points than needed
        phi   = np.random.uniform(0, 2*np.pi, N_randoms)
        theta = np.arccos(1 - 2*np.random.uniform(0, 1, N_randoms))

        NSIDE = self.MapNSIDE_weightrands
        # Remove points that aren't within the galaxy Mask
        hpix = hp.ang2pix(NSIDE, gal_ra, gal_dec, lonlat  = True)
        Ngal = np.bincount(hpix, minlength = hp.nside2npix(NSIDE))
        hpix = hp.ang2pix(NSIDE, theta, phi)
        pix_mask   = Ngal[hpix] > 0
        phi, theta = phi[pix_mask], theta[pix_mask]
        
        
        hpix = hp.ang2pix(NSIDE, fc_ra, fc_dec, lonlat  = True)
        pix_mask   = Ngal[hpix] > 0
        fc_ra, fc_dec = fc_ra[pix_mask], fc_dec[pix_mask]
        
        #Assign weights to the exps and to randoms so it is a uniform field
        hpix = hp.ang2pix(NSIDE, fc_ra, fc_dec, lonlat  = True)
        Ncen = np.bincount(hpix, minlength = hp.nside2npix(NSIDE))
        
        wmap = np.where(Ncen == 0, 0, (Ngal/Ncen)) #Make weights that make field centers roughly match the galaxy distributions.
        hpix = hp.ang2pix(NSIDE, fc_ra, fc_dec, lonlat  = True)
        fc_w = wmap[hpix]
        hpix = hp.ang2pix(NSIDE, theta, phi)
        rand_w = Ngal[hpix] #Don't downweight randoms by Ncen (they're already uniform across sky), just upweight by galaxies
        
        #convert to RA and DEC
        rand_ra  = phi*180/np.pi
        rand_dec = 90 - theta*180/np.pi
        
        #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
        cat_g = treecorr.Catalog(g1 = gal_g1, g2 = gal_g2, ra = gal_ra, dec = gal_dec, w = gal_w, ra_units='deg', dec_units='deg', patch_centers=center_path)
        cat_t = treecorr.Catalog(ra = fc_ra,   dec = fc_dec,   w = fc_w,   ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_r = treecorr.Catalog(ra = rand_ra, dec = rand_dec, w = rand_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        
        del gal_g1, gal_g2, gal_ra, gal_dec, gal_w
        del rand_ra, rand_dec
        
        #Compute the rowe stats
        NG = treecorr.NGCorrelation(nbins = 25, min_sep = 2.5, max_sep = 250, rng = np.random.default_rng(seed = 42),
                                    sep_units = 'arcmin',verbose = 0, bin_slop = 0.1, var_method='jackknife')
        RG = NG.copy()
        
        NG.process(cat_t, cat_g, low_mem=True)
        NG.write(os.path.join(self.output_path, 'fieldcenter_treecorr.txt'))
        cov_jk = treecorr.estimate_multi_cov([NG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
        np.savetxt(os.path.join(self.output_path, 'fieldcenter_cov_treecorr.txt'), cov_jk)
        
        print("FINISHED true_shear")

        
        RG.process(cat_r, cat_g, low_mem=True)
        RG.write(os.path.join(self.output_path, 'fieldcenter_rands_treecorr.txt'))
        cov_jk = treecorr.estimate_multi_cov([RG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
        np.savetxt(os.path.join(self.output_path, 'fieldcenter_rands_cov_treecorr.txt'), cov_jk)
        
        print("FINISHED rand_shear")
        
        cov_jk = treecorr.estimate_multi_cov([NG, RG], 'jackknife', func = lambda x : np.concatenate([x[0].xi - x[1].xi, 
                                                                                                      x[0].xi_im - x[1].xi_im]))
        np.savetxt(os.path.join(self.output_path, 'fieldcenter_diff_cov_treecorr.txt'), cov_jk)

    @timeit
    def star_weights_map(self, gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = 256):

        #NOW COMPUTE STAR WEIGHTS
        star   = np.zeros(hp.nside2npix(NSIDE))
        galaxy = np.zeros(hp.nside2npix(NSIDE))

        pix = hp.ang2pix(NSIDE, gal_ra, gal_dec, lonlat = True)
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        galaxy[unique_pix] += np.bincount(idx_rep, weights = gal_w)
        
        
        pix = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)        
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        star[unique_pix] += np.bincount(idx_rep)

        weight_map = np.zeros_like(star)
        weight_map[star != 0] = galaxy[star != 0]/star[star != 0]
        
        return weight_map
    
    
    @timeit
    def gal_counts_map(self, gal_ra, gal_dec, gal_w, NSIDE = 256):

        #NOW COMPUTE STAR WEIGHTS
        galaxy = np.zeros(hp.nside2npix(NSIDE))

        pix = hp.ang2pix(NSIDE, gal_ra, gal_dec, lonlat = True)
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        galaxy[unique_pix] += np.bincount(idx_rep, weights = gal_w)
        
        weight_map = galaxy

        return weight_map
    
    
    @timeit
    def tangential_shear_stars(self):
        

        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

            #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [gal_g1, gal_g2]:
                a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
        
        print("CALIBRATED/DONE")
            
        
        for m_range, m_name in zip([[-1000, 16.5], [16.5, 10000]], ['bright', 'faint']):
            
            with h5py.File(self.psf_cat, 'r') as f:

                psf_ra   = f['ra'][:][self.psf_cat_inds]
                psf_dec  = f['dec'][:][self.psf_cat_inds]

                band = f['BAND'][:].astype('U1')[self.psf_cat_inds]
                mag  = f['MAGZP'][:][self.psf_cat_inds] - 2.5*np.log10(f['FLUX_AUTO'][:])[self.psf_cat_inds] #Use this instead of MAG_AUTO so we use the better zeropoints
                SNR  = f['FLUX_AUTO'][:][self.psf_cat_inds]/f['FLUXERR_AUTO'][:][self.psf_cat_inds]


                No_Gband  = band != 'g' #We don't use g-band in shear
                SNR_Mask  = SNR > self.Star_SNR_min
                Mag_Mask  = (mag > m_range[0]) & (mag < m_range[1])

                print(np.sum(No_Gband), np.sum(SNR_Mask), np.sum(Mag_Mask))
                Mask = SNR_Mask & No_Gband & Mag_Mask

                print("TOTAL NUM", np.sum(Mask))
                psf_ra   = psf_ra[Mask]
                psf_dec  = psf_dec[Mask]

                del Mask, SNR_Mask, No_Gband, Mag_Mask, band, mag, SNR

            print("LOADED EVERYTHING")


            NSIDE      = self.MapNSIDE_weightrands
            weight_map = self.star_weights_map(gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = NSIDE)
            pix        = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)
            psf_w      = weight_map[pix] #Assign individual stars weights from the map

            #Remove stars that are not in the galaxy sample's footprint
            Mask     = psf_w > 0
            psf_ra   = psf_ra[Mask]
            psf_dec  = psf_dec[Mask]
            psf_w    = psf_w[Mask]
            del pix

            #NOW MAKE A RANDOMS CATALOG
            weight_map = self.gal_counts_map(gal_ra, gal_dec, gal_w, NSIDE = NSIDE)
            N_randoms  = 300_000_000 #Doing rejection sampling so start with many more points than needed
            phi   = np.random.uniform(0, 2*np.pi, N_randoms)
            theta = np.arccos(1 - 2*np.random.uniform(0, 1, N_randoms))

            # Remove points that aren't within the galaxy Mask
            gmask      = np.bincount(hp.ang2pix(4096, gal_ra, gal_dec, lonlat = True), minlength = hp.nside2npix(4096)) > 0
            pix_mask   = gmask[hp.ang2pix(4096, theta, phi)] > 0; del gmask;
            phi, theta = phi[pix_mask], theta[pix_mask]; del pix_mask;
            rand_w     = weight_map[hp.ang2pix(NSIDE, theta, phi)]
            
            print("USING N_RANDOMS = ", len(rand_w))
            
            # psf_w    = np.ones_like(psf_w);  print("FORCEFULLY SETTING STAR WEIGHTS TO 1")
            # rand_w   = np.ones_like(rand_w); print("FORCEFULLY SETTING RAND WEIGHTS TO 1")

            #convert to RA and DEC
            rand_ra  = phi*180/np.pi
            rand_dec = 90 - theta*180/np.pi
            center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'

            Nth    = int(len(gal_ra)/10_000_000) #Select every Nth object such that we end up using 10 million to define patches
            if Nth < 1: Nth = 1
            small_cat = treecorr.Catalog(ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg',dec_units='deg', npatch = self.Npatch)
            small_cat.write_patch_centers(center_path)
            del small_cat 

            #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
            cat_g = treecorr.Catalog(g1 = gal_g1, g2 = gal_g2, ra = gal_ra, dec = gal_dec, w = gal_w, ra_units='deg', dec_units='deg', patch_centers=center_path)
            cat_s = treecorr.Catalog(ra = psf_ra,  dec = psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
            cat_r = treecorr.Catalog(ra = rand_ra, dec = rand_dec, w = rand_w, ra_units='deg',dec_units='deg', patch_centers=center_path)

            del rand_ra, rand_dec

            #Compute the rowe stats
            NG = treecorr.NGCorrelation(nbins = 25, min_sep = 2.5, max_sep = 250, rng = np.random.default_rng(seed = 42),
                                        sep_units = 'arcmin',verbose = 0, bin_slop = 0.1, var_method='jackknife')
            
            RG = NG.copy()

            NG.process(cat_s, cat_g, low_mem=True)
            NG.write(os.path.join(self.output_path, 'starshears_%s_treecorr.txt' % m_name))
            cov_jk = treecorr.estimate_multi_cov([NG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
            np.savetxt(os.path.join(self.output_path, 'starshears_%s_cov_treecorr.txt' % m_name), cov_jk)

            RG.process(cat_r, cat_g, low_mem=True)
            RG.write(os.path.join(self.output_path, 'starshears_%s_rands_treecorr.txt' % m_name))
            cov_jk = treecorr.estimate_multi_cov([RG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
            np.savetxt(os.path.join(self.output_path, 'starshears_%s_rands_cov_treecorr.txt' % m_name), cov_jk)
            
            cov_jk = treecorr.estimate_multi_cov([NG, RG], 'jackknife', func = lambda x : np.concatenate([x[0].xi - x[1].xi, 
                                                                                                          x[0].xi_im - x[1].xi_im])) 
            np.savetxt(os.path.join(self.output_path, 'starshears_%s_diff_cov_treecorr.txt' % m_name), cov_jk)
            
            
    @timeit
    def tangential_shear_coadd_stars(self):
        

        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

            #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [gal_g1, gal_g2]:
                a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
        
        print("CALIBRATED/DONE")
            
        
        for m_range, m_name in zip([[-1000, 16.5], [16.5, 10000]], ['bright', 'faint']):
            
            Mask = self.get_star_Mask()
            
            print(m_range,  np.sum(Mask),)
            with h5py.File(self.galaxy_cat, 'r') as f:

                #Calling this psf for historical reasons in the code.
                #But these are actually coadd stars.
                psf_ra   = f['RA'][:][Mask]
                psf_dec  = f['DEC'][:][Mask]

                mag  = 30 - 2.5*np.log10(f['mcal_flux_noshear'][:, 1])[Mask] # i-band magnitude
                SNR  = f['mcal_s2n_noshear'][:][Mask]

                SNR_Mask  = SNR > 10
                Mag_Mask  = (mag > m_range[0]) & (mag < m_range[1])

                Mask = SNR_Mask & Mag_Mask

                print("TOTAL NUM", np.sum(Mask))
                psf_ra   = psf_ra[Mask]
                psf_dec  = psf_dec[Mask]

                del Mask, SNR_Mask, Mag_Mask, mag, SNR

            print("LOADED EVERYTHING")


            NSIDE      = self.MapNSIDE_weightrands
            weight_map = self.star_weights_map(gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = NSIDE)
            pix        = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)
            psf_w      = weight_map[pix] #Assign individual stars weights from the map

            #Remove stars that are not in the galaxy sample's footprint
            Mask     = psf_w > 0
            psf_ra   = psf_ra[Mask]
            psf_dec  = psf_dec[Mask]
            psf_w    = psf_w[Mask]
            del pix


            #NOW MAKE A RANDOMS CATALOG
            N_randoms = 100_000_000 #Doing rejection sampling so start with many more points than needed
            phi   = np.random.uniform(0, 2*np.pi, N_randoms)
            theta = np.arccos(1 - 2*np.random.uniform(0, 1, N_randoms))

            # Remove points that aren't within the galaxy Mask
            hpix = hp.ang2pix(NSIDE, theta, phi)
            pix_mask   = weight_map[hpix] > 0
            phi, theta = phi[pix_mask], theta[pix_mask]
            rand_w     = weight_map[hpix][pix_mask]

            #convert to RA and DEC
            rand_ra  = phi*180/np.pi
            rand_dec = 90 - theta*180/np.pi
            center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'

            Nth    = int(len(gal_ra)/10_000_000) #Select every Nth object such that we end up using 10 million to define patches
            if Nth < 1: Nth = 1
            small_cat = treecorr.Catalog(ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg',dec_units='deg', npatch = self.Npatch)
            small_cat.write_patch_centers(center_path)
            del small_cat 

            #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
            cat_g = treecorr.Catalog(g1 = gal_g1, g2 = gal_g2, ra = gal_ra, dec = gal_dec, w = gal_w, ra_units='deg', dec_units='deg', patch_centers=center_path)
            cat_s = treecorr.Catalog(ra = psf_ra,  dec = psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
            cat_r = treecorr.Catalog(ra = rand_ra, dec = rand_dec, w = rand_w, ra_units='deg',dec_units='deg', patch_centers=center_path)

            del rand_ra, rand_dec

            #Compute the rowe stats
            NG = treecorr.NGCorrelation(nbins = 25, min_sep = 2.5, max_sep = 250, rng = np.random.default_rng(seed = 42),
                                        sep_units = 'arcmin',verbose = 0, bin_slop = 0.1, var_method='jackknife')

            NG.process(cat_s, cat_g, low_mem=True)
            NG.write(os.path.join(self.output_path, 'coadd_starshears_%s_treecorr.txt' % m_name))
            cov_jk = treecorr.estimate_multi_cov([NG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
            np.savetxt(os.path.join(self.output_path, 'coadd_starshears_%s_cov_treecorr.txt' % m_name), cov_jk)

            NG.process(cat_r, cat_g, low_mem=True)
            NG.write(os.path.join(self.output_path, 'coadd_starshears_%s_rands_treecorr.txt' % m_name))
            cov_jk = treecorr.estimate_multi_cov([NG], 'jackknife', func = lambda x : np.concatenate([x[0].xi, x[0].xi_im])) 
            np.savetxt(os.path.join(self.output_path, 'coadd_starshears_%s_rands_cov_treecorr.txt' % m_name), cov_jk)
            
            
    @timeit
    def Bmodes(self):


        nside = 1024
        npix  = hp.nside2npix(nside)

        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            g1, g2 = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T
            
             #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [g1, g2]:
                a -= np.average(a, weights = w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        g1, g2 = g1/R11, g2/R22

        #Need a -1 for g2 in R (not T) due to Namaster definition
        T = treecorr.Catalog(g1 = g1[::], g2 = g2[::], ra = ra[::], dec = dec[::], w = w[::], ra_units='deg', dec_units='deg', npatch = self.Npatch)
        #R = self.MakeMapFromCat(ra = ra, dec = dec, e1 = g1, e2 = -g2, w = w, NSIDE = nside); del ra, dec, g1, g2
        #C = self.MakeMapFromCls(self.sim_Cls, NSIDE = nside)
        
        
        ##############################################
        # No harmonic-space Bmodes in this test
        ##############################################
        ##Process regular Bmodes
        #X    = self.BmodeRunner(R, C, 42, njobs = 1)
        #data = X.process_data()
        #Cov  = X.process_noise(100) #Make a lot more sims so we dont get hit by Hartlap factor
        #
        #np.save(self.output_path + '/Bmode.npy', np.vstack([data,  X.ell_eff]))
        #np.save(self.output_path + '/Bmode_Noise.npy', Cov)
        #
        #
        ##Process pure Bmodes
        #X    = self.PureBmodeRunner(R, C, 42, njobs = 1)
        #data = X.process_data()
        #Cov  = X.process_noise(100) #Make a lot more sims so we dont get hit by Hartlap factor

        #np.save(self.output_path + '/PureBmode.npy', np.vstack([data,  X.ell_eff]))
        #np.save(self.output_path + '/PureBmode_Noise.npy', Cov)
        
        
        #Process Matt's real-space E/B estimator
        X = self.MRBmodeRunner(T, theta_min = 2.5, theta_max = 250, Ntheta = 250, Nmodes = 40)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_treecorr.txt'))
        np.save(self.output_path + '/MRBmode_E.npy', E)
        np.save(self.output_path + '/MRBmode_B.npy', B)
        np.save(self.output_path + '/MRBmode_E_Cov.npy', cov_E)
        np.save(self.output_path + '/MRBmode_B_Cov.npy', cov_B)
        

    @timeit
    def rho_stats(self):
                
        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

            #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [gal_g1, gal_g2]:
                a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
            

        with h5py.File(self.psf_cat, 'r') as f:
        
            psf_ra   = f['ra'][:][self.psf_cat_inds]
            psf_dec  = f['dec'][:][self.psf_cat_inds]

            g1_star  = f['g1_star_hsm'][:][self.psf_cat_inds]
            g2_star  = f['g2_star_hsm'][:][self.psf_cat_inds]
            g1_model = f['g1_model_hsm'][:][self.psf_cat_inds]
            g2_model = f['g2_model_hsm'][:][self.psf_cat_inds]

            w1 = g1_star * (f['T_star_hsm'][:][self.psf_cat_inds] - f['T_model_hsm'][:][self.psf_cat_inds])/f['T_star_hsm'][:][self.psf_cat_inds]
            w2 = g2_star * (f['T_star_hsm'][:][self.psf_cat_inds] - f['T_model_hsm'][:][self.psf_cat_inds])/f['T_star_hsm'][:][self.psf_cat_inds]

            q1 = g1_star - g1_model
            q2 = g2_star - g2_model

            del g1_star, g2_star
            
            band = f['BAND'][:][self.psf_cat_inds]
            SNR  = f['FLUX_AUTO'][:][self.psf_cat_inds]/f['FLUXERR_AUTO'][:][self.psf_cat_inds]
            
            No_Gband  = band != b'g' #We don't use g-band in shear
            SNR_Mask  = SNR > self.Star_SNR_min

            print(np.sum(No_Gband), np.sum(SNR_Mask))
            Mask = SNR_Mask & No_Gband
            
            print("TOTAL NUM", np.sum(Mask))
            psf_ra   = psf_ra[Mask]
            psf_dec  = psf_dec[Mask]
            g1_model = g1_model[Mask]
            g2_model = g2_model[Mask]
            q1  = q1[Mask]
            q2  = q2[Mask]
            w1  = w1[Mask]
            w2  = w2[Mask]
            
            del Mask, SNR_Mask, No_Gband, band, SNR
        
        print("LOADED EVERYTHING")

        NSIDE      = self.MapNSIDE_weightrands
        weight_map = self.star_weights_map(gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = NSIDE)
        pix        = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)
        psf_w      = weight_map[pix] #Assign individual stars weights from the map

        #Remove stars that are not in the galaxy sample's footprint
        Mask     = psf_w > 0
        psf_ra   = psf_ra[Mask]
        psf_dec  = psf_dec[Mask]
        psf_w    = psf_w[Mask]
        q1, q2   = q1[Mask], q2[Mask]
        w1, w2   = w1[Mask], w2[Mask]
        g1_model = g1_model[Mask]
        g2_model = g2_model[Mask]
        del pix

        print("LOADED EVERYTHING")

        #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
        for a in [gal_g1, gal_g2]:
            a -= np.average(a, weights = gal_w)
            
        for a in [g1_model, g2_model, q1, q2, w1, w2]:
            a -= np.average(a, weights = psf_w)


        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        Nth    = int(len(gal_g1)/20_000_000) #Select every Nth object such that we end up using 20 million to define patches
        if Nth < 1: Nth = 1
        small_cat = treecorr.Catalog(g1=gal_g1[::Nth], g2=gal_g1[::Nth], ra=gal_ra[::Nth], dec=gal_dec[::Nth], 
                                     ra_units='deg', dec_units='deg', npatch = self.Npatch)
        small_cat.write_patch_centers(center_path)
        del small_cat
        
        
        
        ########################################################################################################################
        #NOW MAKE THE CATALOGS
        ########################################################################################################################
        
        #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
        cat_g = treecorr.Catalog(g1=gal_g1,   g2=gal_g2,   ra=gal_ra, dec=gal_dec, w = gal_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_e = treecorr.Catalog(g1=g1_model, g2=g2_model, ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_q = treecorr.Catalog(g1=q1,       g2=q2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_w = treecorr.Catalog(g1=w1,       g2=w2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)

        ########################################################################################################################
        #Compute the shear 2pt
        ########################################################################################################################

        GG = treecorr.GGCorrelation(nbins = 25, min_sep = 0.1, max_sep = 250, rng = np.random.default_rng(seed = 42),
                                    sep_units = 'arcmin',verbose = 0,bin_slop = 0.1, var_method='jackknife')
        GG.process(cat_g, low_mem=True)
        GG.write(os.path.join(self.output_path, 'taustats_shear_2pt_treecorr.txt'))


        ########################################################################################################################
        #Compute the rowe stats
        ########################################################################################################################

        EE = GG.copy()
        EE.process(cat_e, low_mem=True)
        EE.write(os.path.join(self.output_path, 'taustats_rho0_treecorr.txt'))
        

        QQ = GG.copy()
        QQ.process(cat_q, low_mem=True)
        QQ.write(os.path.join(self.output_path, 'taustats_rho1_treecorr.txt'))
        
        EQ = GG.copy()
        EQ.process(cat_e, cat_q, low_mem=True)
        EQ.write(os.path.join(self.output_path, 'taustats_rho2_treecorr.txt'))
        
        WW = GG.copy()
        WW.process(cat_w, low_mem=True)
        WW.write(os.path.join(self.output_path, 'taustats_rho3_treecorr.txt'))
        
        QW = GG.copy()
        QW.process(cat_q, cat_w, low_mem=True)
        QW.write(os.path.join(self.output_path, 'taustats_rho4_treecorr.txt'))
        
        EW = GG.copy()
        EW.process(cat_e, cat_w, low_mem=True)
        EW.write(os.path.join(self.output_path, 'taustats_rho5_treecorr.txt'))
        
        ########################################################################################################################
        #Compute the tau stats
        ########################################################################################################################

        GE = GG.copy()
        GE.process(cat_g, cat_e, low_mem=True)
        GE.write(os.path.join(self.output_path, 'taustats_tau0_treecorr.txt'))
        
        GQ = GG.copy()
        GQ.process(cat_g, cat_q, low_mem=True)
        GQ.write(os.path.join(self.output_path, 'taustats_tau1_treecorr.txt'))
        
        GW = GG.copy()
        GW.process(cat_g, cat_w, low_mem=True)
        GW.write(os.path.join(self.output_path, 'taustats_tau2_treecorr.txt'))
        

        #Get the covariance matrix
        cov_jk = treecorr.estimate_multi_cov([GG, EE, QQ, EQ, WW, QW, EW, GE, GQ, GW], 'jackknife')
        np.savetxt(os.path.join(self.output_path, 'taustats_All_cov_treecorr.txt'), cov_jk)
        
        
    @timeit
    def check_shear_2pt(self):
                
        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

            #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [gal_g1, gal_g2]:
                a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
            
        print("LOADED EVERYTHING")

        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        Nth    = int(len(gal_g1)/20_000_000) #Select every Nth object such that we end up using 20 million to define patches
        if Nth < 1: Nth = 1
        small_cat = treecorr.Catalog(g1=gal_g1[::Nth], g2=gal_g1[::Nth], ra=gal_ra[::Nth], dec=gal_dec[::Nth], 
                                     ra_units='deg', dec_units='deg', npatch = self.Npatch)
        small_cat.write_patch_centers(center_path)
        del small_cat
        
        
        
        ########################################################################################################################
        #NOW MAKE THE CATALOGS
        ########################################################################################################################
        
        #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
        cat_g = treecorr.Catalog(g1=gal_g1,   g2=gal_g2,   ra=gal_ra, dec=gal_dec, w = gal_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        
        ########################################################################################################################
        #Compute the shear 2pt
        ########################################################################################################################

        GG = treecorr.GGCorrelation(nbins = 25, min_sep = 0.1, max_sep = 250, rng = np.random.default_rng(seed = 42),
                                    sep_units = 'arcmin',verbose = 0,bin_slop = 0.1, var_method='jackknife')
        GG.process(cat_g, low_mem=True)
        GG.write(os.path.join(self.output_path, 'taustats_shear_2pt_treecorr.txt'))

        #Get the covariance matrix
        cov_jk = treecorr.estimate_multi_cov([GG], 'jackknife')
        np.savetxt(os.path.join(self.output_path, 'taustats_GG_cov_treecorr.txt'), cov_jk)


    @timeit
    def MRBmode_psf(self):
        
        Mask = self.get_mcal_Mask('noshear')

        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            gal_g1, gal_g2  = f['mcal_g_noshear'][:][self.galaxy_cat_inds][Mask].T

            #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
            for a in [gal_g1, gal_g2]:
                a -= np.average(a, weights = gal_w)

        R11, R22 = self.compute_response(np.ones_like(Mask).astype(bool))
        gal_g1, gal_g2 = gal_g1/R11, gal_g2/R22
        
        
        with h5py.File(self.psf_cat, 'r') as f:
        
            psf_ra   = f['ra'][:][self.psf_cat_inds]
            psf_dec  = f['dec'][:][self.psf_cat_inds]

            g1_star  = f['g1_star_hsm'][:][self.psf_cat_inds]
            g2_star  = f['g2_star_hsm'][:][self.psf_cat_inds]
            g1_model = f['g1_model_hsm'][:][self.psf_cat_inds]
            g2_model = f['g2_model_hsm'][:][self.psf_cat_inds]

            w1 = g1_star * (f['T_star_hsm'][:][self.psf_cat_inds] - f['T_model_hsm'][:][self.psf_cat_inds])/f['T_star_hsm'][:][self.psf_cat_inds]
            w2 = g2_star * (f['T_star_hsm'][:][self.psf_cat_inds] - f['T_model_hsm'][:][self.psf_cat_inds])/f['T_star_hsm'][:][self.psf_cat_inds]

            q1 = g1_star - g1_model
            q2 = g2_star - g2_model

            del g1_star, g2_star
            
            band = f['BAND'][:][self.psf_cat_inds]
            SNR  = f['FLUX_AUTO'][:][self.psf_cat_inds]/f['FLUXERR_AUTO'][:][self.psf_cat_inds]
            
            No_Gband  = band != b'g' #We don't use g-band in shear
            SNR_Mask  = SNR > self.Star_SNR_min

            print(np.sum(No_Gband), np.sum(SNR_Mask))
            Mask = SNR_Mask & No_Gband
            
            print("TOTAL NUM", np.sum(Mask))
            psf_ra   = psf_ra[Mask]
            psf_dec  = psf_dec[Mask]
            g1_model = g1_model[Mask]
            g2_model = g2_model[Mask]
            q1  = q1[Mask]
            q2  = q2[Mask]
            w1  = w1[Mask]
            w2  = w2[Mask]
            
            del Mask, SNR_Mask, No_Gband, band, SNR
        
        print("LOADED EVERYTHING")

        NSIDE      = self.MapNSIDE_weightrands
        weight_map = self.star_weights_map(gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = NSIDE)
        pix        = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)
        psf_w      = weight_map[pix] #Assign individual stars weights from the map

        #Remove stars that are not in the galaxy sample's footprint
        Mask     = psf_w > 0
        psf_ra   = psf_ra[Mask]
        psf_dec  = psf_dec[Mask]
        psf_w    = psf_w[Mask]
        q1, q2   = q1[Mask], q2[Mask]
        w1, w2   = w1[Mask], w2[Mask]
        g1_model = g1_model[Mask]
        g2_model = g2_model[Mask]
        del pix

        print("LOADED EVERYTHING")

        #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
        for a in [gal_g1, gal_g2]:
            a -= np.average(a, weights = gal_w)
            
        for a in [g1_model, g2_model, q1, q2, w1, w2]:
            a -= np.average(a, weights = psf_w)


        if os.path.isfile(os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'):
            os.system('rm %s' % os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp')
        
        center_path = os.environ['TMPDIR'] + '/Patch_centers_TreeCorr_tmp'
        Nth    = int(len(gal_g1)/20_000_000) #Select every Nth object such that we end up using 20 million to define patches
        if Nth < 1: Nth = 1
        small_cat = treecorr.Catalog(ra=gal_ra[::Nth], dec=gal_dec[::Nth], ra_units='deg', dec_units='deg', npatch = self.Npatch)
        small_cat.write_patch_centers(center_path)
        del small_cat
                          
              
        def custom_write(label, E, B, cov_E, cov_B):
        
            np.save(self.output_path + '/MRBmode_%s_E.npy' % label, E)
            np.save(self.output_path + '/MRBmode_%s_B.npy' % label, B)
            np.save(self.output_path + '/MRBmode_%s_E_Cov.npy' % label, cov_E)
            np.save(self.output_path + '/MRBmode_%s_B_Cov.npy' % label, cov_B)
        
        
        ########################################################################################################################
        #NOW MAKE THE CATALOGS
        ########################################################################################################################
        
        #DONT USE SAVE_PATCH_DIR. DOESN'T WORK WELL FOR WHAT WE NEED
        cat_e = treecorr.Catalog(g1=g1_model, g2=g2_model, ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_q = treecorr.Catalog(g1=q1,       g2=q2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)
        cat_w = treecorr.Catalog(g1=w1,       g2=w2,       ra=psf_ra, dec=psf_dec, w = psf_w, ra_units='deg',dec_units='deg', patch_centers=center_path)

        ########################################################################################################################
        #Compute the rowe stats
        ########################################################################################################################

        #Process Matt's real-space E/B estimator
        X = self.MRBmodeRunner(cat_e, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_EE_treecorr.txt'))
        custom_write('EE', E, B, cov_E, cov_B)
        
        
        X = self.MRBmodeRunner(cat_q, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_QQ_treecorr.txt'))
        custom_write('QQ', E, B, cov_E, cov_B)
        
        
        X = self.MRBmodeRunner(cat_e, cat_q, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_EQ_treecorr.txt'))
        custom_write('EQ', E, B, cov_E, cov_B)
        
        
        X = self.MRBmodeRunner(cat_w, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_WW_treecorr.txt'))
        custom_write('WW', E, B, cov_E, cov_B)
        
        
        X = self.MRBmodeRunner(cat_q, cat_w, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_QW_treecorr.txt'))
        custom_write('QW', E, B, cov_E, cov_B)
        
        
        X = self.MRBmodeRunner(cat_e, cat_w, theta_min = 2.5, theta_max = 250, Ntheta = 100, Nmodes = 20)
        E, B, cov_E, cov_B, corr = X.compute_EB()
        corr.write(os.path.join(self.output_path, 'MRBmode_EW_treecorr.txt'))
        custom_write('EW', E, B, cov_E, cov_B)
        
        
    @timeit
    def psf_color(self):


        #Start by building PSF color matching
        with h5py.File(self.psf_cat, 'r') as f:

            ra   = f['ra'][:][self.psf_cat_inds]
            dec  = f['dec'][:][self.psf_cat_inds]
            
            hpix = hp.ang2pix(8192 * 32, ra, dec, lonlat = True)
            unique_hpix, inds, unique_counts = np.unique(hpix, return_inverse = True, return_counts = True)
            
            out  = np.zeros([3, hpix.size], dtype = np.float32) + np.NaN
            
            del hpix, unique_counts
            
            #Get selections for individual bands first
            BAND  = f['BAND'][:][self.psf_cat_inds]
            Masks = [np.where(BAND == b'r')[0], np.where(BAND == b'i')[0], np.where(BAND == b'z')[0]]
            
            flux = f['FLUX_AUTO_DERED_SFD98'][:][self.psf_cat_inds]
            SNR  = flux/f['FLUXERR_AUTO_DERED_SFD98'][:][self.psf_cat_inds]
            ZP   = f['MAGZP'][:][self.psf_cat_inds]
            mag  = ZP - 2.5 * np.log10(flux)
            
            del flux, ZP
            
            #This helps avoid nan errors later down the line
            #We set the mag to 0 but we don't really use it 
            #since the SNR will also be zero
            SNR  = np.where(SNR < 0, 0, SNR)
            mag  = np.where(np.isfinite(mag), mag, 0)
            
            print("LOADED RA, DEC, SNR")
            
            for i in tqdm(range(len(Masks)), desc = 'Matching bands'):
                
                m = Masks[i]

                Band_Matcher = self.StarMatch(ra = ra[m],  dec = dec[m])
                weighted_mag = Band_Matcher.weighted_average(mag[m], weights = SNR[m]) #Average all quantities together
                    
                Masks[i] = None #This just "deletes" the mask in here.
                
                weighted_mag_map    = np.zeros(unique_hpix.size) + np.NaN
                tmp, inds_a, inds_b = np.intersect1d(unique_hpix, Band_Matcher.UniquePix, return_indices = True)
                
                print("Matched index size:", len(tmp))
                
                weighted_mag_map[inds_a] = weighted_mag[inds_b]; del weighted_mag
            
                out[i, :] = weighted_mag_map[inds]; del weighted_mag_map


            matched_ra, matched_dec = hp.pix2ang(8192 * 32, unique_hpix, lonlat = True)

            psf_inds = (treecorr.Catalog(ra = matched_ra, dec = matched_dec, ra_units='deg',dec_units='deg', npatch = self.Npatch)
                                .getNField()
                                .run_kmeans(self.Npatch))
                       

        del Band_Matcher, ra, dec, SNR, Masks

        np.save(os.environ['TMPDIR'] + '/psf_color_matched.npy', out)
        np.save(os.environ['TMPDIR'] + '/psf_color_pos.npy', np.vstack([matched_ra, matched_dec]).T)

        Mask = self.get_mcal_Mask('noshear')
        with h5py.File(self.galaxy_cat, 'r') as f:

            dered   = self.dered
            gal_ra  = f['RA'][:][self.galaxy_cat_inds][Mask]
            gal_dec = f['DEC'][:][self.galaxy_cat_inds][Mask]
            gal_w   = f['mcal_g_w_noshear'][:][self.galaxy_cat_inds][Mask]
            mag_r = 30 -2.5*np.log10(f[f'mcal_flux_noshear{dered}'][:][self.galaxy_cat_inds, 0][Mask])

            mcal_m_r, mcal_m_i, mcal_m_z = 30 - 2.5*np.log10(f[f'mcal_flux_noshear{dered}'][:][self.galaxy_cat_inds][Mask]).T
            
            
            with open(self.output_path + '/PSF_summaries.txt', 'a') as f:
                print("GAL r-z [95% bounds]", np.round(np.nanquantile(mcal_m_r - mcal_m_z, [0.025, 0.5, 0.975]), 3), file = f)
                print("GAL r-i [95% bounds]", np.round(np.nanquantile(mcal_m_r - mcal_m_i, [0.025, 0.5, 0.975]), 3), file = f)
                print("GAL i-z [95% bounds]", np.round(np.nanquantile(mcal_m_i - mcal_m_z, [0.025, 0.5, 0.975]), 3), file = f)
                
        
    
        #'/project/chihway/dhayaa/DECADE/star_psf_shapecatalog_20230510.hdf5'
        with h5py.File(self.psf_cat, 'r') as f:

            psf_ra, psf_dec = f['ra'][:][self.psf_cat_inds], f['dec'][:][self.psf_cat_inds]
            ZE1 = f['g1_model_hsm'][:][self.psf_cat_inds]
            ZE2 = f['g2_model_hsm'][:][self.psf_cat_inds]
            ONE = f['T_star_hsm'][:][self.psf_cat_inds]
            TWO = 1 - f['T_model_hsm'][:][self.psf_cat_inds]/f['T_star_hsm'][:][self.psf_cat_inds]
            TH1 = f['g1_star_hsm'][:][self.psf_cat_inds] - f['g1_model_hsm'][:][self.psf_cat_inds]
            TH2 = f['g2_star_hsm'][:][self.psf_cat_inds] - f['g2_model_hsm'][:][self.psf_cat_inds]

            SNR = f['FLUX_AUTO'][:][self.psf_cat_inds]/f['FLUXERR_AUTO'][:][self.psf_cat_inds]
            
            ########################################################################################################################
            #NOW COMPUTE STAR WEIGHTS
            ########################################################################################################################

            NSIDE      = self.MapNSIDE_weightrands
            weight_map = self.star_weights_map(gal_ra, gal_dec, gal_w, psf_ra, psf_dec, NSIDE = NSIDE)
            pix        = hp.ang2pix(NSIDE, psf_ra, psf_dec, lonlat = True)
            psf_w      = weight_map[pix] #Assign individual stars weights from the map
            
            ########################################################################################################################
            #NOW AVERAGE ACROSS FOOTPRINTS
            ########################################################################################################################
            
            for SNR_threshold in [1, 20, 40, 60, 80, 100, 150]:
                
                with open(self.output_path + '/PSF_summaries.txt', 'a') as f:
                    print("\n--------------------------------------------------------------", file = f)
                    print("--------------------------------------------------------------", file = f)
                    print("AVERAGE PSF QUANTITIES", file = f)
                    print("SNR = ", SNR_threshold, file = f)
                    print("--------------------------------------------------------------", file = f)
                    print("--------------------------------------------------------------", file = f)

                    Mask = (psf_w > 0) & (SNR > SNR_threshold) & (BAND != b'g')

                    print("<p> e1_psf:", np.average(ZE1[Mask], weights = psf_w[Mask]), file = f)
                    print("<p> e2_psf:", np.average(ZE2[Mask], weights = psf_w[Mask]), file = f)


                    print("<q> e1_err:", np.average(TH1[Mask], weights = psf_w[Mask]), file = f)
                    print("<q> e2_err:", np.average(TH2[Mask], weights = psf_w[Mask]), file = f)


                    print("<T> T_err:", np.average(TWO[Mask], weights = psf_w[Mask]), file = f)
                    print("sig(T) T_err:", np.sqrt(np.average(TWO[Mask]**2, weights = psf_w[Mask]) - 
                                                   np.average(TWO[Mask], weights = psf_w[Mask])**2), file = f)
            
            
            ########################################################################################################################
            #GET QUANTITIES FOR  COLOR DEP.
            ########################################################################################################################
                
            Mask = (psf_w > 0) & (SNR > self.Star_SNR_min) & (BAND != b'g')
            
            del psf_ra, psf_dec, SNR, weight_map, gal_ra, gal_dec, BAND
            
            ZE1 = ZE1[Mask]
            ZE2 = ZE2[Mask]
            ONE = ONE[Mask]
            TWO = TWO[Mask]
            TH1 = TH1[Mask]
            TH2 = TH2[Mask]
            
            psf_w = psf_w[Mask] 
            
            
            ########################################################################################################################
            #NOW COMPUTE COLOR DEPENDENT QUANTITIES
            ########################################################################################################################


            #'/project/chihway/dhayaa/DECADE/matched_star_psf_shapecatalog_20230630.npy'
            m_r, m_i, m_z = np.load(os.environ['TMPDIR'] + '/psf_color_matched.npy', mmap_mode = 'r')

            for color, colorname in zip([m_r - m_z, m_i - m_z, m_r - m_i], ['rz', 'iz', 'ri']):
                    
                color = color[Mask]
                    
                bins = np.linspace(-0.6, 2, 20 + 1)
                cen  = (bins[1:] + bins[:-1])/2

                counts_all = np.histogram(color, bins = bins)[0]

                avg1_all = np.histogram(color, bins = bins, weights = ONE)[0]
                avg2_all = np.histogram(color, bins = bins, weights = TWO)[0]            
                avg3_all = np.histogram(color, bins = bins, weights = TH1)[0]
                avg4_all = np.histogram(color, bins = bins, weights = TH2)[0]

                Avg_jack = []
                for j in tqdm(range(self.Npatch), desc = 'color: %s' % colorname):

                    patch_mask = psf_inds == j

                    counts = counts_all - np.histogram(color[patch_mask], bins = bins)[0]

                    avg1 = avg1_all - np.histogram(color[patch_mask], bins = bins, weights = ONE[patch_mask])[0]
                    avg2 = avg2_all - np.histogram(color[patch_mask], bins = bins, weights = TWO[patch_mask])[0]            
                    avg3 = avg3_all - np.histogram(color[patch_mask], bins = bins, weights = TH1[patch_mask])[0]
                    avg4 = avg4_all - np.histogram(color[patch_mask], bins = bins, weights = TH2[patch_mask])[0]
                    
                    avg1 = avg1/counts
                    avg2 = avg2/counts
                    avg3 = avg3/counts
                    avg4 = avg4/counts
                    
                    avg  = [avg1, avg2, avg3, avg4]
                    
                    Avg_jack.append(avg)

                Avg_jack = np.array(Avg_jack)
                np.save(self.output_path + '/PSFColor_%s.npy' % colorname, Avg_jack)
                np.save(self.output_path + '/PSFColorBins_%s.npy' % colorname, cen)


    class StarMatch(object):

        def __init__(self, ra, dec):

            self.ra  = ra
            self.dec = dec

            output    = self.match()
            self.inds = output[1]
            self.UniquePix = output[0]


        def match(self, nside = 8192*32):

            hpix = hp.ang2pix(nside, self.ra, self.dec, lonlat = True)

            unique_hpix, inds, unique_counts = np.unique(hpix, return_inverse = True, return_counts = True)

            print("TOTAL:", len(hpix))
            print("UNIQUE MATCHED:", len(unique_hpix))
            print("MAX REPEATS:", np.max(unique_counts))
            print("MIN REPEATS:", np.min(unique_counts))

            return unique_hpix, inds, unique_counts
        
        
        def weighted_average(self, A, weights = None):

            wgt = np.bincount(self.inds, weights = weights)
            sm  = np.bincount(self.inds, weights = A*weights)        
            
            avg = sm/wgt

            return avg
    

    class MakeMapFromCat(object):

        def __init__(self, ra, dec, e1, e2, w = None, NSIDE = 1024):

            self.NSIDE = NSIDE
            self.pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)
            
            self.unique_pix, self.idx_rep = np.unique(self.pix, return_inverse=True)
            del ra, dec
            
            self.n_map = np.zeros(hp.nside2npix(self.NSIDE))
            self.n_map[self.unique_pix] += np.bincount(self.idx_rep, weights = w)
            
            if w is None:
                
                self.weight_map = np.ones_like(self.n_map)
            else:
                self.weight_map = self.n_map.copy()
                #self.weight_map[self.unique_pix] /= np.bincount(self.idx_rep) #Get mean weight per pixel
                
                
            #Only select pixels where we have at least a single galaxy
            #Rest will have zero ellipticity by default
            self.mask_sims = self.n_map != 0.

            
            if w is None: w = 1
            
            self.w = w
            
            
            self.e1 = e1
            self.e2 = e2


        def process(self, seed, norand = False):
            
            e1_map    = np.zeros(hp.nside2npix(self.NSIDE))
            e2_map    = np.zeros(hp.nside2npix(self.NSIDE))
            
            if norand == True:
                e1, e2 = self.e1, self.e2
            else:
                rot_angle = np.random.default_rng(seed).random(self.e1.size)*2*np.pi
                e1, e2    = self.rotate_ellipticities(self.e1, self.e2, rot_angle)
            
            #Math for getting the weighted shape average per pixel
            e1_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e1 * self.w)
            e2_map[self.unique_pix] += np.bincount(self.idx_rep, weights = e2 * self.w)
            e1_map[self.mask_sims]   = e1_map[self.mask_sims]/(self.n_map[self.mask_sims])
            e2_map[self.mask_sims]   = e2_map[self.mask_sims]/(self.n_map[self.mask_sims])
            
            return e1_map, e2_map
        

        def rotate_ellipticities(self, e1, e2, rot_angle):
            """
            Random rotate ellipticities e1 and e2 over
            angles given in `rot_angle`, which is in
            units of radians
            """
            #Rotate galaxy shapes randomly
            cos = np.cos(rot_angle)
            sin = np.sin(rot_angle)
            e1_new = + e1 * cos + e2 * sin
            e2_new = - e1 * sin + e2 * cos
            return e1_new, e2_new


    class MakeMapFromCls(object):

        def __init__(self, Cls, NSIDE = 1024):

            self.NSIDE = NSIDE
            self.Cls   = Cls
        

        def process(self, seed):
            
            kappaE = hp.synfast(self.Cls, nside = self.NSIDE, pixwin = True, lmax = 2*self.NSIDE)
            kappaB = np.zeros_like(kappaE)  

            ell, emm = hp.Alm.getlm(lmax=2*self.NSIDE)

            with np.errstate(invalid = 'ignore', divide = 'ignore'):
                alms   = hp.map2alm(kappaE, pol=False, use_pixel_weights = True, iter = 0, lmax = 2*self.NSIDE)
                kalmsE = alms/(ell * (ell + 1.) / (ell + 2.) / (ell - 1)) ** 0.5

                alms   = hp.map2alm(kappaB, pol=False, use_pixel_weights = True, iter = 0, lmax = 2*self.NSIDE)  # Spin transform!
                kalmsB = alms/(ell * (ell + 1.) / (ell + 2.) / (ell - 1)) ** 0.5

            #Set monopole terms to 0
            kalmsE[ell == 0] = 0.0
            kalmsB[ell == 0] = 0.0

            #First entry of kalmsE is a dummy entry. Just need 2nd two entries of E and B modes
            _, gamma1, gamma2 = hp.alm2map([kalmsE,kalmsE,kalmsB], nside = self.NSIDE, pol=True)

            return gamma1, gamma2


    class BmodeRunner(object):

        tmp_bins = np.linspace(np.sqrt(8), np.sqrt(2048), 33)**2
        tmp_bins = tmp_bins.astype(int)

        def __init__(self, MakeMapFromCat, MakeMapFromCls, seed, bins = tmp_bins, njobs = -1):

            self.MakeMapFromCat = MakeMapFromCat
            self.MakeMapFromCls = MakeMapFromCls
            
            self.seed    = seed
            self.njobs   = njobs
            self.mask    = MakeMapFromCat.mask_sims #* MakeMapFromCat.weight_map #We don't use weights since it gets kinda weird
            
            self.bins    = bins
            self.bins    = nmt.NmtBin.from_edges(self.bins[:-1], self.bins[1:])
            self.ell_eff = self.bins.get_effective_ells()
            
            m = nmt.NmtField(self.mask, None, spin = 2) #Just mask and we use this to compute coupling matrix
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(m, m, self.bins)
            
            self.w = w
        
        def process_data(self):

            e1, e2 = self.MakeMapFromCat.process(-np.inf, norand = True) #Use -np.inf because seed should never be used in this mode

            return self.measure_NamasterCls(e1, e2)
        
        def process_cov(self, Nrands = 100):

            seeds = np.random.default_rng(self.seed).integers(0, 2**63, Nrands)
            
            with joblib.parallel_backend("loky"):
                
                outputs = [self.single_run_cov(i, seeds[i]) for i in tqdm(np.arange(Nrands), desc = 'Make cov')]
                final   = [0]*Nrands
                for o in outputs: final[o[0]] = o[1]

            return np.array(final)
        

        def process_noise(self, Nrands = 100):

            seeds = np.random.default_rng(self.seed).integers(0, 2**63, Nrands)
            
            with joblib.parallel_backend("loky"):
    #             jobs    = [joblib.delayed(self.single_run_noise)(i, seeds[i]) for i in np.arange(Nrands)]
    #             outputs = joblib.Parallel(n_jobs = self.njobs, verbose = 10)(jobs)

                outputs = [self.single_run_noise(i, seeds[i]) for i in tqdm(np.arange(Nrands), desc = 'Make Rands')]
            
                
                final   = [0]*Nrands
                for o in outputs: final[o[0]] = o[1]

            return np.array(final)
        

        def single_run_cov(self, i, seed):

            A = self.MakeMapFromCat.process(seed, norand = False)
            B = self.MakeMapFromCls.process(seed)
            
            
            e1 = A[0] + B[0]
            e2 = A[1] + B[1]
            
            return i, self.measure_NamasterCls(e1, e2)
        
        
        def single_run_noise(self, i, seed):

            e1, e2 = self.MakeMapFromCat.process(seed, norand = False)
            
            return i, self.measure_NamasterCls(e1, e2)


        def measure_NamasterCls(self, e1, e2):

            field = nmt.NmtField(self.mask, [e1, e2])

            cl_coupled   = nmt.compute_coupled_cell(field, field)
            cl_decoupled = self.w.decouple_cell(cl_coupled)

            return cl_decoupled
        
    
    class PureBmodeRunner(BmodeRunner):

        tmp_bins = np.linspace(np.sqrt(8), np.sqrt(2048), 33)**2
        tmp_bins = tmp_bins.astype(int)

        def __init__(self, MakeMapFromCat, MakeMapFromCls, seed, bins = tmp_bins, apodised_scale = 5/60, njobs = -1):

            super().__init__(MakeMapFromCat, MakeMapFromCls, seed, bins, njobs)

            np.save(os.environ['TMPDIR'] + '/MakeMapFromCat.npy', MakeMapFromCat.mask_sims.astype(float))
            self.mask = nmt.mask_apodization(MakeMapFromCat.mask_sims.astype(np.float64), apodised_scale, apotype='C1')
            print("Apodizing the mask: Needed for purifying B-modes") 

            #Redo mode coupling
            m = nmt.NmtField(self.mask, None, spin = 2)
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(m, m, self.bins)
            
            self.w = w


        def measure_NamasterCls(self, e1, e2):

            field = nmt.NmtField(self.mask, [e1, e2], purify_b = True)
            
            cl_coupled   = nmt.compute_coupled_cell(field, field)
            cl_decoupled = self.w.decouple_cell(cl_coupled)

            return cl_decoupled
        
        
    class MRBmodeRunner(object):
        
        def __init__(self, TreeCat, TreeCat2 = None, theta_min = 2.5, theta_max = 250, Ntheta = 1000, Nmodes = 20):
            
            self.theta_min = theta_min
            self.theta_max = theta_max
            self.Ntheta    = Ntheta
            self.Nmodes    = Nmodes
            self.TreeCat   = TreeCat
            self.TreeCat2  = TreeCat2

        
        def compute_treecorr(self):
            
            Xi = treecorr.GGCorrelation(min_sep = self.theta_min, max_sep = self.theta_max, nbins = self.Ntheta,
                                        rng = np.random.default_rng(seed = 42),
                                        sep_units = 'arcmin', var_method = 'jackknife', bin_slop = 0.1)
            
            if self.TreeCat2 is None:
                Xi.process(self.TreeCat)
            else:
                Xi.process(self.TreeCat, self.TreeCat2)
            
            
            return Xi
        
        
        def setup_kernel(self):
            
            FILE = os.environ['TMPDIR'] + '/EB_Coefficients.npy'
            
            if os.path.isfile(FILE):
                
                print("COEFF FILE (%s) EXISTS. NOT RERUNNING" % FILE)
            
            else:
                import hybrideb

                heb = hybrideb.HybridEB(self.theta_min, self.theta_max, self.Ntheta)
                beb = hybrideb.BinEB(self.theta_min, self.theta_max, self.Ntheta)
                geb = hybrideb.GaussEB(beb, heb, Nl = self.Nmodes)

                np.save(FILE, np.array([geb(i) for i in range(self.Nmodes)], dtype=object), allow_pickle = True)
                
                
        @timeit
        def compute_EB(self):
            
            self.setup_kernel()
            correlator = self.compute_treecorr()
            
            E = self._get_E([correlator])
            B = self._get_B([correlator])
            
            cov_E = treecorr.estimate_multi_cov([correlator], 'jackknife', func = self._get_E)
            cov_B = treecorr.estimate_multi_cov([correlator], 'jackknife', func = self._get_B)
            
            return E, B, cov_E, cov_B, correlator
        

        def _get_E(self, corr): return self._get_EB(corr, E_mode = True)
        def _get_B(self, corr): return self._get_EB(corr, E_mode = False)
        
        def _get_EB(self, corr, E_mode = True):   
        
            xip = corr[0].xip
            xim = corr[0].xim
            
            N = np.ones(self.Nmodes)
            
            for i in range(self.Nmodes):
                
                res = np.load(os.environ['TMPDIR'] + '/EB_Coefficients.npy', allow_pickle = True)[()][i]
                fp  = res[1]
                fm  = res[2]
            
                # X+ = np.sum((fp*xip + fm*xim)/2)
                # X- = np.sum((fp*xip - fm*xim)/2)
                if E_mode == True:
                    N[i] = np.sum(fp*xip + fm*xim)/2
                else:
                    N[i] = np.sum(fp*xip - fm*xim)/2
                
            return N
            
        

if __name__ == '__main__':

    
    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--psf_cat',     action='store', type = str, required = True)
    my_parser.add_argument('--galaxy_cat',  action='store', type = str, required = True)
    my_parser.add_argument('--psf_cat_inds',    action='store', type = str, required = True)
    my_parser.add_argument('--galaxy_cat_inds', action='store', type = str, required = True)
    my_parser.add_argument('--output_path',     action='store', type = str, required = True)
    my_parser.add_argument('--sim_Cls_path',    action='store', type = str, required = True)
    
    
    my_parser.add_argument('--Npatch',       action='store', type = int, default = 150)
    my_parser.add_argument('--Star_SNR_min', action='store', type = int, default = 80)
    my_parser.add_argument('--MapNSIDE_weightrands', action='store', type = int, default = 256)
    
    
    my_parser.add_argument('--All',              action='store_true', default = False)
    my_parser.add_argument('--mean_shear',       action='store_true', default = False)
    my_parser.add_argument('--brighter_fatter',  action='store_true', default = False)
    my_parser.add_argument('--shear_vs_X',       action='store_true', default = False)
    my_parser.add_argument('--gt_field_centers', action='store_true', default = False)
    my_parser.add_argument('--gt_stars',         action='store_true', default = False)
    my_parser.add_argument('--gt_coadd_stars',   action='store_true', default = False)
    my_parser.add_argument('--rho_stats',        action='store_true', default = False)
    my_parser.add_argument('--psf_color',        action='store_true', default = False)
    my_parser.add_argument('--Bmodes',           action='store_true', default = False)
    my_parser.add_argument('--MRBmode_psf',      action='store_true', default = False)
    my_parser.add_argument('--check_shear_2pt',  action='store_true', default = False)
    
    
    args  = vars(my_parser.parse_args())
    cargs = {k:args[k] for k in list(args.keys())[:9]}
    
    
    RUNNER = AllTests(**cargs)
    
    
    if np.logical_or(args['All'], args['brighter_fatter']):  RUNNER.brighter_fatter_effect()
    if np.logical_or(args['All'], args['mean_shear']):       RUNNER.mean_shear()
    if np.logical_or(args['All'], args['shear_vs_X']):       RUNNER.shear_vs_X()
    if np.logical_or(args['All'], args['gt_field_centers']): RUNNER.tangential_shear_field_centers()
    if np.logical_or(args['All'], args['gt_stars']):         RUNNER.tangential_shear_stars()
    if np.logical_or(args['All'], args['rho_stats']):        RUNNER.rho_stats()
    if np.logical_or(args['All'], args['psf_color']):        RUNNER.psf_color()
    if np.logical_or(args['All'], args['Bmodes']):           RUNNER.Bmodes()
    
    
    if args['MRBmode_psf']:      RUNNER.MRBmode_psf() #This takes really long time so we run it separately, always!
    if args['gt_coadd_stars']:   RUNNER.tangential_shear_coadd_stars() #This isn't needed in standard tests, so skip it
    if args['check_shear_2pt']:  RUNNER.check_shear_2pt() #Similarly, this is just for my checks. Not actual test
    
        
    

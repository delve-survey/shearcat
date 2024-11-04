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


class BmodeTomo(object):


    def __init__(self, galaxy_cat, output_path, sim_Cls_path, Ncov):
        
        self.galaxy_cat = galaxy_cat

        self.rng = np.random.default_rng(seed = 42)
        
        self.output_path = output_path

        os.makedirs(output_path, exist_ok = True)

        self.sim_Cls = np.loadtxt(sim_Cls_path) #Cls to use in making Gaussian mocks for covariance 
        self.Ncov    = Ncov
        self.dered = '_dered_sfd98'
        #self.dered = ''

    @timeit
    def compute_response(self, bin):

        dgamma = 0.01*2

        with h5py.File(self.galaxy_cat, 'r') as f:

            if bin is None:
                Mask1p = f['baseline_mcal_mask_1p'][:] > 0
                Mask2p = f['baseline_mcal_mask_2p'][:] > 0
                Mask1m = f['baseline_mcal_mask_1m'][:] > 0
                Mask2m = f['baseline_mcal_mask_2m'][:] > 0
            else:
                Mask1p = f['baseline_mcal_mask_1p'][:] == (1 + bin)
                Mask2p = f['baseline_mcal_mask_2p'][:] == (1 + bin)
                Mask1m = f['baseline_mcal_mask_1m'][:] == (1 + bin)
                Mask2m = f['baseline_mcal_mask_2m'][:] == (1 + bin)


            R11tot = (np.average(f['mcal_g_1p'][:][:, 0][Mask1p],  weights = f['mcal_g_w_1p'][:][:][Mask1p]) - 
                      np.average(f['mcal_g_1m'][:][:, 0][Mask1m],  weights = f['mcal_g_w_1m'][:][:][Mask1m]))/dgamma
            
            R22tot = (np.average(f['mcal_g_2p'][:][:, 1][Mask2p],  weights = f['mcal_g_w_2p'][:][:][Mask2p]) - 
                      np.average(f['mcal_g_2m'][:][:, 1][Mask2m],  weights = f['mcal_g_w_2m'][:][:][Mask2m]))/dgamma
            
        print("BIN", bin, R11tot, R22tot)
        return R11tot, R22tot

    
    @timeit
    def Bmodes_Namaster(self):

        nside = 1024
        npix  = hp.nside2npix(nside)

        Cat   = []
        #Load the shape catalog
        with h5py.File(self.galaxy_cat, 'r') as f:

            for i in range(4):
                Mask   = f['baseline_mcal_mask_noshear'][:] == (i + 1)
                ra     = f['RA'][:][Mask]
                dec    = f['DEC'][:][Mask]
                w      = f['mcal_g_w_noshear'][:][Mask]
                g1, g2 = f['mcal_g_noshear'][:][Mask].T
            
                #Do mean subtraction, following Gatti+ 2020: https://arxiv.org/pdf/2011.03408.pdf
                for a in [g1, g2]:
                    a -= np.average(a, weights = w)

                R11, R22 = self.compute_response(i)
                g1, g2 = g1/R11, g2/R22

                #Need a -1 for g2 in R (not T) due to Namaster definition
                Cat.append(
                    self.MakeMapFromCat(ra = ra, dec = dec, e1 = g1, e2 = -g2, w = w, NSIDE = nside)
                )
            del ra, dec, g1, g2, Mask

            # C = self.MakeMapFromCls(self.sim_Cls, NSIDE = nside)
            C = None
        
        
        ##############################################
        # No harmonic-space Bmodes in this test
        ##############################################
        #Process regular Bmodes
        X    = self.BmodeRunner(Cat, C, 42, njobs = 1)
        data = X.process_data()
        Cov  = X.process_noise(self.Ncov) #Make a lot more sims so we dont get hit by Hartlap factor
        
        np.save(self.output_path + '/Bmode.npy',       data)
        np.save(self.output_path + '/Bmode_Noise.npy', Cov)
        np.save(self.output_path + '/Bmode_ell.npy',   X.ell_eff)
        
        #Process pure Bmodes
#         X    = self.PureBmodeRunner(R, C, 42, njobs = 1)
#         data = X.process_data()
#         Cov  = X.process_noise(100) #Make a lot more sims so we dont get hit by Hartlap factor

#         np.save(self.output_path + '/PureBmode.npy', np.vstack([data,  X.ell_eff]))
#         np.save(self.output_path + '/PureBmode_Noise.npy', Cov)
                
    

    class MakeMapFromCat(object):

        def __init__(self, ra, dec, e1, e2, w = None, NSIDE = 1024):

            self.NSIDE = NSIDE
            self.pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)
            
            self.unique_pix, self.idx_rep = np.unique(self.pix, return_inverse=True)
            del ra, dec
            
            self.n_map = np.bincount(self.pix, minlength = hp.nside2npix(self.NSIDE))
            
            if w is None:
                self.weight_map = np.ones_like(self.n_map)
            else:
                self.weight_map = np.bincount(self.pix, weights = w, minlength = hp.nside2npix(self.NSIDE))
                
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
            e1_map = np.bincount(self.pix, weights = e1 * self.w, minlength = hp.nside2npix(self.NSIDE))
            e2_map = np.bincount(self.pix, weights = e2 * self.w, minlength = hp.nside2npix(self.NSIDE))
            n_map  = np.bincount(self.pix, weights = self.w,      minlength = hp.nside2npix(self.NSIDE))
            e1_map[self.mask_sims] = e1_map[self.mask_sims]/(n_map[self.mask_sims])
            e2_map[self.mask_sims] = e2_map[self.mask_sims]/(n_map[self.mask_sims])
            
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

        def __init__(self, Catalogs, MakeMapFromCls, seed, bins = tmp_bins, njobs = -1):

            self.Catalogs       = Catalogs
            self.MakeMapFromCls = MakeMapFromCls
            
            self.Nbins   = len(self.Catalogs); print("USING BINS", self.Nbins)
            self.seed    = seed
            self.njobs   = njobs
            
            self.bins    = bins
            self.bins    = nmt.NmtBin.from_edges(self.bins[:-1], self.bins[1:])
            self.ell_eff = self.bins.get_effective_ells()

            self.mask    = [X.mask_sims * X.weight_map for X in Catalogs] #We don't use weights since it gets kinda weird
            self.mask    = [X.mask_sims.copy() for X in Catalogs] #We don't use weights since it gets kinda weird
            
            w = [[0 for _ in range(self.Nbins)] for _ in range(self.Nbins)]

            for i in range(self.Nbins):
                for j in range(i, self.Nbins):
                    w[i][j] = w[j][i] = nmt.NmtWorkspace()
                    w[i][j].compute_coupling_matrix(
                        nmt.NmtField(self.mask[i], None, spin = 2),
                        nmt.NmtField(self.mask[j], None, spin = 2),
                        self.bins
                    )
            
            self.w = w
        

        def process_data(self):
            
            RES  = [[0 for _ in range(self.Nbins)] for _ in range(self.Nbins)]
            TMP  = [self.Catalogs[i].process(-np.inf, norand = True)   for i in range(self.Nbins)]
            DATA = [nmt.NmtField(self.mask[i], [TMP[i][0], TMP[i][1]]) for i in range(self.Nbins)]
            for i in range(self.Nbins):
                for j in range(i, self.Nbins):
                    RES[i][j] = RES[j][i] = self.measure_NamasterCls(DATA[i], DATA[j], self.w[i][j]).copy()

            return RES


        def process_cov(self, Nrands = 100):

            seeds = np.random.default_rng(self.seed).integers(0, 2**63, Nrands)
            
            with joblib.parallel_backend("loky"):
                
                outputs = [self.single_run_cov(i, seeds[i]) for i in tqdm(np.arange(Nrands), desc = 'Make cov')]
                final   = [0 for _ in range(Nrands)]
                for o in outputs: final[o[0]] = o[1]

            return np.array(final)
        

        def process_noise(self, Nrands = 100):

            seeds = np.random.default_rng(self.seed).integers(0, 2**63, Nrands)
            
            with joblib.parallel_backend("loky"):
                outputs = [self.single_run_noise(i, seeds[i]) for i in tqdm(np.arange(Nrands), desc = 'Make Rands')]
            
                
                final   = [0 for _ in range(Nrands)]
                for o in outputs: final[o[0]] = o[1]

            return np.array(final)
        

        def single_run_cov(self, i, seed):

            A = self.MakeMapFromCat.process(seed, norand = False)
            B = self.MakeMapFromCls.process(seed)
            
            e1 = A[0] + B[0]
            e2 = A[1] + B[1]
            
            return i, self.measure_NamasterCls(e1, e2)
        
        
        def single_run_noise(self, k, seed):

            SDS  = np.random.default_rng(seed = seed).integers(0, 2**30, 4)
            RES  = [[0 for _ in range(self.Nbins)] for _ in range(self.Nbins)]
            TMP  = [self.Catalogs[i].process(SDS[i], norand = False) for i in range(self.Nbins)]
            DATA = [nmt.NmtField(self.mask[i], [TMP[i][0], TMP[i][1]]) for i in range(self.Nbins)]
            for i in range(self.Nbins):
                for j in range(i, self.Nbins):
                    RES[i][j] = RES[j][i] = self.measure_NamasterCls(DATA[i], DATA[j], self.w[i][j])
            
            return k, RES


        def measure_NamasterCls(self, f_a, f_b, w):

            cl_coupled   = nmt.compute_coupled_cell(f_a, f_b)
            cl_decoupled = w.decouple_cell(cl_coupled)

            return cl_decoupled
    
        

if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--galaxy_cat',      action='store', type = str, required = True)
    my_parser.add_argument('--output_path',     action='store', type = str, required = True)
    my_parser.add_argument('--sim_Cls_path',    action='store', type = str, required = True)
    my_parser.add_argument('--Ncov',            action='store', type = int, default  = 100)
    
    args  = vars(my_parser.parse_args())
    cargs = {k:args[k] for k in list(args.keys())[:]}
    
    RUNNER = BmodeTomo(**cargs)
    RUNNER.Bmodes_Namaster()
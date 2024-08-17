import numpy as np
import healpy as hp, healsparse as hsp
from scipy import stats
import h5py
import gc
import time
from tqdm import tqdm
import treecorr


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


class SysmapTestRunner:
    
    
    def __init__(self, NSIDE, data_path, Npatch = 150):
        
       
        self.NSIDE = NSIDE
        self.data_path = data_path
        
        self.Npatch = Npatch
    
    
    def assign_patches(self, ra, dec, centers = None):

        
        if centers is None:
            centers = treecorr.Catalog(ra = ra, dec = dec, ra_units='deg',dec_units='deg', npatch = self.Npatch)._centers
        
        inds = treecorr.Catalog(ra = ra, dec = dec, ra_units='deg',dec_units='deg', patch_centers = centers)._patch

        return inds, centers
    
    @timeit
    def _pix_avg(self, pix, X, weights):
        
        minlength = hp.nside2npix(self.NSIDE)
        
        num   = np.bincount(pix, weights = X*weights, minlength = minlength)
        denom = np.bincount(pix, weights = weights,   minlength = minlength)
        
        denom[denom == 0] = 1 #To prevent division errors
        
        return num/denom
    
    
    def mask_condition(self, mask):

        return mask > 0
    
    @timeit
    def get_response_maps(self):
        
        
        with h5py.File(self.data_path, 'r') as f:
            
            m    = self.mask_condition(f['baseline_mcal_mask_noshear'][:])
            m_1p = self.mask_condition(f['baseline_mcal_mask_1p'][:])
            m_1m = self.mask_condition(f['baseline_mcal_mask_1m'][:])
            m_2p = self.mask_condition(f['baseline_mcal_mask_2p'][:])
            m_2m = self.mask_condition(f['baseline_mcal_mask_2m'][:])

            print("FINISHED LOADING MASKS")
            

            ra  = f['RA'][::]
            dec = f['DEC'][::]
            
            pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)
            
            print("FINISHED LOADING PIX")
            
            del ra, dec; gc.collect()
            
            
            w   = f['mcal_g_w'][::]
            
            g   = f['mcal_g_noshear'][::]
            g1p = f['mcal_g_1p'][::]
            g1m = f['mcal_g_1m'][::]
            g2p = f['mcal_g_2p'][::]
            g2m = f['mcal_g_2m'][::]
            
            print("FINISHED LOADING SHEAR")
            
            
        dgamma = 0.01*2

        R11    = (self._pix_avg(pix[m], g1p[m, 0], w[m]) - self._pix_avg(pix[m], g1m[m, 0], w[m]))/dgamma
        R11s   = (self._pix_avg(pix[m_1p], g[m_1p, 0], w[m_1p]) - self._pix_avg(pix[m_1m], g[m_1m, 0], w[m_1m]))/dgamma
        
        R22    = (self._pix_avg(pix[m], g2p[m, 1], w[m]) - self._pix_avg(pix[m], g2m[m, 1], w[m]))/dgamma
        R22s   = (self._pix_avg(pix[m_2p], g[m_2p, 1], w[m_2p]) - self._pix_avg(pix[m_2m], g[m_2m, 1], w[m_2m]))/dgamma
        
        
        R11tot = R11 + R11s
        R22tot = R22 + R22s

        
        return R11tot, R22tot
    
    
    @timeit
    def get_data_maps(self):
        
        with h5py.File(self.data_path, 'r') as f:
            
            mask = self.mask_condition(f['baseline_mcal_mask_noshear'][:])

            ra = f['RA'][::][mask]
            dec = f['DEC'][::][mask]
            g  = f['mcal_g_noshear'][::][mask]
            w  = f['mcal_g_w'][::][mask]
            
            
        pix = hp.ang2pix(self.NSIDE, ra, dec, lonlat = True)

        N  = np.bincount(pix, minlength = hp.nside2npix(self.NSIDE), weights = w)
        g1 = np.bincount(pix, minlength = hp.nside2npix(self.NSIDE), weights = w * g[:,0])
        g2 = np.bincount(pix, minlength = hp.nside2npix(self.NSIDE), weights = w * g[:,1])

        m = N > 0

        g1[m] = g1[m]/N[m]
        g2[m] = g2[m]/N[m]
        
        
        del mask, ra, dec, g, w, pix, m; gc.collect()
        
        return g1, g2, N
    
    
    def get_sys_map(self, map_type, band):
        
        p = f'/project/kadrlica/secco/DELVE/combined_dr311+dr312/{map_type}/delve_dr311+dr312_{band}_{map_type}_Nov28th.hsp'
        S = hsp.HealSparseMap.read(p)
        S = S.generate_healpix_map(self.NSIDE, nest = False)
        
        return S
    
    
    @timeit
    def go(self, map_types, bands):
        
        results = {}
        
        g1, g2, N = self.get_data_maps()
        R11, R22  = self.get_response_maps()
        
        g1 = g1/np.where(N > 0, R11, 1)
        g2 = g2/np.where(N > 0, R22, 1)
        
        data_maps = {'g1' : g1, 'g2' : g2, 'N' : N}
        
        ra, dec = hp.pix2ang(self.NSIDE, np.arange(N.size), lonlat = True)
        centers = self.assign_patches(ra[N > 0], dec[N > 0])[1] #Get centers using only good pix
        d_inds  = self.assign_patches(ra, dec, centers = centers)[0] #Get inds for ALL pix
        
        with tqdm(total = len(data_maps.keys()) * len(bands) * len(map_types) * self.Npatch) as pbar:
            #Loop over all data maps (g1, g2, N)
            for d in data_maps.keys():
                results[d] = {}

                for b in bands:
                    results[d][b] = {}

                    for m in map_types:
                        results[d][b][m] = []

                        sys_map = self.get_sys_map(m, b)
                        s_inds  = self.assign_patches(ra, dec, centers = centers)[0]

                        for j in range(self.Npatch):

                            pix_mask = (N > 0) & (sys_map != hp.UNSEEN) & (s_inds != j) & (d_inds != j)
                            res      = stats.linregress(data_maps[d][pix_mask], sys_map[pix_mask])

                            results[d][b][m].append(res.slope)
                            
                            
                            pbar.update(1)
                        
        
        return results


class TomoSysmapTestRunner(SysmapTestRunner):

    def go(self, map_types, bands):

        Results = []
        for bin in range(4):
            
            print(f"STARTING BIN {bin}")
            #Change how the code uses masking
            def tmp(mask): return mask == (bin + 1) #Tomobin is 1-indexed
            setattr(self, 'mask_condition', tmp)

            Results.append(
                super().go(map_types, bands)
            )

        Results = np.array(Results)
        
        return Results


if __name__ == "__main__":

    maps  = ['airmass', 'dcr_e1', 'dcr_e2', 'dcr_ddec', 'dcr_dra', 'maglim',  'exptime', 'fwhm', 'skysigma', 'skybrite', 'nexp']
    bands = ['r', 'i', 'z']

    # RUN   = SysmapTestRunner(NSIDE = 128, data_path = '/project/chihway/data/decade/metacal_gold_combined_20240209.hdf',)
    # corr  = RUN.go(maps, bands)
    # np.save('/home/dhayaa/DECADE/shearcat/shear_tests/Paper_plots/SysMapCorr_20240815.npy',     corr,  allow_pickle = True)
    


    RUN   = TomoSysmapTestRunner(NSIDE = 128, data_path = '/project/chihway/data/decade/metacal_gold_combined_20240209.hdf',)
    tcorr = RUN.go(maps, bands)
    np.save('/home/dhayaa/DECADE/shearcat/shear_tests/Paper_plots/SysMapTomoCorr_20240815.npy', tcorr, allow_pickle = True)
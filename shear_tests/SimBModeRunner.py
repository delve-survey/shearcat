import numpy as np
import healpy as hp
import h5py, gc, glob, yaml
import treecorr
from tqdm import tqdm
from numba import njit

import sys, os, subprocess as sp
sys.path.append('/home/dhayaa/DECADE/')

from MapMaker import utils
from MapMaker.Kappa import Kappa


import time
from contextlib import contextmanager
from functools import wraps
import joblib

@contextmanager
def timed_execution(Name):
    start_time = time.time()  # Record the start time
    yield  # Execute the code block inside the 'with' statement
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("\n===================================================")
    print(f"Finished task {Name} in {elapsed_time:.4f} seconds")
    print("===================================================\n")

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.5} seconds to run.")
        return result
    return wrapper


def parallelize(func):
    @wraps(func)
    def wrapper(array, *args, **kwargs):
        # Function to apply to each Npix-sized array
        def apply_func(a_i):
            return func(a_i, *args, **kwargs)
        
        n_jobs = os.cpu_count()
        n_jobs = np.min([len(array), n_jobs])
        # Parallel execution across Npix-sized arrays
        results = joblib.Parallel(n_jobs = n_jobs, verbose = 10)(joblib.delayed(apply_func)(a_i) for a_i in array)
        
        return results
    
    return wrapper  

@parallelize
def my_map2alm(array): return hp.map2alm(array, use_pixel_weights = True)

@parallelize
def my_alm2map(array): return hp.alm2map(array, use_pixel_weights = True)

@parallelize
def my_rotate_alm(array, angles): return hp.rotator.Rotator(angles, deg = True).rotate_alm(array)

@parallelize
def my_ud_grade(array, NSIDE): return hp.ud_grade(array, NSIDE)

@parallelize
def my_klm_to_rotated_shear(array, angles, NSIDE):

    ell  = hp.Alm.getlm(lmax = NSIDE*3 - 1)[0]
    if np.all(np.array(angles) == 0):
        alms = array
    else:
        alms = hp.rotator.Rotator(angles, deg = True).rotate_alm(array)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        alms = alms / (ell * (ell + 1.) / (ell + 2.) / (ell - 1)) ** 0.5
    alms[ell < 2] = 0.0
    dummy = np.zeros_like(alms)
    return hp.alm2map([dummy, alms, dummy], nside = NSIDE, pol = True)[1:] 


def gen_samples(g, g_cov, rng):
    
    @njit
    def _gen_samples(g, g_cov, rng):
        
        res = np.zeros_like(g)
        
        for i in range(g.shape[0]):
            
            L = np.linalg.cholesky(g_cov[i])
            z = rng.normal(0, 1, g[i].shape)
            res[i] = g[i] + np.dot(L, z)
            
        return res
    
    N_cpu       = os.cpu_count()
    seeds       = rng.integers(2**20, size = N_cpu)
    N_per_batch = int(np.ceil(g.shape[0] / N_cpu))

    def one_process(i):
        s, e = N_per_batch * i, N_per_batch * (i + 1)
        return _gen_samples(g[s:e], g_cov[s:e], np.random.default_rng(seed = seeds[i]))
    
    results = joblib.Parallel(n_jobs = N_cpu, verbose = 10)(joblib.delayed(one_process)(i) for i in range(N_cpu))

    s = 0
    for r in results:
        s += len(r)
    return np.concatenate(results, axis = 0)


def corrs2format(corrs): return np.concatenate([c.xip for c in corrs] + [c.xim for c in corrs])

rots_per_map = 4


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
    

class BaseRunner:

    m_bias       = None
    catalog_path = None
    n_of_z       = None
    z_grid       = None
    map_globpath = None
    NSIDE        = 1024

    def __init__(self, seed, Npatch = 100, homogenize = False, nonoise = False, measurementnoise = False):

        self.seed       = seed
        self.rng        = np.random.default_rng(seed)
        self.Npatch     = Npatch
        self.homogenize = homogenize
        self.nonoise    = nonoise
        self.measurementnoise = measurementnoise


    def load_catalog(self):

        cat = {}
        with h5py.File(self.catalog_path, 'r') as f:
            g1, g2 = f['mcal_g_noshear'][:].T
            g_cov  = f['mcal_g_cov_noshear'][:]
            ra     = f['RA'][:]
            dec    = f['DEC'][:]
            w      = f['mcal_g_w'][:]
            mask   = f['baseline_mcal_mask_noshear'][:]

            if self.homogenize:
                ra, dec = self.random_positions(ra.size, self.seed)

            print("LOADED CATALOG")

            S = self.rng.choice(np.sum(mask > 0), size = 10_000_000, replace = False) #Select 10 million galaxies for building patches
            centers = treecorr.Catalog(ra = ra[mask > 0][S], dec = dec[mask > 0][S], 
                                       ra_units='deg', dec_units='deg', npatch = self.Npatch)._centers
            
            print("GENERATED PATCHES")

            inds = np.where(mask > 0)[0]
            # R    = self.compute_response(None); print(f"RESPONSE (BIN {bin}): {R}")
            R    = (0.7562031022655222, 0.7574298242071931)

            patch = treecorr.Catalog(ra = ra[inds], dec = dec[inds], ra_units='deg', dec_units='deg', patch_centers = centers)._patch
            cat = { 'g1'    : g1[inds] / R[0], 
                    'g2'    : g2[inds] / R[1], 
                    'g_cov' : g_cov[inds] / np.array([[R[0]**2, R[0]*R[1]], [R[0]*R[1], R[1]**2]])[None, ...],
                    'w'     : w[inds],
                    'hpix'  : hp.ang2pix(self.NSIDE, ra[inds], dec[inds], lonlat = True),
                    'ra'    : ra[inds],
                    'dec'   : dec[inds],
                    'patch' : patch}


            print("CONSTRUCTED TOMOGRAPHIC BINNED CATALOG")

            del g1, g2, ra, dec, w, mask, patch, inds; gc.collect()
        

        return cat


    def get_mask(self, NSIDE):

        with h5py.File(self.catalog_path, 'r') as f:
            mask = f['baseline_mcal_mask_noshear'][:] > 0
            hpix = hp.ang2pix(NSIDE, f['RA'][:][mask], f['DEC'][:][mask], lonlat = True)

        Mask = np.bincount(hpix, minlength = hp.nside2npix(NSIDE)) > 0

        print("GENERATED MASK")

        return Mask
            

    def random_positions(self, N, seed):

        NSIDE     = 4096 #Harcoded to high-res for now
        Mask      = self.get_mask(NSIDE)
        rng       = np.random.default_rng(seed)
        
        N_samples = 0
        ra_rand   = []
        dec_rand  = []

        while N_samples < N:
            N_randoms = 100_000_000

            phi   = rng.uniform(0, 2*np.pi, N_randoms)
            theta = np.arccos(1 - 2*rng.uniform(0, 1, N_randoms))

            hpix       = hp.ang2pix(NSIDE, theta, phi)
            pix_mask   = Mask[hpix]
            phi, theta = phi[pix_mask], theta[pix_mask]

            if phi.size > (N - N_samples):
                size  = N - N_samples
                phi   = phi[:size]
                theta = theta[:size] 

            phi   = phi*180/np.pi
            theta = 90 - theta*180/np.pi

            ra_rand.append(phi)
            dec_rand.append(theta)
            N_samples += phi.size

            print(f"ADDED {phi.size} SAMPLES. TOTAL OF {N_samples} SAMPLES.")

        ra_rand  = np.concatenate(ra_rand)
        dec_rand = np.concatenate(dec_rand)

        return ra_rand, dec_rand

    @timeit
    def compute_response(self, bin):

        dgamma = 0.01*2

        with h5py.File(self.catalog_path, 'r') as f:

            if bin is None:
                Mask1p = f['baseline_mcal_mask_1p'][:] > 0
                Mask2p = f['baseline_mcal_mask_2p'][:] > 0
                Mask1m = f['baseline_mcal_mask_1m'][:] > 0
                Mask2m = f['baseline_mcal_mask_2m'][:] > 0
            else:
                Mask1p = f['baseline_mcal_mask_1p'][:] == bin
                Mask2p = f['baseline_mcal_mask_2p'][:] == bin
                Mask1m = f['baseline_mcal_mask_1m'][:] == bin
                Mask2m = f['baseline_mcal_mask_2m'][:] == bin


            R11tot = (np.average(f['mcal_g_1p'][:][:, 0][Mask1p],  weights = f['mcal_g_w_1p'][:][:][Mask1p]) - 
                      np.average(f['mcal_g_1m'][:][:, 0][Mask1m],  weights = f['mcal_g_w_1m'][:][:][Mask1m]))/dgamma
            
            R22tot = (np.average(f['mcal_g_2p'][:][:, 1][Mask2p],  weights = f['mcal_g_w_2p'][:][:][Mask2p]) - 
                      np.average(f['mcal_g_2m'][:][:, 1][Mask2m],  weights = f['mcal_g_w_2m'][:][:][Mask2m]))/dgamma
            
        return R11tot, R22tot

    
    def rotate_shear(self, g1, g2, seed):
        
        rot_angle = np.random.default_rng(seed).random(g1.size)*2*np.pi

        #Rotate galaxy shapes randomly
        cos    = np.cos(rot_angle)
        sin    = np.sin(rot_angle)
        
        g1_new = + g1 * cos + g2 * sin
        g2_new = - g1 * sin + g2 * cos
        
        return g1_new, g2_new
    

    def load_sim(self, path):
        pass


    def make_kappa_maps(self, path):
        
        density, config = self.load_sim(path)

        #Make convergence shells from density shells
        X = Kappa.MakeKappaShells(config['Om'], config['H0']/100, config['w0'], config['z_bin_edges'], n_jobs = -1, max_nbytes = None)
        k = X.process(density); del density; gc.collect()

        X = Kappa.MakeKappaBins(np.mean(self.n_of_z, axis = 0)[None, :], self.z_grid, config['z_bin_edges'], NSIDE = self.NSIDE)
        k = X.process(k)

        return k
    

    def kappa_to_shear(self, M):
        
        g1, g2 = np.zeros_like(M), np.zeros_like(M)
        zero = np.zeros_like(M[0])
        for m in range(M.shape[0]):
                
            g1[m], g2[m] = utils.kappa_to_shear(M[m], zero)
                
        return g1, g2


    def simulated_shears(self, hpix_cat, g1_map, g2_map):

        g1 = g1_map[hpix_cat]
        g2 = g2_map[hpix_cat]
        
        return g1, g2

    
    @timeit
    def process(self, N_i):
        
        assert "*" in self.map_globpath, f"There needs to be a wildcard `*` in your path for us to glob with {self.map_globpath}"
        
        paths  = sorted(glob.glob(self.map_globpath))
        cat    = self.load_catalog()

        xi     = []
        cov    = []
        for p_i in range(len(paths)):

            if N_i // (len(paths) * rots_per_map) != 0: continue

            k = self.make_kappa_maps(paths[p_i])
            k = my_map2alm(k)

            r_i = N_i % (len(paths) * rots_per_map)

            with timed_execution("Rotation"):
                g = my_klm_to_rotated_shear(k, NSIDE = self.NSIDE, angles = [0, 90*r_i, 0])[0]


            print("COMPLETED ROTATIONS OF MAPS")
            
            corr_res = []

            if self.measurementnoise:
                g1_cat, g2_cat = gen_samples(np.array([cat['g1'], cat['g2']]).T, cat['g_cov'], self.rng).T
            else:
                g1_cat, g2_cat = cat['g1'], cat['g2']

            g1_cat, g2_cat = self.rotate_shear(g1_cat, g2_cat, self.rng.integers(2**30))               
            g1_sim, g2_sim = self.simulated_shears(cat['hpix'], g[0], g[1])

            g1_cat, g2_cat = g1_cat + g1_sim, g2_cat + g2_sim
            # g1_cat, g2_cat = g1_sim, g2_sim

            #Mean shear subtraction
            g1_cat = g1_cat - np.average(g1_cat, weights = cat['w'])
            g2_cat = g2_cat - np.average(g2_cat, weights = cat['w'])

            Catalog = (
                treecorr.Catalog(g1  = g1_cat, 
                                g2  = -g2_cat, #Going from healpix convention to treecorr convention
                                w   = cat['w'],
                                ra  = cat['ra'],  ra_units  = 'deg',
                                dec = cat['dec'], dec_units = 'deg',
                                patch = cat['patch'],
                                verbose = 1)
                )

            del g, g1_cat, g2_cat, g1_sim, g2_sim; gc.collect()

            with timed_execution("MRBmode"):

                X = MRBmodeRunner(Catalog, theta_min = 2.5, theta_max = 250, Ntheta = 1000, Nmodes = 20)
                res = X.compute_EB() #E, B, cov_E, cov_B, corr

                break #End loop once we do one run

        return res


class CosmogridRunner:

    NSIDE        = 1024
    map_globpath = "/project2/chihway/dhayaa/COSMOGRID/run_0*"

    def load_sim(self, path):
        
        tar_filelist = glob.glob(path + '/*tar.gz')
        for t in tar_filelist:
            sp.run('tar -xzf %s -C %s' % (t, path), shell = True)
        
        with open(path + '/params.yml', 'r') as file:
            params = yaml.safe_load(file)

        #Load shells + params
        f = np.load(path + '/compressed_shells.npz')
        
        params['z_bin_edges'] = np.concatenate([f['shell_info']['lower_z'], [f['shell_info']['upper_z'][-1]]])

        d = np.array(f['shells'], dtype = np.float32); del f; gc.collect()
        if self.NSIDE != 2048: d = np.array(my_ud_grade(d, self.NSIDE))

        #Convert from particle counts to overdensity
        for i in range(d.shape[0]):
            d[i] = d[i]/np.mean(d[i]) - 1

        return d, params


class UlagamRunner:

    def load_sim(self, path):
        pass


class DELVERunner(CosmogridRunner, BaseRunner):

    catalog_path = '/project/chihway/data/decade/metacal_gold_combined_20240209.hdf'
    m_bias       = [-0.009, -0.024, -0.037, -0.032]
    n_of_z       = np.load('/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20240408/Summary/mean_nz_combined_Final.npy')
    z_grid       = np.arange(0.01, 5.00, 0.05) + 0.05/2


class SysInjection:

    hsp_map   = None
    const     = 0
    map2shear = 1

    def apply_shears(self):
        '''
        Use the inherited apply_shears function, but now also add contributions
        from the systematic maps computed by Eli.
        '''
        pass


OUTPATH = "/project/chihway/dhayaa/DECADE/BmodeLeakage/"

if __name__ == "__main__":

    import argparse

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--Nind',     action='store', type = int, required = True)
    my_parser.add_argument('--Name',     action='store', type = str, required = True)
    my_parser.add_argument('--seed',     action='store', type = int, default = 42)
    my_parser.add_argument('--Npatch',   action='store', type = int, default = 150)

    my_parser.add_argument('--DELVE',    action='store_true')
    my_parser.add_argument('--DES',      action='store_true')
    my_parser.add_argument('--homogenize', action='store_true')
    my_parser.add_argument('--measurementnoise', action='store_true')
    
    args = vars(my_parser.parse_args())

    assert args['DES'] + args['DELVE'] != 2, "You can only use --DELVE or --DES, not both"

    if args['DELVE']:
        res = DELVERunner(seed = args['seed'], Npatch = args['Npatch'], 
                          homogenize = args['homogenize'],
                          measurementnoise = args['measurementnoise']).process(N_i = args['Nind'])
        res = {'E': res[0], 'B': res[1], 'cov_E': res[2], 'cov_B': res[3], 'corr': res[4]}
        np.save(OUTPATH + f"/{args['Name']}.npy", res, allow_pickle = True)
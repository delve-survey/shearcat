import numpy as np
import os
import h5py
from tqdm import tqdm
import joblib
import healpy as hp
import pymaster as nmt
import os


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
            self.weight_map =  self.n_map.copy()
            self.weight_map[self.unique_pix] /= np.bincount(self.idx_rep) #Get mean weight per pixel
            
            
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


class BmodeRunner(object):

    tmp_bins = np.linspace(np.sqrt(8), np.sqrt(2048), 33)**2
    tmp_bins = tmp_bins.astype(int)

    def __init__(self, MakeMapFromCat, seed, bins = tmp_bins, njobs = -1):

        self.MakeMapFromCat = MakeMapFromCat
        
        self.seed    = seed
        self.njobs   = njobs
        self.mask    = MakeMapFromCat.mask_sims * MakeMapFromCat.weight_map
        
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
    

    def process_noise(self, Nrands = 100):

        seeds = np.random.default_rng(self.seed).integers(0, 2**63, Nrands)
        
        with joblib.parallel_backend("loky"):
#             jobs    = [joblib.delayed(self.single_run_noise)(i, seeds[i]) for i in np.arange(Nrands)]
#             outputs = joblib.Parallel(n_jobs = self.njobs, verbose = 10)(jobs)

            outputs = [self.single_run_noise(i, seeds[i]) for i in tqdm(np.arange(Nrands), desc = 'Make Rands')]
        
            
            final   = [0]*Nrands
            for o in outputs: final[o[0]] = o[1]

        return np.array(final)
       

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

    def __init__(self, MakeMapFromCat, seed, bins = tmp_bins, apodised_scale = 2/60, njobs = -1):

        super().__init__(MakeMapFromCat, seed, bins, njobs)

        self.mask = nmt.mask_apodization(MakeMapFromCat.mask_sims.astype(float).byteswap(), apodised_scale, apotype='C1')
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


if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser()

    #Metaparams
    my_parser.add_argument('--NSIDE',     action='store', type = int, default = 1024)
    my_parser.add_argument('--Name',      action='store', type = str, required = True)
    my_parser.add_argument('--njobs',     action='store', type = int, default = -1)
    my_parser.add_argument('--Output',    action='store', type = str, required = True, help = 'Directory to store B-mode data')
    my_parser.add_argument('--Nrands',    action='store', type = int, default = 100)
    my_parser.add_argument('--DECADE',    action='store_true', dest = 'DECADE')
    my_parser.add_argument('--DES',       action='store_true', dest = 'DES')
    my_parser.add_argument('--Foreground_Mask', action='store', type = str, default = '/project2/chihway/dhayaa/DECADE/Gold_Foreground_20230607.fits')
    

    args  = vars(my_parser.parse_args())

    nside = args['NSIDE']
    npix  = hp.nside2npix(nside)

    if args['DECADE']:
        with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230531.hdf', 'r') as f:

            ra      = np.array(f['RA'])
            dec     = np.array(f['DEC'])
            g1, g2  = np.array(f['mcal_g_noshear']).T
            w       = np.array(f['mcal_g_w'])

            mag_r   = 30 - 2.5*np.log10(np.array(f['mcal_flux_noshear'])[:, 0])
            mag_i   = 30 - 2.5*np.log10(np.array(f['mcal_flux_noshear'])[:, 1])
            mag_z   = 30 - 2.5*np.log10(np.array(f['mcal_flux_noshear'])[:, 2])
            
            SNR     = np.array(f['mcal_s2n_noshear'])
            T_ratio = np.array(f['mcal_T_ratio_noshear'])
            T       = np.array(f['mcal_T_noshear'])
            flags   = np.array(f['mcal_flags'])
            
            GOLD_Foreground  = hp.read_map(args['Foreground_Mask'], dtype = int)
            FLAGS_Foreground = GOLD_Foreground[hp.ang2pix(hp.npix2nside(GOLD_Foreground.size), ra, dec, lonlat = True)]

            #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)

            SNR_Mask   = (SNR > 10) & (SNR < 1000)
            Tratio_Mask= T_ratio > 0.5
            T_Mask     = T < 10
            Flag_Mask  = flags == 0
            Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))
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

            g1  = g1[Mask]
            g2  = -g2[Mask]
            ra  = ra[Mask]
            dec = dec[Mask]
            w   = w[Mask]

            del SNR_Mask, Tratio_Mask,T_Mask, Flag_Mask, Other_Mask, GOLD_Mask, SG_Mask, Mask
            del mag_r, SNR, T_ratio, T, flags, FLAGS_Foreground, GOLD_Foreground
            
    elif args['DES']:
        
        with h5py.File('/project2/chihway/dhayaa/DES_Catalogs/DESY3_indexcat.h5') as f:
            mcal_selection = np.array(f['index/select'])

        with h5py.File('/project2/chihway/dhayaa/DES_Catalogs/DESY3_metacal_v03-004.h5') as f:

            g1  = np.array(f['catalog/unsheared/e_1'])[mcal_selection]
            g2  = -np.array(f['catalog/unsheared/e_2'])[mcal_selection]
            ra  = np.array(f['catalog/unsheared/ra'])[mcal_selection]
            dec = np.array(f['catalog/unsheared/dec'])[mcal_selection]
            
            w = np.array(f['catalog/unsheared/weight'])[mcal_selection]

            del mcal_selection



    R = MakeMapFromCat(ra = ra, dec = dec, e1 = g1, e2 = g2, w = w, NSIDE = args['NSIDE']); del ra, dec, g1, g2
#     X = PureBmodeRunner(R, 42, apodised_scale = 2/60, njobs = args['njobs'])
    X = BmodeRunner(R, 42, njobs = args['njobs'])
    print(np.average(R.mask_sims))
    print("FINISHED INITIATING RUNNER")
    data  = X.process_data()
    Noise = X.process_noise(args['Nrands'])

    Name = '_%s' %args['Name'] if args['Name'] is not None else ''
    path = args['Output'] + '/Bmode%s' % Name

    np.savez(path, data = data, noise = Noise, ell = X.ell_eff)
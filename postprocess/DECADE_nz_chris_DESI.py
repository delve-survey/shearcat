import numpy as np, healpy as hp
import h5py, pandas as pd, os, sys
from scipy import interpolate

DEEP_COLS = ['ID', 'RA', 'DEC', 'KNN_CLASS', 'TILENAME',
             'FLAGS', 'FLAGSTR', 'FLAGSTR_NIR', 'FLAGS_NIR', 'MASK_FLAGS', 'MASK_FLAGS_NIR',
             'BDF_FLUX_DERED_CALIB_U', 'BDF_FLUX_DERED_CALIB_G', 'BDF_FLUX_DERED_CALIB_R',
             'BDF_FLUX_DERED_CALIB_I', 'BDF_FLUX_DERED_CALIB_Z', 'BDF_FLUX_DERED_CALIB_J', 
             'BDF_FLUX_DERED_CALIB_H', 'BDF_FLUX_DERED_CALIB_KS', 
             'BDF_FLUX_ERR_DERED_CALIB_U','BDF_FLUX_ERR_DERED_CALIB_G', 'BDF_FLUX_ERR_DERED_CALIB_R',
             'BDF_FLUX_ERR_DERED_CALIB_I', 'BDF_FLUX_ERR_DERED_CALIB_Z', 'BDF_FLUX_ERR_DERED_CALIB_J', 
             'BDF_FLUX_ERR_DERED_CALIB_H', 'BDF_FLUX_ERR_DERED_CALIB_KS', 
             ]



class Processor:

    def __init__(self, deep_catalog_path, balrog_catalog_path, redshift_catalog_path, grid_path):
        self.deep_catalog_path   = deep_catalog_path
        self.balrog_catalog_path = balrog_catalog_path
        self.redshift_catalog_path = redshift_catalog_path
        self.grid_path             = grid_path


    def get_balrog_catalog(self, balrog_classified_df):

        
        #-------------------------- READ OUT ALL DEEP QUANTITIES --------------------------
        f = pd.read_csv(self.deep_catalog_path, usecols = DEEP_COLS).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        deep_sample_cuts  = self.get_deep_sample_cuts(f)
        Deep_df = f[deep_was_detected & deep_sample_cuts]
        
        print("DEEP", len(Deep_df))
        
        
        #-------------------------- READ OUT ALL BALROG QUANTITIES --------------------------
        selection = self.get_wl_sample_mask(self.balrog_catalog_path)
        purity    = self.get_balrog_contam_mask(self.balrog_catalog_path) & self.get_foreground_mask(self.balrog_catalog_path)
        with h5py.File(self.balrog_catalog_path, 'r') as f:

            Balrog_df = pd.DataFrame()
            Balrog_df['ID'] = f['ID'][:]
            Balrog_df['w']  = self.get_shear_weights(f['mcal_s2n_noshear'][:], f['mcal_T_ratio_noshear'][:])
            Balrog_df['id']        = f['id'][:]
            Balrog_df['tilename']  = f['tilename'][:]
            Balrog_df['selection'] = selection & (f['detected'][:] == 1)
            Balrog_df['purity']    = purity
            Balrog_df['true_ra']   = f['true_ra'][:]
            Balrog_df['true_dec']  = f['true_dec'][:]
            
            #Only keep Balrog objects that come from good DF objects
            #And only objects that have no contamination from real objects/sources
            Balrog_df = Balrog_df[np.isin(Balrog_df['ID'], Deep_df['ID'])]
            Balrog_df = Balrog_df[Balrog_df['purity'] == True]
            
            Balrog_df = pd.merge(Balrog_df, balrog_classified_df[['cell', 'true_ra', 'true_dec']], 
                                    how = 'left', on = ['true_ra', 'true_dec'], suffixes = (None, '_classified'), 
                                    validate = "1:1")
            
            Balrog_df['cell_wide_unsheared'] = Balrog_df['cell'] #Rename so it works with Alex's code
            
            counts = pd.DataFrame()
            counts['ID'], counts['injection_counts']  = np.unique(f['ID'][:], return_counts = True)

            detect = pd.DataFrame()
            detect['ID'], detect['detect_counts'] = np.unique(f['ID'][:][selection & (f['detected'][:] == 1)], return_counts = True)


            Balrog_df = pd.merge(Balrog_df, counts, on = 'ID', how = 'left', validate = "m:1")
            Balrog_df = pd.merge(Balrog_df, detect, on = 'ID', how = 'left', validate = "m:1")

            Balrog_df['overlap_weight'] = 1/Balrog_df['injection_counts'] * Balrog_df['w']
            Balrog_df['true_id'] = Balrog_df['ID']
            
            #Use only Balrog objects that were actually detected
            Balrog_df = Balrog_df[Balrog_df['selection'] == True]

        return Balrog_df


    def get_deep_catalog(self,  deep_classified_df):

        f = pd.read_csv(self.deep_catalog_path).reset_index(drop = True)
        deep_was_detected = self.get_deep_mask(self.deep_catalog_path, self.balrog_catalog_path)
        deep_sample_cuts  = self.get_deep_sample_cuts(f)
        Deep_df = f[deep_was_detected & deep_sample_cuts]
        
        #Check masking was ok. I'm paranoid about pandas masking sometimes
        assert len(f) == len(deep_was_detected & deep_sample_cuts), "Mask is not right size"
        assert len(Deep_df) == np.sum(deep_was_detected & deep_sample_cuts), "Masked df doesn't have right size"
        
        #Now check and merge with classifier
        Deep_df = pd.merge(Deep_df, deep_classified_df[['cell', 'ID']], how = 'right', on = 'ID', suffixes = (None, '_classified'))
        
        Deep_df['cell_deep'] = Deep_df['cell']
        Deep_df['true_id']   = Deep_df['ID']
        
        Z_df    = pd.read_csv(self.redshift_catalog_path)
        Deep_df = pd.merge(Deep_df, Z_df[['ID', 'Z', 'SOURCE']], on = "ID", how = 'left')

        return Deep_df


    def get_redshift_catalog(self,  deep_classified_df):
        
        df = self.get_deep_catalog(deep_classified_df)
        df = df[df['Z'] > 0]
        
        return df



    def get_deep_fluxes(self, path, balrog_path):

        #Deep field bands
        bands = [B.upper() for B in ['u', 'g', 'r', 'i', 'z', 'J', 'H', 'KS']]

        f = pd.read_csv(path, usecols = DEEP_COLS)

        flux     = np.array([f['BDF_FLUX_DERED_CALIB_%s' % b].values for b in bands]).T
        flux_err = np.array([f['BDF_FLUX_ERR_DERED_CALIB_%s' % b].values for b in bands]).T
        ID       = f['ID'].values
        tilename = f['TILENAME'].values

        deep_was_detected = self.get_deep_mask(path, balrog_path)
        deep_is_pure      = self.get_deep_sample_cuts(f)

        print("-----------------------")
        print("DEEP FIELD STATS")
        print("-----------------------")
        print("ORIGINAL: %d GALAXIES" % deep_was_detected.size)
        print("DETECTED: %d GALAXIES" % np.sum(deep_was_detected))
        print("PURE: %d GALAXIES" % np.sum(deep_is_pure))
        print("FINAL: %d GALAXIES" % np.sum(deep_was_detected & deep_is_pure))
        print("-----------------------\n\n")
        
        mask     = deep_was_detected & deep_is_pure
        flux     = flux[mask]
        flux_err = flux_err[mask]
        ID       = ID[mask]
        tilename = tilename[mask]

        return flux, flux_err, ID, tilename


    def get_deep_mask(self, path, balrog_path):

        f  = pd.read_csv(path, usecols = DEEP_COLS)
        ID = f['ID'].values
        
        #In Y3, some CCDs have bad chips. So we remove deepfield objects associated with those CCDs alone.
        BAD_CHIPS = ["SN-C3_C01", "SN-C3_C06", "SN-C3_C11", "SN-C3_C54", "SN-C3_C55", "SN-C3_C57", "SN-C3_C58", "SN-C3_C62",
                
                        "SN-X3_C10", "SN-X3_C12", "SN-X3_C15", "SN-X3_C19", "SN-X3_C29", "SN-X3_C46", 
                        "SN-X3_C47", "SN-X3_C49", "SN-X3_C52", "SN-X3_C60", "SN-X3_C62",

                        "SN-E2_C11", "SN-E2_C41", "SN-E2_C49",]
        
        deep_GOOD = np.invert(np.isin(f['TILENAME'].values, BAD_CHIPS))
        
        balrog_gold = (self.get_wl_sample_mask(balrog_path) & 
                        self.get_foreground_mask(balrog_path) & 
                        self.get_balrog_contam_mask(balrog_path))
        
        with h5py.File(balrog_path, 'r') as f:

            balrog_ID   = f['ID'][:]
            balrog_det  = f['detected'][:] == 1

        Mask = balrog_gold & balrog_det
        
        balrog_ID = np.unique(balrog_ID[Mask])
        
        deep_was_detected = np.isin(ID, balrog_ID)
        
        
        return deep_was_detected & deep_GOOD


    def get_deep_sample_cuts(self, deep_catalog):
        '''
        places color cuts on deep field catalog
        Credit: Alex Alarcon
        '''

        #Mask flagged regions -- not needed, saved deep catalog already has flag cuts in place
        mask  = deep_catalog.MASK_FLAGS_NIR==0
        mask &= deep_catalog.MASK_FLAGS==0
        mask &= deep_catalog.FLAGS_NIR==0
        mask &= deep_catalog.FLAGS==0
        
        #These two sometimes fail depending on the catalog and what format it writes
        #string into. Eitherway, this flag is equivalent to the ==0 flags above so
        #using those instead is better.
        #mask &= deep_catalog.FLAGSTR=="ok"
        #mask &= deep_catalog.FLAGSTR_NIR=="ok"
        
        
        deep_bands_ = ["U","G","R","I","Z","J","H","KS"]
        # remove crazy colors, defined as two 
        # consecutive colors (e.g u-g, g-r, r-i, etc) 
        # that have a value smaler than -1
        mags_d = np.zeros((len(deep_catalog),len(deep_bands_)))
        magerrs_d = np.zeros((len(deep_catalog),len(deep_bands_)))

        def flux2mag(flux):    
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return 30 - 2.5*np.log10(flux)
        
        for i,band in enumerate(deep_bands_):
            #print(i,band)
            mags_d[:,i] = flux2mag(deep_catalog['BDF_FLUX_DERED_CALIB_%s'%band].values)

        colors = np.zeros((len(deep_catalog),len(deep_bands_)-1))
        for i in range(len(deep_bands_)-1):
            colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

        normal_colors = np.all(colors > -1, axis=1)
        
        return mask & normal_colors

    def get_shear_weights(self, S2N, T_over_Tpsf):
        
        path = self.grid_path
        res  = np.load(path, allow_pickle = True)[()]
        S    = res['s2n'].flatten()
        T    = res['T_over_Tpsf'].flatten()
        R    = res['R'].flatten()
        w    = res['w'].flatten()

        #Have checked that this what DESY3 uses.
        interp        = interpolate.NearestNDInterpolator((S, T), R * w,)
        shear_weights = interp( (S2N, T_over_Tpsf) )
        
        return shear_weights
    
    def get_wl_sample_mask(self, path, label = 'noshear'):

        with h5py.File(path, 'r') as f:
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
        
                flux_r, flux_i, flux_z = f[f'mcal_flux_{label}_dered_sfd98'][:].T
                
                mag_r = 30 - 2.5*np.log10(flux_r)
                mag_i = 30 - 2.5*np.log10(flux_i)
                mag_z = 30 - 2.5*np.log10(flux_z)

                mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                                (mag_r < 26)   & (mag_r > 15) & 
                                (mag_z < 26)   & (mag_z > 15) & 
                                (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                                (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

                del mag_i, mag_z
                
                SNR     = f[f'mcal_s2n_{label}'][:]
                T_ratio = f[f'mcal_T_ratio_{label}'][:]
                T       = f[f'mcal_T_{label}'][:]
                flags   = f['mcal_flags'][:]
                g1, g2  = f[f'mcal_g_{label}'][:].T

                #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
                Tratio_Mask= T_ratio > 0.5; del T_ratio
                Flag_Mask  = flags == 0; del flags
                SNR_Mask   = (SNR > 10) & (SNR < 1000)
                T_Mask     = T < 10
                
                Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

                del g1, g2, mag_r, T
                
                Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask

        return Mask
    
    
    def de_islandify(self, ra):

        maxdec  = np.piecewise(ra, 
                               [((310 < ra) & (ra <= 360)) | (ra < 50), (ra < 310) & (ra > 180)],
                               [lambda ra: np.where((310 < ra) & (ra < 350), 
                                                    3.5, 
                                                    np.where(ra > 350, 
                                                             (ra - 350) * (18 - 3.5)/(20) + 3.5, 
                                                             (ra + 10)  * (18 - 3.5)/(20) + 3.5
                                                             ) 
                                                    ),
                                lambda ra: np.where(ra < 225, 
                                                    30 - (30 - 12)/(225 - 200) * (ra - 200), 
                                                    12.),
                                lambda ra: 40]
                                )
        
        return maxdec
    
    def get_foreground_mask(self, path):

        Badcolor_map = hp.read_map('/project2/kadrlica/chinyi/DELVE_DR3_1_bad_colour_mask.fits', dtype = int)
        
        with h5py.File(path, 'r') as f:
            FG_mask = f['FLAGS_FOREGROUND'][:] == 0
            
            if 'RA' in f.keys():
                # Region_mask = np.invert(f['DEC'][:] > np.where(f['RA'][:] < 225, 30 - (30 - 12)/(225 - 200) * (f['RA'][:] - 200), 12.))
                Region_mask = np.invert(f['DEC'][:] > self.de_islandify(f['RA'][:])) 
                pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['RA'][:], f['DEC'][:], lonlat = True)
            else:
                # Region_mask = np.invert(f['true_dec'][:] > np.where(f['true_ra'][:] < 225, 30 - (30 - 12)/(225 - 200) * (f['true_ra'][:] - 200), 12.))
                Region_mask = np.invert(f['true_dec'][:] > self.de_islandify(f['true_ra'][:])) 
                pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['true_ra'][:], f['true_dec'][:], lonlat = True)
                
            Color_mask = Badcolor_map[pix_assign] == 0; del pix_assign
        
        Mask = FG_mask & Region_mask & Color_mask; del Badcolor_map, FG_mask, Region_mask, Color_mask
        
        return Mask

    def get_balrog_contam_mask(self, path):

        with h5py.File(path, 'r') as f:
            
            #Only select objects with no GOLD object within 1.5 arcsec
            if 'd_contam_arcsec' in f.keys():
                balrog_cont = f['d_contam_arcsec'][:] > 1.5 
            else:
                balrog_cont = True
            
        return balrog_cont


    def process(self, output_dir, out_path):

        bclass  = pd.DataFrame({'id'       : np.load(output_dir + '/BALROG_DATA_ID.npy'),
                                'true_ra'  : np.load(output_dir + '/BALROG_DATA_TRUE_RA.npy'),
                                'true_dec' : np.load(output_dir + '/BALROG_DATA_TRUE_DEC.npy'),
                                'cell'     : np.load(output_dir + '/collated_balrog_classifier.npy')})
        
        dclass  = pd.DataFrame({'ID'       : np.load(output_dir + '/DEEP_DATA_ID.npy'),
                                'cell'     : np.load(output_dir + '/collated_deep_classifier.npy')})
        
        balrog = self.get_balrog_catalog(bclass)
        deep   = self.get_deep_catalog(dclass)

        Tomo   = np.load(output_dir + '/TomoBinAssign.npy').flatten()
        balrog['tomobin'] = Tomo[balrog['cell'].values.astype(int)]

        balrog_data = pd.merge(balrog[['ID', 'w', 'tomobin']], deep[['ID', 'Z', 'SOURCE'] + [d for d in DEEP_COLS if 'BDF_FLUX_DERED' in d]], 
                               on = 'ID', how = 'left')

        
        balrog_data = balrog_data[balrog_data['Z'] > 0]
        balrog_data.to_csv(out_path, index = False)

        print(balrog_data)



if __name__ == '__main__':
    
    X = Processor(balrog_catalog_path = '/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20240123.hdf5',
                  deep_catalog_path = '/project/chihway/dhayaa/DECADE/Imsim_Inputs/deepfield_Y3_allfields.csv',
                  redshift_catalog_path = '/project/chihway/dhayaa/DECADE/Redshift_files/deepfields_raw_with_redshifts_20240723.csv.gz',
                  grid_path             = '/home/dhayaa/DECADE/CosmicShearPhotoZ/weights_20231212.npy') 
    

    X.process(output_dir = '/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20241113/',
              out_path = '/project/chihway/dhayaa/DECADE/PostprocessForPeople/ForChris_DR3_NGC.csv')




    Y = Processor(balrog_catalog_path = '/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20241223.hdf5',
                  deep_catalog_path = '/project/chihway/dhayaa/DECADE/Imsim_Inputs/deepfield_Y3_allfields.csv',
                  redshift_catalog_path = '/project/chihway/dhayaa/DECADE/Redshift_files/deepfields_raw_with_redshifts_20240723.csv.gz',
                  grid_path             = '/home/dhayaa/DECADE/CosmicShearPhotoZ/weights_20240209.npy') 
    

    Y.process(output_dir = '/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20241223_DR3_2/',
              out_path = '/project/chihway/dhayaa/DECADE/PostprocessForPeople/ForChris_DR3_SGC.csv')
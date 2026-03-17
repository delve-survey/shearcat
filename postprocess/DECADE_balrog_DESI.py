import numpy as np, healpy as hp
import h5py, pandas as pd
import os
from scipy import interpolate


OUT = '/scratch/midway3/dhayaa/Balrog_TMP.hdf5'

KEYOUT = {
    "ID" : "id_y3_deepfield",
    "d_arcsec"        : "d_arcsec",
    "d_contam_arcsec" : "d_contam_arcsec",
    "dec" : "dec",
    "inj_class" : "injection_class",
    "mcal_T_noshear" : "mcal_T_noshear",
    "mcal_T_ratio_noshear"  : "mcal_T_ratio_noshear",
    "mcal_flux_err_noshear_dered_sfd98" : "mcal_flux_err_noshear_dered_sfd98",
    "mcal_flux_noshear_dered_sfd98" : "mcal_flux_noshear_dered_sfd98",
    "mcal_g_noshear" : "mcal_g_noshear",
    "mcal_psf_T_noshear" : "mcal_psf_T_noshear",
    "mcal_psf_g_noshear" : "mcal_psf_g_noshear",
    "mcal_s2n_noshear" : "mcal_s2n_noshear",
    "ra" : "ra",
    "true_FLUX_i" : "true_bdf_flux_i",
    "true_FLUX_r" : "true_bdf_flux_r",
    "true_FLUX_z" : "true_bdf_flux_z",
    "true_dec" : "true_dec",
    "true_ra"  : "true_ra",
}

def get_balrog_contam_mask(path):

    with h5py.File(path, 'r') as f:
        
        #Only select objects with no GOLD object within 1.5 arcsec
        if 'd_contam_arcsec' in f.keys():
            balrog_cont = f['d_contam_arcsec'][:] > 1.5 
        else:
            balrog_cont = True
        
    return balrog_cont


def get_wl_sample_mask(path, label = 'noshear'):

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


def de_islandify(ra):

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


def get_foreground_mask(path):

    Badcolor_map = hp.read_map('/project2/kadrlica/chinyi/DELVE_DR3_1_bad_colour_mask.fits', dtype = int)
    
    with h5py.File(path, 'r') as f:
        FG_mask = f['FLAGS_FOREGROUND'][:] == 0
        
        if 'RA' in f.keys():
            Region_mask = np.invert(f['DEC'][:] > de_islandify(f['RA'][:])) 
            pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['RA'][:], f['DEC'][:], lonlat = True)
        else:
            Region_mask = np.invert(f['true_dec'][:] > de_islandify(f['true_ra'][:])) 
            pix_assign  = hp.ang2pix(hp.npix2nside(Badcolor_map.size), f['true_ra'][:], f['true_dec'][:], lonlat = True)
            
        Color_mask = Badcolor_map[pix_assign] == 0; del pix_assign
    
    Mask = FG_mask & Region_mask & Color_mask; del Badcolor_map, FG_mask, Region_mask, Color_mask
    
    return Mask


DEEP_COLS = ['ID', 'RA', 'DEC', 'KNN_CLASS', 'TILENAME',
             'FLAGS', 'FLAGSTR', 'FLAGSTR_NIR', 'FLAGS_NIR', 'MASK_FLAGS', 'MASK_FLAGS_NIR',
             'BDF_FLUX_DERED_CALIB_U', 'BDF_FLUX_DERED_CALIB_G', 'BDF_FLUX_DERED_CALIB_R',
             'BDF_FLUX_DERED_CALIB_I', 'BDF_FLUX_DERED_CALIB_Z', 'BDF_FLUX_DERED_CALIB_J', 
             'BDF_FLUX_DERED_CALIB_H', 'BDF_FLUX_DERED_CALIB_KS', 
             'BDF_FLUX_ERR_DERED_CALIB_U','BDF_FLUX_ERR_DERED_CALIB_G', 'BDF_FLUX_ERR_DERED_CALIB_R',
             'BDF_FLUX_ERR_DERED_CALIB_I', 'BDF_FLUX_ERR_DERED_CALIB_Z', 'BDF_FLUX_ERR_DERED_CALIB_J', 
             'BDF_FLUX_ERR_DERED_CALIB_H', 'BDF_FLUX_ERR_DERED_CALIB_KS', 
             ]

def get_deep_mask(path):

    f  = pd.read_csv(path, usecols = DEEP_COLS)
    
    #In Y3, some CCDs have bad chips. So we remove deepfield objects associated with those CCDs alone.
    BAD_CHIPS = ["SN-C3_C01", "SN-C3_C06", "SN-C3_C11", "SN-C3_C54", "SN-C3_C55", "SN-C3_C57", "SN-C3_C58", "SN-C3_C62",
            
                    "SN-X3_C10", "SN-X3_C12", "SN-X3_C15", "SN-X3_C19", "SN-X3_C29", "SN-X3_C46", 
                    "SN-X3_C47", "SN-X3_C49", "SN-X3_C52", "SN-X3_C60", "SN-X3_C62",

                    "SN-E2_C11", "SN-E2_C41", "SN-E2_C49",]
    
    deep_GOOD = np.invert(np.isin(f['TILENAME'].values, BAD_CHIPS))
    
    return deep_GOOD


def get_deep_sample_cuts(deep_catalog):
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
    

def get_shear_weights(path, S2N, T_over_Tpsf):
        
    res  = np.load(path)
    S    = res['SNR']
    T    = res['T_ratio']
    R    = (res['R11'] + res['R22'])/2 #Average over both components. No selection response, as in Y3
    w    = res['weight']
    
    #Have checked that this what DESY3 uses.
    interp        = interpolate.NearestNDInterpolator((S, T), R * w,)
    shear_weights = interp((S2N, T_over_Tpsf) )
    
    return shear_weights


deep_catalog_path = '/project/chihway/dhayaa/DECADE/Imsim_Inputs/deepfield_Y3_allfields.csv'
f = pd.read_csv(deep_catalog_path).reset_index(drop = True)
deep_was_detected = get_deep_mask(deep_catalog_path)
deep_sample_cuts  = get_deep_sample_cuts(f)
Deep_df = f[deep_was_detected & deep_sample_cuts]


with h5py.File(OUT, 'w') as b:


    NGC = b.create_group('NGC')
    SGC = b.create_group('SGC')


    ##### NGC ######
    path      = '/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20240123.hdf5'
    wpath     = '/home/dhayaa/DECADE/shearcat/postprocess/grid_quantities_20240827.npy'
    selection = get_wl_sample_mask(path)
    purity    = get_balrog_contam_mask(path) & get_foreground_mask(path)

    with h5py.File(path, 'r') as f:

        for k in KEYOUT.keys():
            NGC.create_dataset(KEYOUT[k], data = f[k][:])

        gooddeep = np.isin(f['ID'][:], Deep_df['ID'].values)
    

        NGC.create_dataset('selection', data = selection,         dtype = bool)
        NGC.create_dataset('good_inj',  data = purity & gooddeep, dtype = bool)
        NGC.create_dataset('z_weights', data = get_shear_weights(wpath, f['mcal_s2n_noshear'][:], f['mcal_T_ratio_noshear'][:]))

    bpath   = '/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20241113/'
    tomob   = np.load(bpath + 'TomoBinAssign.npy').astype(int)
    bclass  = pd.DataFrame({'id'       : np.load(bpath + '/BALROG_DATA_ID.npy'),
                            'true_ra'  : np.load(bpath + '/BALROG_DATA_TRUE_RA.npy'),
                            'true_dec' : np.load(bpath + '/BALROG_DATA_TRUE_DEC.npy'),
                            'cell'     : np.load(bpath + '/collated_balrog_classifier.npy')})
    
    mask = selection & purity
    
    assert np.allclose(bclass['true_ra'].values,  NGC['true_ra'][:][mask])
    assert np.allclose(bclass['true_dec'].values, NGC['true_dec'][:][mask])

    widebins = np.zeros(len(mask)) + -99
    widebins[mask] = tomob[bclass['cell'].values.astype(int)]
    NGC.create_dataset('tomobin', data = widebins)



    ##### SGC ######
    path      = '/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20241223.hdf5'
    wpath     = '/home/dhayaa/DECADE/shearcat/postprocess/grid_quantities_20250206_DR3_2.npy'
    selection = get_wl_sample_mask(path)
    purity    = get_balrog_contam_mask(path) & get_foreground_mask(path)

    with h5py.File(path, 'r') as f:

        for k in KEYOUT.keys():
            SGC.create_dataset(KEYOUT[k], data = f[k][:])

        gooddeep = np.isin(f['ID'][:], Deep_df['ID'].values)
    

        SGC.create_dataset('selection', data = selection,         dtype = bool)
        SGC.create_dataset('good_inj',  data = purity & gooddeep, dtype = bool)
        SGC.create_dataset('z_weights', data = get_shear_weights(wpath, f['mcal_s2n_noshear'][:], f['mcal_T_ratio_noshear'][:]))

    bpath   = '/project/chihway/dhayaa/DECADE/SOMPZ/Runs/20241223_DR3_2/'
    tomob   = np.load(bpath + 'TomoBinAssign.npy').astype(int)
    bclass  = pd.DataFrame({'id'       : np.load(bpath + '/BALROG_DATA_ID.npy'),
                            'true_ra'  : np.load(bpath + '/BALROG_DATA_TRUE_RA.npy'),
                            'true_dec' : np.load(bpath + '/BALROG_DATA_TRUE_DEC.npy'),
                            'cell'     : np.load(bpath + '/collated_balrog_classifier.npy')})
    
    mask = selection & purity
    
    assert np.allclose(bclass['true_ra'].values,  SGC['true_ra'][:][mask])
    assert np.allclose(bclass['true_dec'].values, SGC['true_dec'][:][mask])

    widebins = np.zeros(len(mask)) + -99
    widebins[mask] = tomob[bclass['cell'].values.astype(int)]
    SGC.create_dataset('tomobin', data = widebins)
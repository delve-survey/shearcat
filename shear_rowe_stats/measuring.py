#use this to measure the stars' and psf models' ellipticities

#stuff needed to load PSF files and measure shapes
import galsim
from galsim.des import DES_PSFEx
import ngmix
from ngmix.fitting import Fitter as LMSimple
from ngmix.admom import AdmomFitter as Admom
from ngmix.flags import get_flags_str as flagstr
from ngmix import priors, joint_prior

from astropy.io.fits.verify import VerifyWarning
import warnings

#usual stuff
import numpy as np
from astropy.io import fits
from astropy.table import vstack
from astropy.wcs import WCS
import logging
import pandas as pd
import os, shutil
from time import time
from re import findall
import glob
import pathlib

import joblib

class ImageRunner(object):
    
    def __init__(self, start_ind, end_ind, seed, imagelist, magzp_list, n_jobs = -1):
        '''
            imagelist is a list with format expnum_band_ccdnum, eg. D00284091_z_c11
        '''
        
        self.imagelist  = imagelist[int(start_ind) : int(end_ind)]
        self.seed       = seed
        self.magzp_list = magzp_list[int(start_ind) : int(end_ind)]
        
        self.start_ind  = start_ind
        self.end_ind    = end_ind
        
        
        self.n_jobs = n_jobs
        
        
    def single_run(self, image, magzp):
        
#         MeasurementRunner(self.seed, os.environ['EXP_DIR'], image, magzp).go()
        
        try:
            res = MeasurementRunner(self.seed, os.environ['EXP_DIR'], image, magzp).go()
            return res
        except:
            print("FAILED ON CCD", image)
            return None
        
    def go(self):
        
        jobs = [joblib.delayed(self.single_run)(self.imagelist[i], self.magzp_list[i]) for i in range(len(self.imagelist))]

        with joblib.parallel_backend("loky"):
            out = joblib.Parallel(n_jobs = self.n_jobs, verbose=10)(jobs)
            
        output = [o.data for o in out]
        
        filename = 'output_%s_to_%s.fits' % (self.start_ind, self.end_ind)
        filepath = os.path.join(os.environ['ROWE_STATS_DIR'], filename)
        
        output = vstack(output, join_type = 'exact')
        
        output.write(filepath, overwrite=True)
                    
        return None
        
        
class MeasurementRunner(object):

    def __init__(self, seed, path_to_files, imagename, magzp, n_jobs = 1, write_table = False):

        self.rng = np.random.default_rng(seed)
        self.stampsize = 48
        self.imagename = imagename
        self.magzp     = magzp
        
        self.write_table = write_table
        
        self.image_stem = '_'.join(self.imagename.split('_')[:3])
                
        self.path_to_files = path_to_files
        
        self.n_jobs = 1

        pass


    def load_images(self, path_to_files):
        '''
        Load all the relevant images for PSF shape measuremet
        '''

#         print(glob.glob(path_to_files + '/*'))

        
        
        image_path     = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*immasked*')[0]
        bkg_path       = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*bkg*')[0]
        cat_path       = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*red-fullcat*')[0]
        psf_path       = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*psfexcat*')[0]
        starlist_path  = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*psfex-starlist*')[0]
        
        self.expnum    = int(os.path.basename(image_path).split('_')[0][1:])
        self.ccdnum    = int(os.path.basename(image_path).split('_')[2][1:])
        self.band      = str(os.path.basename(image_path).split('_')[1][0])
        
        self.image     = galsim.fits.read(image_path)
        self.weight    = galsim.fits.read(image_path, hdu = 3)
        self.cat       = fits.open(cat_path)[2].data
        self.starlist  = fits.open(starlist_path)[2].data
        self.bkg       = galsim.fits.read(bkg_path)
        self.des_psfex = galsim.des.DES_PSFEx(psf_path, image_path)

        image_fits = fits.open(image_path)
        self.fwhm  = image_fits['SCI'].header['FWHM']*0.26

        self.wcs_pixel_world = WCS(image_fits[1].header)

        ccdcoords = findall(r'\d+',image_fits['sci'].header['detsec'])
        self.min_x_pix, self.min_y_pix = int(ccdcoords[0]), int(ccdcoords[2])

        return None


    def single_run(self, goodstar_index):

        X = self.starlist['x_image'].astype(int)[goodstar_index]
        Y = self.starlist['y_image'].astype(int)[goodstar_index]
        X_float = self.starlist['x_image'][goodstar_index] #getting them as floats also (as in the file itself)
        Y_float = self.starlist['y_image'][goodstar_index]

        MAGZP  = np.NaN #NEED TO FIX THIS

        #first, check if this star with PSF_FLAGS==0 has a match in the sextractor catalog:
        #this returns nan if the star was not found or if it has a neighbor detection within 4 pixels (apprx 1 arcsec)
        location_in_catalog = get_index_of_star_in_full_catalog(X_float, Y_float, self.cat, pixeldistance=5.0)

        #NEXT FLAGGING: star is safely un-blended and was found in the sextractor catalog
        if np.isnan(location_in_catalog):
            return None

        #if a good match was found, get imaflags_iso and mag_auto
        IMAFLAG_ISO     = self.cat['imaflags_iso'][location_in_catalog]
        MAG_AUTO        = self.cat['mag_auto'][location_in_catalog]
        FLUX_AUTO       = self.cat['FLUX_AUTO'][location_in_catalog]
        FLUXERR_AUTO    = self.cat['FLUXERR_AUTO'][location_in_catalog]
        FLUX_APER_8     = self.cat['FLUX_APER'][location_in_catalog][7]
        FLUXERR_APER_8  = self.cat['FLUXERR_APER'][location_in_catalog][7]

        #now continue: re-bound the image around the right location
        newbounds = galsim.BoundsI( int(X - self.stampsize/2 +0.5),
                                    int(X + self.stampsize/2 +0.5),
                                    int(Y - self.stampsize/2 +0.5),
                                    int(Y + self.stampsize/2 +0.5))

        hsm_input_im = self.image[newbounds] - self.bkg[newbounds]
        hsm_input_wt = self.weight[newbounds].copy()
        hsm_input_wt.array[hsm_input_wt.array<0.0] = 0.0
        hsm_input_wt.wcs = hsm_input_im.wcs

        #position where we want the PSF
        psf_pos   = galsim.PositionD(X, Y)
        psf_model = self.des_psfex.getPSF(psf_pos)
        pixarea   = hsm_input_im.wcs.pixelArea(image_pos=psf_pos)

        copy_stamp = self.image[newbounds].copy() #copies the galsim object with wcs and everything
        psf_image  = psf_model.drawImage(image=copy_stamp,method='no_pixel')

        psf_cutout = psf_image.array #- np.mean(psf_image.array)
        
        #get the wcs at the location
        psf_wcs = self.des_psfex.getLocalWCS(psf_pos)

        ra,dec = self.wcs_pixel_world.pixel_to_world_values(X_float, Y_float)
        ra     = ra.item()
        dec    = dec.item()

        #give the same weights found in the image to the model! (suggested by Mike J.)
        g1_star_hsm, g2_star_hsm, T_star_hsm    = measure_hsm_shear(hsm_input_im, hsm_input_wt, pixarea)
        g1_model_hsm, g2_model_hsm, T_model_hsm = measure_hsm_shear(psf_image, hsm_input_wt, pixarea)

        #if any of these things are NaNs, count it as a failed star
        outputs = np.array([g1_star_hsm, g2_star_hsm, T_star_hsm, g1_model_hsm, g2_model_hsm, T_model_hsm])
        if np.any(np.isnan(outputs)):
            return None

        out = [X_float + self.min_x_pix, Y_float + self.min_y_pix, X_float, Y_float, ra, dec,
               self.expnum, self.ccdnum, self.band, self.magzp,
               g1_star_hsm, g2_star_hsm, T_star_hsm, g1_model_hsm, g2_model_hsm, T_model_hsm,
               MAG_AUTO, IMAFLAG_ISO, FLUX_AUTO, FLUXERR_AUTO, FLUX_APER_8, FLUXERR_APER_8]

        return out

    def go(self):

        self.load_images(self.path_to_files)

        time1  = time()
        Output = []

        N_failed_stars = 0
        N_failed_CCDS  = 0 #number of CCDS for this expnum that did not pass flags
        N_bad_match    = 0 #whether the star was not found in the sextractor catalog or if it has a close neighbor

        goodstar, Ngoodstar, Nall = get_psf_stars_index(self.starlist)
#         print('We have %d PSF stars'%(Ngoodstar))

#         print('--------------------------------------')
#         print('First: A single run to test code')
#         print('--------------------------------------')
#         print(self.single_run(goodstar[0]))
#         print('--------------------------------------')

        jobs = [joblib.delayed(self.single_run)(i) for i in goodstar]

        with joblib.parallel_backend("loky"):
            outputs = joblib.Parallel(n_jobs = self.n_jobs, verbose=0)(jobs)

        for result in outputs:
            if result is not None: Output.append(result)

        Output = np.array(Output)


#         time2 = time()
#         time_it_took = (time2-time1)/60.0
#         print('FINISHED (took %1.2f minutes). Remember to get some rest.'%(time_it_took), flush = True)

        arrays = Output.T #Invert 2D array so format is [[col1], [col2], ...] instead of [[row1], [row2]]

        names  = ['focal_x', 'focal_y', 'pix_x','pix_y','ra','dec',
                  'EXPNUM','CCDNUM', 'BAND', 'MAGZP',
                  'g1_star_hsm','g2_star_hsm','T_star_hsm','g1_model_hsm','g2_model_hsm','T_model_hsm',
                  'MAG_AUTO', 'IMAFLAGS_ISO','FLUX_AUTO','FLUXERR_AUTO','FLUX_APER_8','FLUXERR_APER_8']

        Header = {'FAILED_stars'    : (N_failed_stars, 'number of failed ngmix measurements'),
                  'FAILED_badmatch' : (N_bad_match,    'number of stars not matched to srcextractor cat')
                  }

        
        with warnings.catch_warnings():
            
            warnings.simplefilter('ignore', category=VerifyWarning)
            t = self.make_fits(arrays, names, Header)

            if self.write_table:
                filename = os.path.basename(glob.glob(os.path.join(self.path_to_files,  self.image_stem) + '*immasked*')[0])
                filename = '_'.join(filename.split('_')[:3]) + '.fits'
                t.writeto(os.path.join(os.environ['ROWE_STATS_DIR'], filename), overwrite=True)

        self.cleanup(self.path_to_files)
        
        return t

    def make_fits(self, arrays, names, header_tags):

        columns = []
        for a, n in zip(arrays, names):
            
            if n != 'BAND':
                f = 'e'
            else:
                f = 'a5'
                
            columns.append(fits.Column(name=n, array=a, format = f))

        t = fits.BinTableHDU.from_columns(columns)

        for i in header_tags:
            t.header[i] = header_tags[i]

        return t


    def cleanup(self, path_to_files):
        
        #Could just glob all files related to image, but let's be specific here about which files just to be safe
        image_path    = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*immasked*')[0]
        bkg_path      = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*bkg*')[0]
        cat_path      = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*red-fullcat*')[0]
        psf_path      = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*psfexcat*')[0]
        starlist_path = glob.glob(os.path.join(path_to_files, self.image_stem) +  '*psfex-starlist*')[0]

        for i in [image_path, bkg_path, cat_path, psf_path, starlist_path]:

            os.remove(i)

#         shutil.rmtree(path_to_files)
#         print("FINISHED REMOVING FILES IN", path_to_files)



def match_star_to_cat(starlist, cat, pixeldistance = 1.0):
    good_index = get_psf_stars_index(self.starlist)
    goodstars  = starlist[good_index]

    #now get the X and Y CCD coordinates of each of the "good" stars
    X ,Y       = goodstars['x_image'],    goodstars['y_image']
    Xcat, Ycat = cat['x_image'],  cat['y_image']

    matched_star_indices = []
    match_fail = 0

    for x,y in zip(X,Y):

        #will match stars in starlist by their X,Y position

        wherex  = np.isclose(x, Xcat, atol = pixeldistance)
        wherey  = np.isclose(y, Ycat, atol = pixeldistance)
        product = wherex*wherey

        if np.sum(product) != 1:
            match_fail += 1
            continue
        else:
            matched_star_indices.append(np.where(product)[0])

    print('Failed to find a match in the sextractor fullcat for %1.2f percent of the starlist stars'%(100*len(match_fail)/len(good_index)),flush=True)

    return matched_star_indices

def get_index_of_star_in_full_catalog(Xstar, Ystar, cat, pixeldistance=4.0):

    #pixel distance is 4 because we don't want any overlap between detected objects (and PSF has size approx 4 pixels on average)
    #get the X and Y of all objects in catalog and compare with the input
    Xcat, Ycat = cat['x_image'], cat['y_image']

    wherex  = np.isclose(Xstar, Xcat, atol = pixeldistance)
    wherey  = np.isclose(Ystar, Ycat, atol = pixeldistance)
    product = wherex*wherey

    if np.sum(product) != 1:
        return np.nan

    else:
        indout = np.where(product)[0] #this is the index IN THE FULL CATALOG of the star with coords Xstar, Ystar

        if np.isscalar(indout):
            return indout
        else:
            return indout[0] #Dhayaa: Why select only first one?


def check_if_star_should_have_been_masked(starlist, cat):

    #get the starlist entry, match it to sextractor catalog imaflags_iso and see if it has a mask
    matched_star_indices = match_star_to_cat(starlist, cat)
    matched_imaflags_iso = cat['imaflags_iso'][matched_star_indices]
    print('Flags found for matched stars:',np.unique(matched_imaflags_iso),flush=True)

    return 0


def get_flux_auto_from_cat(starlist, cat):

    matched_star_indices = match_star_to_cat(starlist,cat)
    matched_flux_auto    = cat['flux_auto'][matched_star_indices]

    return matched_flux_auto


def get_psf_stars_index(starlist):

    goodstars = np.where(starlist['flags_psf']==0)[0]
    Ngood     = len(goodstars)
    Nall      = len(starlist['flags_psf'])

    return goodstars, Ngood, Nall


def make_ngmix_prior(T, pixel_scale):
    #from https://github.com/rmjarvis/DESWL/blob/9e73af6c73fcce2a9017d71b01e81bc5aea1eec8/psf/run_piff.py#L732
    # centroid is 1 pixel gaussian in each direction
    cen_prior = priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale,rng)

    # g is Bernstein & Armstrong prior with sigma = 0.1
    gprior    = priors.GPriorBA(0.1,rng)

    # T is log normal with width 0.2
    Tprior    = priors.LogNormal(T, 0.2,rng)

    # flux is the only uninformative prior
    Fprior    = priors.FlatPrior(-10.0, 1.e10,rng)

    prior     = joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)

    return prior

def measure_shear_of_ngmix_obs(obs, fwhm, rng, want_ngmix = False):

    am = Admom(rng = rng)
    T_guess = (fwhm / 2.35482)**2 * 2.
    prior = make_ngmix_prior(T_guess, 0.26)

    #run adaptive moments
    res = am.go(obs, T_guess)

    if res['flags'] != 0:#adaptive moments failed, let's return all nans
        return np.nan, np.nan, np.nan
    else: #adaptive moments succeded, let's either return the values or run LM
        if want_ngmix:
            g1,g2,T = measure_ngmix_lm(obs,res['pars'],prior)
            return g1,g2,T
        else:
            g1,g2 = e1e2_to_g1g2(res['e1'],res['e2']) #transforms admom e1, e2 into reduced shears g1,g2
            if abs(g1) > 0.5 or abs(g2) > 0.5: #bad measurement
                return np.nan, np.nan, np.nan
            else:
                return g1,g2,res['T']


def measure_ngmix_lm(obs,pars,prior):

    lm = LMSimple(model='gauss',prior=prior)

    try:
            lm_res = lm.go(obs) #will not use initial guesses from admom
            if lm_res['flags'] == 0:
                    g1 = lm_res['pars'][2]
                    g2 = lm_res['pars'][3]
                    T = lm_res['pars'][4]
                    return g1, g2, T
            else:
                    return np.nan, np.nan, np.nan
    except:
            return np.nan, np.nan, np.nan


def e1e2_to_g1g2(e1, e2): #from ngmix base code
    """
    convert e1,e2 to reduced shear style ellipticity
    parameters
    ----------
    e1,e2: tuple of scalars
        shapes in (ixx-iyy)/(ixx+iyy) style space
    outputs
    -------
    g1,g2: scalars
        Reduced shear space shapes
    """
    ONE_MINUS_EPS=0.99999999999999999
    e = np.sqrt(e1 * e1 + e2 * e2)
    if isinstance(e1, np.ndarray):
        (w,) = np.where(e >= 1.0)
        if w.size != 0:
            raise Error("some e were out of bounds")
        eta = np.arctanh(e)
        g = np.tanh(0.5 * eta)
        np.clip(g, 0.0, ONE_MINUS_EPS, g)
        g1 = np.zeros(g.size)
        g2 = np.zeros(g.size)
        (w,) = np.where(e != 0.0)
        if w.size > 0:
            fac = g[w] / e[w]
            g1[w] = fac * e1[w]
            g2[w] = fac * e2[w]
    else:
        if e >= 1.0:
            raise Error("e out of bounds: %s" % e)
        if e == 0.0:
            g1, g2 = 0.0, 0.0
        else:
            eta = np.arctanh(e)
            g = np.tanh(0.5 * eta)
            if g >= 1.0:
                # round off?
                g = ONE_MINUS_EPS
            fac = g / e
            g1 = fac * e1
            g2 = fac * e2
    return g1, g2

def measure_hsm_shear(im, wt, pixarea):

    MAX_CENTROID_SHIFT = 1.5 #1.5 pixel recentering at most
    shape_data         = im.FindAdaptiveMom(weight=wt, strict=False) #might need to turn on strict

    if shape_data.moments_status == 0:
        dx = shape_data.moments_centroid.x - im.true_center.x
        dy = shape_data.moments_centroid.y - im.true_center.y
        pixel_scale_in_arcsec = np.sqrt(pixarea)
        if dx**2 + dy**2 > MAX_CENTROID_SHIFT**2:
            #print('cetroid changed by too much in HSM, will return nan',flush=True)
            return np.nan, np.nan, np.nan
        else:
            
            e1 = shape_data.observed_shape.e1
            e2 = shape_data.observed_shape.e2
            s = shape_data.moments_sigma
            
            if galsim.__version__ >= '1.5.1':
                jac = im.wcs.jacobian(im.true_center)
            else:
                jac = im.wcs.jacobian(im.trueCenter())
                
            M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
            J = jac.getMatrix()
            M = J * M * J.T

            e1 = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
            e2 = (2.*M[0,1]) / (M[0,0] + M[1,1])
            T = M[0,0] + M[1,1]

            shear = galsim.Shear(e1=e1, e2=e2)
            g1 = shear.g1
            g2 = shear.g2
            return g1, g2, T
    else:
        return np.nan, np.nan, np.nan

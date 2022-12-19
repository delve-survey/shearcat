#! /bin/csh -f 

#D00992516_r_c20_r5334p01

#conda activate shearDM

# rerun_psfex_2.sh D00522033 36 20

setenv tile $1
setenv ccd $2
setenv chip $1'_r_c'$2'_r5334p01'
#D00522033_r_c36_r3793p01
#D00992516_r_c20_r5334p01
setenv SN $3

#goto continue


rowinterp_nullweight  -i ${chip}_immasked.fits.fz  -o ${chip}_nullweight-immask.fits  \
    --interp_mask TRAIL,CRAY,BPM  --invalid_mask EDGE  --max_cols 50  -v  --null_mask BADAMP,EDGEBLEED,EDGE,CRAY

#/project2/chihway/chihway/sextractor-2.25.0/src/sex 
/project2/chihway/dhayaa/DECADE/Y6DESDM/sextractor-2.24.4/src/sex ${chip}'_nullweight-immask.fits[0]' \
    -FLAG_IMAGE ${chip}'_nullweight-immask.fits[1]' \
    -c config/20180525_default.sex  \
    -CATALOG_NAME cat/${chip}_psfexcat.fits \
    -CATALOG_TYPE FITS_LDAC  -WEIGHT_TYPE MAP_WEIGHT \
    -WEIGHT_IMAGE ${chip}'_nullweight-immask.fits[2]' \
    -PARAMETERS_NAME config/20180525_sex.param_psfex \
    -FILTER_NAME config/20180525_sex.conv \
    -STARNNW_NAME config/20180525_sex.nnw \
    -SATUR_LEVEL 196471.941971  -DETECT_MINAREA 3  -DETECT_THRESH 3.0  -WEIGHT_THRESH 1e-9 -VERBOSE_TYPE NORMAL


/project2/chihway/chihway/psfex-3.21.1/src/psfex cat/${chip}_psfexcat.fits \
    -c config/20180525_default.psfex \
    -SAMPLE_MINSN ${SN} \
    -WRITE_XML Y  -XML_NAME qa/${chip}_psfex.xml \
    -PSF_DIR psf  -OUTCAT_NAME psf/${chip}_psfex-starlist.fits \
    -OUTCAT_TYPE FITS_LDAC  -CHECKPLOT_NAME qa/psfex_selfwhm,qa/psfex_fwhm,qa/psfex_resids 


mv psf/${chip}_psfex-starlist.fits psf/${chip}_psfex-starlist_SN${SN}.fits
mv psf/${chip}_psfexcat.psf psf/${chip}_psfexcat_SN${SN}.psf 




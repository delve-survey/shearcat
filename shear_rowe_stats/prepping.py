import pandas as pd
import os
from time import time
import sys
from time import time

from tqdm import tqdm

def download(web_paths, output_path, exp_filenames, flist_path = './tmp_filelist.txt'):
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    flist_file = flist_path
    
    print("WRITING FILE LIST")
    t1 = time()
    with open(flist_file, 'w') as f:
        
        for i in tqdm(range(len(web_paths))):
            PATH_im = web_paths[i] #filepath to immask
            FILE_im = exp_filenames[i] + '.fz'

            PATH_bkg = PATH_im[0:-6] + 'bkg'
            FILE_bkg = exp_filenames[i][0:-13] + 'bkg.fits.fz'

            PATH_cat = PATH_im[0:-10] + 'cat'
            FILE_cat = exp_filenames[i][0:-13] + 'red-fullcat.fits'

            PATH_psf = PATH_im[0:-10] + 'psf'
            FILE_psf = exp_filenames[i][0:-13] + 'psfexcat.psf'
            FILE_starlist = exp_filenames[i][0:-13] + 'psfex-starlist.fits'

            for j in [os.path.join(PATH_im,  FILE_im),
                      os.path.join(PATH_bkg, FILE_bkg),
                      os.path.join(PATH_cat, FILE_cat),
                      os.path.join(PATH_psf, FILE_psf),
                      os.path.join(PATH_psf, FILE_starlist)]:
                

                f.write(j + "\n")

    _DOWNLOAD_CMD_WGET = "wget \
                            --no-verbose \
                            --input-file=%(flist_file)s \
                            --base=${DECADEREMOTE_WGET} \
                            --directory-prefix=%(source_dir)s \
                            --auth-no-challenge \
                            --no-clobber" % {'flist_file' : flist_file, 'source_dir' : output_path}
    
    
    os.system(_DOWNLOAD_CMD_WGET)
    
    os.remove(flist_file)
    
    t2 = time()
    
    duration = (t2 - t1)/60.0
    
    print("FINISHED DOWNLOADING %d CCDS IN %0.2f MIN" % (len(web_paths), duration))
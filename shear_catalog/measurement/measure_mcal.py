import sys
sys.path.append('/home/chihway/measure_shear/')
sys.path.append('/home/chihway/measure_shear/metacal')
from _step import _run_metacal as run_metacal
import fitsio

tile = sys.argv[1]
seed = 100

dir_meds = '/project2/chihway/data/decade/decade.ncsa.illinois.edu/deca_archive/DEC/multiepoch/shear/r5765/'+tile+'/p01/meds/'
filename = [dir_meds+tile+'_r5765p01_r_meds-shear.fits.fz',
            dir_meds+tile+'_r5765p01_i_meds-shear.fits.fz',
            dir_meds+tile+'_r5765p01_z_meds-shear.fits.fz']

# dir_meds+tile+'_r5765p01_g_meds-shear.fits.fz',

output = run_metacal(filename, seed) #seed can be an integer, for instance

fitsio.write('/project2/chihway/data/decade/shearcat/metacal_output_'+tile+'.fits', output, clobber=True)


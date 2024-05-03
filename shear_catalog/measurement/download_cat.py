
import numpy as np
import os
import sys

i = int(sys.argv[1])
meta = sys.argv[2]

metadata = np.genfromtxt(meta, dtype='str', delimiter=",")[1:][i]

tile = metadata[0]
path = 'DEC'+metadata[1][3:-5]


command1 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/cat/'

#command2 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/wavg/'

#command2 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/shear_SOF_DR3/r6015/'+tile+'/p01/sof/'
command2 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/DEC_Taiga/multiepoch/shear_SOF_DR3/r6128/'+tile+'/p01/sof/'

os.system(command1)
os.system(command2)
#os.system(command3)



import numpy as np
import os
import sys

i = int(sys.argv[1])

metadata = np.genfromtxt('/home/chihway/shearcat/Tilelist_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

tile = metadata[0]
path = 'DEC_Taiga'+metadata[1][3:-5]


#metadata = np.genfromtxt('tile_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

#tile = metadata[0][2:-1]
#path = metadata[1][2:-1]

command1 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/cat/'

command2 = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+path+'/wavg/'


os.system(command1)
os.system(command2)



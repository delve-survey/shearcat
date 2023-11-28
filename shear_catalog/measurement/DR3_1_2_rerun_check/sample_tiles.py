import numpy as np

metadata0 = np.genfromtxt('ReprocessTilelist_20231103.csv', dtype='str')[0]
metadata = np.genfromtxt('ReprocessTilelist_20231103.csv', dtype='str', delimiter=",")[1:]
Ntile = len(metadata)

ids = np.arange(Ntile)
np.random.shuffle(ids)
ids_sample = ids[:100]
print(ids_sample)

with open('DR3_1_2_rerun_check.txt', 'w') as f:
        f.write(metadata0+'\n')

for i in range(100):
    string = metadata[ids_sample[i]+1][0]+','+metadata[ids_sample[i]+1][1]+','+metadata[ids_sample[i]+1][2]+','+metadata[ids_sample[i]+1][3]+'\n'
    with open('DR3_1_2_rerun_check.txt', 'a') as f:
        f.write(string)


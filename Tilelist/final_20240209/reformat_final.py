import numpy as np

# combine all into one meta file
# also mark the location of where the metacal files are stored (we might want to move them eventually)
# based on this file, we want to re-query all the gold catalogs

metadata_dr3_1_1 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]
metadata_dr3_1_2 = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_2.txt', dtype='str', delimiter=",")[1:]
metadata_dr3_1_2_rerun = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/07112023/Tilelist_Reprocess_20231207.csv', dtype='str', delimiter=",")[1:]
metadata_dr3_1_rerun = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/07112023/Tilelist_Reprocess_20240124.csv', dtype='str', delimiter=",")[1:]
output = 'Tilelist_final_DR3_1.csv'

with open(output, 'w') as f:
        f.write('TILENAME,PATH,RA_CENT,DEC_CENT,SHEARCAT_PATH,TAG \n')

# DR3_1_1 original
Ntile_dr3_1_1 = len(metadata_dr3_1_1)

print(Ntile_dr3_1_1)
for i in range(Ntile_dr3_1_1):
    if not np.in1d(metadata_dr3_1_1[i][0], metadata_dr3_1_rerun[:,0]):
        string = metadata_dr3_1_1[i][0]+','+metadata_dr3_1_1[i][1]+','+metadata_dr3_1_1[i][2]+','+metadata_dr3_1_1[i][3]+',2,DR3_1_1_original'+'\n'
        with open(output, 'a') as f:
            f.write(string)

# DR3_1_2 original
Ntile_dr3_1_2 = len(metadata_dr3_1_2)

print(Ntile_dr3_1_2)
for i in range(Ntile_dr3_1_2):
    if not (np.in1d(metadata_dr3_1_2[i][0], metadata_dr3_1_rerun[:,0]) or np.in1d(metadata_dr3_1_2[i][0], metadata_dr3_1_2_rerun[:,0])):
        string = metadata_dr3_1_2[i][0]+','+metadata_dr3_1_2[i][1]+','+metadata_dr3_1_2[i][2]+','+metadata_dr3_1_2[i][3]+',3,DR3_1_2_original'+'\n'
        with open(output, 'a') as f:
            f.write(string)

# DR3_1_2 rerun
Ntile_dr3_1_2_rerun = len(metadata_dr3_1_2_rerun)

print(Ntile_dr3_1_2_rerun)
for i in range(Ntile_dr3_1_2_rerun):
    if not np.in1d(metadata_dr3_1_2_rerun[i][0], metadata_dr3_1_rerun[:,0]):
        string = metadata_dr3_1_2_rerun[i][0]+','+metadata_dr3_1_2_rerun[i][1]+','+metadata_dr3_1_2_rerun[i][2]+','+metadata_dr3_1_2_rerun[i][3]+',5,DR3_1_2_rerun'+'\n'
        with open(output, 'a') as f:
            f.write(string)

# DR3_1 rerun
Ntile_dr3_1_rerun = len(metadata_dr3_1_rerun)

print(Ntile_dr3_1_rerun)
for i in range(Ntile_dr3_1_rerun):
    string = metadata_dr3_1_rerun[i][0]+','+metadata_dr3_1_rerun[i][1]+','+metadata_dr3_1_rerun[i][2]+','+metadata_dr3_1_rerun[i][3]+',6,DR3_1_rerun'+'\n'
    with open(output, 'a') as f:
        f.write(string)




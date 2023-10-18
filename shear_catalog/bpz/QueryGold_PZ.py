import easyaccess
import numpy as np
import fitsio
import os
import pandas as pd

metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]

#metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_2.txt', dtype='str', delimiter=",")[1:]

m0col = 'BDF_MAG_I'
f0col = 'BDF_FLUX_I'

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

cols = ['COADD_OBJECT_ID', 'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z', 'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_MAG_I']

for i in range(10): #len(metadata)):
    tile = metadata[i][0]
    print(i, tile)

    if not os.path.isfile("/project/chihway/data/decade/coaddcat_pz/gold_"+str(tile)+".h5"):
        decade_query = "select a.COADD_OBJECT_ID, a.RA, a.DEC, a.FLUX_AUTO_G, a.FLUX_AUTO_R, a.FLUX_AUTO_I, a.FLUX_AUTO_Z, a.FLUXERR_AUTO_G, a.FLUXERR_AUTO_R, a.FLUXERR_AUTO_I, a.FLUXERR_AUTO_Z, b.BDF_FLUX_G, b.BDF_FLUX_R, b.BDF_FLUX_I, b.BDF_FLUX_Z, b.BDF_FLUX_ERR_G, b.BDF_FLUX_ERR_R, b.BDF_FLUX_ERR_I, b.BDF_FLUX_ERR_Z from DECADE.DR3_1_COADD_OBJECT_SUMMARY a, DR3_1_SOF b where a.COADD_OBJECT_ID=b.COADD_OBJECT_ID and a.TILENAME='"+str(tile)+"';"

        # get the decade qa data as a pandas dataframe
        conn = easyaccess.connect(section='decade')
        decade_df = conn.query_to_pandas(decade_query)
        data = decade_df.to_records()
        print(len(data))

        pos_mask = data[f0col]>0
        data=data[pos_mask]
        dframe_i = pd.DataFrame()

        if len(data)>0:
            for label in cols:
                if label!=m0col: dframe_i[label] = data[label]
                else: dframe_i[m0col] = flux2mag(data[f0col])

            dframe_i.to_hdf("/project/chihway/data/decade/coaddcat_pz/gold_"+str(tile)+".h5", key='df')

            #fitsio.write("/project/chihway/data/decade/coaddcat_pz/gold_"+str(tile)+".fits", data)

    else:
        print("exist")


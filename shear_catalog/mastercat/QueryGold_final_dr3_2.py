import easyaccess
import numpy as np
import fitsio
import os

metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/06042024/dr3_2_final_20241003.csv', dtype='str', delimiter=",")[1:]
print(len(metadata))

for i in range(len(metadata)):
    tile = metadata[i][0]
    print(i, tile)

    if not os.path.isfile("/project/chihway/data/decade/coaddcat_final_dr3_2/gold_"+str(tile)+".fits"):
        decade_query = "select a.COADD_OBJECT_ID, a.RA, a.DEC, a.FLUX_AUTO_G, a.FLUX_AUTO_R, a.FLUX_AUTO_I, a.FLUX_AUTO_Z, a.FLUXERR_AUTO_G, a.FLUXERR_AUTO_R, a.FLUXERR_AUTO_I, a.FLUXERR_AUTO_Z, a.FLUX_RADIUS_G, a.FLUX_RADIUS_R, a.FLUX_RADIUS_I, a.FLUX_RADIUS_Z, b.BDF_T, b.BDF_S2N, b.BDF_FLUX_G, b.BDF_FLUX_R, b.BDF_FLUX_I, b.BDF_FLUX_Z, b.BDF_FLUX_ERR_G, b.BDF_FLUX_ERR_R, b.BDF_FLUX_ERR_I, b.BDF_FLUX_ERR_Z from DR3_2_COADD_OBJECT_SUMMARY a, DR3_2_SOF b where a.COADD_OBJECT_ID=b.COADD_OBJECT_ID and a.FLAGS_G<=3 and a.FLAGS_R<=3 and a.FLAGS_I<=3 and a.FLAGS_Z<=3 and a.IMAFLAGS_ISO_G=0 and a.IMAFLAGS_ISO_R=0 and a.IMAFLAGS_ISO_I=0 and a.IMAFLAGS_ISO_Z=0 and a.NITER_MODEL_G>0 and a.NITER_MODEL_R>0 and a.NITER_MODEL_I>0 and a.NITER_MODEL_Z>0 and a.TILENAME='"+str(tile)+"';"

        # get the decade qa data as a pandas dataframe
        conn = easyaccess.connect(section='decade')
        decade_df = conn.query_to_pandas(decade_query)
        data = decade_df.to_records()
        print(len(data))
        if len(data)>0:
            fitsio.write("/project/chihway/data/decade/coaddcat_final_dr3_2/gold_"+str(tile)+".fits", data)
        else:
            with open('zero_object_tiles.txt', 'a') as f:
                f.write(str(tile)+'\n')

    else:
        print("exist")


import os
import jinja2
import yaml
import pandas as pd, numpy as np

import sys
sys.path.append(os.environ['ROWE_STATS_RUN_DIR'])
from prepping import download
import time
import datetime as dt
import io
import subprocess as sp
import argparse

def current_job_count():

    x = sp.check_output("squeue --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' --sort=+i -u dhayaa", shell = True)
    j = pd.read_csv(io.StringIO(x.decode("utf-8")), delim_whitespace=True)

    count = 0
    batch = []

    #metacal_DES0849+0252_seed0_gminus
    for i in range(len(j)):

        condition1 = 'rowe_stats' in j['NAME'].values[i]
        condition2 = j['STATE'].values[i] == 'RUNNING'

        if condition1:
            count += 1
            batch.append(j['NAME'].values[i])

    for t in batch:
        print("CURRENTLY RUNNING: %s" % t)

    return count

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--rank',     action='store', required = True, type = int)
    my_parser.add_argument('--Nperrank', action='store', required = True, type = int)
    my_parser.add_argument('--Nperjob',  action='store', required = True, type = int)

    args = vars(my_parser.parse_args())

    #Print args for debugging state
    print('-------INPUT PARAMS----------')
    for p in args.keys():
        print('%s : %s'%(p.upper(), args[p]))
    print('-----------------------------')
    print('-----------------------------')

    #Automatically get folder name (assuming certain folder structure)
    name = os.path.basename(os.path.dirname(__file__))

    #Create output directory for rowe_stats
    ROWE_STATS_DIR = os.environ['ROWE_STATS_DIR']
    os.makedirs(ROWE_STATS_DIR, exist_ok=True)

    #Now create all job.sh files for running sims
    with open('job.sh.temp', 'r') as fp:
        tmp = jinja2.Template(fp.read())

    # imgname = pd.concat([pd.read_csv('/home/dhayaa/Desktop/DECADE/DR3_1_1_exp_paths.csv'),
    #                      pd.read_csv('/home/dhayaa/Desktop/DECADE/DR3_1_2_exp_paths.csv')])

    #Run just DR3_1_1 for now since we only have shear for that region
    explist = '/home/dhayaa/Desktop/DECADE/DR3_1_1_explist.csv'
    imgname = pd.read_csv(explist)

    imgname = imgname.drop_duplicates().sort_values('EXPNUM')

    N_exp = args['Nperrank']
    N_per_job = args['Nperjob']

    n_jobs = int(np.ceil(N_exp/N_per_job))

    for n in range(n_jobs):

        start = N_per_job*n       + N_exp*args['rank']
        end   = N_per_job*(n + 1) + N_exp*args['rank']

        if end > len(imgname): end = len(imgname)

        path = os.path.join(os.environ['ROWE_STATS_DIR'], 'output_%d_to_%d.fits' % (start, end))
        if os.path.isfile(path):
            print("CATALOG %s EXISTS. SO SKIPPING" % path)
            continue

        start_time = dt.datetime.now()

        while (dt.datetime.now() - start_time).seconds < np.inf: #FORCE IT INTO LOOP

            if current_job_count() >= 4:

                print("---------------------------")
                print(dt.datetime.now())
                print("---------------------------")
                time.sleep(60*3)

            else:

                flist = os.environ['EXP_DIR'] + '/tmp_filelist_rank%d.txt'%args['rank']
                job   = './job_batch%d_rank%d.txt'%(n,args['rank'])
                download(imgname.PATH.values[start:end], os.environ['EXP_DIR'], imgname.FILENAME.values[start:end], flist)

                with open(job, 'w') as fp:
                    fp.write(tmp.render(start = start, end = end, path_to_explist = explist, seed = 42))

                os.system('chmod u+x %s' % job)
                os.system('sbatch %s' % job)

                #Remove file once you submit to remove clutter
                os.remove(job)

                #Now break out of the loop
                break

        else:

            print("SUBMITTED JOB", job, "SO ENDING LOOP")

#!/bin/bash

Nrank=$1
Nperrank=$2
Nperjob=$3

for((i = 0; i < Nrank; i = i + 1))
do
    nohup python ${ROWE_STATS_RUN_DIR}/runs/render_jobs.py --rank $i --Nperrank $Nperrank --Nperjob $Nperjob >> ${ROWE_STATS_RUN_DIR}/runs/downloading_${i}.log 2>&1 &
    echo $! >> ${ROWE_STATS_RUN_DIR}/runs/save_pid.txt
done

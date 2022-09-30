#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH 

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4

srun -p ${PARTITION} --job-name=${JOB_NAME} --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=5 \ 
    python -u tools/infer.py ${CONFIG} ${CHECKPOINT} ./data/ACCV_workshop/test --out-keys filename pred_class --out result.csv
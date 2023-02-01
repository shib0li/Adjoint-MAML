#!/bin/bash

domain=$1
k_shot=$2
k_query=$3
inner_steps=$4
device=$5

batchsize=1000
test_batchsize=100

meta_batch=5
meta_epochs=50
test_interval=500

max_meta_updates=5000

inner_lr=0.01
meta_lr=0.01

test_max_epochs=5000
verbose=False

python run.py -domain=$domain -k_shot=$k_shot -k_query=$k_query -batchsize=$batchsize \
        -inner_n_steps=$inner_steps -inner_stepsize=$inner_lr \
        -meta_reg=1.0 -meta_lr=0.01 \
        -meta_batch=$meta_batch -meta_shuffle_batch=True -meta_epochs=$meta_epochs \
        -test_batchsize=$test_batchsize -test_max_epochs=$test_max_epochs \
        -test_interval=100 \
        -base_module='fcnn' \
        -max_meta_updates=$max_meta_updates \
        -verbose=$verbose \
        -device="$device" 

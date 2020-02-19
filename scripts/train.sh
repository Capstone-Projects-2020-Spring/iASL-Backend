#! /bin/bash

. ../_runtime_env.sh

PARAMS_DIR="$iASL_PARAMS/params_train.txt"
ODIR_DIR="$iASL_OUT/p1_train/"
SCRIPT_DIR="$iASL_SCRIPTS/train.py"

python $SCRIPT_DIR $PARAMS_DIR $ODIR_DIR

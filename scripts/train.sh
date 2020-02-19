#! /bin/bash

iASL_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $iASL_BASE/../_runtime_env.sh
export iASL_BASE;

PARAMS_DIR="$iASL_PARAMS/params_train.txt"
ODIR_DIR="$iASL_OUT/p1_train/"
SCRIPT_DIR="$iASL_SCRIPTS/train.py"

python $SCRIPT_DIR $PARAMS_DIR $ODIR_DIR

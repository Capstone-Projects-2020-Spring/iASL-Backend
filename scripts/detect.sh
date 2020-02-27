#! /bin/bash

. ../_runtime_env.sh

PARAMS_DIR="$iASL_PARAMS/params_detect.txt"
SCRIPT_DIR="$iASL_SCRIPTS/realtime-detection.py"

python $SCRIPT_DIR $PARAMS_DIR

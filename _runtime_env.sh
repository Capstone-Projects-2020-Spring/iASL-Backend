#!/bin/sh
# file: $iASL_BASE/_runtime_env.sh
#
# revision history:
#  20200219 (TE): initial version
#
# usage:
#  . ./_runtime_env.sh
#
# This script sets up some important environment variables.
#

# define locations for things in this directory
#
iASL_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
iASL_SCRIPTS="$iASL_BASE/scripts";
iASL_OUT="$iASL_BASE/output";
iASL_DATA="$iASL_BASE/data";

# define locations for lists and parameter files
#
iASL_LISTS="$iASL_BASE/lists";
iASL_PARAMS="$iASL_BASE/params";

# export environment variables
#
export iASL_BASE iASL_SCRIPTS;
export iASL_OUT iASL_DATA;
export iASL_LISTS iASL_PARAMS;
#                                                                                                                                                                                                  
# end of file

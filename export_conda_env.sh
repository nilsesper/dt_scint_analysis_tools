#!/bin/bash

### overwrites the env.yml file 

# assume one is already in the conda env and has made changes to it
echo "======================================================================================"
echo "=== DT_SCINT_AT: export_conda_env.sh ================================================="
echo "=== This script will export your current environment into the conda_env.yaml file. ==="
echo "======================================================================================"

cd $(dirname $0) # go to path of this .sh file
# one is in the repo base dir now

source env.sh
conda env export > conda_env.yml


#!/bin/bash

### overwrites the conda env based on env.yaml

echo "==========================================================================================================="
echo "=== DT_SCINT_AT: update_conda_env.sh ======================================================================"
echo "=== This script will update the conda enviroment. Please remain patient and wait until it is completed. ==="
echo "=== You should not type any key, the script runs completely automatic.                                  ==="
echo "==========================================================================================================="

cd $(dirname $0) # go to path of this .sh file
# one is in the repo base dir now
eval "$($HOME/miniconda/bin/conda shell.bash hook)" # conda initialize
conda activate base # go to base env
yes | conda env remove -n dt_scint_at # remove old env
conda env create -f conda_env.yml # install new env

export DT_SCINT_AT=0 # reset env to resource the updated env


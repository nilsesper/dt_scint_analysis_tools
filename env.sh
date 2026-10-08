#!/bin/bash

echo "Checking if environment was already sourced."
if [[ "$DT_SCINT_AT" == 1 ]]
then
    ## DT_SCINT_AT env variable already set -> already sourced, exit
    echo "Environment already sourced. Exiting."
else
    ## DT_SCINT_AT env variable not set -> need to source env
    echo "Environment not yet sourced. Continuing."
    
    echo "Activating conda environment."
    ## comment since we assume now that conda base env is already sourced
    # # start anaconda session
    # eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    # activate conda environment (old)
    #eval "$(conda shell.bash hook)"
    # activate conda environment (new)
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda activate dt_scint_at
    
    echo "Locating the directory."
    # find script abs path when not executing but sourcing the script
    # (taken from https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself)
    SCRIPT_PATH="${BASH_SOURCE[0]}";
    while([ -h "${SCRIPT_PATH}" ]); do
        cd "`dirname "${SCRIPT_PATH}"`"
        SCRIPT_PATH="$(readlink "`basename "${SCRIPT_PATH}"`")";
    done
    cd "`dirname "${SCRIPT_PATH}"`" > /dev/null
    SCRIPT_PATH="`pwd`";
    REPO_PATH=$SCRIPT_PATH
    echo "  REPO_PATH = ${REPO_PATH}"
    
    ## add repo library + dependencies to pythonpath
    echo "Adding variables to PYTHONPATH."
    # add repo directory (minicrate-testing-software) to pythonpath
    export PYTHONPATH="${PYTHONPATH}:${REPO_PATH}"
    echo "  PYTHONPATH += $REPO_PATH"
    
    ## export DT_SCINT_AT bash variable as an indicator that the environment was already sourced
    echo "Exporting DT_SCINT_AT=1 environment variable to indicate sourced environment."
    export DT_SCINT_AT=1
fi


### WORKING ENVIRONMENT FOR ANALYSIS_UTILS

echo "*** ANALYSIS_UTILS environment ***"

# source python env (specific to each computer)
echo "Sourcing base python environment."
source ~/utils/python_env.sh

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

echo "Adding variables to PYTHONPATH."
# add repo directory to python pyth
export PYTHONPATH="${PYTHONPATH}:${REPO_PATH}"
echo "  PYTHONPATH += $REPO_PATH"



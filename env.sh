### WORKING ENVIRONMENT

echo "*** DT_SCINT_ANALYSIS_TOOLS environment ***"

# # source python env (specific to each computer)
# echo "Sourcing base python environment."
# source ~/utils/python_env.sh

# source /home/jstac/pythonBaenv/bin/activate

#eval "$(micromamba shell hook --shell bash)"

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/.automount/home/home__home1/institut_3a/tacke/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/.automount/home/home__home1/institut_3a/tacke/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba activate bajustus

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
export REPO_PATH=${REPO_PATH}
echo "  REPO_PATH = ${REPO_PATH}"

echo "Adding variables to PYTHONPATH."
# add repo directory to python pyth
export PYTHONPATH="${PYTHONPATH}:${REPO_PATH}"
echo "  PYTHONPATH += $REPO_PATH"



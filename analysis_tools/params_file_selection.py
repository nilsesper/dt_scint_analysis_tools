###########################################
### SELECTION OF THE PARAMETER FILE
###########################################
# By default all code uses analysis_tools/params/params.py.
# A different parameter file (e.g. another readout mapping) can be selected without editing any code:
#   - command line:  --params_file /path/to/other_params.py     (every script in scripts/)
#   - environment:   export DT_SCINT_PARAMS_FILE=/path/to/other_params.py   (works for every script)
# The selected file is loaded in place of analysis_tools.params.params, so that every
# "from analysis_tools.params import params" in the code base gets it, and derived_params is derived from it.
# This has to happen before anything imports the parameters, therefore it is called from analysis_tools/__init__.py.

import importlib.util
import os
import sys

ENV_VAR = "DT_SCINT_PARAMS_FILE"
CLI_FLAG = "--params_file"
PARAMS_MODULE_NAME = "analysis_tools.params.params"

### value of --params_file in the command line arguments, or None
def params_file_from_arguments(argv):
    for i in range(len(argv)):
        if argv[i] == CLI_FLAG and i + 1 < len(argv):
            return argv[i + 1]
        if argv[i].startswith(CLI_FLAG + "="):
            return argv[i].split("=", 1)[1]
    return None

### path of the parameter file in use (None = default analysis_tools/params/params.py)
def selected_file():
    return os.environ.get(ENV_VAR) or None

### load the selected parameter file in place of analysis_tools/params/params.py (nothing to do if none is selected)
def apply():
    path = params_file_from_arguments(sys.argv[1:])
    if path is None:
        path = os.environ.get(ENV_VAR)
    if path is None or path == "":
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    loaded = sys.modules.get(PARAMS_MODULE_NAME)
    if loaded is not None:
        if os.path.abspath(getattr(loaded, "__file__", "")) == path:
            return path
        raise RuntimeError(f"Parameters were already loaded from {getattr(loaded, '__file__', '?')}, cannot switch to {path}.")
    os.environ[ENV_VAR] = path  # so that processes started from here use the same file
    import analysis_tools.params as params_package
    spec = importlib.util.spec_from_file_location(PARAMS_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PARAMS_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[PARAMS_MODULE_NAME]
        raise
    params_package.params = module
    print(f"Using parameter file: {path}", flush=True)
    return path

###########################################
### SELECTION OF THE PARAMETER FILE
###########################################
# By default all code uses analysis_tools/params/params.py.
# A different parameter file (e.g. another readout mapping) can be selected without editing any code:
#   - command line:  --params_file /path/to/other_params.py     (scripts in scripts/dt_root/)
#   - environment:   export DT_SCINT_PARAMS_FILE=/path/to/other_params.py   (works for every script)
# The selected file is loaded in place of analysis_tools.params.params, so that every
# "from analysis_tools.params import params" in the code base gets it, and derived_params is derived from it.
# This has to happen before anything imports the parameters, therefore it is called from analysis_tools/__init__.py.

import importlib.util
import os
import sys

ENV_VAR = "DT_SCINT_PARAMS_FILE"
CLI_FLAG = "--params_file"
_MODULE_NAME = "analysis_tools.params.params"

def _from_argv(argv):
    for i, arg in enumerate(argv):
        if arg == CLI_FLAG and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(CLI_FLAG + "="):
            return arg.split("=", 1)[1]
    return None

### path of the parameter file in use (None = default analysis_tools/params/params.py)
def selected_file():
    return os.environ.get(ENV_VAR) or None

def apply(argv=None):
    path = _from_argv(sys.argv[1:] if argv is None else argv) or os.environ.get(ENV_VAR)
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    loaded = sys.modules.get(_MODULE_NAME)
    if loaded is not None:
        if os.path.abspath(getattr(loaded, "__file__", "")) == path:
            return path
        raise RuntimeError(f"Parameters were already loaded from {getattr(loaded, '__file__', '?')}, cannot switch to {path}.")
    os.environ[ENV_VAR] = path  # so that processes started from here use the same file
    import analysis_tools.params as params_package
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_MODULE_NAME]
        raise
    params_package.params = module
    print(f"Using parameter file: {path}", flush=True)
    return path

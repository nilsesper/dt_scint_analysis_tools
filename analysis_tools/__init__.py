### python package repo top file
# select the parameter file first (default: analysis_tools/params/params.py), see params_file_selection.py
from analysis_tools import params_file_selection
params_file_selection.apply()

import analysis_tools.params.params
import analysis_tools.params.derived_params  # noqa: F401 (derived from the selected params)

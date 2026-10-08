### python package repo top file
# select the parameter file first (default: analysis_tools/params/params.py), see _params_select.py
from analysis_tools import _params_select
_params_select.apply()

import analysis_tools.params.params
import analysis_tools.params.derived_params  # noqa: F401 (derived from the selected params)

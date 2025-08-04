###############################
### DERIVED CONSTANTS & PARAMETERS
###############################
# collects "derived parameters", which depend on the content of params.py

import analysis_tools.params.params as params

# -----------------------------------------

### ro_chs lists
# list of ro_chs used for dt chamber
_dt_ro_chs = list(params._dt_mapping.keys())
# list of ro_chs used for scintillator
_scint_ro_chs = list(params._scint_mapping.keys())

### ch lists
# list of allowed channels for dt chamber, by ro_ch
_dt_chs_by_ro_ch = {}
for ro_ch in _dt_ro_chs:
    ch_list = []
    for fe_conn_name, fe_mapping in params._dt_mapping[ro_ch].items():
        ch_list.extend(fe_mapping["chs"])
    _dt_chs_by_ro_ch[ro_ch] = ch_list

### dumpfile keys
_dumpfile_keys = list(params._htg_keys.keys())


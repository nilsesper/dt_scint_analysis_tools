###############################
### DERIVED CONSTANTS & PARAMETERS
###############################
# collects "derived parameters", which depend on the content of params.py

import analysis_tools.params.params as params
import numpy as np

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
# list of allowed channels for scintillator, by ro_ch
_scint_chs_by_ro_ch = {}
for ro_ch in _scint_ro_chs:
    ch_list = []
    for ch_name, ch_mapping in params._scint_mapping[ro_ch].items():
        ch_list.append(ch_mapping["ch"])
    _scint_chs_by_ro_ch[ro_ch] = ch_list

### dumpfile keys
_dumpfile_keys = list(params._htg_keys.keys())

### generate dt remapping table: {ro_ch: {ch: {dt_keys: mapping value}}
_dt_keys = list(params._dt_mapping_keys.keys()) # sl, ly, wi, conn_id, ch_id, fe_id
_fecable2wireoffset = lambda fe: 4*((int(fe[:-1])-1)*2 + (1 if fe[-1]=='B' else 0)) # remapping function: fe connector (XA/B) -> wire offset
_dt_remap_table = {}
for ro_ch in _dt_ro_chs:
    _dt_remap_table[ro_ch] = {}
    for conn_id, (fe_connector_name, fe) in enumerate(params._dt_mapping[ro_ch].items()):
        fe_id = params._fe_idx_list.index(fe["fe"])
        sl = fe["sl"]
        # translate to ly, wi
        wireoffset = _fecable2wireoffset(fe["fe"])
        # fill remap table
        for ch_id in range(16):
            # conductors in the cable come in this order of layers: 4(outer), 2, 3, 1(inner)
            ly = [3,1,2,0][ch_id & 3]
            wi = (ch_id >> 2) + wireoffset
            ch = fe["chs"][ch_id]
            _dt_remap_table[ro_ch][ch] = {
                "conn_id": conn_id, # idx of conn name "J35" in fe_mapping dict
                "fe_id": fe_id, # idx of fe conn name "1A" in order starting at 1A
                "ch_id": ch_id, # ch id wrt connector 0-15, i.e. idx of ch in "chs" list of fe_mapping dict
                "sl": sl, # superlayer 1-3
                "ly": ly, # layer
                "wi": wi, # wire
            }

### generate scint remapping table: {ro_ch: {ch: {scint_keys: mapping value}}
_scint_keys = list(params._scint_mapping_keys.keys()) # ly, st, ch_id
_scint_remap_table = {}
for ro_ch in _scint_ro_chs:
    _scint_remap_table[ro_ch] = {}
    for ch_id, (ch_name, ch_mapping) in enumerate(params._scint_mapping[ro_ch].items()):
        ly = ch_mapping["ly"]
        st = ch_mapping["st"]
        ch = ch_mapping["ch"]
        _scint_remap_table[ro_ch][ch] = {
            "ch_id": ch_id, # idx of coinc channel name in mapping dict
            "ly": ly, # layer id
            "st": st, # strip id
        }

### timestamp conversion
_tdc_to_timestamp = 1
_bx_to_timestamp = params._lhc_tdc_count * _tdc_to_timestamp
_orbit_to_timestamp = params._lhc_bunch_count * _bx_to_timestamp
_orbit_overflow_to_timestamp = np.uint64(params._lhc_orbit_count * _orbit_to_timestamp)
# 1 timestamp unit = 1 TDC = 0.78 ns 



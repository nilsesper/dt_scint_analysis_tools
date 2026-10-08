###############################
### DERIVED PARAMETERS
###############################
# Lookup tables and unit conversions which follow from params.py (the chamber geometry: dt_geometry_utils.py).

import numpy as np

import analysis_tools.params.params as params

# -----------------------------------------
### readout mapping
# -----------------------------------------

_dt_ro_chs = list(params._dt_mapping.keys())

# OBDT channels of the dt chamber per readout channel: {ro_ch: [ch]}
_dt_chs_by_ro_ch = {}
for ro_ch in _dt_ro_chs:
    _dt_chs_by_ro_ch[ro_ch] = []
    for connector in params._dt_mapping[ro_ch].values():
        _dt_chs_by_ro_ch[ro_ch].extend(connector["chs"])

### cell of a channel: {ro_ch: {ch: {"sl", "ly", "wi", "conn_id", "fe_id", "ch_id"}}}
# conductor ch_id of frontend connector fe_id -> wire 4 * fe_id + ch_id // 4, layer params._fe_conductor_layers[ch_id % 4];
# conductors without a wire in the chamber are left out
_dt_remap_table = {}
for ro_ch in _dt_ro_chs:
    _dt_remap_table[ro_ch] = {}
    for conn_id, connector in enumerate(params._dt_mapping[ro_ch].values()):
        if connector["fe"] is None:
            continue
        fe_id = params._fe_idx_list.index(connector["fe"])
        sl = connector["sl"]
        for ch_id, ch in enumerate(connector["chs"]):
            ly = params._fe_conductor_layers[ch_id % 4]
            wi = 4 * fe_id + ch_id // 4
            layer = params._dt_chamber["sls"][sl]["lys"][ly]
            if layer["min_wi"] <= wi <= layer["max_wi"]:
                _dt_remap_table[ro_ch][ch] = {"conn_id": conn_id, "fe_id": fe_id, "ch_id": ch_id, "sl": sl, "ly": ly, "wi": wi}

### channel of a cell: {sl: {ly: {wi: {"ch", "ro_ch", "conn_id", "fe_id", "ch_id"}}}}
_dt_inverted_remap_table = {}
for ro_ch in _dt_ro_chs:
    for ch in _dt_remap_table[ro_ch]:
        cell = _dt_remap_table[ro_ch][ch]
        sl, ly, wi = cell["sl"], cell["ly"], cell["wi"]
        if sl not in _dt_inverted_remap_table:
            _dt_inverted_remap_table[sl] = {}
        if ly not in _dt_inverted_remap_table[sl]:
            _dt_inverted_remap_table[sl][ly] = {}
        _dt_inverted_remap_table[sl][ly][wi] = {"ch": ch, "ro_ch": ro_ch, "conn_id": cell["conn_id"], "fe_id": cell["fe_id"], "ch_id": cell["ch_id"]}

# -----------------------------------------
### units
# -----------------------------------------

# timestamp of one TDC count, BX, orbit and orbit counter range
_tdc_to_timestamp = 1
_bx_to_timestamp = np.uint64(params._lhc_tdc_count * _tdc_to_timestamp)
_orbit_to_timestamp = np.uint64(params._lhc_bunch_count * _bx_to_timestamp)
_orbit_overflow_to_timestamp = np.uint64(params._lhc_orbit_count * _orbit_to_timestamp)

# drift velocity: um/ns -> mm/TU
_drift_velocity_conversion = 0.78 * 1e-3
_drift_velocity_mm_per_timestamp = np.float64(params._drift_velocity * _drift_velocity_conversion)
_drift_velocity_mm_per_timestamp_min = np.float64(params._drift_velocity_min * _drift_velocity_conversion)
_drift_velocity_mm_per_timestamp_max = np.float64(params._drift_velocity_max * _drift_velocity_conversion)

# -----------------------------------------
### plotting
# -----------------------------------------

def color_wheel(i):
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:cyan", "tab:olive"]
    return colors[i % len(colors)]

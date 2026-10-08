###########################################
### DT CHAMBER: superlayers, cells and readout mapping
###########################################

import copy
import collections
import functools
import numpy as np

from analysis_tools.params import params, derived_params

# -----------------------------------------

def superlayers():
    return list(params._dt_chamber["sls"].keys())

def phi_superlayers():
    return [sl for sl in superlayers() if params._dt_chamber["sls"][sl]["orient"] == "phi"]

def theta_superlayer():
    return [sl for sl in superlayers() if params._dt_chamber["sls"][sl]["orient"] == "theta"][0]

def layers(sl):
    return list(params._dt_chamber["sls"][sl]["lys"].keys())

def wires(sl, ly):
    layer = params._dt_chamber["sls"][sl]["lys"][ly]
    return range(layer["min_wi"], layer["max_wi"] + 1)

### all cells of the chamber as (sl, ly, wi)
def chamber_cells():
    return [(sl, ly, wi) for sl in superlayers() for ly in layers(sl) for wi in wires(sl, ly)]

### cells excluded from the analysis: params._dt_wire_mask and params._dt_dead_wires
def excluded_wires(sl, ly):
    return set(params._dt_wire_mask[sl][ly]) | set(params._dt_dead_wires.get(sl, {}).get(ly, []))

### {sl: {ly: {wi: copy of content}}} for every cell of the chamber
def chamber_map(content):
    return {sl: {ly: {wi: copy.deepcopy(content) for wi in wires(sl, ly)} for ly in layers(sl)} for sl in superlayers()}

### chamber map with the default look of a cell in the geometry plots (geoplot_utils)
def cell_display_map():
    return chamber_map({"color": params._color_info["cell"][None], "text": ""})

### readout channel -> cell lookup tables, indexed with [ro_ch, ch] (cell keys) or [sl, ly, wi] (cell masks)
ReadoutTables = collections.namedtuple("ReadoutTables", [
    "is_dt_channel",      # [ro_ch, ch]: the channel belongs to the dt chamber
    "has_cell",           # [ro_ch, ch]: ... and is connected to a cell
    "cell_keys",          # {key of params._dt_mapping_keys: [ro_ch, ch] -> value}
    "is_analysed_cell",   # [sl, ly, wi]: cell of the chamber which is not masked / dead
    "is_chamber_cell",    # [sl, ly, wi]: any cell of the chamber
])

@functools.cache
def readout_tables():
    is_dt_channel = np.zeros((256, 256), dtype=bool)
    has_cell = np.zeros((256, 256), dtype=bool)
    cell_keys = {k: np.zeros((256, 256), dtype=v) for k, v in params._dt_mapping_keys.items()}
    for ro_ch in derived_params._dt_ro_chs:
        is_dt_channel[ro_ch, list(derived_params._dt_chs_by_ro_ch[ro_ch])] = True
        for ch, cell in derived_params._dt_remap_table[ro_ch].items():
            has_cell[ro_ch, ch] = True
            for k in params._dt_mapping_keys.keys():
                cell_keys[k][ro_ch, ch] = cell[k]
    is_chamber_cell = np.zeros((256, 256, 256), dtype=bool)
    for sl, ly, wi in chamber_cells():
        is_chamber_cell[sl, ly, wi] = True
    is_analysed_cell = is_chamber_cell.copy()
    for sl in superlayers():
        for ly in layers(sl):
            for wi in excluded_wires(sl, ly):
                is_analysed_cell[sl, ly, wi] = False
    return ReadoutTables(is_dt_channel, has_cell, cell_keys, is_analysed_cell, is_chamber_cell)

### readout keys (ro_ch, ch, fe_id, conn_id, ch_id) of a cell
def readout_keys_of_cell(sl, ly, wi):
    return {k: derived_params._dt_inverted_remap_table[sl][ly][wi][k] for k in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]}

### frontend connector name (e.g. "5A") of a cell
def fe_name_of_cell(sl, ly, wi):
    return params._fe_idx_list[derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"]]

### one cell id per (sl, ly, wi), to group hits by cell
def cell_id(sl, ly, wi):
    return (np.asarray(sl, dtype=np.int64) * 256 + np.asarray(ly, dtype=np.int64)) * 256 + np.asarray(wi, dtype=np.int64)

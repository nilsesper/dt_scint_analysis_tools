###########################################
### DT CHAMBER: superlayers, layers, wires and readout channels
###########################################

import copy

from analysis_tools.params import params, derived_params

# -----------------------------------------

def superlayers():
    return list(params._dt_chamber["sls"].keys())

def phi_superlayers():
    result = []
    for sl in superlayers():
        if params._dt_chamber["sls"][sl]["orient"] == "phi":
            result.append(sl)
    return result

def theta_superlayer():
    for sl in superlayers():
        if params._dt_chamber["sls"][sl]["orient"] == "theta":
            return sl
    return None

def layers(sl):
    return list(params._dt_chamber["sls"][sl]["lys"].keys())

def wires(sl, ly):
    layer = params._dt_chamber["sls"][sl]["lys"][ly]
    return list(range(layer["min_wi"], layer["max_wi"] + 1))

### all cells of the chamber as list of (sl, ly, wi)
def chamber_cells():
    cells = []
    for sl in superlayers():
        for ly in layers(sl):
            for wi in wires(sl, ly):
                cells.append((sl, ly, wi))
    return cells

### cells left out of the analysis: params._dt_wire_mask and params._dt_dead_wires
def excluded_wires(sl, ly):
    excluded = set(params._dt_wire_mask[sl][ly])
    if sl in params._dt_dead_wires and ly in params._dt_dead_wires[sl]:
        excluded = excluded | set(params._dt_dead_wires[sl][ly])
    return excluded

### all excluded cells as a set of (sl, ly, wi)
def excluded_cells():
    cells = set()
    for sl, ly, wi in chamber_cells():
        if wi in excluded_wires(sl, ly):
            cells.add((sl, ly, wi))
    return cells

### {sl: {ly: {wi: copy of content}}} for every cell of the chamber
def chamber_map(content):
    result = {}
    for sl in superlayers():
        result[sl] = {}
        for ly in layers(sl):
            result[sl][ly] = {}
            for wi in wires(sl, ly):
                result[sl][ly][wi] = copy.deepcopy(content)
    return result

### chamber map with the default look of a cell in the geometry plots (geoplot_utils)
def cell_display_map():
    return chamber_map({"color": params._color_info["cell"][None], "text": ""})

### cell of a readout channel: {"sl", "ly", "wi", "conn_id", "fe_id", "ch_id"}, or None if the channel has no cell
def cell_of_channel(ro_ch, ch):
    if ro_ch not in derived_params._dt_remap_table:
        return None
    if ch not in derived_params._dt_remap_table[ro_ch]:
        return None
    return derived_params._dt_remap_table[ro_ch][ch]

### readout keys (ro_ch, ch, fe_id, conn_id, ch_id) of a cell
def readout_keys_of_cell(sl, ly, wi):
    channel = derived_params._dt_inverted_remap_table[sl][ly][wi]
    keys = {}
    for key in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
        keys[key] = channel[key]
    return keys

### frontend connector name (e.g. "5A") of a cell
def fe_name_of_cell(sl, ly, wi):
    return params._fe_idx_list[derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"]]

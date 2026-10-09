###########################################
### DT PATTERN SEARCH: 4 hits of one superlayer from one muon
###########################################
# The hits of a superlayer are processed in time order. Every cell remembers its last hit. With each new hit,
# every pattern shape (params._dt_sl_patterns: wire offsets per layer) is placed so that it contains the new hit;
# a pattern is found if all 4 cells of the shape have a hit and these hits are close enough in time.
#
# Pattern frame (used in params._dt_sl_patterns): the wire of layer 3 of the pattern is the reference, the other layers
# are given by their wire relative to it ("rel_wis"). See the sketches above _dt_sl_patterns in params.py.

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import data_utils, dt_chamber_utils, timestamp_utils

# -----------------------------------------

### max time difference between the hits of a pattern
# wide_ts_window=True: window for fits with free drift velocity, False: for fits with the nominal drift velocity
def max_ts_difference(wide_ts_window):
    if wide_ts_window:
        return params._dt_sl_patterns_ts_window_fit_vd
    return params._dt_sl_patterns_ts_window

### one found pattern as a row of the sl patterns table (params._sl_pattern_keys)
# pattern_wires: wire of layer 0-3, hit_idx: index of the hit of layer 0-3, new_hit: index of the hit which completed the pattern
# the simulation truth ("sim_...") is taken from the hits, it is 0 in data
def pattern_row(sl, pat_type, pat_name, pattern_wires, hits, hit_idx, new_hit, only_single_muon_patterns):
    reference_hit = hit_idx[3]  # layer 3: reference layer of the pattern frame
    row = {
        "sl": sl,
        "pat_type": pat_type,
        "sim_id": hits["sim_id"][hit_idx[0]],
        "sim_ts": hits["sim_ts"][new_hit],
        "sim_lat_id": 0,
        "sim_x0_loc": hits["sim_dd"][reference_hit] * hits["sim_lat"][reference_hit],
        "sim_tan_alpha": hits["sim_tan_alpha"][reference_hit],
    }
    for key in ["sim_x0", "sim_y0", "sim_z0", "sim_theta", "sim_phi", "sim_vd"]:
        row[key] = hits[key][reference_hit]
    layer_lateralities = []
    for ly in range(4):
        i = hit_idx[ly]
        row[f"wi{ly}"] = pattern_wires[ly]
        row[f"ts{ly}"] = hits["ts"][i]
        row[f"err_ts{ly}"] = hits["err_ts"][i]
        row[f"sim_lat{ly}"] = hits["sim_lat"][i]
        row[f"sim_dt{ly}"] = hits["sim_dt"][i]
        row[f"sim_dd{ly}"] = hits["sim_dd"][i]
        layer_lateralities.append(hits["sim_lat"][i])
    if only_single_muon_patterns:
        lateralities = params._dt_sl_patterns[pat_name]["laterality"]
        if layer_lateralities not in lateralities:
            raise Exception(f"Missing laterality {layer_lateralities} for pattern {pat_type} in params !!!")
        row["sim_lat_id"] = lateralities.index(layer_lateralities)
    return row

### pattern search in the hits of one superlayer (sorted by time); returns a list of pattern rows
# only_single_muon_patterns (simulation): drop patterns made of hits of different simulated muons
def find_patterns_in_superlayer(sl, hits, wide_ts_window, only_single_muon_patterns, verbose):
    ts_window = max_ts_difference(wide_ts_window)
    pattern_names = list(params._dt_sl_patterns.keys())
    last_hit_of_cell = {}  # (ly, wi) -> index of the last hit of this cell
    rows = []
    for i in range(len(hits["ts"])):
        ly, wi = int(hits["ly"][i]), int(hits["wi"][i])
        last_hit_of_cell[(ly, wi)] = i
        for pat_type in range(len(pattern_names)):
            pat_name = pattern_names[pat_type]
            rel_wis = params._dt_sl_patterns[pat_name]["rel_wis"]

            # wires of the 4 layers when the pattern shape contains the new hit
            pattern_wires = []
            for pattern_ly in range(4):
                pattern_wires.append(wi - rel_wis[ly] + rel_wis[pattern_ly])

            # every cell of the pattern must exist and have a hit
            hit_idx = []
            for pattern_ly in range(4):
                if (pattern_ly, pattern_wires[pattern_ly]) in last_hit_of_cell:
                    hit_idx.append(last_hit_of_cell[(pattern_ly, pattern_wires[pattern_ly])])
            if len(hit_idx) < 4:
                continue

            # the 4 hits must be close in time
            pattern_ts = []
            for k in hit_idx:
                pattern_ts.append(hits["ts"][k])
            if max(pattern_ts) - min(pattern_ts) > ts_window:
                continue

            if only_single_muon_patterns:
                sim_ids = set()
                for k in hit_idx:
                    sim_ids.add(hits["sim_id"][k])
                if len(sim_ids) > 1:
                    continue
            if verbose:
                print(f"found pattern: sl={sl}, pattern {pat_name}, wires={pattern_wires}, ts={pattern_ts}")
            rows.append(pattern_row(sl, pat_type, pat_name, pattern_wires, hits, hit_idx, i, only_single_muon_patterns))
    return rows

### find the patterns of all superlayers; returns the sl patterns table (params._sl_pattern_keys), sorted by the time of layer 3
def find_sl_patterns(hits, wide_ts_window=False, only_single_muon_patterns=False, verbose=False):
    rows = []
    for sl in dt_chamber_utils.superlayers():
        sl_hits = data_utils.cut_data(data=hits, conditions=[("sl", "==", sl)], silent=True)
        sl_hits = timestamp_utils.sort_by_timestamp(hits=sl_hits, silent=True)
        rows.extend(find_patterns_in_superlayer(sl, sl_hits, wide_ts_window, only_single_muon_patterns, verbose))

    n_patterns = len(rows)
    sl_patterns = {}
    for key in params._sl_pattern_keys:
        sl_patterns[key] = np.full(n_patterns, 0, dtype=params._sl_pattern_keys[key])
    for i in range(n_patterns):
        for key in rows[i]:
            sl_patterns[key][i] = rows[i][key]
    # sort by the timestamp of layer 3 (the reference cell)
    return data_utils.sort_by_key(data=sl_patterns, sort_key="ts3", silent=True)

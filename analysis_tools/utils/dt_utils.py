###########################################
### DT-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### extract dt hits from hit data
# cut away all hit data not from dt
# add dt specific keys to hits
# take information about this mapping from params.py
def extract_dt_hits(hits, *, silent=False):
    tmp_hits = copy.deepcopy(hits)
    n_hits = len(tmp_hits["ch"])
    if not silent: print(f"Extract DT hits from {n_hits} total hits...")
    # calculate mask to apply to cut away all hits not belonging to dt chamber (wrong ro_ch or invalid ch)
    dt_mask = np.full(n_hits, False, dtype=np.bool)
    for ro_ch in derived_params._dt_ro_chs:
        tmp_mask = np.ma.isin(tmp_hits["ro_ch"], [ro_ch])
        tmp_mask &= np.ma.isin(tmp_hits["ch"], derived_params._dt_chs_by_ro_ch[ro_ch])
        dt_mask |= tmp_mask
    # apply mask
    for k in tmp_hits.keys():
        tmp_hits[k] = tmp_hits[k][dt_mask]
    n_dt_hits = len(tmp_hits["ch"])
    if not silent: print(f"Cut flow: {n_dt_hits}/{n_hits} = {n_dt_hits/n_hits}")
    if not silent: print(f"Found {n_dt_hits} DT hits. Adding DT specific keys...")
    # add specific dt keys
    tmp_hits |= {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()}
    for i in tqdm(range(n_dt_hits)):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in derived_params._dt_keys:
            tmp_hits[k][i] = derived_params._dt_remap_table[ro_ch][ch][k]
    return tmp_hits

### helper: return 3d object to store one value of specified data type for dt chamber
# dt_map = {sl: {ly: [wi: value of dtype]}}
def _empty_dt_chamber_map(dtype):
    dt_map = {}
    for sl in params._dt_chamber["sls"].keys():
        dt_map[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            dt_map[sl][ly] = np.full(params._dt_chamber["sls"][sl]["n_wis"], 0, dtype=dtype)
    return dt_map

### find pattern in dt hits for each superlayer separately, within given timestamp range
# requires timestamps assigned in hits object & sorted hits object wrt timestamps
# returns list of found sl patterns with timestamps and pattern info
#@jit(nopython=True)
def find_sl_patterns(hits, *, silent=False):
    silent = False
    pattern_list = []
    n_hits = len(hits["ch"])
    if not silent: print(f"Extract DT superlayer patterns from {n_hits} total hits...")
    last_hit = _empty_dt_chamber_map(dtype=params._ts_type)
    for i in tqdm(range(n_hits)):
        # update last timestamp of all dt wires
        sl = hits["sl"][i]
        ly = hits["ly"][i]
        wi = hits["wi"][i]
        ts = hits["ts"][i]
        last_hit[sl][ly][wi] = ts
        # check for any pattern only in current superlayer since only in this superlayer something changed wrt to last iteration
        # max value of wire idx for current sl
        max_wi = params._dt_chamber["sls"][sl]["n_wis"]-1
        # loop over all possible patterns
        for pat_type, pat_idcs in enumerate(params._dt_sl_patterns.values()): # pat_idcs = [rel idx wrt base wi for lys 0,1,2,3], pat_type = idx of key in _dt_sl_patterns dict
            # loop over all possible base wires
            for base_wi in range(max_wi+1):
                # calculate relevant wire idcs of all 4 layers for given pattern
                pat_wi = np.full(4, 0, dtype=np.int16) # wi idx of ly 0-3 of pattern
                for ly, rel_wi_idx in enumerate(pat_idcs):
                    pat_wi[ly] = base_wi+rel_wi_idx
                # skip if wire index out of range
                if np.sum(pat_wi < 0) > 0 or np.sum(pat_wi > max_wi) > 0:
                    continue
                pat_wi = np.uint8(pat_wi)
                # collect timestamps of relevant hits for pattern
                pat_ts = np.full(4, 0, dtype=params._ts_type)
                for ly in range(4):
                    pat_ts[ly] = last_hit[sl][ly][ pat_wi[ly] ]
                # skip if any ts is exactly zero (this is simply the initialization value)
                if np.sum(pat_ts == 0) > 0:
                    continue
                # check if timestamps are within specified range
                pat_ts_diff = np.full(6, 0, dtype=params._ts_type)
                pat_ts_diff[0] = pat_ts[0]-pat_ts[1] if pat_ts[0]>pat_ts[1] else pat_ts[1]-pat_ts[0]
                pat_ts_diff[1] = pat_ts[0]-pat_ts[2] if pat_ts[0]>pat_ts[2] else pat_ts[2]-pat_ts[0]
                pat_ts_diff[2] = pat_ts[0]-pat_ts[3] if pat_ts[0]>pat_ts[3] else pat_ts[3]-pat_ts[0]
                pat_ts_diff[3] = pat_ts[1]-pat_ts[2] if pat_ts[1]>pat_ts[2] else pat_ts[2]-pat_ts[1]
                pat_ts_diff[4] = pat_ts[1]-pat_ts[3] if pat_ts[1]>pat_ts[3] else pat_ts[3]-pat_ts[1]
                pat_ts_diff[5] = pat_ts[2]-pat_ts[3] if pat_ts[2]>pat_ts[3] else pat_ts[3]-pat_ts[2]
                #pat_ts_diff2 = pat_ts - pat_ts.reshape(-1,1)
                # no pattern found within time window, continue
                if np.sum(pat_ts_diff > params._dt_sl_patterns_ts_window) > 0:
                    continue
                # if valid pattern, store it
                pattern_list.append([sl, pat_type, pat_wi, pat_ts])
    # convert collected pattern_list to proper output format
    n_patterns = len(pattern_list)
    if not silent: print(f"Found {n_patterns} DT superlayer patterns.")
    sl_patterns = {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_pattern_keys.items()}
    for i in range(n_patterns):
        sl_patterns["sl"][i] = pattern_list[i][0]
        sl_patterns["pat_type"][i] = pattern_list[i][1]
        for j in range(4):
            sl_patterns[f"wi{j}"][i] = pattern_list[i][2][j]
            sl_patterns[f"ts{j}"][i] = pattern_list[i][3][j]
    return sl_patterns

### create empty chamber_data object
def _chamber_data(default={"color": params._color_info["cell"][None], "text": ""}):
    chamber_data = {}
    for sl in params._dt_chamber["sls"].keys():
        chamber_data[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]+1):
            chamber_data[sl][ly] = {}
            for wi in range(params._dt_chamber["sls"][sl]["n_wis"]+1):
                chamber_data[sl][ly][wi] = copy.deepcopy(default)
    return chamber_data




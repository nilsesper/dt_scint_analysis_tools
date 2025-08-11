###########################################
### DT-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm
from scipy.optimize import curve_fit

import analysis_tools.utils.data_utils as data_utils
import analysis_tools.utils.timestamp_utils as timestamp_utils

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
    tmp_hits |= {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_other_keys.items()}
    for i in tqdm(range(n_dt_hits)):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in params._dt_mapping_keys.keys():
            tmp_hits[k][i] = derived_params._dt_remap_table[ro_ch][ch][k]
    # add timestamp and sort by timestamp
    tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    return tmp_hits

### helper: return 3d object to store one value of specified data type for dt chamber
# dt_map = {sl: {ly: [wi: value of dtype]}}
def _empty_dt_chamber_map(dtype, default=0):
    dt_map = {}
    for sl in params._dt_chamber["sls"].keys():
        dt_map[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            dt_map[sl][ly] = np.full(params._dt_chamber["sls"][sl]["n_wis"], default, dtype=dtype)
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
    last_hit = _empty_dt_chamber_map(dtype=params._ts_type, default=0)
    # go through separately for each sl
    for sl in params._dt_chamber["sls"].keys():
        if not silent: print(f"  Progress: SL {sl}...")
        this_sl_hits = data_utils.cut_data(data=hits, conditions=[("sl", "==", sl)])
        n_this_sl_hits = len(this_sl_hits["ch"])
        # max value of wire idx for current sl
        max_wi = params._dt_chamber["sls"][sl]["n_wis"]-1
        for i in tqdm(range(n_this_sl_hits)):
            # update last timestamp of all dt wires
            ly = this_sl_hits["ly"][i]
            wi = this_sl_hits["wi"][i]
            ts = this_sl_hits["ts"][i]
            muon_id = this_sl_hits["muon_id"][i]
            last_hit[sl][ly][wi] = ts
            # check for any pattern only in current superlayer since only in this superlayer something changed wrt to last iteration
            # loop over all possible patterns
            for pat_type, pat_name in enumerate(params._dt_sl_patterns.keys()): # pat_idcs = [rel idx wrt base wi for lys 0,1,2,3], pat_type = idx of key in _dt_sl_patterns dict
                # extract pattern relative wire indices
                pat_idcs = params._dt_sl_patterns[pat_name]["rel_wis"]
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
                    # skip if any ts is exactly zero (this is simply the initialization/reset value)
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
                    pattern_list.append([sl, pat_type, pat_wi, pat_ts, muon_id])
                    # reset the cells which have triggered a pattern (set value to 0)
                    for ly, wi in enumerate(pat_wi):
                        last_hit[sl][ly][wi] = 0
    # convert collected pattern_list to proper output format
    n_patterns = len(pattern_list)
    if not silent: print(f"Found {n_patterns} DT superlayer patterns.")
    sl_patterns = {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_pattern_keys.items()}
    for i in range(n_patterns):
        sl_patterns["sl"][i] = pattern_list[i][0]
        sl_patterns["pat_type"][i] = pattern_list[i][1]
        sl_patterns["muon_id"][i] = pattern_list[i][4]
        for j in range(4):
            sl_patterns[f"wi{j}"][i] = pattern_list[i][2][j]
            sl_patterns[f"ts{j}"][i] = pattern_list[i][3][j]
    # sort pattern list by timestamp of wi3 (ts of ly=3 hit, which later serves as reference cell)
    sl_patterns = data_utils.sort_by_key(hits=sl_patterns, sort_key="wi3")
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

### fit sl patterns
# fit muons to sl patterns, try all lateralities, select best fit
# return list of fit results/parameters
def fit_sl_patterns(patterns, *, silent=False, verbose=False):
    sl_fits = copy.deepcopy(patterns) # keep all pattern keys as well
    n_patterns = len(patterns["sl"])
    if not silent: print(f"Performing SL pattern fits for {n_patterns} patterns...")
    # add other keys
    sl_fits |= {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_fit_keys.items()}
    # fit all patterns
    for i in tqdm(range(n_patterns)):
        pat_type = patterns["pat_type"][i] # idx of key in _dt_sl_patterns
        pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"] # list of [lat for ly0,1,2,3] laterality lists
        # prepare fit data & parameters:
        # arguments are arrays with len=4 i.e. for each layer one hit
        # idx of array = ly idx
        z_arr, x_cell = np.full(4, 0, dtype=np.float64), np.full(4, 0, dtype=np.float64)
        for ly in range(4):
            z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3] #-1*(3-ly)*params._cell_height # z coord for ly0,1,2,3. note coordinate system with ly3 = (z=0)
            rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
            x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2] # x values for fit => x positions of wires / cell centers for each layer, depends on pattern layout
        ts = np.array([patterns[f"ts{ly}"][i] for ly in range(4)], dtype=params._ts_type) # y values for fit => timestamps for hits of each layer
        err_ts = np.full(4, params._err_ts, dtype=np.float64) # ts uncertainty
        t0_start = ts[3] # assume ts of ly=3 rel_wi=0 (reference cell) as t0 starting point
        x0_start = derived_params._sl_pattern_coordinates[3][0][2] # center of ly=3 rel_wi=0 (reference cell)
        tan_alpha_start = 0 # assume straight down muon as start
        p0 = [t0_start, x0_start, tan_alpha_start] # fit start values
        # define parameter bounds
        p_bounds = [
            (np.uint64(t0_start-np.amin([t0_start, params._dt_sl_patterns_ts_window])), derived_params._sl_pattern_coordinates[3][0][0][0], -np.inf), # lower limit for t0, x0, tan_alpha
            (np.uint64(t0_start+params._dt_sl_patterns_ts_window), derived_params._sl_pattern_coordinates[3][0][0][1], np.inf), # upper limit for t0, x0, tan_alpha
        ]
        lat_fits = []
        lat_chi2 = []
        if verbose: print(f"  Fitting pattern {i}...")
        for lat_id, lat in enumerate(lats): # lat_id = idx of laterality list for given pattern
            laterality = np.array(lat)
            # prepare fit function:
            def f_ts_fit_wparams(x_cell, t0, x0, tan_alpha):
                return derived_params.f_ts_fit(x_cell, t0, x0, tan_alpha, z=z_arr, laterality=laterality)
            # execute fit, store results:
            popt, pcov = curve_fit(f=f_ts_fit_wparams, xdata=x_cell, ydata=ts, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds)
            t0_fit, x0_fit, tan_alpha_fit = popt
            ndf = 4 - 3 # no data - no params = 4 - 3
            chi2ndf = np.sum((f_ts_fit_wparams(x_cell, t0_fit, x0_fit, tan_alpha_fit)-ts)**2 / err_ts**2) / ndf
            #if not silent: print(f"pattern no = {i}, pattern name = {pat_name}, laterality = {lat} ---- popt[t0, x0, tan_alpha] = {popt}, chi2/ndf = {chi2ndf}")
            lat_fits.append({"laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "chi2/ndf": chi2ndf})
            if chi2ndf == np.inf: # penalize inf chi2 with high value
                chi2ndf = np.iinfo(np.float32).max
            lat_chi2.append(float(chi2ndf))
            if verbose: print("  Fitting:",{"pattern_id": i, "pattern_name": pat_name, "laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "bounds": p_bounds, "chi2/ndf": chi2ndf})
        lat_chi2 = np.array(lat_chi2)
        # check if more than one fit with minimum chi2 exists
        if (lat_chi2 == lat_chi2.min()).sum() > 1:
            lat_t0 = np.array([lat_fits[i]["chi2/ndf"] for i in range(len(lat_fits))])
            lat_goodness = lat_chi2 + np.log10(np.abs(lat_t0)) # if yes, add t0 bias to goodness param (similar to CIEMAT reco code: https://github.com/magnarex/dtupy-analysis/blob/master/src/dtupy_analysis/dqm/reco/classes/MuSE.py)
        else:
            lat_goodness = lat_chi2 # else use red chi2 as goodness param
        # select fit with best lat_goodness value, store results:
        best_fit_idx = np.argmin(lat_goodness)
        for k in params._sl_fit_keys.keys():
            sl_fits[k][i] = lat_fits[best_fit_idx][k]
    return sl_fits





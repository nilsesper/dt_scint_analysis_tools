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
import analysis_tools.utils.muon_utils as muon_utils
import analysis_tools.utils.hist_utils as hist_utils
import analysis_tools.utils.combination_utils as combination_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### extract dt hits from hit data
# cut away all hit data not from dt
# add dt specific keys to hits
# take information about this mapping from params.py
# if has_timestamp = True, assume that timestamp was already assigned, do not add it here
def extract_dt_hits(hits, *, silent=False, has_timestamp=False, ignore_deadtime=False):
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
    tmp_hits |= {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_other_keys.items() if ((not has_timestamp) or k != "ts")}
    for i in tqdm(range(n_dt_hits), disable=silent):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # skip hit if from "invalid" unconnected channel
        if ch not in derived_params._dt_remap_table[ro_ch].keys():
            continue
        # add keys according to remapping table
        for k in params._dt_mapping_keys.keys():
            tmp_hits[k][i] = derived_params._dt_remap_table[ro_ch][ch][k]
    # add timestamp and sort by timestamp
    if not has_timestamp:
        tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    # add ts uncertainty
    err_ts_com = np.sqrt( (1/np.sqrt(12))**2 + params.dt_hit_add_ts_unc**2)
    if not silent:  print(f"add common ts uncertainty of {err_ts_com} TU...")
    for i in tqdm(range(n_dt_hits), disable=silent):
        tmp_hits["err_ts"][i] = err_ts_com
    ### ------------------------
    # cut away undefined wires (if wi >= n_wis in sl)
    # can happen due to bad remapping tables...
    if not silent: print("cut away undefined wires...")
    keep_ly = []
    for sl in params._dt_chamber["sls"].keys():
        for ly in params._dt_chamber["sls"][sl]["lys"].keys():
            keep_ly.append( data_utils.cut_data(data=tmp_hits, conditions=[("sl","==",sl), ("ly","==",ly), ("wi",">=",params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]), ("wi","<=",params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"])]) )
            #print(f"wi < min_wi:", data_utils.cut_data(data=tmp_hits, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","<",params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"])]) )
            #print(f"wi > max_wi:", data_utils.cut_data(data=tmp_hits, conditions=[("sl","==",sl), ("ly","==",ly), ("wi",">",params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"])]) )
    tmp_hits = data_utils.merge_dataset(split_data=keep_ly)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    ### -----------------------
    # apply dead time constraint to all individual channels (if specified dead time is > 0)
    if params._dt_ts_individual_dead_time > 0 and ignore_deadtime == False:
        if not silent: print(f"apply dead time constraint for all individual channels of {params._dt_ts_individual_dead_time} TU")
        cut_tmp_hits = {}
        for sl in derived_params._dt_inverted_remap_table.keys():
            cut_tmp_hits[sl] = {}
            for ly in derived_params._dt_inverted_remap_table[sl].keys():
                cut_tmp_hits[sl][ly] = {}
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    cut_tmp_hits[sl][ly][wi] = data_utils.cut_data(data=tmp_hits, conditions=[("sl","==",sl),("ly","==",ly),("wi","==",wi)], silent=True)
                    n_cut_hits = len(cut_tmp_hits[sl][ly][wi]["ts"])
                    allowed_indices = []
                    ts_list = np.array(cut_tmp_hits[sl][ly][wi]["ts"])
                    if len(ts_list) > 0:
                        cur_ts = ts_list[0]
                        for i in range(n_cut_hits):
                            if (ts_list[i]) - (cur_ts) < params._dt_ts_individual_dead_time:
                                cur_ts = ts_list[i] # deadtime wrt last hit
                                continue
                            cur_ts = ts_list[i] # deadtime wrt first hit
                            allowed_indices.append(i)
                    for k in cut_tmp_hits[sl][ly][wi].keys():
                        cut_tmp_hits[sl][ly][wi][k] = cut_tmp_hits[sl][ly][wi][k][allowed_indices]
                    n_cut_hits_after = len(cut_tmp_hits[sl][ly][wi]['ts'])
                    if not silent: print(f"sl{sl} ly{ly} wi{wi} dead time cut flow: {n_cut_hits_after} / {n_cut_hits} = {n_cut_hits_after/max(1,n_cut_hits)}")
        # merge back to tmp_hits
        if not silent: print("merging data after applying individual dead time...")
        merge_data = []
        for sl in derived_params._dt_inverted_remap_table.keys():
            for ly in derived_params._dt_inverted_remap_table[sl].keys():
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    merge_data.append(cut_tmp_hits[sl][ly][wi])
        tmp_hits = data_utils.merge_dataset(split_data=merge_data)
        if not silent: print("sort data by timestamp...")
        tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    return tmp_hits

### helper: return 3d object to store one value of specified data type for dt chamber
# dt_map = {sl: {ly: [wi: value of dtype]}}
def _empty_dt_chamber_map(content):
    dt_map = {}
    for sl in params._dt_chamber["sls"].keys():
        dt_map[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            dt_map[sl][ly] = {}
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                dt_map[sl][ly][wi] = copy.deepcopy(content)
    return dt_map

### calculate dt chamber hits caused by muons
# simply propagate it to all layers of the chamber
# returns dt hits with keys {ts = ts of muon + drift time, sl, ly, wi}
# dt hits are being sorted by ts value of hits
# if noise_ampl > 0: add gaussian noise to drift distance (in unit of ts units), sampled individually for each wire and for each event
# if sys_miscalib_ampl > 0: add constant time shift (in unit of ts units) to all individual wire timestamps. sampled only once in the beginning (simulating miscalibration in timestamps)
def hits_from_muons(muons, *, silent=False, noise_ampl=0, sys_miscalib_ampl=0):
    dt_hit_list = []
    n_muons = len(muons["x0"])
    lat_dict = {True: 1, False: -1} # laterality dict: -1: left of wire (l), +1: right of wire (r)
    if not silent: print(f"Calculating DT hits by {n_muons} muons...")
    # simulate constant time offset (miscalib)
    miscalib_map = _empty_dt_chamber_map(content=0)
    if sys_miscalib_ampl > 0:
        for sl in params._dt_chamber["sls"].keys():
            for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    miscalib_map[sl][ly][wi] = np.random.normal(loc=0, scale=1) * sys_miscalib_ampl # constant gaussian noise with sigma = noise_ampl, applied on drift time in ts units
    # propagate muons
    for sl in params._dt_chamber["sls"].keys():
        # calculate x0, tan_alpha projection for simulated muons
        #z_wi_idx = 1
        #z_pos = derived_params._dt_cell_coordinates[sl][3][z_wi_idx][5]
        #(x_ly3,y_ly3,z_ly3) = muon_utils.propagate_muons(muons=muons, z=z_pos) # propagate all muons together
        #z_pos = derived_params._dt_cell_coordinates[sl][2][z_wi_idx][5]
        #(x_ly2,y_ly2,z_ly2) = muon_utils.propagate_muons(muons=muons, z=z_pos) # propagate all muons together
        #muon_tan_alpha = ((-x_ly3+x_ly2) / params._cell_height) if (params._dt_chamber["sls"][sl]["orient"] == "phi") else ((-y_ly3+y_ly2) / params._cell_height) # np.arctan()
        # generate hits
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            z_wi_idx = 1 # all wis have same z therefore save some time here
            z_pos = derived_params._dt_cell_coordinates[sl][ly][z_wi_idx][5] # use center z position (idx 5) of each layer
            (x,y,z) = muon_utils.propagate_muons(muons=muons, z=z_pos) # propagate all muons together
            if not silent: print(f"  Progress: SL {sl}, LY {ly}...")
            # check for all muons separately
            for i in tqdm(range(n_muons), disable=silent):
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    # check if muon propagated inside of x and y range of cell, use >= but < to suppress double hits
                    if not ((x[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][0][0] and x[i] < derived_params._dt_cell_coordinates[sl][ly][wi][0][1]) and (y[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][1][0] and y[i] < derived_params._dt_cell_coordinates[sl][ly][wi][1][1])):
                        continue
                    # consider cell efficiency: sample random number [0,1] and accept only if <= than given efficiency
                    if np.random.uniform(low=0, high=1) > params._dt_cell_efficiency:
                        continue
                    # calculate drift distance
                    hit_coord = x[i] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else y[i]
                    wire_coord = derived_params._dt_cell_coordinates[sl][ly][wi][3] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else derived_params._dt_cell_coordinates[sl][ly][wi][4]
                    noise = 0
                    if noise_ampl > 0:
                        noise = np.random.normal(loc=0, scale=1) * noise_ampl # gaussian noise with sigma = noise_ampl, applied on drift time in ts units
                    drift_distance = np.float64(np.clip(np.abs(hit_coord-wire_coord), a_min=0, a_max=params._cell_width/2)) # in mm
                    drift_time = np.float64(drift_distance / derived_params._drift_velocity_mm_per_timestamp) + noise + miscalib_map[sl][ly][wi] # in timestamp units, cast to int value
                    muon_ts = muons["ts"][i]
                    hit_ts = muon_ts + drift_time # hit timestamp = muon timestamp + drift time
                    # determine laterality of hit: -1 if left of wire, +1 if right of wire
                    laterality = lat_dict[x[i] >= wire_coord] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else lat_dict[y[i] >= wire_coord]
                    # determine tan alpha
                    muon_theta = muons["theta"][i]
                    muon_phi = muons["phi"][i]
                    muon_tan_alpha_x = np.tan(muon_theta)*np.cos(muon_phi)
                    muon_tan_alpha_y = np.tan(muon_theta)*np.sin(muon_phi)
                    muon_tan_alpha = muon_tan_alpha_x if (params._dt_chamber["sls"][sl]["orient"] == "phi") else muon_tan_alpha_y
                    # store this hit
                    dt_hit_list.append({"muon_ts": muon_ts, "sl": sl, "ly": ly, "wi": wi, "muon_dd": drift_distance, "muon_dt": drift_time, "hit_ts": hit_ts, "muon_id": i, "muon_lat": laterality, "muon_tan_alpha": muon_tan_alpha, "muon_vd": derived_params._drift_velocity_mm_per_timestamp , "muon_x0": muons["x0"][i], "muon_y0": muons["y0"][i], "muon_z0": muons["z0"][i], "muon_theta": muon_theta, "muon_phi": muon_phi})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(dt_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    if not silent: print(f"Adding all keys to calculated {n_hits} DT hits...")
    dt_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_other_keys.items()} 
    for i in range(n_hits):
        # copy existing keys
        for k in ["sl", "ly", "wi", "muon_ts", "muon_dt", "muon_dd", "muon_id", "muon_lat", "muon_tan_alpha", "muon_vd", "muon_x0", "muon_y0", "muon_z0", "muon_theta", "muon_phi"]:
            dt_hits[k][i] = dt_hit_list[i][k]
        # map back htg timestamp from drift time
        hit_ts = dt_hit_list[i]["hit_ts"]
        dt_hits["ts"][i] = hit_ts
        dt_hits["err_ts"][i] = np.sqrt( (1/np.sqrt(12))**2 + params.dt_hit_add_ts_unc**2)
        (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(hit_ts)
        dt_hits["oc"][i], dt_hits["bx"][i], dt_hits["tdc"][i] = oc, bx, tdc
        # map back htg parameters
        sl, ly, wi = dt_hit_list[i]["sl"], dt_hit_list[i]["ly"], dt_hit_list[i]["wi"]
        for k in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
            dt_hits[k][i] = derived_params._dt_inverted_remap_table[sl][ly][wi][k]
    # sort hits by their timestamp value
    dt_hits = timestamp_utils.sort_by_timestamp(hits=dt_hits)
    return dt_hits

### apply timing calibration to dt hits
# generated by testpulse run, see functions below: correction object "dt_tp_corrections"
# corrections in dt_tp_corrections should be applied with a plus sign
#   as follows: ts(wi)_corrected = ts(wi)_uncorrected + correction(wi)
# dt_tp_corrections = {"ts_corr": sl_ts_target - ch_ts_mean, "err_ts_corr": ch_ts_err}
def apply_timing_calibration(hits, *, dt_tp_corrections, silent=False):
    n_hits = len(hits["ch"])
    corr_hits = copy.deepcopy(hits)
    if not silent: print(f"Applying testpulse timing correction to {n_hits} DT hits...")
    for i in tqdm(range(n_hits), disable=silent):
            sl = hits["sl"][i]
            ly = hits["ly"][i]
            wi = hits["wi"][i]
            ts = hits["ts"][i]
            err_ts = hits["err_ts"][i]
            # correct timestamp
            ts_corr = np.float64(ts) + dt_tp_corrections[sl][ly][wi]["ts_corr"]
            err_ts_corr = np.sqrt(err_ts**2 + dt_tp_corrections[sl][ly][wi]["err_ts_corr"]**2)
            corr_hits["ts"][i] = ts_corr
            corr_hits["err_ts"][i] = err_ts_corr
            # remap correction to bx oc tdc (htg timestamp), for consistency reasons
            (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(ts_corr)
            corr_hits["oc"][i], corr_hits["bx"][i], corr_hits["tdc"][i] = oc, bx, tdc
    return corr_hits

### find pattern in dt hits for each superlayer separately, within given timestamp range
# requires timestamps assigned in hits object
# returns list of found sl patterns with timestamps and pattern info
#@jit(nopython=True)
# can pass different dt_sl_patterns dictionaries (e.g. fake patterns)
# simulation_only_muon_patterns = True: for simulation reject patterns which come from coincidence of multiple muons, may have wrong laterality
# simulation_only_muon_patterns = False: for simulation keep all patterns (more like data)
def find_sl_patterns(hits, *, dt_sl_patterns=params._dt_sl_patterns, silent=False, verbose=False, simulation_only_muon_patterns=False, fit_vd=False):
    pattern_list = []
    n_hits = len(hits["ch"])
    if not silent: print(f"Extract DT superlayer patterns from {n_hits} total hits...")
    dummy_dt_hit = {k: np.array(0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.array(0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.array(0, dtype=v) for k,v in params._dt_other_keys.items()}
    if not fit_vd:
        delta_ts_max = params._dt_sl_patterns_ts_window
    else:
        delta_ts_max = params._dt_sl_patterns_ts_window_fit_vd
    # go through separately for each sl
    for sl in params._dt_chamber["sls"].keys():
        last_hit = _empty_dt_chamber_map(content=dummy_dt_hit) # holds dict of hits
        if not silent: print(f"  Progress: SL {sl}...")
        this_sl_hits = data_utils.cut_data(data=hits, conditions=[("sl", "==", sl)], silent=silent)
        n_this_sl_hits = len(this_sl_hits["ch"])
        # sort hits by timestamp
        this_sl_hits = timestamp_utils.sort_by_timestamp(hits=this_sl_hits, silent=silent)
        # max value of wire idx for current sl
        min_wi, max_wi  = [params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"] for ly in params._dt_chamber["sls"][sl]["lys"].keys()], [params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] for ly in params._dt_chamber["sls"][sl]["lys"].keys()]
        for i in tqdm(range(n_this_sl_hits), disable=silent):
            # update last timestamp of all dt wires
            ly = this_sl_hits["ly"][i]
            wi = this_sl_hits["wi"][i]
            ts = this_sl_hits["ts"][i]
            muon_ts = this_sl_hits["muon_ts"][i]
            if verbose: print(f"hit: sl={sl} ly={ly} wi={wi} ts={ts}")
            last_hit[sl][ly][wi] = {k: this_sl_hits[k][i] for k in this_sl_hits.keys()} # store dict of current hit
            last_hit_ly, last_hit_wi = ly, wi
            # check for any pattern only in current superlayer since only in this superlayer something changed wrt to last iteration
            # loop over all possible base wires (max. +- 3 away from wire coordinate, no matter which layer)
            # base wi = wi in ly 3
            for base_wi in range(min_wi[3], max_wi[3]+1):
                # loop over all possible patterns
                for pat_type, pat_name in enumerate(dt_sl_patterns.keys()): # pat_idcs = [rel idx wrt base wi for lys 0,1,2,3], pat_type = idx of key in dt_sl_patterns dict
                    # extract pattern relative wire indices
                    pat_idcs = dt_sl_patterns[pat_name]["rel_wis"]
                    # calculate relevant wire idcs of all 4 layers for given pattern
                    pat_wi = np.full(4, 0, dtype=np.int16) # wi idx of ly 0-3 of pattern
                    for ly, rel_wi_idx in enumerate(pat_idcs):
                        pat_wi[ly] = base_wi+rel_wi_idx
                    # skip if last hit has nothing to do with the pattern (i.e. skip if last hit is not in current pattern)
                    #print(pat_wi[last_hit_ly], last_hit_wi)
                    if pat_wi[last_hit_ly] != last_hit_wi:
                        continue
                    # skip if wire index out of range
                    if pat_wi[0] < min_wi[0] or pat_wi[0] > max_wi[0]:
                        continue
                    if pat_wi[1] < min_wi[1] or pat_wi[1] > max_wi[1]:
                        continue
                    if pat_wi[2] < min_wi[2] or pat_wi[2] > max_wi[2]:
                        continue
                    if pat_wi[3] < min_wi[3] or pat_wi[3] > max_wi[3]:
                        continue
                    pat_wi = np.uint8(pat_wi)
                    # collect timestamps of relevant hits for pattern
                    pat_ts = np.full(4, 0, dtype=params._ts_type)
                    pat_err_ts = np.full(4, 0, dtype=np.float64)
                    for ly in range(4):
                        pat_ts[ly] = (last_hit[sl][ly][ pat_wi[ly] ]["ts"])
                        pat_err_ts[ly] = last_hit[sl][ly][ pat_wi[ly] ]["err_ts"]
                    # skip if any ts is exactly zero (this is simply the initialization/reset value)
                    if np.sum(pat_ts == 0) > 0:
                        continue
                    # check if timestamps are within specified range
                    pat_ts_diff = np.full(6, 0, dtype=params._ts_type)
                    pat_ts_diff[0] = np.abs((pat_ts[0])-(pat_ts[1]))
                    pat_ts_diff[1] = np.abs((pat_ts[0])-(pat_ts[2]))
                    pat_ts_diff[2] = np.abs((pat_ts[0])-(pat_ts[3]))
                    pat_ts_diff[3] = np.abs((pat_ts[1])-(pat_ts[2]))
                    pat_ts_diff[4] = np.abs((pat_ts[1])-(pat_ts[3]))
                    pat_ts_diff[5] = np.abs((pat_ts[2])-(pat_ts[3]))
                    if verbose: print(f"check pat: sl={sl}, pat_type={pat_type}, pat_wi={pat_wi}, pat_ts={pat_ts}, pat_ts_diff={pat_ts_diff}")
                    # no pattern found within time window, continue
                    if np.sum(pat_ts_diff > delta_ts_max) > 0:
                        continue
                    if verbose: print(f"found pat: sl={sl}, pat_wi={pat_wi}, pat_ts={pat_ts}")
                    # additional keys
                    dt = [last_hit[sl][ly][ pat_wi[ly] ]["muon_dt"] for ly in range(4)]
                    dd = [last_hit[sl][ly][ pat_wi[ly] ]["muon_dd"] for ly in range(4)]
                    x0_loc = dd[3] * last_hit[sl][ly][ pat_wi[ly] ]["muon_lat"] # x0 is dd in ly3 (reference cell)
                    ly_muon_id = [last_hit[sl][ly][ pat_wi[ly] ]["muon_id"] for ly in range(4)]
                    if simulation_only_muon_patterns:
                        if len(set(ly_muon_id)) > 1: # check if really the same muon
                            #print("non-equal muon_id, reject pattern: muon_id =",ly_muon_id)
                            continue
                    # now can use common attributes of this hit since ensured same muon id above
                    muon_id = 0
                    if simulation_only_muon_patterns:
                        muon_id = ly_muon_id[0]
                    tan_alpha = last_hit[sl][ly][ pat_wi[ly] ]["muon_tan_alpha"]
                    # lateralities
                    ly_lats = [last_hit[sl][ly][ pat_wi[ly] ]["muon_lat"] for ly in range(4)]
                    lat = 0
                    if simulation_only_muon_patterns:
                        if ly_lats not in params._dt_sl_patterns[pat_name]["laterality"]:
                            raise Exception(f"Missing laterality {ly_lats} for pattern {pat_type} in params !!!")
                        lat = params._dt_sl_patterns[pat_name]["laterality"].index(ly_lats) # laterality id of this pattern (index of laterality list in params for this pat_id)
                    # sim muon data
                    muon_x0 = last_hit[sl][ly][ pat_wi[ly] ]["muon_x0"]
                    muon_y0 = last_hit[sl][ly][ pat_wi[ly] ]["muon_y0"]
                    muon_z0 = last_hit[sl][ly][ pat_wi[ly] ]["muon_z0"]
                    muon_theta = last_hit[sl][ly][ pat_wi[ly] ]["muon_theta"]
                    muon_phi = last_hit[sl][ly][ pat_wi[ly] ]["muon_phi"]
                    muon_vd = last_hit[sl][ly][ pat_wi[ly] ]["muon_vd"]
                    # if valid pattern, store it
                    pattern_list.append([sl, pat_type, pat_wi, pat_ts, muon_id, muon_ts, lat, dt, x0_loc, tan_alpha, ly_lats, dd, muon_x0, muon_y0, muon_z0, muon_theta, muon_phi, muon_vd, pat_err_ts])
                    # reset the cells which have triggered a pattern (set value to 0)
                    #for ly, wi in enumerate(pat_wi):
                    #    last_hit[sl][ly][wi] = 0
    # convert collected pattern_list to proper output format
    n_patterns = len(pattern_list)
    if not silent: print(f"Found {n_patterns} DT superlayer patterns.")
    sl_patterns = {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_pattern_keys.items()}
    for i in range(n_patterns):
        sl_patterns["sl"][i] = pattern_list[i][0]
        sl_patterns["pat_type"][i] = pattern_list[i][1]
        sl_patterns["muon_id"][i] = pattern_list[i][4]
        sl_patterns["muon_ts"][i] = pattern_list[i][5]
        sl_patterns[f"muon_lat_id"][i] = pattern_list[i][6]
        sl_patterns[f"muon_x0_loc"][i] = pattern_list[i][8]
        sl_patterns[f"muon_tan_alpha"][i] = pattern_list[i][9]
        sl_patterns[f"muon_x0"][i] = pattern_list[i][12]
        sl_patterns[f"muon_y0"][i] = pattern_list[i][13]
        sl_patterns[f"muon_z0"][i] = pattern_list[i][14]
        sl_patterns[f"muon_theta"][i] = pattern_list[i][15]
        sl_patterns[f"muon_phi"][i] = pattern_list[i][16]
        sl_patterns[f"muon_vd"][i] = pattern_list[i][17]
        for j in range(4):
            sl_patterns[f"wi{j}"][i] = pattern_list[i][2][j]
            sl_patterns[f"ts{j}"][i] = pattern_list[i][3][j]
            sl_patterns[f"err_ts{j}"][i] = pattern_list[i][18][j]
            sl_patterns[f"muon_lat{j}"][i] = pattern_list[i][10][j]
            sl_patterns[f"muon_dt{j}"][i] = pattern_list[i][7][j]
            sl_patterns[f"muon_dd{j}"][i] = pattern_list[i][11][j]
    # sort pattern list by timestamp of wi3 (ts of ly=3 hit, which later serves as reference cell)
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="wi3", silent=silent)
    return sl_patterns

### create empty chamber_data object
def _chamber_data(default={"color": params._color_info["cell"][None], "text": ""}):
    chamber_data = {}
    for sl in params._dt_chamber["sls"].keys():
        chamber_data[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            chamber_data[sl][ly] = {}
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                chamber_data[sl][ly][wi] = copy.deepcopy(default)
    return chamber_data

### fit sl patterns
# fit muons to sl patterns, try all lateralities, select best fit
# return list of fit results/parameters
def fit_sl_patterns(patterns, *, silent=False, verbose=False, fit_vd=False):
    sl_fits = copy.deepcopy(patterns) # keep all pattern keys as well
    n_patterns = len(patterns["sl"])
    if not silent: print(f"Performing SL pattern fits for {n_patterns} patterns...")
    # add other keys
    sl_fits |= {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_fit_keys.items()} | {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_fit_other_keys.items()}
    # fit all patterns
    for i in tqdm(range(n_patterns), disable=silent):
        pat_type = patterns["pat_type"][i] # idx of key in _dt_sl_patterns
        pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"] # list of [lat for ly0,1,2,3] laterality lists
        # prepare fit data & parameters:
        # arguments are arrays with len=4 i.e. for each layer one hit
        # idx of array = ly idx
        z_arr, x_cell = np.full(4, 0, dtype=np.float64), np.full(4, 0, dtype=np.float64)
        lys = np.arange(0, 4)
        for ly in lys:
            z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3] #-1*(3-ly)*params._cell_height # z coord for ly0,1,2,3. note coordinate system with ly3 = (z=0)
            rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
            x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2] # x values for fit => x positions of wires / cell centers for each layer, depends on pattern layout
        ts = np.array([np.float64(patterns[f"ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type) # y values for fit => timestamps for hits of each layer
        err_ts = np.array([np.float64(patterns[f"err_ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type) # ts uncertainty
        ts_min = np.amin(ts)
        ts_max = np.amax(ts)
        # scale timestamps by subtracting min timestamp
        ts_offset = ts_min
        ts_for_fit = ts - ts_offset
        ts_min_for_fit = ts_min - ts_offset
        ts_max_for_fit = ts_max - ts_offset
        # -- define parameter bounds,: t0, vd
        # t0 bound
        if not fit_vd:
            t0_min_bound = ts_max_for_fit-params._dt_max_drift_time-params._t0_tolerance
            t0_max_bound = ts_min_for_fit+params._t0_tolerance
        else:
            t0_min_bound = ts_max_for_fit-params._dt_max_drift_time_vd_min-params._t0_tolerance
            t0_max_bound = ts_min_for_fit+params._t0_tolerance
        # check impossible timestamps
        impossible_pattern = False
        if t0_min_bound >= t0_max_bound:
            impossible_pattern = True
        if verbose: print(f"\n ********** Fitting pattern {i}:")
        if not impossible_pattern: # fit only if possible bounds / timestamps
            # vd bound
            vd_min_bound = derived_params._drift_velocity_mm_per_timestamp_min
            vd_max_bound = derived_params._drift_velocity_mm_per_timestamp_max
            # tan alpha bound
            alpha_min_bound = params._dt_pattern_alpha_range[pat_type][0]
            alpha_max_bound = params._dt_pattern_alpha_range[pat_type][1]
            tan_alpha_min_bound = np.tan(alpha_min_bound)
            tan_alpha_max_bound = np.tan(alpha_max_bound)
            # --- fitting
            lat_fits = []
            lat_chi2 = []
            # fit all paterality possibilities
            for lat_id, lat in enumerate(lats): # lat_id = idx of laterality list for given pattern
                laterality = np.array(lat)
                # define parameter bounds: x0
                # set x0 bounds depending on laterality (l = -1: left of wire i.e. x0 < x_wire, r = 1: right of wire i.e x0 > x_wire)
                x0_min_bound = derived_params._sl_pattern_coordinates[3][0][0][0] if (laterality[3] == -1) else derived_params._sl_pattern_coordinates[3][0][2]
                x0_max_bound = derived_params._sl_pattern_coordinates[3][0][0][1] if (laterality[3] == 1) else derived_params._sl_pattern_coordinates[3][0][2]
                # write into concatenated p_bounds variable
                if not fit_vd:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound), # lower limit for (t0, x0, tan_alpha)
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound), # upper limit for (t0, x0, tan_alpha)
                    ])
                else:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound, vd_min_bound), # lower limit for (t0, x0, tan_alpha, vd)
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound, vd_max_bound), # upper limit for (t0, x0, tan_alpha, vd)
                    ])
                # prepare fit initial params
                t0_start = np.mean([p_bounds[0][0], p_bounds[1][0]]) # t0 starting point
                x0_start = np.mean([p_bounds[0][1], p_bounds[1][1]]) #+ 10*laterality[3]
                tan_alpha_start = np.tan(np.mean([alpha_min_bound, alpha_max_bound])) # tan alpha starting point: tan of mean of angle
                vd_start = derived_params._drift_velocity_mm_per_timestamp
                if not fit_vd:
                    p0 = np.float64([t0_start, x0_start, tan_alpha_start]) 
                else:
                    p0 = np.float64([t0_start, x0_start, tan_alpha_start, vd_start])
                # prepare fit function
                if not fit_vd:
                    def f_ts_fit_wparams(ly, t0, x0, tan_alpha):
                        ly = np.uint64(ly)
                        return derived_params.f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=derived_params._drift_velocity_mm_per_timestamp)
                    def err_f_ts_fit_wparams(ly, t0, x0, tan_alpha, err_t0, err_x0, err_tan_alpha, corr_t0_x0, corr_t0_tan_alpha, corr_x0_tan_alpha):
                        ly = np.uint64(ly)
                        return derived_params.err_f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=derived_params._drift_velocity_mm_per_timestamp, err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=0, corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=0, corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=0, corr_tan_alpha_vd=0)
                else:
                    def f_ts_fit_wparams(ly, t0, x0, tan_alpha, vd):
                        ly = np.uint64(ly)
                        return derived_params.f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd)
                    def err_f_ts_fit_wparams(ly, t0, x0, tan_alpha, vd, err_t0, err_x0, err_tan_alpha, err_vd, corr_t0_x0, corr_t0_tan_alpha, corr_x0_tan_alpha, corr_x0_vd, corr_tan_alpha_vd, corr_t0_vd):
                        ly = np.uint64(ly)
                        return derived_params.err_f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd, err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd, corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=corr_t0_vd, corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=corr_x0_vd, corr_tan_alpha_vd=corr_tan_alpha_vd)
                # execute fit, store results: parameters = (t0, x0, tan_alpha)
                popt, pcov, infodict, mesg, _ = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds, full_output=True, )
                # extract fit result
                if not fit_vd:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit = popt
                    err_t0_fit = np.sqrt(pcov[0][0])
                    err_x0_fit = np.sqrt(pcov[1][1])
                    err_tan_alpha_fit = np.sqrt(pcov[2][2])
                    err_vd_fit = 0
                    corr_t0_x0_fit = pcov[0][1]
                    corr_t0_tan_alpha_fit = pcov[0][2]
                    corr_t0_vd_fit = 0
                    corr_x0_tan_alpha_fit = pcov[1][2]
                    corr_x0_vd_fit = 0
                    corr_tan_alpha_vd_fit = 0
                else:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit, vd_from_fit = popt
                    err_t0_fit = np.sqrt(pcov[0][0])
                    err_x0_fit = np.sqrt(pcov[1][1])
                    err_tan_alpha_fit = np.sqrt(pcov[2][2])
                    err_vd_fit = np.sqrt(pcov[3][3])
                    corr_t0_x0_fit = pcov[0][1]
                    corr_t0_tan_alpha_fit = pcov[0][2]
                    corr_t0_vd_fit = pcov[0][3]
                    corr_x0_tan_alpha_fit = pcov[1][2]
                    corr_x0_vd_fit = pcov[1][3]
                    corr_tan_alpha_vd_fit = pcov[2][3]
                # calculate chi2 value
                if not fit_vd:
                    ndf = 1
                else:
                    ndf = 1
                if not fit_vd:
                    ts_from_fit = f_ts_fit_wparams(lys, t0_from_fit, x0_from_fit, tan_alpha_from_fit)
                else:
                    ts_from_fit = f_ts_fit_wparams(lys, t0_from_fit, x0_from_fit, tan_alpha_from_fit, vd_from_fit)
                # convert fit output to real parameters
                ts_fit = ts_from_fit + ts_offset
                ts_residuals = ts_from_fit - np.float64(ts_for_fit)
                chi2ndf = np.sum(ts_residuals**2 / err_ts**2) / ndf
                t0_fit = t0_from_fit + ts_offset
                x0_fit = x0_from_fit
                tan_alpha_fit = tan_alpha_from_fit
                if not fit_vd:
                    vd_fit = derived_params._drift_velocity_mm_per_timestamp
                else:
                    vd_fit = vd_from_fit
                # estimate drift times from fit
                td = [ts_fit[ly]-t0_fit for ly in range(4)]
                # store results
                lat_fits.append({"impossible": 0, "laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "chi2/ndf": chi2ndf, "dt0": td[0], "dt1": td[1], "dt2": td[2], "dt3": td[3], "vd": vd_fit, "err_t0": err_t0_fit, "err_x0": err_x0_fit, "err_tan_alpha": err_tan_alpha_fit, "err_vd": err_vd_fit, "corr_t0_x0": corr_t0_x0_fit, "corr_t0_tan_alpha": corr_t0_tan_alpha_fit, "corr_t0_vd": corr_t0_vd_fit, "corr_x0_tan_alpha": corr_x0_tan_alpha_fit, "corr_x0_vd": corr_x0_vd_fit, "corr_tan_alpha_vd": corr_tan_alpha_vd_fit})
                if chi2ndf == np.inf: # penalize inf chi2 with high value
                    chi2ndf = 999999999
                lat_chi2.append(chi2ndf)
                if verbose:
                    print(f" **** Pattern name {pat_name}, laterality {lat_id}:")
                    print(f"    Data x:", [lys[ly] for ly in range(4)])
                    print(f"    Data y:", [ts[ly] for ly in range(4)])
                    print(f"    Error y:", [err_ts[ly] for ly in range(4)])
                    print(f"    Fit impossible (bound error): {impossible_pattern}")
                    print(f"    Fit input:",{ "p0": p0, "bounds": p_bounds})
                    print(f"    Fitted y:", [ts_fit[ly] for ly in range(4)])
                    print(f"    Residuals y:", [ts_residuals[ly] for ly in range(4)])
                    print(f"    Result:",{"popt": popt, "infodict": infodict, "mesg": mesg})
                    print(f"    Values:",{"t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "vd": vd_fit, "chi2/ndf": chi2ndf})
                    print(f"\n    Chi2 / Ndf: {chi2ndf}\n")
            # round chi2 value to given fixed digits
            for j in range(len(lat_chi2)):
                lat_chi2[j] = float('{:0.3e}'.format(lat_chi2[j])) # round to 4 significant digits in total
            lat_chi2 = np.array(lat_chi2)
            # check if more than one fit with minimum chi2 exists
            lat_goodness = []
            if (lat_chi2 == lat_chi2.min()).sum() > 1:
                lat_t0 = np.array([lat_fits[k]["t0"] for k in range(len(lat_fits))])
                lat_goodness = lat_chi2 + np.log10(np.abs(lat_t0)) # if yes, add t0 bias to goodness param (similar to CIEMAT reco code: https://github.com/magnarex/dtupy-analysis/blob/master/src/dtupy_analysis/dqm/reco/classes/MuSE.py)
            else:
                lat_goodness = lat_chi2 # else use red chi2 as goodness param
            best_fit_idx = np.argmin(lat_goodness)
            # store results of best fit
            for k in params._sl_fit_keys.keys():
                sl_fits[k][i] = lat_fits[best_fit_idx][k]
            # store results of all laterality fits
            for lat_id in range(len(lats)):
                for k1,k2 in [(f"lat{lat_id}_impossible", "impossible"), (f"lat{lat_id}_t0", "t0"), (f"lat{lat_id}_x0", "x0"), (f"lat{lat_id}_tan_alpha", "tan_alpha"), (f"lat{lat_id}_chi2/ndf", "chi2/ndf"), (f"lat{lat_id}_dt0", "dt0"), (f"lat{lat_id}_dt1", "dt1"), (f"lat{lat_id}_dt2", "dt2"), (f"lat{lat_id}_dt3", "dt3"), (f"lat{lat_id}_vd", "vd"), 
                    (f"lat{lat_id}_err_t0", "err_t0"), (f"lat{lat_id}_err_x0", "err_x0"), (f"lat{lat_id}_err_tan_alpha", "err_tan_alpha"), (f"lat{lat_id}_err_vd", "err_vd"), (f"lat{lat_id}_corr_t0_x0", "corr_t0_x0"), (f"lat{lat_id}_corr_t0_tan_alpha", "corr_t0_tan_alpha"), (f"lat{lat_id}_corr_t0_vd", "corr_t0_vd"), (f"lat{lat_id}_corr_x0_tan_alpha", "corr_x0_tan_alpha"), (f"lat{lat_id}_corr_x0_vd", "corr_x0_vd"), (f"lat{lat_id}_corr_tan_alpha_vd", "corr_tan_alpha_vd")]:
                    sl_fits[k1][i] = lat_fits[lat_id][k2]
        else: # if fit impossible
            print(f" **** Impossible to fit timestamps.")
            sl_fits["impossible"][i] = 1
        # NOTE:
        # if one wants to use also the +-d patterns, one might need to keep several fit results since there are possibilities of ambiguities between rrll and llrr
        # but if not using the +-d pattern, do not care :)
    return sl_fits

"""
### fit sl patterns WITH MEANTIMER METHOD
def fit_sl_patterns_meantimer(patterns, *, silent=False, verbose=False):
    patterns = data_utils.cut_data(data=patterns, conditions=[("pat_type","in",list(params._meantimer_patterns.keys()))], silent=silent)
    sl_fits = copy.deepcopy(patterns) # keep all pattern keys as well
    n_patterns = len(patterns["sl"])
    if not silent: print(f"Performing SL pattern fits for {n_patterns} patterns...")
    # add other keys
    sl_fits |= {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_fit_keys.items()} | {k: np.full(n_patterns, 0, dtype=v) for k,v in params._sl_fit_other_keys.items()}
    # fit all patterns
    for i in tqdm(range(n_patterns), disable=silent):
        pat_type = patterns["pat_type"][i] # idx of key in _dt_sl_patterns
        pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"] # list of [lat for ly0,1,2,3] laterality lists
        ts = np.array([np.float64(patterns[f"ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type) # y values for fit => timestamps for hits of each layer
        err_ts = np.full(4, params._err_ts, dtype=np.float64) # ts uncertainty
        if verbose: print(f"\n ********** Fitting pattern {i}:")
        t0_list_mt = [[] for lat_id in range(len(lats))] # t0 = t_muon results for all lateralities for all mt_equations
        tan_alpha_list_mt = [[] for lat_id in range(len(lats))] # t0 = t_muon results for all lateralities for all mt_equations
        max_deviation_t0 = [0 for lat_id in range(len(lats))] # max deviation between meantimer values
        max_deviation_tan_alpha = [0 for lat_id in range(len(lats))]
        if verbose: print("---------------------------")
        if verbose: print(f"pat_type = {pat_type}")
        if verbose: print(f"muon_lat_id = {patterns['muon_lat_id'][i]}")
        for lat_id, lat in enumerate(lats): # lat_id = idx of laterality list for given pattern
            laterality = np.array(lat)
            ## calculate meantimer equation results for the timestamps of this pattern
            # meantimer functions have format func(t0, t1, t2, t3, h, vd, t_max)
            # t_muon = t0
            var_str = f"t_muon"
            for j, func in enumerate(derived_params.meantimer_functions[pat_type][lat_id][var_str]):
                t0_list_mt[lat_id].append( func(t0=ts[0], t1=ts[1], t2=ts[2], t3=ts[3], h=params._cell_height, vd=derived_params._drift_velocity_mm_per_timestamp, t_max=params._dt_max_drift_time) )
            # tan_alpha
            var_str = f"tan_alpha"
            for j, func in enumerate(derived_params.meantimer_functions[pat_type][lat_id][var_str]):
                tan_alpha_list_mt[lat_id].append( func(t0=ts[0], t1=ts[1], t2=ts[2], t3=ts[3], h=params._cell_height, vd=derived_params._drift_velocity_mm_per_timestamp, t_max=params._dt_max_drift_time) )
            ## check if meantimer results lie closer than specified tolerances within each other
            for j in range(len(t0_list_mt[lat_id])):
                for k in range(len(t0_list_mt[lat_id])):
                    mt_deviation = np.abs(t0_list_mt[lat_id][j] - t0_list_mt[lat_id][k])
                    if mt_deviation > max_deviation_t0[lat_id]:
                        max_deviation_t0[lat_id] = mt_deviation
            for j in range(len(tan_alpha_list_mt[lat_id])):
                for k in range(len(tan_alpha_list_mt[lat_id])):
                    mt_deviation = np.abs(tan_alpha_list_mt[lat_id][j] - tan_alpha_list_mt[lat_id][k])
                    if mt_deviation > max_deviation_tan_alpha[lat_id]:
                        max_deviation_tan_alpha[lat_id] = mt_deviation
        if verbose: print(f"t0_list_mt = {t0_list_mt}")
        if verbose: print(f"tan_alpha_list_mt = {tan_alpha_list_mt}")
        best_lat = 99 # default value if no lat is found good
        if verbose: print(f"max_deviation_t0 = {max_deviation_t0}  --  max_deviation_tan_alpha = {max_deviation_tan_alpha}")
        # get lateralies where t0 deviation is within tolerance
        best_t0_lats = []
        for lat_id in range(len(lats)):
            if max_deviation_t0[lat_id] <= params._meantimer_tolerance_t0:
                best_t0_lats.append(lat_id)
        if verbose: print(f"best_t0_lats = {best_t0_lats}")
        # try for all these lats which tan alpha has lowest tolerance
        best_tan_alpha_lats = []
        for lat_id in best_t0_lats:
            if max_deviation_tan_alpha[lat_id] <= params._meantimer_tolerance_tan_alpha:
                best_tan_alpha_lats.append(lat_id)
        if verbose: print(f"best_tan_alpha_lats = {best_tan_alpha_lats}")
        if len(best_tan_alpha_lats) > 0: # if any laterality works
            # use the one with smallest tan alpha deviation
            best_lat_idx = np.argmin([max_deviation_tan_alpha[lat_id] for lat_id in best_tan_alpha_lats])
            best_lat = best_tan_alpha_lats[best_lat_idx]
        # collect info about selected laterality
        if best_lat != 99:
            t0_result = np.mean(t0_list_mt[best_lat])
            tan_alpha_result = np.mean(tan_alpha_list_mt[best_lat])
            td = ts - t0_result
            x0_result = td[3]*derived_params._drift_velocity_mm_per_timestamp * lats[best_lat][3]
            if verbose: print(f"best_lat = {best_lat}  --  t0_result = {t0_result}  --  tan_alpha_result = {tan_alpha_result}")
            if verbose: print(f"muon_lat_id = {patterns['muon_lat_id'][i]}  --  muon_ts = {patterns['muon_ts'][i]}  --  muon_tan_alpha = {patterns['muon_tan_alpha'][i]}")
            if verbose: print(f"best_lat - muon_lat_id = {best_lat - patterns['muon_lat_id'][i]}  --  t0_result - muon_ts = {t0_result - patterns['muon_ts'][i]}  --  tan_alpha_result - muon_tan_alpha = {tan_alpha_result - patterns['muon_tan_alpha'][i]}")
            # estimate ts chi2 value
            z_arr, x_cell = np.full(4, 0, dtype=np.float64), np.full(4, 0, dtype=np.float64)
            lys = np.arange(0, 4)
            for ly in lys:
                z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3] #-1*(3-ly)*params._cell_height # z coord for ly0,1,2,3. note coordinate system with ly3 = (z=0)
                rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
                x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2] # x values for fit => x positions of wires / cell centers for each layer, depends on pattern layout
            def f_ts_fit_wparams(ly, t0, x0, tan_alpha):
                ly = np.uint64(ly)
                return derived_params.f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly])
            ts_from_fit = f_ts_fit_wparams(lys, t0_result, x0_result, tan_alpha_result)
            ts_residuals = ts_from_fit - np.float64(ts)
            ndf = 4-3
            chi2ndf = np.sum(ts_residuals**2 / err_ts**2) / ndf
            result = {
                "laterality": best_lat, # laterality = 99 --> means no good laterality value found, invalid hit
                "t0": t0_result,
                "x0": x0_result,
                "tan_alpha": tan_alpha_result,
                "vd": derived_params._drift_velocity_mm_per_timestamp,
                "chi2/ndf": chi2ndf,
                "dt0": td[0],
                "dt1": td[1],
                "dt2": td[2],
                "dt3": td[3],
            }
        sl_fits["laterality"][i] = best_lat
        if best_lat != 99:
            for k in params._sl_fit_keys.keys():
                sl_fits[k][i] = result[k]

    # cut away invalid meantimer fits (with laterality = 99)
    sl_fits = data_utils.cut_data(data=sl_fits, conditions=[("laterality","!=",99)], silent=silent)
    return sl_fits
#"""


### group sl fits of one sl together in time
def group_sl_fits_of_one_sl(sl_fits, idx_offset=0, *, silent=False):
    # measurement duration
    duration = 0.78e-9 * (np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"])) # secs
    if not silent: print(f"measurement duration = {duration} s")
    # group fits
    sl_fits_sl = {} # sl fits of one sl
    idx_grouped = {} # {sl: [group idx: [idx list of patterns in pattern list which belong to group]]}
    ts_group = {} # {sl: [group idx: timestamp of group (mean of group member timestamps)]}
    n_groups = {}
    group_rate = {}
    for sl in params._dt_chamber["sls"].keys():
        if not silent: print(f"grouping sl fits of sl = {sl}...")
        sl_fits_sl[sl] = data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl)], silent=True)
        idx_grouped[sl], ts_group[sl] = combination_utils.time_grouping_indices_2(data=sl_fits_sl[sl], ts_tolerance=params._sl_fit_group_ts_tolerance, data_ts_key="t0")
        n_groups[sl] = len(idx_grouped[sl])
        group_rate[sl] = n_groups[sl] / duration
    n_groups_sum = np.sum([n_groups[sl] for sl in params._dt_chamber["sls"].keys()])
    if not silent: print(f"n_fit_groups per sl = {n_groups}")
    if not silent: print(f"fit group rate per sl = {group_rate} Hz")
    # sl_fit_groups = { "sl": superlayer, "tgroup": mean t0 of fits in group, "idcs": [indices of group member sl_fits], "n_fits": no of sl fits in group }
    ### translate idcs of sl_fits_sl (only one sl) back to idcs of sl_fits (all sls together)
    sl_fit_groups = {
        "sl": np.zeros(n_groups_sum),
        "tgroup": np.zeros(n_groups_sum),
        "idcs": [[] for i in range(n_groups_sum)],
        "n_fits": np.zeros(n_groups_sum),
    }
    # get original indices
    for sl in params._dt_chamber["sls"].keys():
        if not silent: print(f"translating back indices of groups in sl = {sl}...")
        idx_shift = int( np.sum([n_groups[sl_i] for sl_i in params._dt_chamber["sls"].keys() if sl_i < sl]) )
        for i in range(n_groups[sl]):
            j = int( i + idx_shift )
            glob_idcs = []
            for loc_idx in idx_grouped[sl][i]:
                glob_idx = np.where((sl_fits["t0"] == sl_fits_sl[sl]["t0"][loc_idx]) & (sl_fits["sl"] == sl))[0][0]
                glob_idcs.append(glob_idx + idx_offset)
            sl_fit_groups["idcs"][j] = glob_idcs
            sl_fit_groups["tgroup"][j] = ts_group[sl][i]
            sl_fit_groups["sl"][j] = sl
            sl_fit_groups["n_fits"][j] = len(glob_idcs)
    # sort sl_fit_groups by tgroup
    sl_fit_groups = data_utils.sort_by_key(data=sl_fit_groups, sort_key="tgroup", silent=silent)
    return sl_fit_groups

### combine sl fit groups for full chamber, generate muon object as output
def reco_muons_from_sl_fit_groups(fits, fit_groups, *, silent=False, verbose=False):
    n_fits = data_utils.length(fits)
    n_fit_groups = data_utils.length(fit_groups)
    reco_muon_list = []
    counter_groups = 0
    counter_3sl_candidates = 0
    counter_tgroup = 0
    counter_n_fits = 0
    counter_chi2 = 0
    counter_tan_alpha = 0
    counter_xproj = 0
    duration = (np.amax(fit_groups["tgroup"]) - np.amin(fit_groups["tgroup"]))*0.78e-9 # duration of data set in s
    # extract is of sls in phi & theta orientation
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]
    phi_sl1, phi_sl2 = phi_sls[0], phi_sls[1]
    theta_sl = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "theta"][0]
    ### sort fit groups by tgroup value
    fit_groups = data_utils.sort_by_key(data=fit_groups, sort_key="tgroup", silent=silent)
    ### combine sl fit groups of different superlayers to "muon"
    last_fit_group = {sl: None for sl in params._dt_chamber["sls"].keys()} # holds last fit group of each sl
    if not silent: print(f"combine sl fit groups to muons...")
    for i in tqdm(range(n_fit_groups), disable=silent):
        counter_groups += 1
        ### search for fit groups close in time
        sl = fit_groups["sl"][i]
        last_fit_group[sl] = {k: fit_groups[k][i] for k in fit_groups.keys()} | {"idx": i} # store all info of fit group
        # continue of not one fit group in each sl found until now
        if None in last_fit_group.values():
            continue
        counter_3sl_candidates += 1
        # check max n_fits condition
        if last_fit_group[1]["n_fits"] > params._muon_n_fits_max:
            continue
        if last_fit_group[2]["n_fits"] > params._muon_n_fits_max:
            continue
        if last_fit_group[3]["n_fits"] > params._muon_n_fits_max:
            continue
        counter_n_fits += 1
        # check if fit group timestamps are within specified tolerance, else continue
        if np.abs(last_fit_group[1]["tgroup"] - last_fit_group[2]["tgroup"]) > params._muon_tgroup_tolerance:
            continue
        if np.abs(last_fit_group[1]["tgroup"] - last_fit_group[3]["tgroup"]) > params._muon_tgroup_tolerance:
            continue
        if np.abs(last_fit_group[2]["tgroup"] - last_fit_group[3]["tgroup"]) > params._muon_tgroup_tolerance:
            continue
        counter_tgroup += 1
        # if code made it to here, a muon (3 fit groups reasonably close in time) was found
        ###### attempt muon reco
        ### currently only supports 1 pattern/fit per sl in selected fit group - if multiple one would need to add more selection logic in order to decide which fit to choose...
        if params._muon_n_fits_max > 1:
            raise Exception(f"_muon_n_fits_max > 1 is not supported until now. can only reco muon if only 1 pattern/fit per pattern group is found")
        fit = {} # {sl: {sl_fit obj keys & values}
        for sl in params._dt_chamber["sls"].keys():
            fit_idx = last_fit_group[sl]["idcs"][0]
            fit[sl] =  {k: fits[k][fit_idx] for k in fits.keys()}
        # check max chi2/ndf condition
        if fit[1]["chi2/ndf"] > params._muon_chi2_ndf_max:
            continue
        if fit[2]["chi2/ndf"] > params._muon_chi2_ndf_max:
            continue
        if fit[3]["chi2/ndf"] > params._muon_chi2_ndf_max:
            continue
        counter_chi2 += 1
        # reject muon if tan alpha tolerance not met
        if np.abs(fit[phi_sl1]["tan_alpha"] - fit[phi_sl2]["tan_alpha"]) > params._muon_slphi_tan_alpha_tolerance:
            continue
        counter_tan_alpha += 1
        for sl in params._dt_chamber["sls"].keys():
            if fit[sl]["sl"] != sl:
                raise Exception(fit)
        ### tan_alpha_phi
        tan_alpha_phi = np.mean([fit[phi_sl1]["tan_alpha"], fit[phi_sl2]["tan_alpha"]])
        ### tan_alpha_phi uncertainty
        # calc deviaion
        dev_tan_alpa_phi = np.abs(fit[phi_sl1]["tan_alpha"] - fit[phi_sl2]["tan_alpha"])
        # check if max compatible within all errors
        min_tan_alpa_phi_sigma = np.amin([fit[sl]["err_tan_alpha"] for sl in phi_sls])
        # if compatible do error on mean (1/sqrt(N))
        if dev_tan_alpa_phi <= min_tan_alpa_phi_sigma:
            err_tan_alpha_phi = np.sqrt(
                (fit[phi_sl1]["err_tan_alpha"]/2)**2
                + (fit[phi_sl2]["err_tan_alpha"]/2)**2
            )
        # if not compatible do error as max deviation between ts
        else:
            err_tan_alpha_phi = dev_tan_alpa_phi
        ### tan_alpha_theta
        tan_alpha_theta = fit[theta_sl]["tan_alpha"]
        err_tan_alpha_theta = fit[theta_sl]["err_tan_alpha"]
        ### do propagation of local sl fits to z = _muon_reco_z0 (z reco target coordinate)
        z0_reco = params._muon_reco_z0 # z0 target of muon in global coord frame
        err_z0_reco = 0
        x0_reco_sl, err_x0_reco_sl = {}, {}
        skip_this_combination = False
        for sl in params._dt_chamber["sls"].keys():
            x0_fit, tan_alpha_fit = fit[sl]["x0"], fit[sl]["tan_alpha"]
            err_x0_fit, err_tan_alpha_fit, corr_x0_tan_alpha_fit = fit[sl]["err_x0"], fit[sl]["err_tan_alpha"], fit[sl]["corr_x0_tan_alpha"]
            orient = "phi" if (sl in phi_sls) else "theta"
            x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
            # transform coordinates from local coordinate frame (with (0,0) at center (wire) position of cell ly=3, rel_wi=0) into global coordinate frame of dt chamber (used in params.py file)
            base_wi = fit[sl]["wi3"] # wi idx of ly=3 (base wi)
            if base_wi not in derived_params._dt_cell_coordinates[sl][3].keys():
                skip_this_combination = True
                continue
            z_wire = derived_params._sl_pattern_coordinates[3][0][3] # z_cell (wire position) of ly=3
            _coord_transform = [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ]
            # calculate muon track in coord frame
            x0_reco_sl[sl] = derived_params.f_x_muon(z=-_coord_transform[1]+z0_reco, x0=x0_fit, tan_alpha=tan_alpha_fit) + _coord_transform[0]
            err_x0_reco_sl[sl] = derived_params.err_f_x_muon(z=-_coord_transform[1]+z0_reco, x0=x0_fit, tan_alpha=tan_alpha_fit, err_x0=err_x0_fit, err_tan_alpha=err_tan_alpha_fit, corr_x0_tan_alpha=corr_x0_tan_alpha_fit)
        if skip_this_combination:
            continue
        # reject muon if xproj tolerance not met
        if np.abs(x0_reco_sl[phi_sl1] - x0_reco_sl[phi_sl2]) > params._muon_slphi_xproj_tolerance:
            continue
        counter_xproj += 1
        # reco x, y positions of muon in 2d slice of global coord frame
        ### x0 phi
        x0_reco_phi = np.mean([x0_reco_sl[sl] for sl in phi_sls])
        ### x0 phi uncertainty
        # calc deviaion
        max_x0_dev = np.abs(x0_reco_sl[phi_sls[0]] - x0_reco_sl[phi_sls[1]])
        # check if max compatible within all errors
        min_x0_phi_sigma = np.amin([fit[sl]["err_x0"] for sl in phi_sls])
        # if compatible do error on mean (1/sqrt(N))
        if max_x0_dev <= min_x0_phi_sigma:
            err_x0_reco_phi = np.sqrt(
                  (err_x0_reco_sl[phi_sl1]/2)**2
                + (err_x0_reco_sl[phi_sl2]/2)**2
            )
        # if not compatible do error as max deviation between ts
        else:
            err_x0_reco_phi = max_x0_dev
        ### x0 theta and uncertainty
        x0_reco_theta = x0_reco_sl[theta_sl]
        err_x0_reco_theta = err_x0_reco_sl[theta_sl]
        x0_reco, err_x0_reco = (x0_reco_phi, err_x0_reco_phi) if (params._orientation["phi"][0] == 0) else (x0_reco_theta, err_x0_reco_theta)
        y0_reco, err_y0_reco = (x0_reco_phi, err_x0_reco_phi) if (params._orientation["phi"][0] == 1) else (x0_reco_theta, err_x0_reco_theta)
        ### combine phi + theta planes to (x0, y0, theta, phi) muon in global coord system
        # the combined (x0_reco, y0_reco, z0_reco, theta_reco, phi_reco) is in global coord system at z0 = params._muon_reco_z0
        # calculate theta, phi from projection angles alpha_phi, alpha_theta
        tan_alpha_x, err_tan_alpha_x = (tan_alpha_phi, err_tan_alpha_phi) if (params._orientation["phi"][0] == 0) else (tan_alpha_theta, err_tan_alpha_theta) # proj on global x axis
        tan_alpha_y, err_tan_alpha_y = (tan_alpha_phi, err_tan_alpha_phi) if (params._orientation["phi"][0] == 1) else (tan_alpha_theta, err_tan_alpha_theta) # proj on global y axis
        # tan_alpha_x = tan_theta*cos_phi, tan_alpha_y = tan_theta*sin_phi
        # => phi = arctan(tan_alpha_x/tan_alpha_y), theta = arctan(tan_alpha_x/cos_phi)
        phi_reco_prelim = np.atan2( tan_alpha_y, tan_alpha_x ) #np.atan2(y/x) #np.arctan( tan_alpha_y / tan_alpha_x ) # np.atan2( tan_alpha_y, tan_alpha_x ) # use atan2 to cover full 360 degrees
        # make sure phi is in range [0, 2*np.pi]
        phi_periodicity = 2*np.pi
        phi_reco = phi_reco_prelim - phi_periodicity*(phi_reco_prelim//phi_periodicity)
        err_phi_reco = np.sqrt( # np.atan2(y/x): d atan2/dx = x/(x^2+y^2) and d atan2/dx = -y/(x^2+y^2)
              (-tan_alpha_x/(tan_alpha_y**2+tan_alpha_x**2))**2 * err_tan_alpha_x**2 # dy
            + (tan_alpha_y/(tan_alpha_y**2+tan_alpha_x**2))**2 * err_tan_alpha_y**2 # dx
        )
        theta_reco = np.arctan( tan_alpha_x / np.cos(phi_reco) ) #print( np.arctan( tan_alpha_x / np.cos(phi_reco) ) , np.arctan( tan_alpha_y / np.sin(phi_reco) ) )
        err_theta_reco = np.sqrt(
              ( (2*np.cos(phi_reco)) / (2*tan_alpha_x**2+np.cos(2*phi_reco)+1) )**2 * err_tan_alpha_x**2
            + ( (tan_alpha_x*np.sin(phi_reco)) / (tan_alpha_x**2+np.cos(phi_reco)**2) )**2 * err_phi_reco**2
        )
        ### combine t0 to muon arrival time ts (averaging)
        ts_reco = np.mean([fit[sl]["t0"] for sl in params._dt_chamber["sls"].keys()])
        ### ts uncertainty
        # check if max ts difference compatible within all errors
        max_ts_diff = np.amax([
            np.abs(fit[1]["t0"]-fit[2]["t0"]),
            np.abs(fit[1]["t0"]-fit[3]["t0"]),
            np.abs(fit[2]["t0"]-fit[3]["t0"]),
        ])
        min_ts_sigma = np.amin([fit[sl]["err_t0"] for sl in params._dt_chamber["sls"].keys()])
        # if compatible do error on mean (1/sqrt(N))
        if max_ts_diff <= min_ts_sigma:
            err_ts_reco = np.sqrt(np.sum([(fit[sl]["err_t0"]/3)**2 for sl in params._dt_chamber["sls"].keys()]))
        # if not compatible do error as max deviation between ts
        else:
            err_ts_reco = max_ts_diff
        ### combine muon_id of hits (if there is one from simulation)
        # raise error of muon_id of combined sl patters is not single value
        muon_id = fit[sl]["muon_id"]
        skip_this_combination = False
        for this_muon_id in [fit[sl]["muon_id"] for sl in params._dt_chamber["sls"].keys()]:
            if muon_id != this_muon_id:
                if verbose: print("Different muon_ids for the sl fits which should be combined to dt muon. Skip this combination...")
                skip_this_combination = True
        if skip_this_combination:
            continue
        ### store reco muon
        reco_muon_list.append({
            # reco values
            "x0":x0_reco, "y0":y0_reco, "z0":z0_reco, "theta":theta_reco, "phi":phi_reco, "ts":ts_reco, "muon_id":muon_id,
            "sl1_fit_group": last_fit_group[1]["idx"], "sl2_fit_group": last_fit_group[2]["idx"], "sl3_fit_group": last_fit_group[3]["idx"], # indices of fit groups used for this muon
            # errors
            "err_x0":err_x0_reco, "err_y0":err_y0_reco, "err_z0":err_z0_reco, "err_theta":err_theta_reco, "err_phi":err_phi_reco, "err_ts":err_ts_reco,
            # also extract sim muon keys (from one pattern since fine because have ensured that it is from same muon)
            "muon_ts": fit[1]["muon_ts"],
            "muon_phi": fit[1]["muon_phi"],
            "muon_theta": fit[1]["muon_theta"],
            "muon_x0": fit[1]["muon_x0"],
            "muon_y0": fit[1]["muon_y0"],
            "muon_z0": fit[1]["muon_z0"],
        })
        ### printing for debugging
        if verbose: print("------")
        for sl in params._dt_chamber["sls"].keys():
            if verbose:
                print(f"sl = {sl}:")
                print(f"  fit_group =", last_fit_group[sl])
                print(f"  fit =", fit[sl])
                for k in ["wi3", "t0", "tan_alpha", "x0"]:
                    print(f"  {k} = {fit[sl][k]}")
                print(f"x0_reco = {x0_reco_sl[sl]}")
                print(f"z0_reco = {z0_reco}")
        if verbose:
            print(f"muon:")
            print(f"  ( x0_reco , y0_reco , z0_reco ) = ( {x0_reco} +- {err_x0_reco} , {y0_reco} +- {err_y0_reco} , {z0_reco} +- {err_z0_reco} )")
            print(f"  ( phi_reco , theta_reco ) = ( {phi_reco} +- {err_phi_reco} , {theta_reco} +- {err_theta_reco} )")
            print(f"  ts_reco = {ts_reco} +- {err_ts_reco}")
        ### make sure no fit group is double counted, remove all 3 used fit groups from last_fit_group list...
        for sl in params._dt_chamber["sls"].keys():
            last_fit_group[sl] = None
        #print(np.arctan(fit[phi_sl1]["tan_alpha"])*180/np.pi , np.arctan(fit[phi_sl2]["tan_alpha"])*180/np.pi, np.arctan(fit[theta_sl]["tan_alpha"])*180/np.pi)
    ### store into reco_muons object
    # !!! for muon the name of the timestamp key is "ts" and not "t0"
    n_reco_muons = len(reco_muon_list)
    if not silent: print(f"Reconstructed {n_reco_muons} muons from {n_fits} SL patterns.")
    reco_muons = {k: np.full(n_reco_muons, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    for i in range(n_reco_muons):
        for k in params._muon_obj_keys.keys():
            reco_muons[k][i] = reco_muon_list[i][k]
    if not silent:
        print(f"muon reco cut flow:")
        print(f"duration: {duration} s")
        print(f"initial sl fit group rate: {counter_groups/duration} Hz")
        print(f"3 sl candidates:  {counter_3sl_candidates/duration} Hz")
        print(f"after n_fits condition:  {counter_n_fits/duration} Hz")
        print(f"after tgroup condition:  {counter_tgroup/duration} Hz")
        print(f"after chi2/ndf condition:  {counter_chi2/duration} Hz")
        print(f"after tan_alpha_condition:  {counter_tan_alpha/duration} Hz")
        print(f"after x_proj condition:  {counter_xproj/duration} Hz")
    return reco_muons

    """
    ## re-sort data
    fits = data_utils.sort_by_key(data=fits, sort_key="t0", silent=silent)
    if not silent: print(f"Combining {n_fits} fitted SL patterns to reconstruct muons...")
    # extract is of sls in phi & theta orientation
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]
    theta_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "theta"]
    # grouping by timestamp, check if the fitted t0 timestamps of the patterns are within given acceptance interval params._t0_acceptance_interval
    # if 2 phi patterns: combine phi patterns, check for spatial coincidence of projected x values onto other sl within params._xproj_acceptance_interval
    # combine the patterns theta + phi
    # calculate muon object (similar to the muon objects that one can simulate)
    # NOTE:
    # the algorithm can only cope one muon after another (strictly in order), not multiple muon fits simultaneously :(
    last_sl_pattern = {sl: None for sl in params._dt_chamber["sls"].keys()} # last sl pattern for all sls
    t0_ref = 0
    for i in tqdm(range(n_fits), disable=silent):
        ### fitted sl pattern grouping
        sl = fits["sl"][i]
        t0 = fits["t0"][i]
        # apply time correction given by sl time offset in params
        t0 = np.uint64(int(t0) - int(params._sl_time_offset[sl]))
        if t0_ref == 0: # if t0_ref was reset, take first timestamp t0 here as reference (can do this since dataset is ordered...)
            t0_ref = t0
        last_sl_pattern[sl] = {k: fits[k][i] for k in fits.keys()} # store current column
        # continue to "fill up" last_sl_pattern, if next hit also is within time window
        if i < n_fits-1: # only do it if there is a "next hit"
            t0_next = fits["t0"][i+1]
            if np.abs(t0_next - t0) <= params._t0_acceptance_interval:
                continue
        # if not: continue, the combination of collected sl fits starts
        # check for at least 1 phi + 1 theta pattern within t0 interval
        # if 2 phi patterns, also accept it
        phi_patterns = [last_sl_pattern[sl] for sl in params._dt_chamber["sls"].keys() if (params._dt_chamber["sls"][sl]["orient"] == "phi" and last_sl_pattern[sl] != None)]
        theta_patterns = [last_sl_pattern[sl] for sl in params._dt_chamber["sls"].keys() if (params._dt_chamber["sls"][sl]["orient"] == "theta" and last_sl_pattern[sl] != None)]
        # need to reset t0_ref, last_sl_pattern afterwards (for next iteration)
        last_sl_pattern = {sl: None for sl in params._dt_chamber["sls"].keys()} # last sl pattern for all sls
        t0_ref = 0
        ### muon reco
        n_phi_patterns, n_theta_patterns = len(phi_patterns), len(theta_patterns)
        ## check for at least 1 phi + 1 theta pattern, else discard and continue
        if(n_phi_patterns not in [1, 2]) or (n_theta_patterns not in [1]):
            continue
        if verbose: print("")
        candidate = False
        if np.abs(np.arctan(phi_patterns[0]["tan_alpha"])) < 0.1 and np.abs(np.arctan(theta_patterns[0]["tan_alpha"])) < 0.1:
            candidate = True
            if verbose: print(f"candidate for reco    theta_proj_phi = {np.arctan(phi_patterns[0]['tan_alpha'])}   theta_proj_theta = {np.arctan(theta_patterns[0]['tan_alpha'])}")
            #print(phi_patterns, theta_patterns)
        ## prepare coord trafo for each sl pattern (local sl pattern coord frame to global dt chamber coord frame)
        # for phi
        x_axis, y_axis = params._orientation["phi"][0], params._orientation["phi"][1]
        _coord_transform_phi_patterns = [] # [_coord_transform = [x_trafo, y_trafo] for j in range(n_phi_patterns)]
        _z_distance_phi_patterns = [] # relative distance between ref cell (cur sl, ly=3, rel_wi=0) and global z0 reference params._muon_reco_z0
        for j in range(n_phi_patterns):
            sl = phi_patterns[j]["sl"]
            base_wi = phi_patterns[j]["wi3"] # wi idx of ly=3 (base wi)
            _coord_transform_phi_patterns.append( [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ] )
            _z_distance_phi_patterns.append( params._muon_reco_z0 - derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] )
        # for theta
        x_axis, y_axis = params._orientation["theta"][0], params._orientation["theta"][1]
        _coord_transform_theta_patterns = [] # [_coord_transform[x_trafo, y_trafo] for j in range(n_phi_patterns)]
        _z_distance_theta_patterns = [] # relative distance between ref cell (cur sl, ly=3, rel_wi=0) and global z0 reference params._muon_reco_z0
        for j in range(n_theta_patterns):
            sl = theta_patterns[j]["sl"]
            base_wi = theta_patterns[j]["wi3"] # wi idx of ly=3 (base wi)
            _coord_transform_theta_patterns.append( [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ] )
            _z_distance_theta_patterns.append( params._muon_reco_z0 - derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] )
        ## combine (x0, tan_alpha) within phi plane, if > 1 phi pattern
        ## the resulting (x0_phi, z0_phi, tan_alpha_phi) is in global coord system at z0 = params._muon_reco_z0
        if n_phi_patterns == 1:
            tan_alpha_phi = phi_patterns[0]["tan_alpha"]
            x0_phi = derived_params.f_x_muon(z=_z_distance_phi_patterns[0], x0=phi_patterns[0]["x0"], tan_alpha=phi_patterns[0]["tan_alpha"]) + _coord_transform_phi_patterns[0][0]
            z0_phi = derived_params._sl_pattern_coordinates[3][0][3] + params._muon_reco_z0
        elif n_phi_patterns == 2:
            tan_alpha_phi, x0_phi, z0_phi = [], [], []
            for j in range(n_phi_patterns):
                tan_alpha_phi.append( phi_patterns[j]["tan_alpha"] )
                x0_phi.append( derived_params.f_x_muon(z=_z_distance_phi_patterns[j], x0=phi_patterns[j]["x0"], tan_alpha=phi_patterns[j]["tan_alpha"]) + _coord_transform_phi_patterns[j][0] )
                z0_phi.append( derived_params._sl_pattern_coordinates[3][0][3] + params._muon_reco_z0 )
            if verbose: print("phi", (x0_phi, z0_phi, tan_alpha_phi, ))
            # check if two are compatible, else skip this full group since unclear which one should be chosen...
            if np.abs(x0_phi[1]-x0_phi[0]) > params._xproj_acceptance_interval: # use xproj threshold (max distance on proj x axis for specified z0) to discriminate
                continue
            # combine if compatible, by averaging x0 and slope (tan) values
            x0_phi = np.mean(x0_phi)
            tan_alpha_phi = np.mean(tan_alpha_phi)
        else:
            raise Exception(f"Wrong number of phi patterns ({n_phi_patterns}). Expect value in [1, 2].")
        if verbose: print("phi comb", (x0_phi, z0_phi, tan_alpha_phi))
        ## the resulting (x0_theta, z0_theta, tan_alpha_theta) is in global coord system at z0 = params._muon_reco_z0
        if n_theta_patterns == 1:
            tan_alpha_theta = theta_patterns[0]["tan_alpha"]
            x0_theta = derived_params.f_x_muon(z=_z_distance_theta_patterns[0], x0=theta_patterns[0]["x0"], tan_alpha=theta_patterns[0]["tan_alpha"]) + _coord_transform_theta_patterns[0][0]
            z0_theta = derived_params._sl_pattern_coordinates[3][0][3] + params._muon_reco_z0
        else:
            raise Exception(f"Wrong number of theta patterns ({n_theta_patterns}). Expect value in [1].")
        if verbose: print("theta comb", (x0_theta, z0_theta, tan_alpha_theta, ))
        if verbose and candidate: print("successful reco")
        ## combine phi + theta planes to (x0, y0, theta, phi) muon in global coord system
        # the combined (x0_reco, y0_reco, z0_reco, theta_reco, phi_reco) is in global coord system at z0 = params._muon_reco_z0
        # calculate theta, phi from projection angles alpha_phi, alpha_theta
        tan_alpha_x = tan_alpha_phi if (params._orientation["phi"][0] == 0) else tan_alpha_theta # proj on global x axis
        tan_alpha_y = tan_alpha_theta if (params._orientation["phi"][0] == 0) else tan_alpha_phi # proj on global y axis
        # tan_alpha_x = tan_theta*cos_phi, tan_alpha_y = tan_theta*sin_phi
        # => phi = arctan(tan_alpha_x/tan_alpha_y), theta = arctan(tan_alpha_x/cos_phi)
        phi_reco_prelim = np.atan2( tan_alpha_y, tan_alpha_x ) #np.arctan( tan_alpha_y / tan_alpha_x ) # np.atan2( tan_alpha_y, tan_alpha_x ) # use atan2 to cover full 360 degrees
        # make sure phi is in range [0, 2*np.pi]
        phi_periodicity = 2*np.pi
        phi_reco = phi_reco_prelim - phi_periodicity*(phi_reco_prelim//phi_periodicity)
        theta_reco = np.arctan( tan_alpha_x / np.cos(phi_reco) )
        x0_reco = x0_phi if (params._orientation["phi"][0] == 0) else x0_theta # proj on global x axis
        y0_reco = x0_theta if (params._orientation["phi"][0] == 0) else x0_phi # proj on global y axis
        z0_reco = params._muon_reco_z0
        ### combine t0 to muon arrival time (averaging)
        t0_reco = np.uint64(np.round(np.mean([int(phi_patterns[j]["t0"]) for j in range(n_phi_patterns)] + [int(theta_patterns[j]["t0"]) for j in range(n_theta_patterns)]),0))
        ### combine muon_id of hits (if there is one from simulation)
        # raise error of muon_id of combined sl patters is not single value
        muon_id = phi_patterns[0]["muon_id"]
        skip_this_combination = False
        for this_muon_id in [phi_patterns[j]["muon_id"] for j in range(n_phi_patterns)] + [theta_patterns[j]["muon_id"] for j in range(n_theta_patterns)]:
            if muon_id != this_muon_id:
                if verbose: print("Different muon_ids for the sl fits which should be combined to dt muon. Skip this combination...")
                #raise Exception(f"Expect hits of same muon_id {muon_id}, not {this_muon_id}.")
                skip_this_combination = True
        if skip_this_combination:
            continue
        # store reco muon
        if verbose: print("muon reco", (x0_reco, y0_reco, z0_reco, theta_reco, phi_reco, t0_reco, muon_id))
        reco_muon_list.append({
            # reco values
            "x0":x0_reco, "y0":y0_reco, "z0":z0_reco, "theta":theta_reco, "phi":phi_reco, "ts":t0_reco, "muon_id":muon_id,
            # also extract sim muon keys (from one pattern since fine because have ensured that it is from same muon)
            "muon_ts": phi_patterns[0]["muon_ts"],
            "muon_phi": phi_patterns[0]["muon_phi"],
            "muon_theta": phi_patterns[0]["muon_theta"],
            "muon_x0": phi_patterns[0]["muon_x0"],
            "muon_y0": phi_patterns[0]["muon_y0"],
            "muon_z0": phi_patterns[0]["muon_z0"],
        })
        # !!! for muon the name of the timestamp key is "ts" and not "t0"
    n_reco_muons = len(reco_muon_list)
    if not silent: print(f"Reconstructed {n_reco_muons} muons from {n_fits} SL patterns.")
    reco_muons = {k: np.full(n_reco_muons, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    for i in range(n_reco_muons):
        for k in params._muon_obj_keys.keys():
            reco_muons[k][i] = reco_muon_list[i][k]
    #"""

### extract timestamps of testpulse hits for full readout system
# fe connector granularity
def analyze_testpulses(hits, *, rel_thres=0.2, plot_hists=False, correct_for_offsets=True, silent=False):
    tp_timing = {}
    for sl in params._dt_chamber["sls"].keys():
        tp_timing[sl] = {}
        if not silent: print(f"Analyzing testpulses for SL {sl}.")
        for fe_id in tqdm(derived_params._dt_fe_id_remap_table[sl], disable=silent):
            tp_timing[sl][int(fe_id)] = {}
            ch_ts_mean, ch_ts_err = 0, 0 # default values
            # select hits of one channel
            fec_hits = data_utils.cut_data(data=hits, conditions=[("sl","==",sl),("fe_id","==",fe_id)], silent=True)
            if data_utils.length(fec_hits) > 0:
                # calculate histogram of hit timing (bin width = 1 ts unit)
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=fec_hits, key="ts_orbit", bin_centers="step1", silent=True)      
                # select first peak of histogram (with lowest ts), the higher ts hits are due to ringing of the testpulse circuit
                peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=rel_thres) # 20% of max amplitude for peak
                if len(peak_indices) > 0:
                    sel_peak_indices = peak_indices[0] # first peak
                    hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
                    err_hists_peak = np.sqrt(hists_peak)
                    err_centers_peak = np.full( len(centers_peak), 1/np.sqrt(12) )
                    # calculate peak position (weighted mean)
                    ch_ts_mean, ch_ts_err = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
                # plot hist if desired
                plot_hists = True if fe_id == 0 else False ### HARDCODED FOR NOW
                if plot_hists:
                    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=fec_hits, key="ts_orbit", bin_centers=np.arange(0,4000+1), silent=True)
                    xlabel = params._key_symbols["ts_orbit"]
                    xlabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
                    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, show=True, title=f"Testpulse timing (SL {sl}, FEC ID {fe_id})")  
            # store result
            tp_timing[sl][int(fe_id)] = {"tp_ts_mean": ch_ts_mean, "tp_ts_err": ch_ts_err}    
    ## correct for offsets if desired
    if correct_for_offsets:
        if not silent: print(f"Correcting global/constant offsets between superlayers and fe connectors...")
        for sl in params._dt_chamber["sls"].keys():
            for fe_id in tqdm(derived_params._dt_fe_id_remap_table[sl], disable=silent):
                ch_ts_mean, ch_ts_err = tp_timing[sl][int(fe_id)]["tp_ts_mean"], tp_timing[sl][int(fe_id)]["tp_ts_err"]
                fe_name = params._fe_idx_list[ derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"] ]
                ch_ts_mean_corr = ch_ts_mean - params._tp_time_offset[sl][fe_name]
                ch_ts_err_corr = np.sqrt(params._tp_time_offset_err**2 + ch_ts_err**2)
                ch_ts_err_corr = np.sqrt(params._tp_time_offset_err**2 + ch_ts_err**2)
                tp_timing[sl][int(fe_id)] = {"tp_ts_mean": ch_ts_mean_corr, "tp_ts_err": ch_ts_err_corr}    
    return tp_timing

### extract timestamps of testpulse hits for full readout system
# single channel (wire) granularity
def analyze_testpulses_per_wire(hits, *, rel_thres=0.2, plot_hists=False, correct_for_offsets=True, silent=False):
    tp_timing = {}
    for sl in params._dt_chamber["sls"].keys():
        tp_timing[sl] = {}
        for ly in params._dt_chamber["sls"][sl]["lys"].keys():
            if not silent: print(f"Analyzing testpulses for SL {sl} LY {ly}.")
            tp_timing[sl][ly] = {}
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                tp_timing[sl][ly][wi] = {}
                ch_ts_mean, ch_ts_err = 0, 0 # default values
                # select hits of one channel
                fec_hits = data_utils.cut_data(data=hits, conditions=[("sl","==",sl),("ly","==",ly), ("wi","==",wi)], silent=True)
                if data_utils.length(fec_hits) > 0:
                    # calculate histogram of hit timing (bin width = 1 ts unit)
                    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=fec_hits, key="ts_orbit", bin_centers="step1", silent=True)      
                    # select first peak of histogram (with lowest ts), the higher ts hits are due to ringing of the testpulse circuit
                    peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=rel_thres) # 20% of max amplitude for peak
                    if len(peak_indices) > 0:
                        sel_peak_indices = peak_indices[0] # first peak
                        hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
                        err_hists_peak = np.sqrt(hists_peak)
                        err_centers_peak = np.full( len(centers_peak), 1/np.sqrt(12) )
                        # calculate peak position (weighted mean)
                        ch_ts_mean, ch_ts_err = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
                    # plot hist if desired
                    plot_hists = True if wi == 0 else False ### HARDCODED FOR NOW
                    if plot_hists:
                        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=fec_hits, key="ts_orbit", bin_centers=np.arange(0,4000+1), silent=True)
                        xlabel = params._key_symbols["ts_orbit"]
                        xlabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
                        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, show=True, title=f"Testpulse timing (SL {sl}, LY {ly}, WI {wi})")  
                # store result
                tp_timing[sl][ly][wi] = {"tp_ts_mean": ch_ts_mean, "tp_ts_err": ch_ts_err}    
    ## correct for offsets if desired
    if correct_for_offsets:
        if not silent: print(f"Correcting global/constant offsets between superlayers and fe connectors...")
        for sl in params._dt_chamber["sls"].keys():
            for ly in params._dt_chamber["sls"][sl]["lys"].keys():
                for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                    ch_ts_mean, ch_ts_err = tp_timing[sl][ly][wi]["tp_ts_mean"], tp_timing[sl][ly][wi]["tp_ts_err"]
                    fe_name = params._fe_idx_list[ derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"] ]
                    ch_ts_mean_corr = ch_ts_mean - params._tp_time_offset[sl][fe_name]
                    ch_ts_err_corr = np.sqrt(params._tp_time_offset_err**2 + ch_ts_err**2)
                    tp_timing[sl][ly][wi] = {"tp_ts_mean": ch_ts_mean_corr, "tp_ts_err": ch_ts_err_corr}  
    return tp_timing

### calculate corrections to channel timing from the testpulse timing of the channels
# do not globally align the superlayers, but only the channels within the superlayer...
# align channels to average time of superlayer
# corrections in dt_tp_corrections should be applied with a plus sign
#   as follows: ts(wi)_corrected = ts(wi)_uncorrected + correction(wi)
#   ( since correction(wi) = ts_target=ts(wi)_corrected - ts(wi)_uncorrected )
def calculate_sl_tp_corrections(tp_timing, *, silent=False):
    dt_tp_corrections = {}
    if not silent: print(f"Calculating corrections to the timestamps of all channels based on the testpulse response time - separately for each SL...")
    for sl in params._dt_chamber["sls"].keys():
        dt_tp_corrections[sl] = {}
        # do this step independently for each superlayer (therefore only local timing alignment within superlayer will be acheived when applying this correction later to data)
        # calculate target timing: mean of sl timing of testpulses
        sl_ts_target = 0
        n_ts_chs = 0
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            dt_tp_corrections[sl][ly] = {}
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                ch_ts_mean, ch_ts_err = tp_timing[sl][ly][wi]["tp_ts_mean"], tp_timing[sl][ly][wi]["tp_ts_err"]
                if ch_ts_mean > 0:
                    n_ts_chs += 1
                    sl_ts_target += ch_ts_mean
        sl_ts_target /= n_ts_chs # average timing
        # calculate corrections for individual channels
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                ch_ts_mean, ch_ts_err = tp_timing[sl][ly][wi]["tp_ts_mean"], tp_timing[sl][ly][wi]["tp_ts_err"]
                if ch_ts_mean > 0:
                    dt_tp_corrections[sl][ly][wi] = {"ts_corr": sl_ts_target - ch_ts_mean, "err_ts_corr": ch_ts_err}
                else:
                    dt_tp_corrections[sl][ly][wi] = {"ts_corr": 0, "err_ts_corr": 0}
    return dt_tp_corrections

### calculate corrections to channel timing from the testpulse timing of the channels
# align channels to average time of full chamber
# corrections in dt_tp_corrections should be applied with a plus sign
#   as follows: ts(wi)_corrected = ts(wi)_uncorrected + correction(wi)
#   ( since correction(wi) = ts_target=ts(wi)_corrected - ts(wi)_uncorrected )
def calculate_chamber_tp_corrections(tp_timing, *, silent=False):
    dt_tp_corrections = {}
    if not silent: print(f"Calculating corrections to the timestamps of all channels based on the testpulse response time - for full chamber...")
    # do this step for full chamber
    # calculate target timing: mean of timing of testpulses
    ts_target = 0
    n_ts_chs = 0
    for sl in derived_params._dt_inverted_remap_table.keys():
        dt_tp_corrections[sl] = {}
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            dt_tp_corrections[sl][ly] = {}
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                ch_ts_mean, ch_ts_err = tp_timing[sl][ly][wi]["tp_ts_mean"], tp_timing[sl][ly][wi]["tp_ts_err"]
                if ch_ts_mean > 0:
                    n_ts_chs += 1
                    ts_target += ch_ts_mean
    ts_target /= n_ts_chs # average timing
    # calculate corrections for individual channels
    for sl in derived_params._dt_inverted_remap_table.keys():
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                ch_ts_mean, ch_ts_err = tp_timing[sl][ly][wi]["tp_ts_mean"], tp_timing[sl][ly][wi]["tp_ts_err"]
                if ch_ts_mean > 0:
                    dt_tp_corrections[sl][ly][wi] = {"ts_corr": ts_target - ch_ts_mean, "err_ts_corr": ch_ts_err}
                else:
                    dt_tp_corrections[sl][ly][wi] = {"ts_corr": 0, "err_ts_corr": 0}
    return dt_tp_corrections

### add noise hits to dt cells, separately for all channels
# ref_cell_noise_rate = cell noise rate in Hz
# ts_range = [ts_min, ts_max] where noise should be added
def add_noise(hits, *, ts_range, ref_cell_noise_rate, silent=False):
    t_start = ts_range[0]
    t_sim = ts_range[1]-ts_range[0]
    noise_hit_list = []
    for sl in derived_params._dt_inverted_remap_table.keys():
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            if not silent: print(f"  Progress: SL {sl}, LY {ly}...")
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1), disable=silent):
                noise_rate = ref_cell_noise_rate * 0.78e-9 # 1/tu
                noise_lambda = noise_rate * t_sim # expected muon count in simulation time
                # number of muons in time interval is poisson distributed
                n_noise = np.random.poisson(lam = noise_lambda)
                # simulate: for poisson, the time between events is exponentially distributed (!)
                inter_arrival_times = np.random.exponential(1.0 / noise_rate, n_noise) # in tu
                # generate muon timestamps from time differences between muon events
                noise_ts = t_start + np.cumsum(inter_arrival_times) # timestamp units
                ### generate dt_hits subset object with only noise hits
                noise_hits = {k: np.full(n_noise, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_noise, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_noise, 0, dtype=v) for k,v in params._dt_other_keys.items()} 
                for i in range(n_noise):
                    # copy existing keys
                    noise_hits["sl"][i] = sl
                    noise_hits["ly"][i] = ly
                    noise_hits["wi"][i] = wi
                    ts = noise_ts[i]
                    noise_hits["ts"][i] = ts
                    (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(ts)
                    noise_hits["oc"][i], noise_hits["bx"][i], noise_hits["tdc"][i] = oc, bx, tdc
                    # default values for other things
                    for k in ["dt", "dd", "muon_id", "hit_lat"]:
                        noise_hits[k][i] = 0
                    # map back htg parameters
                    for k in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
                        noise_hits[k][i] = derived_params._dt_inverted_remap_table[sl][ly][wi][k]
                # store subset of noise hits for this cell
                noise_hit_list.append(noise_hits)
    ### merge noise of all cells and previous hits
    noise_hit_list.append( copy.deepcopy(hits) )
    hits = data_utils.merge_dataset(split_data=noise_hit_list)
    ### sort hits by timestamp
    hits = timestamp_utils.sort_by_timestamp(hits=hits)
    return hits

### add secondary hits which follow the "real" dt hits
# physics: delta ray or photo-inization
# at a given probability, a second hit occurs
# it will be generated in a uniformly distributed time window which can be specified
def add_secondary_hits(hits, *, secondary_hit_window, secondary_hit_probability, silent=False):
    n_hits = data_utils.length(hits)
    t_sec_min, t_sec_max = secondary_hit_window
    secondary_hit_list = []
    for i in tqdm(range(n_hits), disable=silent):
        sl, ly, wi = hits["sl"][i], hits["ly"][i], hits["wi"][i]
        primary_ts = hits["ts"][i]
        # uniform probability sampling if secondary hit should be generated
        if np.random.uniform(low=0, high=1) < secondary_hit_probability:
            # uniform probability sampling for timing of secondary hit within window
            secondary_ts = primary_ts + np.random.uniform(low=t_sec_min, high=t_sec_max)
            secondary_hit_list.append([sl, ly, wi, secondary_ts])
    n_secondaries = len(secondary_hit_list)
    ### generate dt_hits subset object with only noise hits
    secondary_hits = {k: np.full(n_secondaries, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_secondaries, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_secondaries, 0, dtype=v) for k,v in params._dt_other_keys.items()} 
    for i in range(n_secondaries):
        # copy existing keys
        secondary_hits["sl"][i] = secondary_hit_list[i][0]
        secondary_hits["ly"][i] = secondary_hit_list[i][1]
        secondary_hits["wi"][i] = secondary_hit_list[i][2]
        ts = secondary_hit_list[i][3]
        secondary_hits["ts"][i] = ts
        (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(ts)
        secondary_hits["oc"][i], secondary_hits["bx"][i], secondary_hits["tdc"][i] = oc, bx, tdc
        # default values for other things
        for k in ["dt", "dd", "muon_id", "hit_lat"]:
            secondary_hits[k][i] = 0
        # map back htg parameters
        for k in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
            secondary_hits[k][i] = derived_params._dt_inverted_remap_table[sl][ly][wi][k]
    ### merge secondary hits of all cells and previous hits
    merge_list = [hits, secondary_hits]
    hits = data_utils.merge_dataset(split_data=merge_list)
    ### sort hits by timestamp
    hits = timestamp_utils.sort_by_timestamp(hits=hits)
    return hits





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
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="wi3")
    return sl_patterns

### create empty chamber_data object
def _chamber_data(default={"color": params._color_info["cell"][None], "text": ""}):
    chamber_data = {}
    for sl in params._dt_chamber["sls"].keys():
        chamber_data[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            chamber_data[sl][ly] = {}
            for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
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
        ts = np.array([patterns[f"ts{ly}"][i] for ly in range(4)], dtype=params._ts_float_type) # y values for fit => timestamps for hits of each layer
        err_ts = np.full(4, params._err_ts, dtype=np.float64) # ts uncertainty
        t0_start = ts[3] # assume ts of ly=3 rel_wi=0 (reference cell) as t0 starting point
        x0_start = derived_params._sl_pattern_coordinates[3][0][2] # center of ly=3 rel_wi=0 (reference cell)
        tan_alpha_start = 0 # assume straight down muon as start
        p0 = [t0_start, x0_start, tan_alpha_start] # fit start values
        # define parameter bounds
        p_bounds = [
            (np.uint64(t0_start-np.amin([t0_start, params._dt_sl_patterns_ts_window])), derived_params._sl_pattern_coordinates[3][0][0][0], -np.inf), # lower limit for (t0, x0, tan_alpha)
            (np.uint64(t0_start+params._dt_sl_patterns_ts_window), derived_params._sl_pattern_coordinates[3][0][0][1], np.inf), # upper limit for (t0, x0, tan_alpha)
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
            chi2ndf = np.sum((f_ts_fit_wparams(x_cell, t0_fit, x0_fit, tan_alpha_fit) - ts)**2 / err_ts**2) / ndf
            lat_fits.append({"laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "chi2/ndf": chi2ndf})
            if chi2ndf == np.inf: # penalize inf chi2 with high value
                chi2ndf = np.iinfo(np.float64).max
            lat_chi2.append(np.float64(chi2ndf))
            if verbose: print("  Fitting:",{"pattern_id": i, "pattern_name": pat_name, "laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "bounds": p_bounds, "chi2/ndf": chi2ndf})
        # round chi2 value to given fixed digits
        for j in range(len(lat_chi2)):
            lat_chi2[j] = float('{:0.3e}'.format(lat_chi2[j])) # round to 4 significant digits in total
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
        # NOTE:
        # if one wants to use also the +-d patterns, one might need to keep several fit results since there are possibilities of ambiguities between rrll and llrr
        # but if not using the +-d pattern, do not care :)
    return sl_fits

### combine fitted sl patterns for full chamber, generate muon object as output
# only works if at least one phi & theta pattern exists
def reco_muons_from_sl_fits(fits, *, silent=False, verbose=False):
    reco_muon_list = []
    # sort fits by t0 value
    fits = data_utils.sort_by_key(data=fits, sort_key="t0")
    n_fits = len(fits["t0"])
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
    for i in tqdm(range(n_fits)):
        ### fitted sl pattern grouping
        sl = fits["sl"][i]
        t0 = fits["t0"][i]
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
        if n_phi_patterns == 0 or n_theta_patterns == 0:
            continue
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
            if verbose: print("phi", (x0_phi, z0_phi, tan_alpha_phi))
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
        if verbose: print("theta comb", (x0_theta, z0_theta, tan_alpha_theta))
        ## combine phi + theta planes to (x0, y0, theta, phi) muon in global coord system
        # the combined (x0_reco, y0_reco, z0_reco, theta_reco, phi_reco) is in global coord system at z0 = params._muon_reco_z0
        # calculate theta, phi from projection angles alpha_phi, alpha_theta
        tan_alpha_x = tan_alpha_phi if (params._orientation["phi"][0] == 0) else tan_alpha_theta # proj on global x axis
        tan_alpha_y = tan_alpha_theta if (params._orientation["phi"][0] == 0) else tan_alpha_phi # proj on global y axis
        # tan_alpha_x = tan_theta*cos_phi, tan_alpha_y = tan_theta*sin_phi
        # => phi = arctan(tan_alpha_x/tan_alpha_y), theta = arctan(tan_alpha_x/cos_phi)
        phi_reco_prelim = np.atan2( tan_alpha_y, tan_alpha_x ) # use atan2 to cover full 360 degrees
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
        for this_muon_id in [phi_patterns[j]["muon_id"] for j in range(n_phi_patterns)] + [theta_patterns[j]["muon_id"] for j in range(n_theta_patterns)]:
            if muon_id != this_muon_id:
                raise Exception(f"Expect hits of same muon_id {muon_id}, not {this_muon_id}.")
        if verbose: print("muon reco", (x0_reco, y0_reco, z0_reco, theta_reco, phi_reco, t0_reco, muon_id))
        reco_muon_list.append({"x0":x0_reco, "y0":y0_reco, "z0":z0_reco, "theta":theta_reco, "phi":phi_reco, "ts":t0_reco, "muon_id":muon_id})
        # !!! for muon the name of the timestamp key is "ts" and not "t0"
    n_reco_muons = len(reco_muon_list)
    if not silent: print(f"Reconstructed {n_reco_muons} muons from {n_fits} SL patterns.")
    reco_muons = {k: np.full(n_reco_muons, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    for i in range(n_reco_muons):
        for k in params._muon_obj_keys.keys():
            reco_muons[k][i] = reco_muon_list[i][k]
    return reco_muons

### calculate dt chamber hits caused by muons
# simply propagate it to all layers of the chamber
# returns dt hits with keys {ts = ts of muon + drift time, sl, ly, wi}
# dt hits are being sorted by ts value of hits
# if noise_ampl > 0: add gaussian noise to drift distance (in unit of mm)
def hits_from_muons(muons, *, silent=False, noise_ampl=0):
    dt_hit_list = []
    n_muons = len(muons["x0"])
    lat_dict = {True: 1, False: -1} # laterality dict: -1: left of wire (l), +1: right of wire (r)
    if not silent: print(f"Calculating DT hits by {n_muons} muons...")
    for sl in params._dt_chamber["sls"].keys():
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            z_wi_idx = 0 # all wis have same z therefore save some time here
            z_pos = derived_params._dt_cell_coordinates[sl][ly][z_wi_idx][5] # use center z position (idx 5) of each layer
            (x,y,z) = muon_utils.propagate_muons(muons=muons, z=z_pos) # propagate all muons together
            if not silent: print(f"  Progress: SL {sl}, LY {ly}...")
            for wi in tqdm(range(params._dt_chamber["sls"][sl]["n_wis"])):
                # check for all muons separately
                for i in range(n_muons):
                    # check if muon propagated inside of x and y range of cell, use >= but < to suppress double hits
                    if (x[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][0][0] and x[i] < derived_params._dt_cell_coordinates[sl][ly][wi][0][1]) and (y[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][1][0] and y[i] < derived_params._dt_cell_coordinates[sl][ly][wi][1][1]):
                        # calculate drift distance
                        hit_coord = x[i] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else y[i]
                        wire_coord = derived_params._dt_cell_coordinates[sl][ly][wi][3] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else derived_params._dt_cell_coordinates[sl][ly][wi][4]
                        noise = 0
                        if noise_ampl > 0:
                            noise = np.random.normal(loc=0, scale=1) * noise_ampl # gaussian noise with sigma = noise_ampl, applied on drift distance in unit mm
                        drift_distance = np.float64(np.clip(np.abs(hit_coord-wire_coord+noise), a_min=0, a_max=params._cell_width/2)) # in mm
                        drift_time = np.uint64(np.round(drift_distance / derived_params._drift_velocity_mm_per_timestamp, 0)) # in timestamp units, cast to int value
                        muon_ts = muons["ts"][i]
                        hit_ts = np.uint64(muon_ts + drift_time) # hit timestamp = muon timestamp + drift time
                        # determine laterality of hit: -1 if left of wire, +1 if right of wire
                        laterality = lat_dict[x[i] >= wire_coord] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else lat_dict[y[i] >= wire_coord]
                        # store this hit
                        dt_hit_list.append({"muon_ts": muon_ts, "sl": sl, "ly": ly, "wi": wi, "dd": drift_distance, "dt": drift_time, "hit_ts": hit_ts, "muon_id": i, "hit_lat": laterality})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(dt_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    if not silent: print(f"Adding all keys to calculated {n_hits} DT hits...")
    dt_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_other_keys.items()} 
    for i in range(n_hits):
        # copy existing keys
        for k in ["sl", "ly", "wi", "muon_ts", "dt", "dd", "muon_id", "hit_lat"]:
            dt_hits[k][i] = dt_hit_list[i][k]
        # map back htg timestamp from drift time
        hit_ts = dt_hit_list[i]["hit_ts"]
        dt_hits["ts"][i] = hit_ts
        (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(hit_ts)
        dt_hits["oc"][i], dt_hits["bx"][i], dt_hits["tdc"][i] = oc, bx, tdc
        # map back htg parameters
        sl, ly, wi = dt_hit_list[i]["sl"], dt_hit_list[i]["ly"], dt_hit_list[i]["wi"]
        for k in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
            dt_hits[k][i] = derived_params._dt_inverted_remap_table[sl][ly][wi][k]
    # sort hits by their timestamp value
    dt_hits = timestamp_utils.sort_by_timestamp(hits=dt_hits)
    return dt_hits



###########################################
### SCINTILLATOR-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

import analysis_tools.utils.data_utils as data_utils
import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.muon_utils as muon_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### extract scintillator hits from hit data
# cut away all hit data not from scintillator
# add scintillator specific keys to hits
# take information about this mapping from params.py
def extract_scint_hits(hits, *, silent=False):
    tmp_hits = copy.deepcopy(hits)
    n_hits = len(tmp_hits["ch"])
    if not silent: print(f"Extract scintillator hits from {n_hits} total hits...")
    # calculate mask to apply to cut away all hits not belonging to dt chamber (wrong ro_ch or invalid ch)
    scint_mask = np.full(n_hits, False, dtype=np.bool)
    for ro_ch in derived_params._scint_ro_chs:
        tmp_mask = np.ma.isin(tmp_hits["ro_ch"], [ro_ch])
        tmp_mask &= np.ma.isin(tmp_hits["ch"], derived_params._scint_chs_by_ro_ch[ro_ch])
        scint_mask |= tmp_mask
    # apply mask
    for k in tmp_hits.keys():
        tmp_hits[k] = tmp_hits[k][scint_mask]
    n_scint_hits = len(tmp_hits["ch"])
    if not silent: print(f"Cut flow: {n_scint_hits}/{n_hits} = {n_scint_hits/n_hits}")
    if not silent: print(f"Found {n_scint_hits} scintillator hits. Adding scintillator specific keys...")
    # add specific scint keys
    tmp_hits |= {k: np.full(n_scint_hits, 0, dtype=v) for k,v in params._scint_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._scint_other_keys.items()} 
    for i in tqdm(range(n_scint_hits), disable=silent):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in derived_params._scint_keys:
            tmp_hits[k][i] = derived_params._scint_remap_table[ro_ch][ch][k]
    # add timestamp and sort by timestamp
    tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    return tmp_hits

### create empty scint_data object
def _scint_data(default={"color": params._color_info["cell"][None], "text": ""}):
    scint_data = {}
    for ly in range(params._scintillator["n_lys"]):
        scint_data[ly] = {}
        for st in range(params._scintillator["lys"][ly]["n_sts"]):
            scint_data[ly][st] = copy.deepcopy(default)
    return scint_data

### calculate scintillator hits caused by muons
# simply propagate it to all layers of the scintillator
# returns sciunt hits with keys {ts = ts of muon, ly, st}
# hits are being sorted by ts value of hits
def hits_from_muons(muons, *, silent=False):
    scint_hit_list = []
    n_muons = len(muons["x0"])
    if not silent: print(f"Calculating scintillator hits by {n_muons} muons...")
    for ly in params._scintillator["lys"].keys():
        z_st_idx = 0 # all sts have same z therefore save some time here
        z_pos = derived_params._scintillator_strip_coordinates[ly][z_st_idx][5] # use center z position (idx 5) of each layer
        (x,y,z) = muon_utils.propagate_muons(muons=muons, z=z_pos) # propagate all muons together
        if not silent: print(f"  Progress: LY {ly}...")
        for st in tqdm(range(params._scintillator["lys"][ly]["n_sts"]), disable=silent):
            # check for all muons separately
            for i in range(n_muons):
                # check if muon propagated inside of x and y range of cell, use >= but < to suppress double hits
                if (x[i] >= derived_params._scintillator_strip_coordinates[ly][st][0][0] and x[i] < derived_params._scintillator_strip_coordinates[ly][st][0][1]) and (y[i] >= derived_params._scintillator_strip_coordinates[ly][st][1][0] and y[i] < derived_params._scintillator_strip_coordinates[ly][st][1][1]):
                    # calculate drift distance
                    hit_coord = x[i] if (params._scintillator["lys"][ly]["orient"] == "phi") else y[i]
                    xleft_strip_coord = derived_params._scintillator_strip_coordinates[ly][st][0][0] if (params._scintillator["lys"][ly]["orient"] == "phi") else derived_params._scintillator_strip_coordinates[ly][st][1][0]
                    xhit = np.float64(np.clip(np.abs(hit_coord-xleft_strip_coord), a_min=0, a_max=params._strip_width)) # in mm
                    # drift distance does not make much sense in this context, but want to store coordinate of hit. with dd one can calculate it: x_hit = x_left(smaller x coord) + xhit
                    muon_ts = muons["ts"][i]
                    hit_ts = np.uint64(muon_ts + params._scintillator_hit_delay) # assume constant delay: hit timestamp = muon timestamp + scint delay
                    # store this hit
                    scint_hit_list.append({"muon_ts": muon_ts, "ly": ly, "st": st, "xhit": xhit, "hit_ts": hit_ts, "muon_id": i})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(scint_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    if not silent: print(f"Adding all keys to calculated {n_hits} scintillator hits...")
    scint_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._scint_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._scint_other_keys.items()} 
    for i in range(n_hits):
        # copy existing keys
        for k in ["ly", "st", "muon_ts", "xhit", "muon_id"]:
            scint_hits[k][i] = scint_hit_list[i][k]
        # map back htg timestamp from drift time
        hit_ts = scint_hit_list[i]["hit_ts"]
        scint_hits["ts"][i] = hit_ts
        (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(hit_ts)
        scint_hits["oc"][i], scint_hits["bx"][i], scint_hits["tdc"][i] = oc, bx, tdc
        # map back htg parameters
        ly, st = scint_hit_list[i]["ly"], scint_hit_list[i]["st"]
        for k in ["ro_ch", "ch", "ch_id"]:
            scint_hits[k][i] = derived_params._scint_inverted_remap_table[ly][st][k]
    # sort hits by their timestamp value
    scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    return scint_hits

### group scintillator hits and then reco area of muon hit from scintillator hits
# group together if hits of all layers are close enough in time
# return x and y interval and z coordinate (mean of z pos of layers) that the hit could have been (by assessing all layers hit)
def reco_muon_area_from_hits(hits, *, silent=False, verbose=False):
    reco_muon_area_list = []
    # sort hits by timestamp
    hits = data_utils.sort_by_key(data=hits, sort_key="ts")
    n_hits = len(hits["ts"])
    if not silent: print(f"Combining {n_hits} scintillator hits to reconstruct muons...")
    # extract sls in phi & theta orientation
    phi_lys = [ly for ly in params._scintillator["lys"].keys() if params._scintillator["lys"][ly]["orient"] == "phi"]
    theta_lys = [ly for ly in params._scintillator["lys"].keys() if params._scintillator["lys"][ly]["orient"] == "theta"]
    # grouping by timestamp, check if the ts timestamps of the htis are within given acceptance interval params._scintillator_ts_acceptance_interval
    # if multiple hits per orientation, combine them
    # combine the hits in theta + phi
    # calculate muon area object (xrange, yrange, z, ts) where muon should have been
    # NOTE:
    # the algorithm can only cope one muon after another (strictly in order), not multiple muon fits simultaneously :(
    last_scint_hits = {ly: None for ly in params._scintillator["lys"].keys()} # last sl pattern for all sls
    ts_ref = 0
    for i in tqdm(range(n_hits), disable=silent):
        ### fitted sl pattern grouping
        ly = hits["ly"][i]
        ts = hits["ts"][i]
        if ts_ref == 0: # if t0_ref was reset, take first timestamp t0 here as reference (can do this since dataset is ordered...)
            ts_ref = ts
        last_scint_hits[ly] = {k: hits[k][i] for k in hits.keys()} # store current column
        # continue to "fill up" last_sl_pattern, if next hit also is within time window
        if i < n_hits-1: # only do it if there is a "next hit"
            ts_next = hits["ts"][i+1]
            if np.abs(ts_next - ts) <= params._scintillator_ts_acceptance_interval:
                continue
        # if not: continue, the combination of collected sl fits starts
        # check for at least 1 phi + 1 theta pattern within t0 interval
        # if 2 phi patterns, also accept it
        phi_hits = [last_scint_hits[ly] for ly in params._scintillator["lys"].keys() if (params._scintillator["lys"][ly]["orient"] == "phi" and last_scint_hits[ly] != None)]
        theta_hits = [last_scint_hits[ly] for ly in params._scintillator["lys"].keys() if (params._scintillator["lys"][ly]["orient"] == "theta" and last_scint_hits[ly] != None)]
        # need to reset ts_ref, last_scint_hits afterwards (for next iteration)
        last_scint_hits = {ly: None for ly in params._scintillator["lys"].keys()} # last sl pattern for all sls
        ts_ref = 0
        ### muon area reco
        n_phi_hits, n_theta_hits = len(phi_hits), len(theta_hits)
        ## check for at least 1 phi + 1 theta pattern, else discard and continue
        if n_phi_hits == 0 or n_theta_hits == 0:
            continue
        ## combine hits within phi plane, if > 1 phi pattern
        ## the resulting (xmin, xmax, z)_phi is in global coord system
        xmin_phi, xmax_phi, z_phi = [], [], []
        phi_axis = params._orientation["phi"][0]
        for j in range(n_phi_hits):
            ly = phi_hits[j]["ly"]
            st = phi_hits[j]["st"]
            xmin_phi.append( derived_params._scintillator_strip_coordinates[ly][st][phi_axis][0] ) # min x value
            xmax_phi.append( derived_params._scintillator_strip_coordinates[ly][st][phi_axis][1] ) # max x value
            z_phi.append( derived_params._scintillator_strip_coordinates[ly][st][5] ) # center z value
            if verbose: print("phi", ([xmin_phi, xmax_phi], z_phi))
        # combine if compatible, by averaging z and selecting tightest x interval
        z_phi = np.mean(z_phi)
        xmin_phi = np.amax(xmin_phi)
        xmax_phi = np.amin(xmax_phi)
        if xmax_phi <= xmin_phi:
            raise Exception(f"xmin_phi = {xmin_phi} must not be larger or equal to xmax_phi = {xmax_phi}.")
        if verbose: print("phi comb", ([xmin_phi, xmax_phi], z_phi))
        ## combine hits within phi plane, if > 1 phi pattern
        ## the resulting (xmin, xmax, z)_phi is in global coord system
        xmin_theta, xmax_theta, z_theta = [], [], []
        theta_axis = params._orientation["theta"][0]
        for j in range(n_theta_hits):
            ly = theta_hits[j]["ly"]
            st = theta_hits[j]["st"]
            xmin_theta.append( derived_params._scintillator_strip_coordinates[ly][st][theta_axis][0] ) # min x value
            xmax_theta.append( derived_params._scintillator_strip_coordinates[ly][st][theta_axis][1] ) # max x value
            z_theta.append( derived_params._scintillator_strip_coordinates[ly][st][5] ) # center z value
            if verbose: print("theta", ([xmin_theta, xmax_theta], z_theta))
        # combine if compatible, by averaging z and selecting tightest x interval
        z_theta = np.mean(z_theta)
        xmin_theta = np.amax(xmin_theta)
        xmax_theta = np.amin(xmax_theta)
        if xmax_theta <= xmin_theta:
            raise Exception(f"xmin_theta = {xmin_theta} must not be larger or equal to xmax_theta = {xmax_theta}.")
        if verbose: print("phi comb", ([xmin_theta, xmax_theta], z_theta))
        ### combine theta + phi and finally form a muon area object
        z0_reco = np.mean([z_theta, z_phi]) # average z
        xmin_reco = xmin_phi if (params._orientation["phi"][0] == 0) else xmin_theta
        xmax_reco = xmax_phi if (params._orientation["phi"][0] == 0) else xmax_theta
        ymin_reco = xmin_theta if (params._orientation["phi"][0] == 0) else xmin_phi
        ymax_reco = xmax_theta if (params._orientation["phi"][0] == 0) else xmax_phi
        ### combine ts to muon arrival time (averaging)
        ts_reco = np.uint64(np.round(np.mean([int(phi_hits[j]["ts"]) for j in range(n_phi_hits)] + [int(theta_hits[j]["ts"]) for j in range(n_theta_hits)]),0))
        ### combine muon_id of hits (if there is one from simulation)
        # raise error of muon_id of combined sl patters is not single value
        muon_id = phi_hits[0]["muon_id"]
        for this_muon_id in [phi_hits[j]["muon_id"] for j in range(n_phi_hits)] + [theta_hits[j]["muon_id"] for j in range(n_theta_hits)]:
            if muon_id != this_muon_id:
                raise Exception(f"Expect hits of same muon_id {muon_id}, not {this_muon_id}.")
        if verbose: print("muon area reco", ([xmin_reco, xmax_reco], [ymin_reco, ymax_reco], z0_reco, ts_reco, muon_id))
        reco_muon_area_list.append({"xmin":xmin_reco, "xmax":xmax_reco, "ymin":ymin_reco, "ymax":ymax_reco, "z0":z0_reco, "ts":ts_reco, "muon_id":muon_id})
        # !!! for muon the name of the timestamp key is "ts" and not "t0"
    n_reco_muon_areas = len(reco_muon_area_list)
    if not silent: print(f"Reconstructed {n_reco_muon_areas} muon areas from {n_hits} scintillator hits.")
    reco_muon_areas = {k: np.full(n_reco_muon_areas, 0, dtype=v) for k,v in params._muon_area_obj_keys.items()}
    for i in range(n_reco_muon_areas):
        for k in params._muon_area_obj_keys.keys():
            reco_muon_areas[k][i] = reco_muon_area_list[i][k]
    return reco_muon_areas







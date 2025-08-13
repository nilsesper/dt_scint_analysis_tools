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
    for i in tqdm(range(n_scint_hits)):
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
        for st in tqdm(range(params._scintillator["lys"][ly]["n_sts"])):
            # check for all muons separately
            for i in range(n_muons):
                # check if muon propagated inside of x and y range of cell, use >= but < to suppress double hits
                if (x[i] >= derived_params._scintillator_strip_coordinates[ly][st][0][0] and x[i] < derived_params._scintillator_strip_coordinates[ly][st][0][1]) and (y[i] >= derived_params._scintillator_strip_coordinates[ly][st][1][0] and y[i] < derived_params._scintillator_strip_coordinates[ly][st][1][1]):
                    # calculate drift distance
                    hit_coord = x[i] if (params._scintillator["lys"][ly]["orient"] == "phi") else y[i]
                    xleft_strip_coord = derived_params._scintillator_strip_coordinates[ly][st][0][0] if (params._scintillator["lys"][ly]["orient"]) else derived_params._scintillator_strip_coordinates[ly][st][1][0]
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







###########################################
### MUON RECONSTRUCTION / DUMMY DATA UTILS
###########################################

import numpy as np
import copy
import os.path

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

"""
### generate random cosmic muon
def generate_cosmic_muon():
    muon = {k: np.full(1, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    muon["x0"], muon["y0"], muon["z0"] = 0, 0, 0
    muon["theta"], muon["phi"] = 0, 0
    muon["ts"] = 0
    return muon

"""

### propagate muon to given z coordinate
def propagate_muon(muon, z): # propagate spherical coordinates
    x = muon["x0"] + (z-muon["z0"]) * np.tan(muon["theta"])*np.cos(muon["phi"])
    y = muon["y0"] + (z-muon["z0"]) * np.tan(muon["theta"])*np.sin(muon["phi"])
    return (x,y,z)

### calculate dt chamber hits caused by muon
# simply propagate it to all layers of the chamber
# returns dt hits with keys {ts = ts of muon + drift time, sl, ly, wi}
def dt_hits_from_muon(muon):
    dt_hit_list = []
    for sl in params._dt_chamber["sls"].keys():
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            z_wi_idx = 0 # all lys have same z therefore save some time here
            z_pos = derived_params._dt_cell_coordinates[sl][ly][z_wi_idx][5] # use center z position (idx 5) of each layer
            (x,y,z) = propagate_muon(muon, z_pos)
            for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
                # check if muon propagated inside of x and y range of cell
                if (x >= derived_params._dt_cell_coordinates[sl][ly][wi][0][0] and x <= derived_params._dt_cell_coordinates[sl][ly][wi][0][1]) and (y >= derived_params._dt_cell_coordinates[sl][ly][wi][1][0] and y <= derived_params._dt_cell_coordinates[sl][ly][wi][1][1]):
                    # calculate drift distance
                    hit_coord = x if (params._dt_chamber["sls"][sl]["orient"] == "phi") else y
                    wire_coord = derived_params._dt_cell_coordinates[sl][ly][wi][3] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else derived_params._dt_cell_coordinates[sl][ly][wi][4]
                    drift_distance = np.float16(np.abs(hit_coord-wire_coord))
                    drift_time = np.uint64(drift_distance / derived_params._drift_velocity_mm_per_timestamp) # in timestamp units, cast to int value
                    hit_ts = np.uint64(muon["ts"] + drift_time) # hit timestamp = muon timestamp + drift time
                    # store this hit
                    dt_hit_list.append({"muon_ts": muon["ts"], "sl": sl, "ly": ly, "wi": wi, "dd": drift_distance, "dt": drift_time, "hit_ts": hit_ts})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(dt_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    dt_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()}
    dt_hits |= {"ts": np.full(n_hits, 0, dtype=params._ts_type)} | {"muon_ts": np.full(n_hits, 0, dtype=params._ts_type)} | {"dt": np.full(n_hits, 0, dtype=np.uint16)}
    for i in range(n_hits):
        # copy existing keys
        for k in ["sl", "ly", "wi", "muon_ts", "dt"]:
            dt_hits[k][i] = dt_hit_list[i][k]
        # map back htg timestamp from drift time
        hit_ts = dt_hit_list[i]["hit_ts"]
        dt_hits["ts"][i] = hit_ts
        # map back obdt ch
        sl, ly, wi = dt_hit_list[i]["sl"], dt_hit_list[i]["ly"], dt_hit_list[i]["wi"]
        dt_hits["ch"][i] = derived_params._dt_inverted_remap_table[sl][ly][wi]["ch"]
    return dt_hits













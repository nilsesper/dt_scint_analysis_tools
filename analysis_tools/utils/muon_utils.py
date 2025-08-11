###########################################
### MUON RECONSTRUCTION / DUMMY DATA UTILS
###########################################

import numpy as np
import copy
import os.path

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params
import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.math_utils as math_utils

# -----------------------------------------

### propagate all muons to given z coordinate
def propagate_muons(muons, z): # propagate spherical coordinates
    x = muons["x0"] + (z-muons["z0"]) * np.tan(muons["theta"])*np.cos(muons["phi"])
    y = muons["y0"] + (z-muons["z0"]) * np.tan(muons["theta"])*np.sin(muons["phi"])
    return (x,y,z)

### propagate one muon to given z coordinate
def propagate_muon(muons, muon_id, z): # propagate spherical coordinates
    x = muons["x0"][muon_id] + (z-muons["z0"][muon_id]) * np.tan(muons["theta"][muon_id])*np.cos(muons["phi"][muon_id])
    y = muons["y0"][muon_id] + (z-muons["z0"][muon_id]) * np.tan(muons["theta"][muon_id])*np.sin(muons["phi"][muon_id])
    return (x,y,z)

### calculate dt chamber hits caused by muons
# simply propagate it to all layers of the chamber
# returns dt hits with keys {ts = ts of muon + drift time, sl, ly, wi}
# dt hits are being sorted by ts value of hits
def dt_hits_from_muons(muons, *, silent=False):
    dt_hit_list = []
    n_muons = len(muons["x0"])
    if not silent: print(f"Calculating DT hits by {n_muons} muons...")
    for sl in params._dt_chamber["sls"].keys():
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            z_wi_idx = 0 # all lys have same z therefore save some time here
            z_pos = derived_params._dt_cell_coordinates[sl][ly][z_wi_idx][5] # use center z position (idx 5) of each layer
            (x,y,z) = propagate_muons(muons=muons, z=z_pos) # propagate all muons together
            for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
                # check for all muons separately
                for i in range(n_muons):
                    # check if muon propagated inside of x and y range of cell
                    if (x[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][0][0] and x[i] <= derived_params._dt_cell_coordinates[sl][ly][wi][0][1]) and (y[i] >= derived_params._dt_cell_coordinates[sl][ly][wi][1][0] and y[i] <= derived_params._dt_cell_coordinates[sl][ly][wi][1][1]):
                        # calculate drift distance
                        hit_coord = x[i] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else y[i]
                        wire_coord = derived_params._dt_cell_coordinates[sl][ly][wi][3] if (params._dt_chamber["sls"][sl]["orient"] == "phi") else derived_params._dt_cell_coordinates[sl][ly][wi][4]
                        drift_distance = np.float16(np.abs(hit_coord-wire_coord))
                        drift_time = np.uint64(drift_distance / derived_params._drift_velocity_mm_per_timestamp) # in timestamp units, cast to int value
                        muon_ts = muons["ts"][i]
                        hit_ts = np.uint64(muon_ts + drift_time) # hit timestamp = muon timestamp + drift time
                        # store this hit
                        dt_hit_list.append({"muon_ts": muon_ts, "sl": sl, "ly": ly, "wi": wi, "dd": drift_distance, "dt": drift_time, "hit_ts": hit_ts, "muon_id": i})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(dt_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    if not silent: print(f"Adding HTG keys to {n_hits} DT hits...")
    dt_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_other_keys.items()} 
    for i in range(n_hits):
        # copy existing keys
        for k in ["sl", "ly", "wi", "muon_ts", "dt", "dd", "muon_id"]:
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

### generate random cosmic muons
# with spawnpoint range same for all muons: xrange = [xmin, xmax], yrange = [ymin, ymax], z0
# generate n muons
# pass separate timestamp for all muons i.e. ts = [ts[i] for i in range(n)]
def generate_cosmic_muons(n, ts, xrange, yrange, z0, *, silent=False):
    if not silent: print(f"Generating {n} cosmic muons...")
    muons = {k: np.full(n, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    muons["x0"] = np.random.uniform(low=xrange[0], high=xrange[1], size=n).astype(dtype=params._muon_obj_keys["x0"])
    muons["y0"] = np.random.uniform(low=yrange[0], high=yrange[1], size=n).astype(dtype=params._muon_obj_keys["y0"]) # x,y uniformly distributed inside xrange, yrange
    muons["z0"] = np.full(n, z0, dtype=params._muon_obj_keys["z0"])
    muons["theta"] = math_utils.draw_from_pdf(pdf=params.cosmic_muon_theta_weight, val_range=[0, np.pi], n=n, dtype=params._muon_obj_keys["theta"]) # theta distributed according to distribution
    muons["phi"] = np.random.uniform(low=0, high=2*np.pi, size=n).astype(dtype=params._muon_obj_keys["phi"]) # phi uniformly distributed in [0, 2*pi]
    muons["ts"] = np.array(ts).astype(dtype=params._muon_obj_keys["ts"])
    return muons









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

### propagate muon to given z coordinate
def propagate_muon(muon, z): # propagate spherical coordinates
    x = muon["x0"] + (z-muon["z0"]) * np.tan(muon["theta"])*np.cos(muon["phi"])
    y = muon["y0"] + (z-muon["z0"]) * np.tan(muon["theta"])*np.sin(muon["phi"])
    return (x,y,z)

### calculate dt chamber hits caused by muon
# simply propagate it to all layers of the chamber
# returns dt hits with keys {ts = ts of muon + drift time, sl, ly, wi}
def dt_hits_from_muon(muon, *, muon_id=0): # optionally store field "muon_id" in hits
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
                    dt_hit_list.append({"muon_ts": muon["ts"], "sl": sl, "ly": ly, "wi": wi, "dd": drift_distance, "dt": drift_time, "hit_ts": hit_ts, "muon_id": muon_id})
    # convert dt_hit_list to proper format object dt_hits
    n_hits = len(dt_hit_list)
    # map sl,ly,wi to all other keys of dt -> map back to obdt channels & oc,bx,tdc timestamp
    dt_hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()}
    dt_hits |= {"ts": np.full(n_hits, 0, dtype=params._ts_type)} | {"muon_ts": np.full(n_hits, 0, dtype=params._ts_type)} | {"dt": np.full(n_hits, 0, dtype=np.uint16)} | {"muon_id": np.full(n_hits, 0, dtype=np.uint16)}
    for i in range(n_hits):
        # copy existing keys
        for k in ["sl", "ly", "wi", "muon_ts", "dt", "muon_id"]:
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
    return dt_hits

### generate random cosmic muon
# with spawnpoint at given xyrange and z0 and timestamp
def generate_cosmic_muon(xrange, yrange, z0, ts):
    muon = {k: np.full(1, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    muon["x0"], muon["y0"] = np.random.uniform(low=xrange[0], high=xrange[1]), np.random.uniform(low=yrange[0], high=yrange[1]) # x,y uniformly distributed inside xrange, yrange
    muon["z0"] = z0
    muon["theta"] = math_utils.draw_from_pdf(pdf=params.cosmic_muon_theta_weight, range=[0, np.pi]) # theta distributed according to distribution
    muon["phi"] = np.random.uniform(low=0, high=2*np.pi) # phi uniformly distributed in [0, 2*pi]
    muon["ts"] = ts
    return muon

### same for multiple muons
def dt_hits_from_muons(muons):
    n_muons = len(muons)
    _dt_hits = []
    for i in range(n_muons):
        _dt_hits.append( dt_hits_from_muon(muons[i], muon_id=i) )
    keys = list(params._htg_keys.keys()) + list(params._dt_mapping_keys.keys()) + ["ts", "muon_ts", "dt", "muon_id"]
    dt_hits = {k: np.concatenate([_dt_hits[i][k] for i in range(n_muons)]) for k in keys}
    return dt_hits








###########################################
### SCINTILLATOR-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm
from itertools import combinations

import analysis_tools.utils.data_utils as data_utils
import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.muon_utils as muon_utils
import analysis_tools.utils.hist_utils as hist_utils
import analysis_tools.utils.combination_utils as combination_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

##### FUNCTIONS FOR SCINTILLATOR HITS = STRIP HITS (with 2 sipm coincidence of sipms of same strip)

### extract scintillator hits from hit data
# cut away all hit data not from scintillator
# add scintillator specific keys to hits
# take information about this mapping from params.py
def extract_scint_hits(hits, *, silent=False, has_timestamp=False):
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
    tmp_hits |= {k: np.full(n_scint_hits, 0, dtype=v) for k,v in params._scint_mapping_keys.items()} | {k: np.full(n_hits, 0, dtype=v) for k,v in params._scint_other_keys.items() if ((not has_timestamp) or k != "ts")} 
    for i in tqdm(range(n_scint_hits), disable=silent):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in derived_params._scint_keys:
            tmp_hits[k][i] = derived_params._scint_remap_table[ro_ch][ch][k]
    # add timestamp
    if not has_timestamp:
        tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits)
    # sort by timestamp
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    scint_hits = tmp_hits
    ### -----------------------
    # apply dead time constraint to all individual channels (if specified dead time is > 0)
    if params._scintillator_ts_individual_dead_time > 0:
        print(f"apply dead time constraint for all individual channels of {params._scintillator_ts_individual_dead_time} TU")
        cut_tmp_hits = {}
        for ly in derived_params._scint_inverted_remap_table.keys():
            cut_tmp_hits[ly] = {}
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                cut_tmp_hits[ly][st] = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly), ("st","==",st)], silent=True)
                cut_tmp_hits[ly][st] = timestamp_utils.sort_by_timestamp(hits=cut_tmp_hits[ly][st])
                n_cut_hits = len(cut_tmp_hits[ly][st]["ts"])
                allowed_indices = []
                ts_list = np.array(cut_tmp_hits[ly][st]["ts"])
                if len(ts_list) > 0:
                    cur_ts = ts_list[0]
                    for i in range(n_cut_hits): 
                        if (ts_list[i]) - (cur_ts) < params._scintillator_ts_individual_dead_time:
                            cur_ts = ts_list[i] # deadtime wrt last hit
                            continue
                        cur_ts = ts_list[i] # deadtime wrt first hit
                        allowed_indices.append(i)
                    for k in cut_tmp_hits[ly][st].keys():
                        cut_tmp_hits[ly][st][k] = cut_tmp_hits[ly][st][k][allowed_indices]
                    n_cut_hits_after = len(cut_tmp_hits[ly][st]['ts'])
                    print(f"ly{ly} st{st} dead time cut flow: {n_cut_hits_after} / {n_cut_hits} = {n_cut_hits_after/max(1,n_cut_hits)}")
        # merge back to tmp_hits
        print("merging data after applying individual dead time...")
        merge_data = []
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                merge_data.append(cut_tmp_hits[ly][st])
        scint_hits = data_utils.merge_dataset(split_data=merge_data)
        print("sort data by timestamp...")
        scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    return scint_hits

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
            # scint coordinates
            scint_x_min = derived_params._scintillator_strip_coordinates[ly][st][0][0]
            scint_x_max = derived_params._scintillator_strip_coordinates[ly][st][0][1]
            scint_y_min = derived_params._scintillator_strip_coordinates[ly][st][1][0]
            scint_y_max = derived_params._scintillator_strip_coordinates[ly][st][1][1]
            # check for all muons separately
            for i in range(n_muons):
                # check if muon propagated inside of x and y range of cell, use >= but < to suppress double hits
                if (x[i] >= scint_x_min and x[i] < scint_x_max) and (y[i] >= scint_y_min and y[i] < scint_y_max):
                    # calculate drift distance
                    hit_coord = x[i] if (params._scintillator["lys"][ly]["orient"] == "phi") else y[i]
                    xleft_strip_coord = scint_x_min if (params._scintillator["lys"][ly]["orient"] == "phi") else scint_y_min
                    xhit = np.float64(np.clip(np.abs(hit_coord-xleft_strip_coord), a_min=0, a_max=params._strip_width)) # in mm
                    # drift distance does not make much sense in this context, but want to store coordinate of hit. with dd one can calculate it: x_hit = x_left(smaller x coord) + xhit
                    muon_ts = muons["ts"][i]
                    hit_ts = np.uint64(muon_ts + params._scintillator_hit_delay) # assume constant delay: hit timestamp = muon timestamp + scint delay
                    # store this hit
                    scint_hit_list.append({"muon_ts": muon_ts, "ly": ly, "st": st, "xhit": xhit, "hit_ts": hit_ts, "muon_id": i})
                #print(f"ly = {ly} , st={st} , muon = ( {x[i]}, {y[i]}, {z_pos} )  ,  scint = ( {derived_params._scintillator_strip_coordinates[ly][st][0][0]} - {derived_params._scintillator_strip_coordinates[ly][st][0][1]} , {derived_params._scintillator_strip_coordinates[ly][st][1][0]} - {derived_params._scintillator_strip_coordinates[ly][st][1][1]} , {derived_params._scintillator_strip_coordinates[ly][st][5]} )")
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

"""
### group scintillator hits with strip coincidence and then reco area of muon hit from scintillator hits
# group together if hits of all layers are close enough in time
# return x and y interval and z coordinate (mean of z pos of layers) that the hit could have been (by assessing all layers hit)
def reco_muon_area_from_hits(hits, *, silent=False, verbose=False):
    reco_muon_area_list = []
    # sort hits by timestamp
    hits = data_utils.sort_by_key(data=hits, sort_key="ts")
    n_hits = len(hits["ts"])
    if not silent: print(f"Combining {n_hits} scintillator hits to reconstruct muons...")
    # extract sls in phi & theta orientation
    phi_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "phi") else 1
    theta_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "theta") else 1
    # grouping by timestamp, check if the ts timestamps of the htis are within given acceptance interval params._scintillator_ts_acceptance_interval
    # calculate muon area object (xrange, yrange, z, ts) where muon should have been
    # NOTE:
    # the algorithm can only cope one muon after another (strictly in order), not multiple muon fits simultaneously :(
    # HARDCODED TO 2 LAYERS IN OPPOSITE ORIENTATION
    last_scint_hits = {ly: None for ly in params._scintillator["lys"].keys()} # last sl pattern for all sls
    for i in tqdm(range(n_hits), disable=silent):
        ### fitted sl pattern grouping
        ly = hits["ly"][i]
        ts = hits["ts"][i]
        # store data of cur hit
        last_scint_hits[ly] = {k: hits[k][i] for k in hits.keys()} # store current column
        ### check continue conditions
        if None in last_scint_hits.values(): # if not have hits of 2 different layers then continue
            continue
        if np.abs(int(last_scint_hits[0]["ts"]) - int(last_scint_hits[1]["ts"])) > params._scintillator_ts_acceptance_interval: # if 2 hits not within time interval, continue
            continue
        ### if not continue: have found 2 matching hits in different layers
        ### muon area reco
        ly_phi, st_phi = last_scint_hits[phi_ly_idx]["ly"], last_scint_hits[phi_ly_idx]["st"]
        ly_theta, st_theta = last_scint_hits[theta_ly_idx]["ly"], last_scint_hits[theta_ly_idx]["st"]
        xmin_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][0]
        xmax_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][1]
        ymin_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][0]
        ymax_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][1]
        z0_reco = np.mean([derived_params._scintillator_strip_coordinates[ly_phi][st_phi][5], derived_params._scintillator_strip_coordinates[ly_theta][st_theta][5]])
        ly0_st = st_phi if (ly_phi == 0) else st_theta
        ly1_st = st_phi if (ly_phi == 1) else st_theta
        pixel_index = derived_params._scint_pixel_mapping[(ly0_st, ly1_st)]
        #### ----      
        xcenter_reco = np.mean([xmin_reco, xmax_reco])
        ycenter_reco = np.mean([ymin_reco, ymax_reco])
        ### combine ts to muon arrival time (averaging)
        ts_phi, ts_theta = last_scint_hits[phi_ly_idx]["ts"], last_scint_hits[theta_ly_idx]["ts"]
        ts_reco = np.uint64(np.round(np.mean([ts_phi, ts_theta]), 0))
        ### calculate ts difference between hits in both layers (absolute value)
        ly_delta_ts = np.uint64(np.abs(int(ts_phi) - int(ts_theta)))
        ### combine muon_id of hits (if there is one from simulation)
        # raise error of muon_id of combined sl patters is not single value
        muon_id =  last_scint_hits[phi_ly_idx]["muon_id"]
        if muon_id != last_scint_hits[theta_ly_idx]["muon_id"]:
            raise Exception(f"Expect hits of same muon_id {muon_id}, not {last_scint_hits[theta_ly_idx]['muon_id']}.")
        if verbose: print("muon area reco", ([xmin_reco, xmax_reco], [ymin_reco, ymax_reco], z0_reco, ts_reco, muon_id))
        ### store reco obj
        reco_muon_area_list.append({
            "xmin": xmin_reco, 
            "xmax": xmax_reco, 
            "ymin": ymin_reco, 
            "ymax": ymax_reco, 
            "z0": z0_reco, 
            "ts": ts_reco,
            "muon_id": muon_id, 
            "pixel": pixel_index, 
            "xcenter": xcenter_reco, 
            "ycenter": ycenter_reco, 
            "ly_delta_ts": ly_delta_ts,
            "st0": ly0_st,
            "st1": ly1_st,
        })
        # !!! for muon the name of the timestamp key is "ts" and not "t0"
        ##### need to reset ts_ref, last_scint_hits afterwards (for next iteration)
        last_scint_hits = {ly: None for ly in params._scintillator["lys"].keys()} # last hit for all lys
    # store in proper format
    n_reco_muon_areas = len(reco_muon_area_list)
    if not silent: print(f"Reconstructed {n_reco_muon_areas} muon areas from {n_hits} scintillator hits.")
    reco_muon_areas = {k: np.full(n_reco_muon_areas, 0, dtype=v) for k,v in params._muon_area_obj_keys.items()}
    for i in range(n_reco_muon_areas):
        for k in params._muon_area_obj_keys.keys():
            reco_muon_areas[k][i] = reco_muon_area_list[i][k]
    return reco_muon_areas
#"""

### group scintillator hits with strip coincidence and then reco area of muon hit from scintillator hits
# group together if hits of all layers are close enough in time
# return x and y interval and z coordinate (mean of z pos of layers) that the hit could have been (by assessing all layers hit)
def reco_muon_area_from_hits(hits, *, silent=False, verbose=False):
    reco_muon_area_list = []
    # sort hits by timestamp
    hits = data_utils.sort_by_key(data=hits, sort_key="ts")
    n_hits = len(hits["ts"])
    if not silent: print(f"Combining {n_hits} scintillator hits to reconstruct muons...")
    # extract sls in phi & theta orientation
    phi_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "phi") else 1
    theta_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "theta") else 1
    # grouping by timestamp, check if the ts timestamps of the htis are within given acceptance interval params._scintillator_ts_acceptance_interval
    # calculate muon area object (xrange, yrange, z, ts) where muon should have been
    # NOTE:
    # the algorithm can only cope all hits
    # HARDCODED TO 2 LAYERS IN OPPOSITE ORIENTATION
    dummy_scint_hit = {k: np.array(0, dtype=v) for k,v in params._htg_keys.items()} | {k: np.array(0, dtype=v) for k,v in params._scint_mapping_keys.items()} | {k: np.array(0, dtype=v) for k,v in params._scint_other_keys.items()}
    last_hits = _scint_data(default=dummy_scint_hit) # last hit for all lys, sts
    for i in tqdm(range(n_hits), disable=silent):
        ### fitted sl pattern grouping
        ly = hits["ly"][i]
        st = hits["st"][i]
        ts = hits["ts"][i]
        # store data of cur hit
        last_hits[ly][st] = {k: hits[k][i] for k in hits.keys()} # store current column
        ### check if hit for any pixel
        for st0 in derived_params._scint_inverted_remap_table[0].keys():
            for st1 in derived_params._scint_inverted_remap_table[1].keys():
                ### skip pixels which are not relevant
                if not((ly == 1 and st == st1) or (ly == 0 and st == st0)):
                    continue
                ### check time interval
                if np.abs((last_hits[0][st0]["ts"]) - (last_hits[1][st1]["ts"])) > params._scintillator_ts_acceptance_interval: # if 2 hits not within time interval, continue
                    continue
                # if timestamp = 0, continue since it is default value
                if (last_hits[0][st0]["ts"]) == 0 or (last_hits[1][st1]["ts"]) == 0:
                    continue
                ##### store pixel hit
                ### muon area reco
                pixel_index = derived_params._scint_pixel_mapping[(st0, st1)]
                ly_phi = phi_ly_idx
                st_phi = st0 if (ly_phi == 0) else st1
                ly_theta = theta_ly_idx
                st_theta = st0 if (ly_theta == 0) else st1
                xmin_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][0]
                xmax_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][1]
                ymin_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][0]
                ymax_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][1]
                z0_reco = np.mean([derived_params._scintillator_strip_coordinates[ly_phi][st_phi][5], derived_params._scintillator_strip_coordinates[ly_theta][st_theta][5]])
                #### ----      
                xcenter_reco = np.mean([xmin_reco, xmax_reco])
                ycenter_reco = np.mean([ymin_reco, ymax_reco])
                ### combine ts to muon arrival time (averaging)
                ts_phi, ts_theta = last_hits[ly_phi][st_phi]["ts"], last_hits[ly_theta][st_theta]["ts"]
                ts_reco = np.mean([ts_phi, ts_theta])
                ### calculate ts difference between hits in both layers (absolute value)
                ly_delta_ts = np.abs((ts_phi) - (ts_theta))
                ### combine muon_id of hits (if there is one from simulation)
                # raise error of muon_id of combined sl patters is not single value
                muon_id =  last_hits[ly_phi][st_phi]["muon_id"]
                if muon_id != last_hits[ly_theta][st_theta]["muon_id"]:
                    raise Exception(f"Expect hits of same muon_id {muon_id}, not {last_hits[ly_theta][st_theta]['muon_id']}.")
                if verbose: print("muon area reco", ([xmin_reco, xmax_reco], [ymin_reco, ymax_reco], z0_reco, ts_reco, muon_id))
                ### store reco obj
                reco_muon_area_list.append({
                    "xmin": xmin_reco, 
                    "xmax": xmax_reco, 
                    "ymin": ymin_reco, 
                    "ymax": ymax_reco, 
                    "z0": z0_reco, 
                    "ts": ts_reco,
                    "muon_id": muon_id, 
                    "pixel": pixel_index, 
                    "xcenter": xcenter_reco, 
                    "ycenter": ycenter_reco, 
                    "ly_delta_ts": ly_delta_ts,
                    "st0": st0,
                    "st1": st1,
                })
                ### remove the hits to not use them twice for multiple pixels
                #last_hits[0][st0] = copy.deepcopy(dummy_scint_hit)
                #last_hits[1][st1] = copy.deepcopy(dummy_scint_hit)
    # store in proper format
    n_reco_muon_areas = len(reco_muon_area_list)
    if not silent: print(f"Reconstructed {n_reco_muon_areas} muon areas from {n_hits} scintillator hits.")
    reco_muon_areas = {k: np.full(n_reco_muon_areas, 0, dtype=v) for k,v in params._muon_area_obj_keys.items()}
    for i in range(n_reco_muon_areas):
        for k in params._muon_area_obj_keys.keys():
            reco_muon_areas[k][i] = reco_muon_area_list[i][k]
    # sort by timestamp
    reco_muon_areas = data_utils.sort_by_key(data=reco_muon_areas, sort_key="ts")
    return reco_muon_areas

### remove all muon areas which occur within short time window params._scint_area_clear_interval
# keep only "isolated" areas i.e. where no other pixels were hit
def remove_crosstalk_areas(areas, * , silent=False):
    if not silent: print(f"Apply crosstalk mitigation / hit isolation criterion with isolation window (down, up) = {params._scint_area_clear_interval_down} TU, {params._scint_area_clear_interval_up} TU)...")
    cleaned_areas = copy.deepcopy(areas)
    n_areas = data_utils.length(areas)
    # sort by timestamp
    cleaned_areas = data_utils.sort_by_key(data=cleaned_areas, sort_key="ts")
    ## find isolated hits and put them in index mask
    mask = []
    for i in tqdm(range(1,n_areas-1), disable=silent):
        delta_ts_down = (cleaned_areas["ts"][i]) - (cleaned_areas["ts"][i-1])
        delta_ts_up = (cleaned_areas["ts"][i+1]) - (cleaned_areas["ts"][i])
        if delta_ts_down < params._scint_area_clear_interval_down or delta_ts_up < params._scint_area_clear_interval_up:
            continue
        mask.append(i)
    if not silent: print(f"Crosstalk / hit isolation cut flow: {len(mask)} / {n_areas} = {len(mask)/n_areas}")
    ## apply mask of isolated hits
    for k in cleaned_areas.keys():
        cleaned_areas[k] = cleaned_areas[k][mask]
    return cleaned_areas


###### FUNCTIONS FOR RAW SCINTILLATOR HITS = SIPM HITS (individual sipm hits, no coincidence criterea applied)

### extract raw scintillator hits from hit data
# cut away all hit data not from scintillator
# add scintillator specific keys to hits
# take information about this mapping from params.py
# if has_timestamp = True, assume that timestamp was already assigned, do not add it here
def extract_raw_scint_hits(hits, *, silent=False, has_timestamp=False):
    tmp_hits = copy.deepcopy(hits)
    n_hits = len(tmp_hits["ch"])
    if not silent: print(f"Extract raw scintillator hits from {n_hits} total hits...")
    # calculate mask to apply to cut away all hits not belonging to dt chamber (wrong ro_ch or invalid ch)
    scint_mask = np.full(n_hits, False, dtype=np.bool)
    for ro_ch in derived_params._raw_scint_ro_chs:
        tmp_mask = np.ma.isin(tmp_hits["ro_ch"], [ro_ch])
        tmp_mask &= np.ma.isin(tmp_hits["ch"], derived_params._raw_scint_chs_by_ro_ch[ro_ch])
        scint_mask |= tmp_mask
    # apply mask
    for k in tmp_hits.keys():
        tmp_hits[k] = tmp_hits[k][scint_mask]
    n_scint_hits = len(tmp_hits["ch"])
    if not silent: print(f"Cut flow: {n_scint_hits}/{n_hits} = {n_scint_hits/n_hits}")
    if not silent: print(f"Found {n_scint_hits} raw scintillator hits. Adding raw scintillator specific keys...")
    # add specific scint keys
    tmp_hits |= {k: np.full(n_scint_hits, 0, dtype=v) for k,v in params._raw_scint_mapping_keys.items()} | {k: np.full(n_scint_hits, 0, dtype=v) for k,v in params._raw_scint_other_keys.items() if ((not has_timestamp) or k != "ts")} 
    for i in tqdm(range(n_scint_hits), disable=silent):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in derived_params._raw_scint_keys:
            tmp_hits[k][i] = derived_params._raw_scint_remap_table[ro_ch][ch][k]
    # add timestamp and sort by timestamp
    if not has_timestamp:
        tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    ### -----------------------
    # apply dead time constraint to all individual channels (if specified dead time is > 0)
    if params._raw_scintillator_ts_individual_dead_time > 0:
        print(f"apply dead time constraint for all individual channels of {params._raw_scintillator_ts_individual_dead_time} TU")
        cut_tmp_hits = {}
        for ly in derived_params._scint_inverted_remap_table.keys():
            cut_tmp_hits[ly] = {}
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                cut_tmp_hits[ly][st] = {}
                for sipm in [0,1]:
                    cut_tmp_hits[ly][st][sipm] = data_utils.cut_data(data=tmp_hits, conditions=[("ly","==",ly), ("st","==",st), ("sipm","==",sipm)], silent=True)
                    cut_tmp_hits[ly][st][sipm] = timestamp_utils.sort_by_timestamp(hits=cut_tmp_hits[ly][st][sipm])
                    n_cut_hits = len(cut_tmp_hits[ly][st][sipm]["ts"])
                    allowed_indices = []
                    ts_list = np.array(cut_tmp_hits[ly][st][sipm]["ts"])
                    if len(ts_list) > 0:
                        cur_ts = ts_list[0]
                        for i in range(n_cut_hits): 
                            if (ts_list[i]) - (cur_ts) < params._raw_scintillator_ts_individual_dead_time:
                                cur_ts = ts_list[i] # deadtime wrt last hit
                                continue
                            cur_ts = ts_list[i] # deadtime wrt first hit
                            allowed_indices.append(i)
                        for k in cut_tmp_hits[ly][st][sipm].keys():
                            cut_tmp_hits[ly][st][sipm][k] = cut_tmp_hits[ly][st][sipm][k][allowed_indices]
                        n_cut_hits_after = len(cut_tmp_hits[ly][st][sipm]['ts'])
                        print(f"ly{ly} st{st} sipm{sipm} dead time cut flow: {n_cut_hits_after} / {n_cut_hits} = {n_cut_hits_after/max(1,n_cut_hits)}")
        # merge back to tmp_hits
        print("merging data after applying individual dead time...")
        merge_data = []
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                for sipm in [0,1]:
                    merge_data.append(cut_tmp_hits[ly][st][sipm])
        tmp_hits = data_utils.merge_dataset(split_data=merge_data)
        print("sort data by timestamp...")
        tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits)
    return tmp_hits

### match raw scint hits (single sipm hits) to scint hits (2 sipm coincidences of strips)
# apply coincidence criterea
# hardcoded to use sipms 0 and 1 (2-fold coincidence)
# coincidence window hardcoded at params._raw_scintillator_ts_acceptance_interval
def reco_hits_from_raw_hits(hits, *, silent=False):
    scint_hit_list = []
    # sort hits by timestamp
    hits = data_utils.sort_by_key(data=hits, sort_key="ts")
    n_hits = len(hits["ts"])
    if not silent: print(f"Combining {n_hits} raw scintillator hits to scintillator hits...")
    # for all strips separately
    for ly in params._scintillator["lys"].keys():
        for st in range(params._scintillator["lys"][ly]["n_sts"]):
            if not silent: print(f"  ly={ly}, st={st}...")
            # check sipm masking
            if ly in params._scint_masked_sipms.keys() and st in params._scint_masked_sipms[ly].keys():
            ### use only 1 sipm of strip if other was masked (in params._scint_masked_sipms)
                masked_sipm = params._scint_masked_sipms[ly][st]
                unmasked_sipm = 1 if (params._scint_masked_sipms[ly][st] == 0) else 0
                st_hits = data_utils.cut_data(data=hits, conditions=[("ly","==",ly), ("st","==",st), ("sipm","==",unmasked_sipm)], silent=True)
                for i in range(data_utils.length(st_hits)):
                    sipm = st_hits["sipm"][i]
                    ts = st_hits["ts"][i]
                    err_ts = st_hits["err_ts"][i]
                    ### combine ts time (averaging)
                    ts_reco = ts
                    err_ts_reco = err_ts
                    (oc_reco, bx_reco, tdc_reco) = timestamp_utils.remap_htg_timestamp(ts_reco)
                    ### calculate ts difference between hits (absolute value)
                    sipm_delta_ts = np.uint64(0)
                    ### calculate ts difference to last hit of this strip / of this sipm (to check ringing)
                    st_delta_last_ts0, st_delta_last_ts1, st_delta_last_ts = 0, 0, 0
                    ### scint ch_id if coincidence would have been active
                    # extract from scint mapping table 
                    scint_ch_id = derived_params._scint_inverted_remap_table[ly][st]["ch_id"]
                    scint_ch = derived_params._scint_inverted_remap_table[ly][st]["ch"]
                    ### store reco obj
                    scint_hit_list.append({
                        "ly": ly,
                        "st": st,
                        "ts": ts_reco,
                        "err_ts": err_ts_reco,
                        "ch_id": scint_ch_id,
                        "muon_id": st_hits["muon_id"][i],
                        "ro_ch": st_hits["ro_ch"][i],
                        "muon_ts": st_hits["muon_ts"][i],
                        "xhit": st_hits["xhit"][i],
                        "oc": oc_reco,
                        "bx": bx_reco,
                        "tdc": tdc_reco,
                        "ch": scint_ch,
                        "sipm_delta_ts": sipm_delta_ts,
                        "st_delta_last_ts0": st_delta_last_ts0,
                        "st_delta_last_ts1": st_delta_last_ts1,
                        "st_delta_last_ts": st_delta_last_ts,
                    })
            else:
            ### build coincidence from 2 sipms of strip
                st_hits = data_utils.cut_data(data=hits, conditions=[("ly","==",ly), ("st","==",st)], silent=True)
                # match hits of same ly, st for sipm = 0 and sipm = 1 if within temporal coincidence window
                SIPMS = [0, 1]
                last_hits = {sipm: None for sipm in SIPMS}
                ts0_old, ts1_old, ts_reco_old = None, None, None
                for i in range(data_utils.length(st_hits)):
                    sipm = st_hits["sipm"][i]
                    ts = st_hits["ts"][i]
                    # store data of cur hit
                    last_hits[sipm] = {k: st_hits[k][i] for k in hits.keys()} # store current column
                    ### check continue conditions
                    if None in last_hits.values(): # if not have hits of 2 different layers then continue
                        continue
                    if np.abs((last_hits[0]["ts"]) - (last_hits[1]["ts"])) > params._raw_scintillator_ts_acceptance_interval: # if 2 hits not within time interval, continue
                        continue
                    ### if not continue: have found 2 matching hits of same strip
                    #--- build scintillator hit from this object
                    ### combine ts time (averaging)
                    ts0, ts1 = last_hits[0]["ts"], last_hits[1]["ts"]
                    err_ts0, err_ts1 = last_hits[0]["err_ts"], last_hits[1]["err_ts"]
                    ts_reco = np.mean([ts0, ts1])
                    err_ts_reco = np.sqrt(err_ts0**2 + err_ts1**2)
                    (oc_reco, bx_reco, tdc_reco) = timestamp_utils.remap_htg_timestamp(ts_reco)
                    ### calculate ts difference between hits (absolute value)
                    sipm_delta_ts = np.abs((ts0) - (ts1))
                    ### calculate ts difference to last hit of this strip / of this sipm (to check ringing)
                    st_delta_last_ts0, st_delta_last_ts1, st_delta_last_ts = 0, 0, 0
                    if ts0_old != None:
                        st_delta_last_ts0 = np.abs((ts0) - (ts0_old))
                        st_delta_last_ts1 = np.abs((ts1) - (ts1_old))
                        st_delta_last_ts = np.abs((ts_reco) - (ts_reco_old))
                    ts0_old, ts1_old, ts_reco_old = ts0, ts1, ts_reco
                    ### combine muon_id of hits (if there is one from simulation)
                    # raise error of muon_id of combined sl patters is not single value
                    muon_id =  last_hits[0]["muon_id"]
                    if muon_id != last_hits[1]["muon_id"]:
                        raise Exception(f"Expect hits of same muon_id {muon_id}, not {last_hits[1]['muon_id']}.")
                    ### scint ch_id if coincidence would have been active
                    # extract from scint mapping table 
                    scint_ch_id = derived_params._scint_inverted_remap_table[ly][st]["ch_id"]
                    scint_ch = derived_params._scint_inverted_remap_table[ly][st]["ch"]
                    ### store reco obj
                    scint_hit_list.append({
                        "ly": ly,
                        "st": st,
                        "ts": ts_reco,
                        "err_ts": err_ts_reco,
                        "ch_id": scint_ch_id,
                        "muon_id": muon_id,
                        "ro_ch": last_hits[0]["ro_ch"],
                        "muon_ts": last_hits[0]["muon_ts"],
                        "xhit": last_hits[0]["xhit"],
                        "oc": oc_reco,
                        "bx": bx_reco,
                        "tdc": tdc_reco,
                        "ch": scint_ch,
                        "sipm_delta_ts": sipm_delta_ts,
                        "st_delta_last_ts0": st_delta_last_ts0,
                        "st_delta_last_ts1": st_delta_last_ts1,
                        "st_delta_last_ts": st_delta_last_ts,
                    })
                    ##### need to reset ts_ref, last_scint_hits afterwards (for next iteration)
                    last_hits = {sipm: None for sipm in SIPMS}
    # store in proper format
    n_scint_hits = len(scint_hit_list)
    if not silent: print(f"Reconstructed {n_scint_hits} scintillator hits from {n_hits} raw scintillator hits.")
    scint_keys_types = copy.deepcopy(params._scint_mapping_keys) | copy.deepcopy(params._scint_other_keys) | copy.deepcopy(params._htg_keys)
    scint_hits = {k: np.full(n_scint_hits, 0, dtype=v) for k,v in scint_keys_types.items()}
    for i in range(n_scint_hits):
        for k in scint_keys_types.keys():
            scint_hits[k][i] = scint_hit_list[i][k]
    scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    ### -----------------------
    # apply dead time constraint to all individual channels (if specified dead time is > 0)
    if params._scintillator_ts_individual_dead_time > 0:
        print(f"apply dead time constraint for all individual channels of {params._scintillator_ts_individual_dead_time} TU")
        cut_tmp_hits = {}
        for ly in derived_params._scint_inverted_remap_table.keys():
            cut_tmp_hits[ly] = {}
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                cut_tmp_hits[ly][st] = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly), ("st","==",st)], silent=True)
                cut_tmp_hits[ly][st] = timestamp_utils.sort_by_timestamp(hits=cut_tmp_hits[ly][st])
                n_cut_hits = len(cut_tmp_hits[ly][st]["ts"])
                allowed_indices = []
                ts_list = np.array(cut_tmp_hits[ly][st]["ts"])
                if len(ts_list) > 0:
                    cur_ts = ts_list[0]
                    for i in range(n_cut_hits): 
                        if (ts_list[i]) - (cur_ts) < params._scintillator_ts_individual_dead_time:
                            cur_ts = ts_list[i] # deadtime wrt last hit
                            continue
                        cur_ts = ts_list[i] # deadtime wrt first hit
                        allowed_indices.append(i)
                    for k in cut_tmp_hits[ly][st].keys():
                        cut_tmp_hits[ly][st][k] = cut_tmp_hits[ly][st][k][allowed_indices]
                    n_cut_hits_after = len(cut_tmp_hits[ly][st]['ts'])
                    print(f"ly{ly} st{st} dead time cut flow: {n_cut_hits_after} / {n_cut_hits} = {n_cut_hits_after/max(1,n_cut_hits)}")
        # merge back to tmp_hits
        print("merging data after applying individual dead time...")
        merge_data = []
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                merge_data.append(cut_tmp_hits[ly][st])
        scint_hits = data_utils.merge_dataset(split_data=merge_data)
        print("sort data by timestamp...")
        scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    return scint_hits


### group closely lying raw scint hits (to analyze crosstalk)
# ts_tolerance = 
def group_raw_scint_hits(hits, *, ts_tolerance=200, idx_offset=0, silent=False):
    # group all hits by ts_tolerance
    if not silent: print(f"grouping raw scint hits...")
    idx_grouped, ts_group = combination_utils.time_grouping_indices_2(data=hits, ts_tolerance=ts_tolerance, data_ts_key="ts")
    n_groups = len(idx_grouped)
    # storage format of scint hit groups
    raw_scint_groups = {
        "tgroup": np.zeros(n_groups),
        "idcs": [[] for i in range(n_groups)], # idx in raw_scint_hits object
        "tuples": [[] for i in range(n_groups)], # hit tuples (ly,st,sipm)
        "n_hits": np.zeros(n_groups), # no of hits in group
        "tuples_nodupl": [[] for i in range(n_groups)], # hit tuples (ly,st,sipm) with removed double hits of same channel
        "n_hits_nodupl": np.zeros(n_groups), # no of hits in group counting each channel only once also if there are multiple hits of same channel
        "idcs_nodupl": [[] for i in range(n_groups)], # idx in raw_scint_hits object, earliest hit of channel
    }
    # fill content into data structure
    if not silent: print(f"filling information in data structure...")
    for i in tqdm(range(n_groups), disable=silent):
        # extract group hits
        initial_idcs = idx_grouped[i]
        n_initial_idcs = len(initial_idcs)
        hit_tuples = []
        hit_tuples_nodupl = []
        idcs = []
        idcs_nodupl = []
        for j in range(n_initial_idcs):
            idx = initial_idcs[j]
            glob_idx = idx + idx_offset
            idcs.append(glob_idx)
            hit_tuple = (hits["ly"][idx], hits["st"][idx], hits["sipm"][idx])
            hit_tuples.append(hit_tuple)
            if hit_tuple not in hit_tuples_nodupl:
                hit_tuples_nodupl.append(hit_tuple)
                idcs_nodupl.append(glob_idx)
        raw_scint_groups["tgroup"][i] = ts_group[i]
        raw_scint_groups["tuples"][i] = hit_tuples
        raw_scint_groups["idcs"][i] = idcs
        raw_scint_groups["n_hits"][i] = n_initial_idcs
        # remove double entries
        n_idcs_nodupl = len(hit_tuples_nodupl)
        raw_scint_groups["tuples_nodupl"][i] = hit_tuples_nodupl
        raw_scint_groups["n_hits_nodupl"][i] = n_idcs_nodupl
        raw_scint_groups["idcs_nodupl"][i] = idcs_nodupl
    return raw_scint_groups

### raw scint groups to strips / scint hits (in both layers)
# groups: raw scint hit groups
# hits: raw scint hits with indices matching to those given in raw scint group data object
# isolation_criterion: if True only accept strip coincidence if no other coincidence in same layer is found, if False do not care
# coincidence window given by: params._scintillator_ts_acceptance_interval
def raw_scint_groups_to_strips(groups, hits, *, silent=False, isolation_criterion=True):
    n_groups = data_utils.length(groups)
    n_hits = data_utils.length(hits)
    strip_hits = []
    # strip coincidence
    if not silent: print(f"applying strip coincidence...")
    ### loop twice: looking for coincidences in ly0 and ly1
    for ref_ly in derived_params._scint_inverted_remap_table.keys():
        if not silent: print(f"for ly {ref_ly}:")
        # check all groups for possible coincidence of sipm0 and sipm1
        for i in tqdm(range(n_groups), disable=silent):
            # list of idcs of hits of each strip
            cur_st_hits = [[] for st in derived_params._scint_inverted_remap_table[ref_ly].keys()]
            # use nodupl idcs, since do not want two hits of same channel...
            idcs = groups["idcs_nodupl"][i]
            n_hits = 0 # no of hits in group of current layer ly=ref_ly
            for idx in idcs:
                ly, st = hits["ly"][idx], hits["st"][idx]
                # only consider current layer (ref_ly)
                if ly != ref_ly:
                    continue
                n_hits += 1
                cur_st_hits[st].append(idx)
            # check if there are strips with 2 hits
            n_coinc = 0
            for st in derived_params._scint_inverted_remap_table[ref_ly].keys():
                if len(cur_st_hits[st]) < 2:
                    continue
                n_coinc += 1
            if n_coinc == 0:
                continue
            if (n_hits > 2 or n_coinc > 1) and isolation_criterion == True: # ignore coincidence if more than one coincidence found in group and isolation criterion active
                continue 
            for st in derived_params._scint_inverted_remap_table[ref_ly].keys():
                if len(cur_st_hits[st]) < 2:
                    continue
                idx0, idx1 = cur_st_hits[st][0], cur_st_hits[st][1]
                ##### check ts coincindence window
                ts0 = hits["ts"][idx0]
                ts1 = hits["ts"][idx1]
                delta_ts = np.abs(ts0 - ts1)
                if delta_ts > params._scintillator_ts_acceptance_interval:
                    continue
                ###### if strip hit found: collect all info and store hit
                # reco timestamp
                ts_reco = np.mean([ ts0, ts1 ])
                #print(ts0, ts1, ts_reco, delta_ts)
                err_ts_reco = np.sqrt(hits["err_ts"][idx0]**2 + hits["err_ts"][idx1]**2)
                (oc_reco, bx_reco, tdc_reco) = timestamp_utils.remap_htg_timestamp(ts_reco)
                # calculate ts difference between hits (absolute value)
                sipm_delta_ts = delta_ts
                # signed time difference sipm0 - sipm1
                sipm_0_idx = idx0 if (hits["sipm"][idx0] == 0) else idx1
                sipm_1_idx = idx0 if (hits["sipm"][idx0] == 1) else idx1
                sipm_delta_ts_signed = hits["ts"][sipm_0_idx] - hits["ts"][sipm_1_idx]
                # extract from scint mapping table 
                scint_ch_id = derived_params._scint_inverted_remap_table[ly][st]["ch_id"]
                scint_ch = derived_params._scint_inverted_remap_table[ly][st]["ch"]
                print(f"ly={ref_ly}, st={st}, delta_ts={sipm_delta_ts_signed}, n_hits_st={len(cur_st_hits[st])}, hit0=[ly={hits['ly'][sipm_0_idx]}, ly={hits['ly'][sipm_0_idx]}, st={hits['st'][sipm_0_idx]}, sipm={hits['sipm'][sipm_0_idx]}, ts={hits['ts'][sipm_0_idx]}], hit1=[ly={hits['ly'][sipm_1_idx]}, ly={hits['ly'][sipm_1_idx]}, st={hits['st'][sipm_1_idx]}, sipm={hits['sipm'][sipm_1_idx]}, ts={hits['ts'][sipm_1_idx]}]")
                # store strip hit
                strip_hits.append({
                    "ly": ref_ly,
                    "st": st,
                    "ts": ts_reco,
                    "err_ts": err_ts_reco,
                    "ch_id": scint_ch_id,
                    "oc": oc_reco,
                    "bx": bx_reco,
                    "tdc": tdc_reco,
                    "sipm_delta_ts": sipm_delta_ts,
                    "sipm_delta_ts_signed": sipm_delta_ts_signed,
                })
    # store in proper format
    n_scint_hits = len(strip_hits)
    if not silent: print(f"Reconstructed {n_scint_hits} scintillator hits from {n_groups} raw scintillator hit groups.")
    scint_keys_types = copy.deepcopy(params._scint_mapping_keys) | copy.deepcopy(params._scint_other_keys) | copy.deepcopy(params._htg_keys)
    scint_hits = {k: np.full(n_scint_hits, 0, dtype=v) for k,v in scint_keys_types.items()}
    if n_scint_hits > 0:
        avail_keys = strip_hits[0].keys()
    for i in range(n_scint_hits):
        for k in avail_keys:
            scint_hits[k][i] = strip_hits[i][k]
    scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    return scint_hits

### raw scint groups to pixels / scint areas (by combining 4 sipms of 2 layers and 2 strips)
# groups: raw scint hit groups
# hits: raw scint hits with indices matching to those given in raw scint group data object
# isolation_criterion: if True only accept strip coincidence if no other coincidence in same layer is found, if False do not care
# coincidence window given by: params._scintillator_ts_acceptance_interval
def raw_scint_groups_to_pixels(groups, hits, *, silent=False, isolation_criterion=True):
    n_groups = data_utils.length(groups)
    n_hits = data_utils.length(hits)
    reco_muon_area_list = []
    # extract sls in phi & theta orientation
    phi_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "phi") else 1
    theta_ly_idx = 0 if (params._scintillator["lys"][0]["orient"] == "theta") else 1
    # pixel coincidence
    if not silent: print(f"applying pixel coincidence...")
    # check all groups for possible coincidence of sipm0 and sipm1
    for i in tqdm(range(n_groups), disable=silent):
        # list of idcs of hits of each strip
        cur_ly_st_hits = [[[] for st in derived_params._scint_inverted_remap_table[0].keys()] for ly in derived_params._scint_inverted_remap_table.keys()]
        # use nodupl idcs, since do not want two hits of same channel...
        idcs = groups["idcs_nodupl"][i]
        n_hits = 0 # no of hits in group of current layer ly=ref_ly
        for idx in idcs:
            ly, st = hits["ly"][idx], hits["st"][idx]
            n_hits += 1
            cur_ly_st_hits[ly][st].append(idx)
        # check if there are strips with 2 hits in both layers
        n_coinc = [0, 0] # no. coinc in ly 0, 1
        coinc_ly_st = [[] for ly in derived_params._scint_inverted_remap_table.keys()] # ly: (st, (idx0, idx1) of sipm hits of this st)
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                if len(cur_ly_st_hits[ly][st]) < 2:
                    continue
                coinc_ly_st[ly].append((st, cur_ly_st_hits[ly][st]))
                n_coinc[ly] += 1
        if n_coinc[0] == 0 or n_coinc[1] == 0:
            continue
        if (n_hits > 4 or n_coinc[0] > 1 or n_coinc[1] > 1) and isolation_criterion == True: # ignore coincidence if more than one coincidence found in group and isolation criterion active
            continue
        for j0 in range(len(coinc_ly_st[0])):
            for j1 in range(len(coinc_ly_st[1])):
                st0, (idx00, idx01) = coinc_ly_st[0][j0]
                st1, (idx10, idx11) = coinc_ly_st[1][j1]
                ##### check ts coincindence window
                max_delta_ts = np.amax([
                    np.abs(hits["ts"][idx00] - hits["ts"][idx01]),
                    np.abs(hits["ts"][idx00] - hits["ts"][idx10]),
                    np.abs(hits["ts"][idx00] - hits["ts"][idx11]),
                    np.abs(hits["ts"][idx01] - hits["ts"][idx10]),
                    np.abs(hits["ts"][idx01] - hits["ts"][idx11]),
                    np.abs(hits["ts"][idx10] - hits["ts"][idx11]),
                ])
                if max_delta_ts > params._scintillator_ts_acceptance_interval:
                    continue
                ###### if pixel hit found: collect all info and store hit
    ### muon area reco
                pixel_index = derived_params._scint_pixel_mapping[(st0, st1)]
                ly_phi = phi_ly_idx
                st_phi = st0 if (ly_phi == 0) else st1
                ly_theta = theta_ly_idx
                st_theta = st0 if (ly_theta == 0) else st1
                xmin_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][0]
                xmax_reco = derived_params._scintillator_strip_coordinates[ly_phi][st_phi][0][1]
                ymin_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][0]
                ymax_reco = derived_params._scintillator_strip_coordinates[ly_theta][st_theta][1][1]
                z0_reco = np.mean([derived_params._scintillator_strip_coordinates[ly_phi][st_phi][5], derived_params._scintillator_strip_coordinates[ly_theta][st_theta][5]])
                #### ----      
                xcenter_reco = np.mean([xmin_reco, xmax_reco])
                ycenter_reco = np.mean([ymin_reco, ymax_reco])
                ### combine ts to muon arrival time (averaging)
                ts_reco = np.mean([hits["ts"][idx00], hits["ts"][idx01], hits["ts"][idx10], hits["ts"][idx11]])
                ### calculate max ts difference between hits in both layers (absolute value)
                ly_delta_ts = max_delta_ts
                ### store reco obj
                reco_muon_area_list.append({
                    "xmin": xmin_reco, 
                    "xmax": xmax_reco, 
                    "ymin": ymin_reco, 
                    "ymax": ymax_reco, 
                    "z0": z0_reco, 
                    "ts": ts_reco,
                    "pixel": pixel_index, 
                    "xcenter": xcenter_reco, 
                    "ycenter": ycenter_reco, 
                    "ly_delta_ts": ly_delta_ts,
                    "st0": st0,
                    "st1": st1,
                })
                ### remove the hits to not use them twice for multiple pixels
                #last_hits[0][st0] = copy.deepcopy(dummy_scint_hit)
                #last_hits[1][st1] = copy.deepcopy(dummy_scint_hit)
    # store in proper format
    n_reco_muon_areas = len(reco_muon_area_list)
    if not silent: print(f"Reconstructed {n_reco_muon_areas} muon areas from {n_hits} scintillator hits.")
    reco_muon_areas = {k: np.full(n_reco_muon_areas, 0, dtype=v) for k,v in params._muon_area_obj_keys.items()}
    if n_reco_muon_areas > 0:
        avail_keys = reco_muon_area_list[0].keys()
    for i in range(n_reco_muon_areas):
        for k in avail_keys:
            reco_muon_areas[k][i] = reco_muon_area_list[i][k]
    # sort by timestamp
    reco_muon_areas = data_utils.sort_by_key(data=reco_muon_areas, sort_key="ts")
    return reco_muon_areas

### apply dead time to raw scint hits
def deadtime_raw_hits(hits, *, silent=False, dead_time=params._raw_scintillator_ts_individual_dead_time):
    tmp_hits = copy.deepcopy(hits)
    ### -----------------------
    # apply dead time constraint to all individual channels (if specified dead time is > 0)
    if dead_time > 0:
        if not silent: print(f"apply dead time constraint for all individual channels of {dead_time} TU")
        cut_tmp_hits = {}
        for ly in derived_params._scint_inverted_remap_table.keys():
            cut_tmp_hits[ly] = {}
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                cut_tmp_hits[ly][st] = {}
                for sipm in [0,1]:
                    cut_tmp_hits[ly][st][sipm] = data_utils.cut_data(data=tmp_hits, conditions=[("ly","==",ly), ("st","==",st), ("sipm","==",sipm)], silent=True)
                    cut_tmp_hits[ly][st][sipm] = timestamp_utils.sort_by_timestamp(hits=cut_tmp_hits[ly][st][sipm], silent=True)
                    n_cut_hits = len(cut_tmp_hits[ly][st][sipm]["ts"])
                    allowed_indices = []
                    ts_list = np.array(cut_tmp_hits[ly][st][sipm]["ts"])
                    if len(ts_list) > 0:
                        cur_ts = ts_list[0]
                        for i in range(n_cut_hits): 
                            if (ts_list[i]) - (cur_ts) < dead_time:
                                cur_ts = ts_list[i] # deadtime wrt last hit
                                continue
                            cur_ts = ts_list[i] # deadtime wrt first hit
                            allowed_indices.append(i)
                        for k in cut_tmp_hits[ly][st][sipm].keys():
                            cut_tmp_hits[ly][st][sipm][k] = copy.deepcopy(cut_tmp_hits[ly][st][sipm][k][allowed_indices])
                        n_cut_hits_after = len(cut_tmp_hits[ly][st][sipm]['ts'])
                        if not silent: print(f"ly{ly} st{st} sipm{sipm} dead time cut flow: {n_cut_hits_after} / {n_cut_hits} = {n_cut_hits_after/max(1,n_cut_hits)}")
        # merge back to tmp_hits
        if not silent: print("merging data after applying individual dead time...")
        merge_data = []
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                for sipm in [0,1]:
                    merge_data.append(cut_tmp_hits[ly][st][sipm])
        tmp_hits = data_utils.merge_dataset(split_data=merge_data)
        if not silent: print("sort data by timestamp...")
        tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits, silent=True)
    return tmp_hits



########### TESTPULSES

### extract timestamps of testpulse hits for full readout system
def analyze_testpulses(hits, *, rel_thres=0.2, plot_hists=False, silent=False):
    tp_timing = {}
    for ly in derived_params._scint_inverted_remap_table.keys():
            tp_timing[ly] = {}
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                tp_timing[ly][st] = {}
                for sipm in [0,1]:
                    ch_ts_mean, ch_ts_err = 0, 0 # default values
                    # select hits of one channel
                    ch_hits = data_utils.cut_data(data=hits, conditions=[("ly","==",ly),("st","==",st),("sipm","==",sipm)], silent=True)
                    if data_utils.length(ch_hits) > 0:
                        # calculate histogram of hit timing (bin width = 1 ts unit)
                        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=ch_hits, key="ts_orbit", bin_centers="step1", silent=True)
                        # plot hist if desired
                        if plot_hists:
                            xlabel = params._key_symbols["ts_orbit"]
                            xlabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
                            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, show=True, title=f"Testpulse timing histogram", figsize=(12,5)) #  (Ly {ly}, St {st}, SiPM {sipm})
                        # select first peak of histogram (with lowest ts), the higher ts hits are due to ringing of the testpulse circuit
                        peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=rel_thres) # 20% of max amplitude for peak
                        if len(peak_indices) > 0:
                            sel_peak_indices = peak_indices[0] # first peak
                            hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
                            err_hists_peak = np.sqrt(hists_peak)
                            err_centers_peak = np.full( len(centers_peak), 8/np.sqrt(12) )
                            # calculate peak position (weighted mean)
                            ch_ts_mean, ch_ts_err = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
                            # # reject data if tp ts mean outside ts accept range
                            # if ch_ts_mean < accept_ts_range[0] or ch_ts_mean > accept_ts_range[1]:
                            #     ch_ts_mean, ch_ts_err = 0, 0
                    # store result
                    tp_timing[ly][st][sipm] = {"tp_ts_mean": ch_ts_mean, "tp_ts_err": ch_ts_err}    
    return tp_timing








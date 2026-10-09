###########################################
### DT HITS: extraction from the data words, dead time, time between hits of a cell, hits per cell
###########################################

import copy
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import data_utils, dt_chamber_utils, timestamp_utils

# -----------------------------------------

### extract dt hits from hit data (one block of the dumpfile, see data_utils.import_raw_lines)
# cut away all hit data not from dt, add dt specific keys (mapping from params.py), add the timestamp
# all_cells=False: remove the hits of masked / dead cells (params._dt_wire_mask, params._dt_dead_wires)
# overflow_state: orbit counter overflows of the earlier blocks, see timestamp_utils.add_timestamp
# returns the dt hits sorted by timestamp
def extract_dt_hits(hits, *, all_cells=False, overflow_state=None, silent=True):
    tmp_hits = copy.deepcopy(hits)
    n_hits = len(tmp_hits["ch"])
    if not silent: print(f"Extract DT hits from {n_hits} total hits...")
    # calculate mask to apply to cut away all hits not belonging to dt chamber (wrong ro_ch or invalid ch)
    dt_mask = np.full(n_hits, False, dtype=bool)
    for ro_ch in derived_params._dt_ro_chs:
        tmp_mask = np.ma.isin(tmp_hits["ro_ch"], [ro_ch])
        tmp_mask &= np.ma.isin(tmp_hits["ch"], list(derived_params._dt_remap_table[ro_ch].keys()))
        dt_mask |= tmp_mask
    # apply mask
    for k in tmp_hits.keys():
        tmp_hits[k] = tmp_hits[k][dt_mask]
    n_dt_hits = len(tmp_hits["ch"])
    if not silent: print(f"Found {n_dt_hits} DT hits. Adding DT specific keys...")
    # add specific dt keys
    tmp_hits |= {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_mapping_keys.items()} | {k: np.full(n_dt_hits, 0, dtype=v) for k,v in params._dt_other_keys.items() if k != "ts"}
    for i in range(n_dt_hits):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        # add keys according to remapping table
        for k in params._dt_mapping_keys.keys():
            tmp_hits[k][i] = derived_params._dt_remap_table[ro_ch][ch][k]
    # remove masked / dead cells
    if not all_cells:
        excluded_cells = dt_chamber_utils.excluded_cells()
        keep = np.full(n_dt_hits, True)
        for i in range(n_dt_hits):
            if (int(tmp_hits["sl"][i]), int(tmp_hits["ly"][i]), int(tmp_hits["wi"][i])) in excluded_cells:
                keep[i] = False
        for k in tmp_hits.keys():
            tmp_hits[k] = tmp_hits[k][keep]
    # add timestamp and sort by timestamp
    tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits, silent=silent, overflow_state=overflow_state)
    # add ts uncertainty
    err_ts_com = np.sqrt( (1/np.sqrt(12))**2 + params.dt_hit_add_ts_unc**2)
    tmp_hits["err_ts"][:] = err_ts_com
    return tmp_hits

### apply dead time constraint to all individual channels (if specified dead time is > 0): a hit is dropped if it comes less
# than params._dt_ts_individual_dead_time after the previous hit of the same cell; the first hit of every cell is dropped
# as well (its previous hit is unknown). returns the remaining hits, sorted by timestamp
def apply_dead_time(hits, *, silent=True):
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=hits, silent=True)
    if params._dt_ts_individual_dead_time <= 0 or data_utils.length(tmp_hits) == 0:
        return tmp_hits
    merge_data = []
    for sl in dt_chamber_utils.superlayers():
        for ly in dt_chamber_utils.layers(sl):
            for wi in dt_chamber_utils.wires(sl, ly):
                cell_hits = data_utils.cut_data(data=tmp_hits, conditions=[("sl","==",sl),("ly","==",ly),("wi","==",wi)], silent=True)
                n_cut_hits = len(cell_hits["ts"])
                allowed_indices = []
                ts_list = np.array(cell_hits["ts"])
                if len(ts_list) > 0:
                    cur_ts = ts_list[0]
                    for i in range(n_cut_hits):
                        if (ts_list[i]) - (cur_ts) < params._dt_ts_individual_dead_time:
                            cur_ts = ts_list[i] # deadtime wrt last hit
                            continue
                        cur_ts = ts_list[i] # deadtime wrt first hit
                        allowed_indices.append(i)
                for k in cell_hits.keys():
                    cell_hits[k] = cell_hits[k][allowed_indices]
                if not silent: print(f"sl{sl} ly{ly} wi{wi} dead time cut flow: {len(allowed_indices)} / {n_cut_hits}")
                if len(allowed_indices) > 0:
                    merge_data.append(cell_hits)
    if len(merge_data) == 0:
        return data_utils.cut_data(data=tmp_hits, conditions=[("ts", "<", -1)], silent=True)  # no hit left: empty table
    # merge back and sort by timestamp
    tmp_hits = data_utils.merge_dataset(split_data=merge_data, silent=True)
    tmp_hits = timestamp_utils.sort_by_timestamp(hits=tmp_hits, silent=True)
    return tmp_hits

### time between consecutive hits of the same cell, for all cells: list of time differences
def hit_time_differences(hits):
    # hit indices of every cell in time order: {(sl, ly, wi): [index, ...]}
    time_order = np.argsort(np.asarray(hits["ts"], dtype=np.float64), kind="stable")
    indices_of_cell = {}
    for i in time_order:
        cell = (int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i]))
        if cell not in indices_of_cell:
            indices_of_cell[cell] = []
        indices_of_cell[cell].append(i)

    differences = []
    for cell in sorted(indices_of_cell.keys()):
        indices = indices_of_cell[cell]
        for k in range(1, len(indices)):
            differences.append(np.float64(hits["ts"][indices[k]]) - np.float64(hits["ts"][indices[k - 1]]))
    return differences

### number of hits of every cell, added to cell_counts = {sl: {ly: {wi: count}}} (start with dt_chamber_utils.chamber_map(0))
def count_hits_per_cell(cell_counts, hits):
    for i in range(len(hits["ts"])):
        sl, ly, wi = int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i])
        cell_counts[sl][ly][wi] += 1

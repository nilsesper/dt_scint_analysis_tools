###########################################
### DT HITS: dead time, time between hits of a cell, hits per cell
###########################################

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils, hist_utils, root_utils
from analysis_tools.utils.root_utils import log

# -----------------------------------------

def sort_by_time(hits):
    order = np.argsort(np.asarray(hits["ts"], dtype=np.float64), kind="stable")
    sorted_hits = {}
    for key in hits:
        sorted_hits[key] = hits[key][order]
    return sorted_hits

### dead time of the readout channel: a hit is dropped if it comes less than params._dt_ts_individual_dead_time after the
# previous hit of the same cell. The first hit of every cell is dropped as well (its previous hit is unknown).
# returns the remaining hits, sorted by time
def apply_dead_time(hits):
    n_hits = len(hits["ts"])
    if n_hits == 0:
        return hits
    hits = sort_by_time(hits)
    if params._dt_ts_individual_dead_time <= 0:
        return hits
    keep = np.zeros(n_hits, dtype=bool)
    last_ts_of_cell = {}
    for i in range(n_hits):
        cell = (int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i]))
        ts = np.float64(hits["ts"][i])
        if cell in last_ts_of_cell and ts - last_ts_of_cell[cell] >= params._dt_ts_individual_dead_time:
            keep[i] = True
        last_ts_of_cell[cell] = ts
    kept_hits = {}
    for key in hits:
        kept_hits[key] = hits[key][keep]
    return kept_hits

# -----------------------------------------
# time between consecutive hits of the same cell
# -----------------------------------------

### the hit indices of every cell, in time order: {(sl, ly, wi): [index, ...]}
def hit_indices_per_cell(hits):
    time_order = np.argsort(np.asarray(hits["ts"], dtype=np.float64), kind="stable")
    indices_of_cell = {}
    for i in time_order:
        cell = (int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i]))
        if cell not in indices_of_cell:
            indices_of_cell[cell] = []
        indices_of_cell[cell].append(i)
    return indices_of_cell

### histogram of the time between consecutive hits of the same cell, summed over all cells
# filled chunk by chunk (fill_hit_diff_histogram), final uncertainties by finish_hit_diff_histogram
def empty_hit_diff_histogram(n_bins, ts_max):
    edges = np.linspace(0, ts_max, n_bins + 1)
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)
    return {"edges": edges, "centers": centers, "hist": hist, "entries": entries, "underflow": underflow, "overflow": overflow,
            "hist_err_right": hist_err_right, "hist_err_left": hist_err_left}

def fill_hit_diff_histogram(hit_diff_hist, hits):
    if len(hits["ts"]) < 2:
        return
    diffs = []
    err_diffs = []
    indices_of_cell = hit_indices_per_cell(hits)
    for cell in sorted(indices_of_cell.keys()):
        indices = indices_of_cell[cell]
        for k in range(1, len(indices)):
            previous, current = indices[k - 1], indices[k]
            diffs.append(np.float64(hits["ts"][current]) - np.float64(hits["ts"][previous]))
            err_diffs.append(np.sqrt(np.float64(hits["err_ts"][current]) ** 2 + np.float64(hits["err_ts"][previous]) ** 2))
    if len(diffs) == 0:
        return
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(
        data=np.array(diffs, dtype=np.float64), edges=hit_diff_hist["edges"], err_data=np.array(err_diffs, dtype=np.float64))
    hit_diff_hist["hist"] += hist
    hit_diff_hist["entries"] += entries
    hit_diff_hist["underflow"] += underflow
    hit_diff_hist["overflow"] += overflow
    hit_diff_hist["hist_err_right"] += hist_err_right
    hit_diff_hist["hist_err_left"] += hist_err_left

### the histogram with its uncertainties: {"edges", "centers", "hist", "err_hist", "err_hist_stat", "err_hist_down",
# "err_hist_up", "entries", "underflow", "overflow"}
def finish_hit_diff_histogram(hit_diff_hist):
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
        hist=hit_diff_hist["hist"], hist_err_right=hit_diff_hist["hist_err_right"], hist_err_left=hit_diff_hist["hist_err_left"], do_stat_err=True)
    return {"edges": hit_diff_hist["edges"], "centers": hit_diff_hist["centers"], "hist": hit_diff_hist["hist"],
            "err_hist": err_hist, "err_hist_stat": np.sqrt(hit_diff_hist["hist"]), "err_hist_down": err_hist_down, "err_hist_up": err_hist_up,
            "entries": hit_diff_hist["entries"], "underflow": hit_diff_hist["underflow"], "overflow": hit_diff_hist["overflow"]}

# -----------------------------------------
# hits per cell
# -----------------------------------------

### number of hits per cell of a dt hits file, read in chunks of step_size
# returns (cell_counts = {sl: {ly: {wi: count}}}, ts_min, ts_max, n_hits)
def count_hits_per_cell(dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    cell_counts = dt_chamber_utils.chamber_map(0)
    ts_min, ts_max, n_hits = None, None, 0
    chunks = root_utils.chunk_ranges(dt_hits_file, root_utils.DT_HITS_TREE, step_size)
    for i_chunk in range(len(chunks)):
        start, stop = chunks[i_chunk]
        hits = root_utils.read_entries(dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        if root_utils.length(hits) == 0:
            continue
        n_hits += root_utils.length(hits)
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(hits):,} hits counted")
        for i in range(len(hits["ts"])):
            sl, ly, wi = int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i])
            cell_counts[sl][ly][wi] += 1
        if ts_min is None or np.amin(hits["ts"]) < ts_min:
            ts_min = np.amin(hits["ts"])
        if ts_max is None or np.amax(hits["ts"]) > ts_max:
            ts_max = np.amax(hits["ts"])
    return cell_counts, ts_min, ts_max, n_hits

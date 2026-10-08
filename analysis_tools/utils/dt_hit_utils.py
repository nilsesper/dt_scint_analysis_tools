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
    return {k: v[order] for k, v in hits.items()}

### the hits of every cell next to each other, in time order inside a cell
# returns (order, same_cell_as_previous): hits[order] is that sequence, same_cell_as_previous[k] tells whether hit k + 1
# of the sequence belongs to the same cell as hit k
def cell_sequences(hits):
    cell = dt_chamber_utils.cell_id(hits["sl"], hits["ly"], hits["wi"])
    order = np.lexsort((np.asarray(hits["ts"], dtype=np.float64), cell))
    return order, cell[order][1:] == cell[order][:-1]

### dead time of the readout channel: a hit is dropped if it comes less than params._dt_ts_individual_dead_time after the
# previous hit of the same cell. The first hit of every cell is dropped as well (its previous hit is unknown).
# returns the remaining hits, sorted by time
def apply_dead_time(hits):
    if len(hits["ts"]) == 0:
        return hits
    hits = sort_by_time(hits)
    if params._dt_ts_individual_dead_time <= 0:
        return hits
    order, same_cell = cell_sequences(hits)
    ts = np.asarray(hits["ts"][order], dtype=np.float64)
    keep = np.zeros(len(order), dtype=bool)
    keep[order[1:]] = same_cell & (ts[1:] - ts[:-1] >= params._dt_ts_individual_dead_time)
    return {k: v[keep] for k, v in hits.items()}

### histogram of the time between consecutive hits of the same cell, summed over all cells, filled chunk by chunk
class HitTimeDifferenceHistogram:
    def __init__(self, n_bins, ts_max):
        self.edges = np.linspace(0, ts_max, n_bins + 1)
        self.centers, self.hist, self.entries, self.underflow, self.overflow, self.hist_err_right, self.hist_err_left = \
            hist_utils.create_empty_histogram(edges=self.edges)

    def fill(self, hits):
        if len(hits["ts"]) < 2:
            return
        order, same_cell = cell_sequences(hits)
        ts, err_ts = np.asarray(hits["ts"][order], dtype=np.float64), np.asarray(hits["err_ts"][order], dtype=np.float64)
        diffs = (ts[1:] - ts[:-1])[same_cell]
        err_diffs = np.sqrt(err_ts[1:] ** 2 + err_ts[:-1] ** 2)[same_cell]
        if len(diffs) == 0:
            return
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = \
            hist_utils.calculate_histogram_and_shifted_histograms(data=diffs, edges=self.edges, err_data=err_diffs)
        self.hist += hist
        self.entries += entries
        self.underflow += underflow
        self.overflow += overflow
        self.hist_err_right += hist_err_right
        self.hist_err_left += hist_err_left

    def result(self):
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
            hist=self.hist, hist_err_right=self.hist_err_right, hist_err_left=self.hist_err_left, do_stat_err=True)
        return {"edges": self.edges, "centers": self.centers, "hist": self.hist, "err_hist": err_hist, "err_hist_stat": np.sqrt(self.hist),
                "err_hist_down": err_hist_down, "err_hist_up": err_hist_up,
                "entries": self.entries, "underflow": self.underflow, "overflow": self.overflow}

### number of hits per cell of a dt hits file: returns ({sl: {ly: {wi: count}}}, ts_min, ts_max, n_hits)
def count_hits_per_cell(dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    counts = {sl: {ly: {wi: 0 for wi in dt_chamber_utils.wires(sl, ly)} for ly in range(0, 4)} for sl in range(1, 4)}
    ts_min, ts_max, n_hits = None, None, 0
    for i_chunk, n_chunks, chunk in root_utils.iterate_chunks(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size):
        if root_utils.length(chunk) == 0:
            continue
        n_hits += root_utils.length(chunk)
        log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(chunk):,} hits counted")
        cells, cell_counts = np.unique(np.stack([chunk["sl"], chunk["ly"], chunk["wi"]], axis=1), axis=0, return_counts=True)
        for (sl, ly, wi), count in zip(cells, cell_counts):
            counts[int(sl)][int(ly)][int(wi)] += int(count)
        ts_min = chunk["ts"].min() if ts_min is None else min(ts_min, chunk["ts"].min())
        ts_max = chunk["ts"].max() if ts_max is None else max(ts_max, chunk["ts"].max())
    return counts, ts_min, ts_max, n_hits

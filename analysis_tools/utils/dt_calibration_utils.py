###########################################
### DT TIMING CALIBRATION WITH TESTPULSES
###########################################
# Calibration = one time offset per cell, applied with a plus sign:
#   ts -> ts + ts_corr(sl, ly, wi),  err_ts -> sqrt(err_ts^2 + err_ts_corr^2)
# In memory a calibration is the dict dt_tp_corrections = {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}}
# Calibration files:
#   .root: tree "tree" with one row per cell (sl, ly, wi, ts_corr, err_ts_corr, ...), see calibrate_cells
#   .pcl:  the dict dt_tp_corrections (format of the old testpulse script)

import copy
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import data_utils, dt_chamber_utils, hist_utils, root_utils, timestamp_utils
from analysis_tools.utils.root_utils import log

# -----------------------------------------
# reading / writing calibrations
# -----------------------------------------

def load_tp_corrections(dt_tp_corrections_file):
    root_utils.check_input_file(dt_tp_corrections_file)
    if dt_tp_corrections_file.endswith(".pcl"):
        return data_utils.load_pickle(file=dt_tp_corrections_file, silent=True)
    rows = root_utils.read_tree(dt_tp_corrections_file, root_utils.DEFAULT_TREE, branches=["sl", "ly", "wi", "ts_corr", "err_ts_corr"])
    dt_tp_corrections = {}
    for i in range(len(rows["sl"])):
        sl, ly, wi = int(rows["sl"][i]), int(rows["ly"][i]), int(rows["wi"][i])
        if sl not in dt_tp_corrections:
            dt_tp_corrections[sl] = {}
        if ly not in dt_tp_corrections[sl]:
            dt_tp_corrections[sl][ly] = {}
        dt_tp_corrections[sl][ly][wi] = {"ts_corr": float(rows["ts_corr"][i]), "err_ts_corr": float(rows["err_ts_corr"][i])}
    return dt_tp_corrections

### True if obj looks like a calibration dict {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}}
def is_tp_corrections_dict(obj):
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    first_sl = list(obj.values())[0]
    if not isinstance(first_sl, dict) or len(first_sl) == 0:
        return False
    first_ly = list(first_sl.values())[0]
    if not isinstance(first_ly, dict) or len(first_ly) == 0:
        return False
    first_cell = list(first_ly.values())[0]
    return isinstance(first_cell, dict) and "ts_corr" in first_cell and "err_ts_corr" in first_cell

### all cells (sl, ly, wi) of a calibration dict, sorted
def cells_of_tp_corrections(dt_tp_corrections):
    cells = []
    for sl in dt_tp_corrections:
        for ly in dt_tp_corrections[sl]:
            for wi in dt_tp_corrections[sl][ly]:
                cells.append((int(sl), int(ly), int(wi)))
    return sorted(cells)

### write a calibration dict as ROOT file (one row per cell), e.g. to convert an old .pcl calibration
def store_tp_corrections_root(dt_tp_corrections, dt_tp_corrections_file):
    cells = cells_of_tp_corrections(dt_tp_corrections)
    n_cells = len(cells)
    rows = {}
    for key in ["sl", "ly", "wi"]:
        rows[key] = np.zeros(n_cells, dtype=np.int32)
    for key in ["ts_corr", "err_ts_corr"]:
        rows[key] = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        sl, ly, wi = cells[i]
        rows["sl"][i], rows["ly"][i], rows["wi"][i] = sl, ly, wi
        rows["ts_corr"][i] = dt_tp_corrections[sl][ly][wi]["ts_corr"]
        rows["err_ts_corr"][i] = dt_tp_corrections[sl][ly][wi]["err_ts_corr"]
    root_utils.write_file(dt_tp_corrections_file, rows)
    return n_cells

# -----------------------------------------
# applying a calibration
# -----------------------------------------

### correction {"ts_corr", "err_ts_corr"} of one cell, or None if the calibration has no value for it
def tp_correction_of_cell(dt_tp_corrections, sl, ly, wi):
    if sl not in dt_tp_corrections:
        return None
    if ly not in dt_tp_corrections[sl]:
        return None
    if wi not in dt_tp_corrections[sl][ly]:
        return None
    return dt_tp_corrections[sl][ly][wi]

### load a calibration file and print what it contains
def read_tp_corrections(dt_tp_corrections_file, label):
    dt_tp_corrections = load_tp_corrections(dt_tp_corrections_file)
    cells = cells_of_tp_corrections(dt_tp_corrections)
    if len(cells) == 0:
        raise RuntimeError(f"No timing corrections found in {dt_tp_corrections_file}.")
    corrections = []
    for sl, ly, wi in cells:
        corrections.append(dt_tp_corrections[sl][ly][wi]["ts_corr"])
    n_chamber_cells_missing = 0
    for sl, ly, wi in dt_chamber_utils.chamber_cells():
        if tp_correction_of_cell(dt_tp_corrections, sl, ly, wi) is None:
            n_chamber_cells_missing += 1
    message = f"[{label}] timing corrections for {len(cells):,} cells, between {np.amin(corrections):.2f} and {np.amax(corrections):.2f} ts units"
    if n_chamber_cells_missing > 0:
        message += f"; {n_chamber_cells_missing} cells of the chamber have no correction (left uncorrected)"
    log(message)
    return dt_tp_corrections

### apply timing calibration to dt hits
# generated by testpulse run, see functions below: correction object "dt_tp_corrections"
# corrections in dt_tp_corrections should be applied with a plus sign
#   as follows: ts(wi)_corrected = ts(wi)_uncorrected + correction(wi)
# dt_tp_corrections = {"ts_corr": sl_ts_target - ch_ts_mean, "err_ts_corr": ch_ts_err}
# hits of cells without correction are left as they are, these cells are added to the set cells_without_correction
def apply_timing_calibration(hits, *, dt_tp_corrections, cells_without_correction, silent=True):
    n_hits = len(hits["ch"])
    corr_hits = copy.deepcopy(hits)
    if not silent: print(f"Applying testpulse timing correction to {n_hits} DT hits...")
    for i in range(n_hits):
        sl = int(hits["sl"][i])
        ly = int(hits["ly"][i])
        wi = int(hits["wi"][i])
        ts = hits["ts"][i]
        err_ts = hits["err_ts"][i]
        correction = tp_correction_of_cell(dt_tp_corrections, sl, ly, wi)
        if correction is None:
            cells_without_correction.add((sl, ly, wi))
            continue
        # correct timestamp
        ts_corr = np.float64(ts) + correction["ts_corr"]
        err_ts_corr = np.sqrt(err_ts**2 + correction["err_ts_corr"]**2)
        corr_hits["ts"][i] = ts_corr
        corr_hits["err_ts"][i] = err_ts_corr
        # remap correction to bx oc tdc (htg timestamp), for consistency reasons
        (oc, bx, tdc) = timestamp_utils.remap_htg_timestamp(ts_corr)
        corr_hits["oc"][i], corr_hits["bx"][i], corr_hits["tdc"][i] = oc, bx, tdc
    return corr_hits

def warn_about_cells_without_correction(cells_without_correction, label):
    if len(cells_without_correction) == 0:
        return
    cells = sorted(cells_without_correction)
    message = f"[{label}] WARNING: hits of {len(cells):,} cells (sl, ly, wi) without timing correction were left uncorrected: {cells[:10]}"
    if len(cells) > 10:
        message += " ..."
    log(message)

# -----------------------------------------
# calibration from a testpulse run
# -----------------------------------------

### histograms of ts_orbit (bins of 1 ts unit), filled block by block (fill_testpulse_histograms):
#   "per_cell": {(sl, ly, wi): {ts_orbit: number of hits}}  (only the occupied bins)
#   "per_sl":   {sl: array with the number of hits for every ts_orbit}
def empty_testpulse_histograms():
    orbit_length = int(derived_params._orbit_to_timestamp)
    tp_histograms = {"orbit_length": orbit_length, "per_cell": {}, "per_sl": {}}
    for sl in dt_chamber_utils.superlayers():
        tp_histograms["per_sl"][sl] = np.zeros(orbit_length, dtype=np.int64)
    return tp_histograms

def fill_testpulse_histograms(tp_histograms, hits, ts_orbit):
    for i in range(len(ts_orbit)):
        sl, ly, wi = int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i])
        ts = int(ts_orbit[i])
        if (sl, ly, wi) not in tp_histograms["per_cell"]:
            tp_histograms["per_cell"][(sl, ly, wi)] = {}
        cell_hist = tp_histograms["per_cell"][(sl, ly, wi)]
        if ts not in cell_hist:
            cell_hist[ts] = 0
        cell_hist[ts] += 1
        if sl in tp_histograms["per_sl"]:
            tp_histograms["per_sl"][sl][ts] += 1
    return tp_histograms

### occupied ts_orbit values of one cell (sorted) and their number of hits, as two arrays
def testpulse_times_of_cell(tp_histograms, sl, ly, wi):
    cell_hist = {}
    if (sl, ly, wi) in tp_histograms["per_cell"]:
        cell_hist = tp_histograms["per_cell"][(sl, ly, wi)]
    values = np.array(sorted(cell_hist.keys()), dtype=np.int64)
    counts = np.zeros(len(values), dtype=np.int64)
    for k in range(len(values)):
        counts[k] = cell_hist[int(values[k])]
    return values, counts

### first peak of the testpulse timing of one cell (values, counts: occupied ts_orbit values and their number of hits)
# calculate histogram of hit timing (bin width = 1 ts unit), select first peak of histogram (with lowest ts), the higher
# ts hits are due to ringing of the testpulse circuit; peak position = weighted mean of the bin centres
# returns (mean, err, number of hits in the peak, first bin, last bin), or None if the cell has no hits / no peak
def first_testpulse_peak(values, counts, rel_thres):
    if len(values) == 0:
        return None
    cell_hits = {"ts_orbit": np.repeat(values, counts)}
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cell_hits, key="ts_orbit", bin_centers="step1", silent=True)
    peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=rel_thres)
    if len(peak_indices) == 0:
        return None
    sel_peak_indices = peak_indices[0] # first peak
    hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
    err_hists_peak = np.sqrt(hists_peak)
    err_centers_peak = np.full(len(centers_peak), 1/np.sqrt(12))
    mean, err = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
    return mean, err, int(np.sum(hists_peak)), int(centers_peak[0]), int(centers_peak[-1])

CELL_INT_KEYS = ["sl", "ly", "wi", "fe_id", "n_hits", "n_peak_hits", "peak_ts_min", "peak_ts_max", "valid", "masked"]
CELL_FLOAT_KEYS = ["tp_ts_mean_raw", "tp_ts_err_raw", "tp_offset", "tp_ts_mean", "tp_ts_err", "ts_target", "ts_corr", "err_ts_corr"]

### calibration of every cell of the chamber from its testpulse time histogram
# 1. testpulse time of the cell = position of the first peak
# 2. correct_for_offsets: subtract the known testpulse delay of the frontend connector (params._tp_time_offset,
#    uncertainty params._tp_time_offset_err), e.g. the longer theta testpulse latency and old cables
# 3. ts_corr = target - testpulse time, err_ts_corr = its uncertainty; target = mean over the valid cells which are not
#    masked / dead, of the whole chamber (alignment "chamber") or of the superlayer (alignment "sl")
# cells without testpulse peak: ts_corr = err_ts_corr = 0, valid = 0
# returns one row per cell with the keys CELL_INT_KEYS + CELL_FLOAT_KEYS
def calibrate_cells(tp_histograms, *, rel_thres, alignment, correct_for_offsets, label):
    cells = dt_chamber_utils.chamber_cells()
    n_cells = len(cells)
    excluded_cells = dt_chamber_utils.excluded_cells()
    calib = {}
    for key in CELL_INT_KEYS:
        calib[key] = np.zeros(n_cells, dtype=np.int32)
    for key in CELL_FLOAT_KEYS:
        calib[key] = np.zeros(n_cells, dtype=np.float64)

    # 1. + 2. testpulse time of every cell
    for i in range(n_cells):
        sl, ly, wi = cells[i]
        values, counts = testpulse_times_of_cell(tp_histograms, sl, ly, wi)
        peak = first_testpulse_peak(values, counts, rel_thres)
        if peak is None: print(f"no data for {(sl, ly, wi)}")
        mean, err = 0.0, 0.0
        if peak is not None:
            mean, err, calib["n_peak_hits"][i], calib["peak_ts_min"][i], calib["peak_ts_max"][i] = peak
        calib["sl"][i], calib["ly"][i], calib["wi"][i] = sl, ly, wi
        calib["fe_id"][i] = derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"]
        if (sl, ly, wi) in excluded_cells:
            calib["masked"][i] = 1
        calib["n_hits"][i] = int(counts.sum())
        calib["tp_ts_mean_raw"][i], calib["tp_ts_err_raw"][i] = mean, err
        if correct_for_offsets:
            offset = params._tp_time_offset[sl][dt_chamber_utils.fe_name_of_cell(sl, ly, wi)]
            calib["tp_offset"][i] = offset
            mean = mean - offset
            err = np.sqrt(params._tp_time_offset_err ** 2 + err ** 2)
        calib["tp_ts_mean"][i], calib["tp_ts_err"][i] = mean, err
        if peak is not None and mean > 0:
            calib["valid"][i] = 1

    if np.sum(calib["valid"]) == 0:
        raise RuntimeError("No cell with a testpulse peak found -- is this a testpulse run?")

    # 3. groups of cells which are aligned to a common target time
    groups = []
    if alignment == "chamber":
        groups.append(list(range(n_cells)))
    else:
        for sl in dt_chamber_utils.superlayers():
            group = []
            for i in range(n_cells):
                if calib["sl"][i] == sl:
                    group.append(i)
            groups.append(group)

    for group in groups:
        times_for_target = []
        for i in group:
            if calib["valid"][i] == 1 and calib["masked"][i] == 0:
                times_for_target.append(calib["tp_ts_mean"][i])
        if len(times_for_target) == 0:
            log(f"[{label}] WARNING: no valid cell in SL {calib['sl'][group[0]]}, its cells are not corrected")
            continue
        target = np.mean(np.array(times_for_target, dtype=np.float64))
        for i in group:
            calib["ts_target"][i] = target
            if calib["valid"][i] == 1:
                calib["ts_corr"][i] = target - calib["tp_ts_mean"][i]
                calib["err_ts_corr"][i] = calib["tp_ts_err"][i]
    return calib

### calibration dict {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}} from the rows of calibrate_cells
def tp_corrections_dict(calib):
    dt_tp_corrections = {}
    for i in range(len(calib["sl"])):
        sl, ly, wi = int(calib["sl"][i]), int(calib["ly"][i]), int(calib["wi"][i])
        if sl not in dt_tp_corrections:
            dt_tp_corrections[sl] = {}
        if ly not in dt_tp_corrections[sl]:
            dt_tp_corrections[sl][ly] = {}
        dt_tp_corrections[sl][ly][wi] = {"ts_corr": calib["ts_corr"][i], "err_ts_corr": calib["err_ts_corr"][i]}
    return dt_tp_corrections

### ROOT histograms of a calibration: per superlayer TH2D (x = wire, y = layer) of ts_corr, tp_ts_mean, n_peak_hits
# and TH1D of ts_orbit of all hits
def calibration_histograms(calib, tp_histograms):
    root_histograms = {}
    for sl in dt_chamber_utils.superlayers():
        rows_of_sl = []
        for i in range(len(calib["sl"])):
            if calib["sl"][i] == sl:
                rows_of_sl.append(i)
        wi_min, wi_max = np.amin(calib["wi"][rows_of_sl]), np.amax(calib["wi"][rows_of_sl])
        ly_min, ly_max = np.amin(calib["ly"][rows_of_sl]), np.amax(calib["ly"][rows_of_sl])
        wi_edges = np.arange(wi_min - 0.5, wi_max + 1.5)
        ly_edges = np.arange(ly_min - 0.5, ly_max + 1.5)
        for key in ["ts_corr", "tp_ts_mean", "n_peak_hits"]:
            hist2d = np.zeros((len(wi_edges) - 1, len(ly_edges) - 1))
            for i in rows_of_sl:
                hist2d[calib["wi"][i] - wi_min, calib["ly"][i] - ly_min] = calib[key][i]
            root_histograms[f"{key}_sl{sl}"] = (hist2d, wi_edges, ly_edges)
        ts_orbit_edges = np.arange(tp_histograms["orbit_length"] + 1) - 0.5
        root_histograms[f"ts_orbit_sl{sl}"] = (tp_histograms["per_sl"][sl], ts_orbit_edges)
    return root_histograms

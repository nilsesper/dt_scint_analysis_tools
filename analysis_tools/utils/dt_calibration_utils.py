###########################################
### DT TIMING CALIBRATION WITH TESTPULSES
###########################################
# Calibration = one time offset per cell, applied with a plus sign:
#   ts -> ts + ts_corr(sl, ly, wi),  err_ts -> sqrt(err_ts^2 + err_ts_corr^2)
# Calibration files:
#   .root: tree "tree" with one row per cell (sl, ly, wi, ts_corr, err_ts_corr, ...), see calibrate_cells
#   .pcl:  {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}} (format of the old testpulse script)

import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import data_utils, dt_chamber_utils, hist_utils, root_utils
from analysis_tools.utils.root_utils import log

# -----------------------------------------
# reading / writing calibrations
# -----------------------------------------

def load_tp_corrections(dt_tp_corrections_file):
    root_utils.check_input_file(dt_tp_corrections_file)
    if dt_tp_corrections_file.endswith(".pcl"):
        return data_utils.load_pickle(file=dt_tp_corrections_file, silent=True)
    rows = root_utils.read_branches(dt_tp_corrections_file, ["sl", "ly", "wi", "ts_corr", "err_ts_corr"], root_utils.DEFAULT_TREE)
    dt_tp_corrections = {}
    for sl, ly, wi, corr, err_corr in zip(rows["sl"], rows["ly"], rows["wi"], rows["ts_corr"], rows["err_ts_corr"]):
        dt_tp_corrections.setdefault(int(sl), {}).setdefault(int(ly), {})[int(wi)] = {"ts_corr": float(corr), "err_ts_corr": float(err_corr)}
    return dt_tp_corrections

def is_tp_corrections_dict(obj):
    try:
        first_cell = next(iter(next(iter(next(iter(obj.values())).values())).values()))
        return isinstance(first_cell, dict) and "ts_corr" in first_cell and "err_ts_corr" in first_cell
    except (AttributeError, StopIteration, TypeError):
        return False

### write a calibration dict as ROOT file (one row per cell), e.g. to convert an old .pcl calibration
def store_tp_corrections_root(dt_tp_corrections, dt_tp_corrections_file):
    cells = sorted((int(sl), int(ly), int(wi)) for sl, lys in dt_tp_corrections.items() for ly, wis in lys.items() for wi in wis.keys())
    rows = {key: np.array([cell[i] for cell in cells], dtype=np.int32) for i, key in enumerate(["sl", "ly", "wi"])}
    for key in ["ts_corr", "err_ts_corr"]:
        rows[key] = np.array([dt_tp_corrections[sl][ly][wi][key] for sl, ly, wi in cells], dtype=np.float64)
    root_utils.write_tree(dt_tp_corrections_file, rows, tree=root_utils.DEFAULT_TREE)
    return len(cells)

# -----------------------------------------
# applying a calibration
# -----------------------------------------

### the calibration as lookup arrays [sl, ly, wi]; NaN for cells without correction
class TimingCorrection:
    def __init__(self, dt_tp_corrections):
        self.ts_corr = np.full((256, 256, 256), np.nan, dtype=np.float64)
        self.err_ts_corr = np.full((256, 256, 256), np.nan, dtype=np.float64)
        for sl, lys in dt_tp_corrections.items():
            for ly, wis in lys.items():
                for wi, corr in wis.items():
                    self.ts_corr[int(sl), int(ly), int(wi)] = corr["ts_corr"]
                    self.err_ts_corr[int(sl), int(ly), int(wi)] = corr["err_ts_corr"]
        self.cells_without_correction = set()  # cells with hits but without correction, filled by apply

    @classmethod
    def from_file(cls, dt_tp_corrections_file, *, label):
        correction = cls(load_tp_corrections(dt_tp_corrections_file))
        n_cells = int(np.sum(~np.isnan(correction.ts_corr)))
        if n_cells == 0:
            raise RuntimeError(f"No timing corrections found in {dt_tp_corrections_file}.")
        n_chamber_cells_missing = int(np.sum(dt_chamber_utils.readout_tables().is_chamber_cell & np.isnan(correction.ts_corr)))
        log(f"[{label}] timing corrections for {n_cells:,} cells, between {np.nanmin(correction.ts_corr):.2f} and {np.nanmax(correction.ts_corr):.2f} ts units"
            + (f"; {n_chamber_cells_missing} cells of the chamber have no correction (left uncorrected)" if n_chamber_cells_missing > 0 else ""))
        return correction

    ### corrected copy of the hits; oc / bx / tdc are recalculated from the corrected timestamp
    # hits of cells without correction are left as they are
    def apply(self, hits):
        sl, ly, wi = hits["sl"].astype(np.intp), hits["ly"].astype(np.intp), hits["wi"].astype(np.intp)
        corr, err_corr = self.ts_corr[sl, ly, wi], self.err_ts_corr[sl, ly, wi]
        missing = np.isnan(corr)
        if np.any(missing):
            self.cells_without_correction |= {(int(a), int(b), int(c)) for a, b, c in zip(sl[missing], ly[missing], wi[missing])}
            corr, err_corr = np.where(missing, 0.0, corr), np.where(missing, 0.0, err_corr)
        corrected = dict(hits)
        ts = np.asarray(hits["ts"], dtype=np.float64) + corr
        corrected["ts"] = ts.astype(hits["ts"].dtype)
        corrected["err_ts"] = np.sqrt(hits["err_ts"] ** 2 + err_corr ** 2).astype(hits["err_ts"].dtype)
        ts_int = np.round(ts, 0).astype(np.uint64)
        corrected["oc"] = ((ts_int % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp).astype(hits["oc"].dtype)
        corrected["bx"] = ((ts_int % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp).astype(hits["bx"].dtype)
        corrected["tdc"] = ((ts_int % derived_params._bx_to_timestamp) // np.uint64(derived_params._tdc_to_timestamp)).astype(hits["tdc"].dtype)
        return corrected

    def warn_about_cells_without_correction(self, label):
        if len(self.cells_without_correction) > 0:
            cells = sorted(self.cells_without_correction)
            log(f"[{label}] WARNING: hits of {len(cells):,} cells (sl, ly, wi) without timing correction were left uncorrected: "
                f"{cells[:10]}{' ...' if len(cells) > 10 else ''}")

# -----------------------------------------
# calibration from a testpulse run
# -----------------------------------------

### time inside the orbit: ts_orbit = tdc + 32 * bx
def ts_in_orbit(hits):
    return (hits["tdc"].astype(np.int64) * int(derived_params._tdc_to_timestamp)
            + hits["bx"].astype(np.int64) * int(derived_params._bx_to_timestamp))

### histograms of ts_orbit (bins of 1 ts unit): per cell (sparse: only the occupied bins are stored) and per superlayer
class TestpulseTimeHistograms:
    def __init__(self):
        self.orbit_length = int(derived_params._orbit_to_timestamp)
        self.ts_bits = int(np.ceil(np.log2(self.orbit_length)))
        self.cell_and_ts = np.zeros(0, dtype=np.int64)  # occupied bins: (cell id << ts_bits) | ts_orbit, sorted
        self.counts = np.zeros(0, dtype=np.int64)
        self.per_sl = {sl: np.zeros(self.orbit_length, dtype=np.int64) for sl in dt_chamber_utils.superlayers()}

    def add(self, hits, ts_orbit):
        cell = dt_chamber_utils.cell_id(hits["sl"], hits["ly"], hits["wi"])
        bins, counts = np.unique((cell << self.ts_bits) | ts_orbit, return_counts=True)
        self.cell_and_ts, inverse = np.unique(np.concatenate((self.cell_and_ts, bins)), return_inverse=True)
        self.counts = np.bincount(inverse, weights=np.concatenate((self.counts, counts)), minlength=len(self.cell_and_ts)).astype(np.int64)
        for sl, hist in self.per_sl.items():
            hist += np.bincount(ts_orbit[hits["sl"] == sl], minlength=self.orbit_length)[:self.orbit_length]

    ### occupied ts_orbit values of one cell (sorted) and their counts
    def of_cell(self, sl, ly, wi):
        cells = self.cell_and_ts >> self.ts_bits
        this_cell = dt_chamber_utils.cell_id(sl, ly, wi)
        start, stop = int(np.searchsorted(cells, this_cell)), int(np.searchsorted(cells, this_cell, side="right"))
        return self.cell_and_ts[start:stop] & ((1 << self.ts_bits) - 1), self.counts[start:stop]

### first peak of the testpulse timing of one cell (values, counts: occupied ts_orbit values and their number of hits)
# histogram with bins of 1 ts unit, peaks = groups of neighbouring bins with at least rel_thres * (highest bin).
# The first peak (lowest time) is the direct testpulse response, the later ones come from ringing of the testpulse
# circuit. Peak position = weighted mean of the bin centres.
# returns (mean, err, number of hits in the peak, first bin, last bin), or None if the cell has no hits
def first_testpulse_peak(values, counts, rel_thres):
    if len(values) == 0:
        return None
    hist = np.zeros(int(values[-1] - values[0]) + 1, dtype=np.int64)
    hist[(values - values[0]).astype(np.intp)] = counts
    above = hist >= np.amax(hist) * rel_thres
    start = int(np.argmax(above))
    stop = start + (int(np.argmin(above[start:])) if not above[start:].all() else len(above) - start)
    hist_peak = hist[start:stop].astype(np.float64)
    centers_peak = values[0] + np.arange(start, stop, dtype=np.float64)
    mean, err = hist_utils.weighted_mean_peak_position(hist=hist_peak, centers=centers_peak, err_hist=np.sqrt(hist_peak),
                                                       err_centers=np.full(len(hist_peak), 1 / np.sqrt(12)))
    return mean, err, int(hist_peak.sum()), int(centers_peak[0]), int(centers_peak[-1])

CELL_INT_KEYS = ("sl", "ly", "wi", "fe_id", "n_hits", "n_peak_hits", "peak_ts_min", "peak_ts_max", "valid", "masked")
CELL_FLOAT_KEYS = ("tp_ts_mean_raw", "tp_ts_err_raw", "tp_offset", "tp_ts_mean", "tp_ts_err", "ts_target", "ts_corr", "err_ts_corr")

### calibration of every cell of the chamber from its testpulse time histogram
# 1. testpulse time of the cell = position of the first peak
# 2. correct_for_offsets: subtract the known testpulse delay of the frontend connector (params._tp_time_offset,
#    uncertainty params._tp_time_offset_err), e.g. the longer theta testpulse latency and old cables
# 3. ts_corr = target - testpulse time, err_ts_corr = its uncertainty; target = mean over the valid cells which are not
#    masked / dead, of the whole chamber (alignment "chamber") or of the superlayer (alignment "sl")
# cells without testpulse peak: ts_corr = err_ts_corr = 0, valid = 0
# returns one row per cell with the keys CELL_INT_KEYS + CELL_FLOAT_KEYS
def calibrate_cells(histograms, *, rel_thres, alignment, correct_for_offsets, label):
    cells = dt_chamber_utils.chamber_cells()
    calib = {k: np.zeros(len(cells), dtype=np.int32) for k in CELL_INT_KEYS} | {k: np.zeros(len(cells), dtype=np.float64) for k in CELL_FLOAT_KEYS}
    is_analysed_cell = dt_chamber_utils.readout_tables().is_analysed_cell
    for i, (sl, ly, wi) in enumerate(cells):
        values, counts = histograms.of_cell(sl, ly, wi)
        peak = first_testpulse_peak(values, counts, rel_thres)
        mean, err = 0.0, 0.0
        if peak is not None:
            mean, err, calib["n_peak_hits"][i], calib["peak_ts_min"][i], calib["peak_ts_max"][i] = peak
        calib["sl"][i], calib["ly"][i], calib["wi"][i] = sl, ly, wi
        calib["fe_id"][i] = derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"]
        calib["masked"][i] = int(not is_analysed_cell[sl, ly, wi])
        calib["n_hits"][i] = int(counts.sum())
        calib["tp_ts_mean_raw"][i], calib["tp_ts_err_raw"][i] = mean, err
        if correct_for_offsets:
            offset = params._tp_time_offset[sl][dt_chamber_utils.fe_name_of_cell(sl, ly, wi)]
            calib["tp_offset"][i] = offset
            mean, err = mean - offset, np.sqrt(params._tp_time_offset_err ** 2 + err ** 2)
        calib["tp_ts_mean"][i], calib["tp_ts_err"][i] = mean, err
        calib["valid"][i] = int(peak is not None and mean > 0)

    valid = calib["valid"] == 1
    if not valid.any():
        raise RuntimeError("No cell with a testpulse peak found -- is this a testpulse run?")
    used_for_target = valid & (calib["masked"] == 0)
    groups = [np.full(len(cells), True)] if alignment == "chamber" else [calib["sl"] == sl for sl in dt_chamber_utils.superlayers()]
    for group in groups:
        if not (group & used_for_target).any():
            log(f"[{label}] WARNING: no valid cell in SL {calib['sl'][group][0]}, its cells are not corrected")
            continue
        target = np.mean(calib["tp_ts_mean"][group & used_for_target])
        calib["ts_target"][group] = target
        calib["ts_corr"][group & valid] = target - calib["tp_ts_mean"][group & valid]
        calib["err_ts_corr"][group & valid] = calib["tp_ts_err"][group & valid]
    return calib

### calibration dict {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}} from the rows of calibrate_cells
def tp_corrections_dict(calib):
    corrections = {}
    for i in range(len(calib["sl"])):
        sl, ly, wi = int(calib["sl"][i]), int(calib["ly"][i]), int(calib["wi"][i])
        corrections.setdefault(sl, {}).setdefault(ly, {})[wi] = {"ts_corr": calib["ts_corr"][i], "err_ts_corr": calib["err_ts_corr"][i]}
    return corrections

### ROOT histograms of a calibration: per superlayer TH2D (x = wire, y = layer) of ts_corr, tp_ts_mean, n_peak_hits
# and TH1D of ts_orbit of all hits
def calibration_histograms(calib, histograms):
    root_histograms = {}
    for sl in dt_chamber_utils.superlayers():
        in_sl = calib["sl"] == sl
        wi, ly = calib["wi"][in_sl], calib["ly"][in_sl]
        wi_edges = np.arange(wi.min() - 0.5, wi.max() + 1.5)
        ly_edges = np.arange(ly.min() - 0.5, ly.max() + 1.5)
        for key in ("ts_corr", "tp_ts_mean", "n_peak_hits"):
            hist2d = np.zeros((len(wi_edges) - 1, len(ly_edges) - 1))
            hist2d[wi - wi.min(), ly - ly.min()] = calib[key][in_sl]
            root_histograms[f"{key}_sl{sl}"] = (hist2d, wi_edges, ly_edges)
        root_histograms[f"ts_orbit_sl{sl}"] = (histograms.per_sl[sl], np.arange(histograms.orbit_length + 1) - 0.5)
    return root_histograms

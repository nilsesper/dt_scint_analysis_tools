###########################################
### DT WORKFLOW STAGES ON ROOT FILES
###########################################
# One function per workflow stage. Every stage reads one (or two) ROOT files and writes
# one ROOT file, in chunks, so that memory use does not grow with the size of the run.
# The command line scripts in scripts/dt_root/ are thin wrappers around these functions.
#
#   dumpfile (.txt)
#     -> dt hits                 convert_dumpfile_to_dt_hits
#          -> corrected dt hits  apply_timing_correction   (optional, testpulse timing calibration)
#          -> hit diff hist      dt_hits_to_hit_diff_hist  (side product)
#          -> cell counts        dt_hits_to_cell_counts    (side product)
#          -> sl patterns        dt_hits_to_sl_patterns    (applies dead time)
#               -> sl fits       fit_sl_patterns_file      (fixed drift velocity)
#                    -> cut sl fits         apply_cuts_file
#                         -> super fits     sl_fits_to_super_fits   (both phi sls combined, fixed drift velocity by default)
#                              -> cut super fits   apply_cuts_file
#                                   -> dt muons    super_fits_to_dt_muons (cut super fits + theta sl fits of the cut sl fits)
#
# The reconstruction functions themselves live in dt_utils.py.

import collections
import contextlib
import itertools
import multiprocessing
import os
import gc
import time
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import data_utils, hist_utils, root_utils
from analysis_tools.utils.root_utils import log

# default suffix of the super fit result branches (e.g. "t0_super_fits", "chi2/ndf_super_fits")
DEFAULT_SUPER_FIT_SUFFIX = "_super_fits"

# -----------------------------------------
# helpers
# -----------------------------------------

### parse cut string "key1,operator1,value1;key2,operator2,value2;..." into [(key, operator, value)]
# value is a number or a parameter name written as "params._name"
# all cuts are AND-ed together (see data_utils.cut_data for the allowed operators)
def parse_cuts(cuts_str):
    cuts_list = []
    if cuts_str is None or cuts_str.strip() == "":
        return cuts_list
    for cut_str in cuts_str.split(";"):
        if cut_str.strip() == "":
            continue
        parts = cut_str.split(",")
        if len(parts) != 3:
            raise ValueError(f"Cannot read cut \"{cut_str}\". Expected format: key,operator,value")
        key, operator, value = [p.strip() for p in parts]
        if value.startswith("params."):
            value = getattr(params, value.split("params.")[1])
        else:
            value = float(value)
        cuts_list.append((key, operator, value))
    return cuts_list

def _check_cut_keys(cuts_list, data, file):
    for key, _, _ in cuts_list:
        if key not in data:
            raise KeyError(f"Cut key \"{key}\" is not a branch of {file}.")

def _split_rows(data, n_parts):
    parts = [{} for _ in range(n_parts)]
    for k, v in data.items():
        for i, piece in enumerate(np.array_split(v, n_parts)):
            parts[i][k] = piece
    return parts

### a pool of n_proc worker processes, or None if n_proc <= 1 (to be used in a with statement)
@contextlib.contextmanager
def _optional_pool(n_proc):
    if n_proc <= 1:
        yield None
        return
    pool = multiprocessing.Pool(n_proc)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def _call(job):
    function, data, data_key, kwargs = job
    return function(**{data_key: data}, **kwargs)

### run a row-wise function (one output row per input row, rows independent) on n_proc processes
# the result is the same as calling function(data) directly
def _run_rowwise(function, data, data_key, kwargs, n_proc):
    n_rows = root_utils.length(data)
    if n_proc <= 1 or n_rows < 4 * n_proc:
        return function(**{data_key: data}, **kwargs)
    jobs = [(function, part, data_key, kwargs) for part in _split_rows(data, n_proc)]
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(_call, jobs)
    return {k: np.concatenate([r[k] for r in results]) for k in results[0].keys()}

# -----------------------------------------
# chunked helpers
# -----------------------------------------

# -----------------------------------------
# stage: dumpfile -> dt hits
# -----------------------------------------

### number of lines of a text file (fast, without decoding)
def _count_lines(file_name, buffer_size=16 * 1024 * 1024):
    n_lines, last = 0, b""
    with open(file_name, "rb") as f:
        while True:
            buffer = f.read(buffer_size)
            if not buffer:
                break
            n_lines += buffer.count(b"\n")
            last = buffer[-1:]
    return n_lines + (1 if last not in (b"", b"\n") else 0)

### read the dumpfile in blocks of block_n_lines lines, as bytes (never holds the whole file in memory)
def _read_raw_blocks(file_name, block_n_lines, n_lines_to_skip):
    with open(file_name, "rb") as f:
        for _ in itertools.islice(f, n_lines_to_skip):
            pass
        while True:
            block = list(itertools.islice(f, block_n_lines))
            if not block:
                return
            yield b"".join(block)

### lookup tables for the mapping of readout channel and channel to the dt cell: {key: array[ro_ch, ch]}
# and for the cells which are kept (inside the chamber, not masked, not dead): array[sl, ly, wi]
_DT_TABLES = None
def _dt_tables():
    global _DT_TABLES
    if _DT_TABLES is None:
        mapped = np.zeros((256, 256), dtype=bool)   # readout channel and channel belong to the dt chamber
        has_cell = np.zeros((256, 256), dtype=bool)  # ... and are connected to a cell
        mapping = {k: np.zeros((256, 256), dtype=v) for k, v in params._dt_mapping_keys.items()}
        for ro_ch in derived_params._dt_ro_chs:
            mapped[ro_ch, list(derived_params._dt_chs_by_ro_ch[ro_ch])] = True
            for ch, cell in derived_params._dt_remap_table[ro_ch].items():
                has_cell[ro_ch, ch] = True
                for k in params._dt_mapping_keys.keys():
                    mapping[k][ro_ch, ch] = cell[k]
        keep_cell = np.zeros((256, 256, 256), dtype=bool)
        for sl in params._dt_chamber["sls"].keys():
            for ly in params._dt_chamber["sls"][sl]["lys"].keys():
                min_wi = params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]
                max_wi = params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]
                keep_cell[sl, ly, min_wi:max_wi + 1] = True
                excluded_wis = set(params._dt_wire_mask[sl][ly]) | set(params._dt_dead_wires.get(sl, {}).get(ly, []))
                for wi in excluded_wis:
                    keep_cell[sl, ly, wi] = False
        _DT_TABLES = (mapped, has_cell, mapping, keep_cell)
    return _DT_TABLES

### convert one block of dumpfile lines (bytes) into dt hits
# - decodes the data words into their fields
# - calculates the timestamp of every hit without the orbit counter overflows before this block
# - keeps only dt channels, adds the cell (sl, ly, wi) and removes masked / dead wires
# returns (dt hits or None, number of raw hits, first orbit number, last orbit number, overflows inside the block)
# the dt hits carry two helper columns which the caller turns into the final timestamp (see _finish_block):
#   "_ts_base": timestamp from tdc, bx and orbit number, "_n_overflow": orbit counter overflows inside this block
#   up to this hit
def _convert_block(block_bytes):
    raw = np.array(block_bytes.split(), dtype=np.uint64)
    n_raw = len(raw)
    if n_raw == 0:
        return None, 0, None, None, 0
    hits = {}
    for k, dtype in params._htg_keys.items():
        hits[k] = ((raw & np.uint64(params._htg_shifted_mask[k])) >> np.uint64(params._htg_bitshift[k])).astype(dtype)
    del raw
    # orbit counter overflow: the orbit number jumps back by more than params._oc_difference_for_overflow
    oc = hits["oc"].astype(np.int64)
    n_overflow = np.zeros(n_raw, dtype=np.int64)
    n_overflow[1:] = np.cumsum((oc[:-1] - oc[1:]) > params._oc_difference_for_overflow)
    ts_base = (hits["tdc"].astype(np.uint64) * np.uint64(derived_params._tdc_to_timestamp)
               + hits["bx"].astype(np.uint64) * np.uint64(derived_params._bx_to_timestamp)
               + hits["oc"].astype(np.uint64) * np.uint64(derived_params._orbit_to_timestamp))
    first_oc, last_oc, n_overflow_block = int(oc[0]), int(oc[-1]), int(n_overflow[-1])
    # dt channels only, with the cell they belong to
    mapped, has_cell, mapping, keep_cell = _dt_tables()
    ro_ch, ch = hits["ro_ch"].astype(np.intp), hits["ch"].astype(np.intp)
    sl, ly, wi = mapping["sl"][ro_ch, ch], mapping["ly"][ro_ch, ch], mapping["wi"][ro_ch, ch]
    keep = mapped[ro_ch, ch] & has_cell[ro_ch, ch] & keep_cell[sl, ly, wi]
    n_dt = int(keep.sum())
    if n_dt == 0:
        return None, n_raw, first_oc, last_oc, n_overflow_block
    dt_hits = {k: v[keep] for k, v in hits.items()}
    dt_hits["ts"] = np.zeros(n_dt, dtype=params._ts_type)  # filled by _finish_block
    dt_hits["err_ts"] = np.full(n_dt, np.sqrt((1 / np.sqrt(12)) ** 2 + params.dt_hit_add_ts_unc ** 2), dtype=np.float64)
    ro_ch, ch = ro_ch[keep], ch[keep]
    for k in params._dt_mapping_keys.keys():
        dt_hits[k] = mapping[k][ro_ch, ch]
    for k, v in params._dt_other_keys.items():
        if k not in ("ts", "err_ts"):
            dt_hits[k] = np.full(n_dt, 0, dtype=v)
    dt_hits["_ts_base"] = ts_base[keep]
    dt_hits["_n_overflow"] = n_overflow[keep]
    return dt_hits, n_raw, first_oc, last_oc, n_overflow_block

### carries the orbit counter overflow count from one block to the next
class _TimestampState:
    def __init__(self):
        self.oc_overflow = 0
        self.last_oc = None

### add the orbit counter overflows of all blocks before this one and set the final timestamps
def _finish_block(result, state):
    dt_hits, n_raw, first_oc, last_oc, n_overflow_block = result
    if n_raw == 0:
        return None, 0
    if state.last_oc is not None and (state.last_oc - first_oc) > params._oc_difference_for_overflow:
        state.oc_overflow += 1  # overflow between the last hit of the previous block and the first hit of this one
    if dt_hits is not None:
        n_overflow = (dt_hits.pop("_n_overflow") + state.oc_overflow).astype(np.uint64)
        ts = dt_hits.pop("_ts_base") + n_overflow * np.uint64(derived_params._orbit_overflow_to_timestamp)
        dt_hits["ts"] = ts.astype(params._ts_type)
    state.oc_overflow += n_overflow_block
    state.last_oc = last_oc
    return dt_hits, n_raw

### convert a raw dumpfile (.txt) into a ROOT file holding only the dt hits (tree "dt_hits")
# decodes the data words, adds the timestamp, keeps only dt channels, adds the chamber mapping
# (sl, ly, wi) and removes masked / dead wires
# n_proc > 1: the blocks are converted on several processes and written in the order of the file; the result
#   is the same as on one process
def convert_dumpfile_to_dt_hits(input_dumpfile, dt_hits_file, *, n_lines_to_skip=999, block_n_lines=500_000, n_proc=1):
    root_utils.check_input_file(input_dumpfile)
    log(f"[dumpfile -> dt hits] START \"{input_dumpfile}\" -> \"{dt_hits_file}\"")
    log(f"[dumpfile -> dt hits] block_n_lines={block_n_lines}, n_lines_to_skip={n_lines_to_skip}, n_proc={n_proc}, "
        f"input size {os.path.getsize(input_dumpfile) / 1e6:.1f} MB")
    n_lines = _count_lines(input_dumpfile)
    n_blocks = int(np.ceil(max(0, n_lines - n_lines_to_skip) / block_n_lines))
    log(f"[dumpfile -> dt hits] {n_lines} lines in the dumpfile -> {n_blocks} blocks")
    ts_state = _TimestampState()
    writer = root_utils.TreeWriter(dt_hits_file, root_utils.DT_HITS_TREE)
    totals = {"raw": 0, "blocks": 0}
    t_start = time.perf_counter()

    def _write(result):
        dt_block, n_raw = _finish_block(result, ts_state)
        totals["blocks"] += 1
        totals["raw"] += n_raw
        n_dt = 0
        if dt_block is not None:
            n_dt = root_utils.length(dt_block)
            writer.write(dt_block)
        elapsed = time.perf_counter() - t_start
        log(f"[dumpfile -> dt hits] block {totals['blocks']} / {n_blocks}: {n_raw} raw hits -> {n_dt} dt hits "
            f"({100 * n_dt / max(1, n_raw):.1f}%) | totals: {totals['raw']} raw, {writer.n_written} dt | "
            f"{totals['raw'] / max(elapsed, 1e-9):.0f} raw hits/s")

    blocks = _read_raw_blocks(input_dumpfile, block_n_lines, n_lines_to_skip)
    with _optional_pool(n_proc) as pool:
        if pool is None:
            for block in blocks:
                _write(_convert_block(block))
        else:
            # keep at most 2 * n_proc blocks in work at a time, write the results in the order of the file
            pending = collections.deque()
            for block in blocks:
                pending.append(pool.apply_async(_convert_block, (block,)))
                if len(pending) >= 2 * n_proc:
                    _write(pending.popleft().get())
            while pending:
                _write(pending.popleft().get())
    writer.close()
    if ts_state.oc_overflow > 0:
        log(f"[dumpfile -> dt hits] orbit counter overflows found: {ts_state.oc_overflow}")
    log(f"[dumpfile -> dt hits] DONE. {totals['blocks']} blocks, {totals['raw']} raw hits read, {writer.n_written} dt hits written, "
        f"took {time.perf_counter() - t_start:.1f}s")
    if writer.n_written == 0:
        raise RuntimeError(f"No dt hits found in {input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")
    return writer.n_written


def _apply_individual_dead_time_chunk(hits_chunk):
    """Dead-time cut applied to ONE CHUNK only -- state resets at every chunk boundary (no carry-over).
    The hits are first sorted by their timestamp (the order in the input file is not kept), then for every cell a
    hit is dropped if it comes less than the dead time after the previous hit of the same cell.
    Returns the surviving hits, sorted by timestamp."""
    n_cur = len(hits_chunk["ts"])
    if n_cur == 0:
        return hits_chunk
    time_order = np.argsort(np.asarray(hits_chunk["ts"], dtype=np.float64), kind="stable")
    hits_chunk = {k: v[time_order] for k, v in hits_chunk.items()}
    if params._dt_ts_individual_dead_time <= 0:
        return hits_chunk
    dead_time = params._dt_ts_individual_dead_time

    combo = np.stack([hits_chunk["sl"], hits_chunk["ly"], hits_chunk["wi"]], axis=1)
    _, inverse = np.unique(combo, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    group_bounds = np.flatnonzero(np.diff(sorted_inverse)) + 1
    group_starts = np.concatenate(([0], group_bounds))
    group_ends = np.concatenate((group_bounds, [n_cur]))

    keep = np.zeros(n_cur, dtype=bool)
    for g in range(len(group_starts)):
        start, end = group_starts[g], group_ends[g]
        idx = order[start:end]
        ts_list = hits_chunk["ts"][idx]
        n_group = len(ts_list)
        diffs = np.empty(n_group, dtype=np.float64)
        diffs[0] = 0
        if n_group > 1:
            diffs[1:] = np.asarray(ts_list[1:], dtype=np.float64) - np.asarray(ts_list[:-1], dtype=np.float64)
        allowed_local = diffs >= dead_time
        keep[idx[allowed_local]] = True

    return {k: v[keep] for k, v in hits_chunk.items()}


def _fold_histogram_chunk(hits_chunk, running):
    """Fold ONE CHUNK's hit-diff histogram into running totals: for every cell the time differences between
    consecutive hits of that cell (hits of the cell ordered in time), all cells summed in one histogram."""
    n_cur = len(hits_chunk["ts"])
    if n_cur < 2:
        return
    combo = np.stack([hits_chunk["sl"], hits_chunk["ly"], hits_chunk["wi"]], axis=1)
    _, inverse = np.unique(combo, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    order = np.lexsort((np.asarray(hits_chunk["ts"], dtype=np.float64), inverse))  # hits grouped by cell, ordered in time inside a cell
    same_cell = inverse[order][1:] == inverse[order][:-1]
    ts_sorted = np.asarray(hits_chunk["ts"][order], dtype=np.float64)
    err_ts_sorted = np.asarray(hits_chunk["err_ts"][order], dtype=np.float64)
    diffs = (ts_sorted[1:] - ts_sorted[:-1])[same_cell]
    err_diffs = np.sqrt(err_ts_sorted[1:] ** 2 + err_ts_sorted[:-1] ** 2)[same_cell]
    if len(diffs) == 0:
        return
    hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = \
        hist_utils.calculate_histogram_and_shifted_histograms(data=diffs, edges=running["edges"], err_data=err_diffs)
    running["hist"] += hist_
    running["entries"] += entries_
    running["underflow"] += underflow_
    running["overflow"] += overflow_
    running["hist_err_right"] += hist_err_right_
    running["hist_err_left"] += hist_err_left_

# -----------------------------------------
# stage: dt hits -> timing corrected dt hits
# -----------------------------------------

### build lookup arrays [sl][ly][wi] from a testpulse correction object {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}}
def _timing_correction_tables(dt_tp_corrections):
    ts_corr = np.full((256, 256, 256), np.nan, dtype=np.float64)
    err_ts_corr = np.full((256, 256, 256), np.nan, dtype=np.float64)
    for sl, lys in dt_tp_corrections.items():
        for ly, wis in lys.items():
            for wi, corr in wis.items():
                ts_corr[int(sl), int(ly), int(wi)] = corr["ts_corr"]
                err_ts_corr[int(sl), int(ly), int(wi)] = corr["err_ts_corr"]
    return ts_corr, err_ts_corr

### apply the testpulse timing calibration to one chunk of dt hits
# same result as dt_utils.apply_timing_calibration, without the loop over hits:
#   ts_corrected = ts + ts_corr(sl, ly, wi),  err_ts_corrected = sqrt(err_ts^2 + err_ts_corr^2),
#   oc / bx / tdc are recalculated from the corrected timestamp
def _apply_timing_calibration_chunk(hits, ts_corr_table, err_ts_corr_table):
    sl, ly, wi = hits["sl"].astype(np.intp), hits["ly"].astype(np.intp), hits["wi"].astype(np.intp)
    corr, err_corr = ts_corr_table[sl, ly, wi], err_ts_corr_table[sl, ly, wi]
    missing = np.isnan(corr)
    if np.any(missing):
        cells = sorted({(int(a), int(b), int(c)) for a, b, c in zip(sl[missing], ly[missing], wi[missing])})
        raise KeyError(f"No timing correction for {len(cells)} cells (sl, ly, wi), first ones: {cells[:10]}")
    corr_hits = dict(hits)
    ts = np.asarray(hits["ts"], dtype=np.float64) + corr
    corr_hits["ts"] = ts.astype(hits["ts"].dtype)
    corr_hits["err_ts"] = np.sqrt(hits["err_ts"] ** 2 + err_corr ** 2).astype(hits["err_ts"].dtype)
    ts_int = np.round(ts, 0).astype(np.uint64)
    corr_hits["oc"] = ((ts_int % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp).astype(hits["oc"].dtype)
    corr_hits["bx"] = ((ts_int % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp).astype(hits["bx"].dtype)
    corr_hits["tdc"] = ((ts_int % derived_params._bx_to_timestamp) // np.uint64(derived_params._tdc_to_timestamp)).astype(hits["tdc"].dtype)
    return corr_hits

### apply the testpulse timing calibration (one time offset per wire) to the dt hits
# dt_tp_corrections_file: .pcl file made by the testpulse scripts, {sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}}
def apply_timing_correction(dt_hits_file, dt_tp_corrections_file, corr_dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    root_utils.check_input_file(dt_tp_corrections_file)
    log(f"[dt hits -> corrected dt hits] START \"{dt_hits_file}\" + \"{dt_tp_corrections_file}\" -> \"{corr_dt_hits_file}\"")
    dt_tp_corrections = data_utils.load_pickle(file=dt_tp_corrections_file, silent=True)
    ts_corr_table, err_ts_corr_table = _timing_correction_tables(dt_tp_corrections)
    n_hits = 0
    n_chunks_total = root_utils.n_steps(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size)
    with root_utils.TreeWriter(corr_dt_hits_file, root_utils.DT_HITS_TREE) as writer:
        for i_chunk, (_, chunk) in enumerate(root_utils.iterate_tree(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size), start=1):
            n_hits += root_utils.length(chunk)
            writer.write(_apply_timing_calibration_chunk(chunk, ts_corr_table, err_ts_corr_table))
            log(f"    chunk {i_chunk} / {n_chunks_total}: {root_utils.length(chunk)} hits corrected")
    log(f"[dt hits -> corrected dt hits] DONE. {n_hits} hits, corrections between "
        f"{np.nanmin(ts_corr_table):.2f} and {np.nanmax(ts_corr_table):.2f} ts units")
    return n_hits

# -----------------------------------------
# stage: dt hits -> hit difference histogram (side product)
# -----------------------------------------

### histogram of the time difference between consecutive hits of the same cell, summed over all cells
# for every cell separately: its hits are ordered in time, then the differences between neighbours are histogrammed
# uses the hits before the dead time cut
# output format is chosen by the file ending: .root or .pcl (.pcl = format of the old pipeline)
def dt_hits_to_hit_diff_hist(dt_hits_file, hit_diff_hist_file, *, step_size=root_utils.DEFAULT_STEP_SIZE, n_bins=5000, ts_max=5000):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> hit diff hist] START \"{dt_hits_file}\" -> \"{hit_diff_hist_file}\"")
    edges = np.linspace(0, ts_max, n_bins + 1)
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)
    running = {"edges": edges, "hist": hist, "entries": entries, "underflow": underflow, "overflow": overflow,
               "hist_err_right": hist_err_right, "hist_err_left": hist_err_left}
    n_hits = 0
    n_chunks_total = root_utils.n_steps(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size)
    for i_chunk, (_, chunk) in enumerate(root_utils.iterate_tree(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size), start=1):
        n_hits += root_utils.length(chunk)
        _fold_histogram_chunk(chunk, running)
        log(f"    chunk {i_chunk} / {n_chunks_total}: {root_utils.length(chunk)} hits, {int(running['entries'])} entries in histogram so far")
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
        hist=running["hist"], hist_err_right=running["hist_err_right"], hist_err_left=running["hist_err_left"], do_stat_err=True,
    )
    err_hist_stat = np.sqrt(running["hist"])
    hist_data = {
        "edges": running["edges"], "centers": centers, "hist": running["hist"],
        "err_hist": err_hist, "err_hist_stat": err_hist_stat, "err_hist_down": err_hist_down, "err_hist_up": err_hist_up,
        "entries": running["entries"], "underflow": running["underflow"], "overflow": running["overflow"],
    }
    root_utils.prepare_output_file(hit_diff_hist_file)
    if hit_diff_hist_file.endswith(".pcl"):
        data_utils.store_pickle(data=hist_data, file=hit_diff_hist_file, silent=True)
    else:
        # one row per bin in tree "tree" (branch "hist" = content of the bin), run level numbers in tree "summary"
        bins = {
            "edge_low": np.asarray(edges[:-1], dtype=np.float64), "edge_high": np.asarray(edges[1:], dtype=np.float64),
            "center": np.asarray(centers, dtype=np.float64),
        } | {k: np.asarray(hist_data[k], dtype=np.float64) for k in ["hist", "err_hist", "err_hist_stat", "err_hist_down", "err_hist_up"]}
        summary = {k: np.float64(hist_data[k]) for k in ["entries", "underflow", "overflow"]} | {"n_hits": np.int64(n_hits)}
        # the same histogram as TH1D object "hit_diff_hist", to be drawn directly in ROOT
        root_utils.write_tree(hit_diff_hist_file, bins, summary=summary, histograms={"hit_diff_hist": (hist_data["hist"], edges)})
    log(f"[dt hits -> hit diff hist] DONE. {n_hits} hits, {int(running['entries'])} entries in histogram")
    return hist_data

# -----------------------------------------
# stage: dt hits -> cell counts (side product)
# -----------------------------------------

### number of hits per cell and duration of the run
# uses the hits before the dead time cut
# output format is chosen by the file ending: .root or .pcl (.pcl = format of the old pipeline)
def count_dt_cells(dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    """Count the hits per cell. Returns (cell_counts {sl: {ly: {wi: count}}}, ts_min, ts_max, n_hits)."""
    root_utils.check_input_file(dt_hits_file)
    cell_counts = {
        sl: {
            ly: {
                wi: 0
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1)
            }
            for ly in range(0, 4)
        }
        for sl in range(1, 4)
    }
    ts_min, ts_max = None, None
    n_hits = 0
    n_chunks_total = root_utils.n_steps(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size)
    for i_chunk, (_, chunk) in enumerate(root_utils.iterate_tree(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size), start=1):
        if root_utils.length(chunk) == 0:
            continue
        n_hits += root_utils.length(chunk)
        log(f"    chunk {i_chunk} / {n_chunks_total}: {root_utils.length(chunk)} hits counted")
        combo = np.stack([chunk["sl"], chunk["ly"], chunk["wi"]], axis=1)
        uniq_cells, uniq_counts = np.unique(combo, axis=0, return_counts=True)
        for (sl_u, ly_u, wi_u), c in zip(uniq_cells, uniq_counts):
            cell_counts[int(sl_u)][int(ly_u)][int(wi_u)] += int(c)
        chunk_ts_min, chunk_ts_max = chunk["ts"].min(), chunk["ts"].max()
        ts_min = chunk_ts_min if ts_min is None else min(ts_min, chunk_ts_min)
        ts_max = chunk_ts_max if ts_max is None else max(ts_max, chunk_ts_max)
    return cell_counts, ts_min, ts_max, n_hits

def dt_hits_to_cell_counts(dt_hits_file, cell_counts_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> cell counts] START \"{dt_hits_file}\" -> \"{cell_counts_file}\"")
    cell_counts, ts_min, ts_max, n_hits = count_dt_cells(dt_hits_file, step_size=step_size)
    if n_hits == 0:
        raise RuntimeError(f"No dt hits in {dt_hits_file}.")
    duration_seconds = float(ts_max - ts_min) * 0.78 * 1e-9
    counts_data = {"cell_counts": cell_counts, "duration_seconds": duration_seconds, "ts_min": ts_min, "ts_max": ts_max}
    root_utils.prepare_output_file(cell_counts_file)
    if cell_counts_file.endswith(".pcl"):
        data_utils.store_pickle(data=counts_data, file=cell_counts_file, silent=True)
    else:
        # one row per cell in tree "tree", run level numbers in tree "summary"
        rows = [(sl, ly, wi, c) for sl in cell_counts for ly in cell_counts[sl] for wi, c in cell_counts[sl][ly].items()]
        cells = {
            "sl": np.array([r[0] for r in rows], dtype=np.uint8), "ly": np.array([r[1] for r in rows], dtype=np.uint8),
            "wi": np.array([r[2] for r in rows], dtype=np.uint8), "count": np.array([r[3] for r in rows], dtype=np.int64),
        }
        summary = {"duration_seconds": np.float64(duration_seconds), "ts_min": np.float64(ts_min), "ts_max": np.float64(ts_max), "n_hits": np.int64(n_hits)}
        # the same numbers as TH2D object "cell_counts": x = wire, y = 4 * (sl - 1) + ly
        n_wires = int(cells["wi"].max()) + 1
        counts_2d = np.zeros((n_wires, 12))
        counts_2d[cells["wi"].astype(int), 4 * (cells["sl"].astype(int) - 1) + cells["ly"].astype(int)] = cells["count"]
        root_utils.write_tree(cell_counts_file, cells, summary=summary,
                              histograms={"cell_counts": (counts_2d, np.arange(n_wires + 1) - 0.5, np.arange(13) - 0.5)})
    log(f"[dt hits -> cell counts] DONE. {n_hits} hits, duration {duration_seconds:.3f} s")
    return counts_data

# -----------------------------------------
# stage: dt hits -> sl patterns
# -----------------------------------------

### find 4-layer hit patterns inside each superlayer
# - applies the individual dead time cut (per cell, hits ordered in time) before the pattern search, unless apply_dead_time=False
# - wide_ts_window=True uses the time window for a free drift velocity (params._dt_sl_patterns_ts_window_fit_vd),
#   False the one for the fixed drift velocity (params._dt_sl_patterns_ts_window)
# - hits are processed in chunks of step_size; dead time and pattern search do not look across chunk borders
# - n_proc > 1: inside a chunk the three superlayers and pieces in time of each superlayer are searched in
#   parallel. The result is exactly the same as on one process (see dt_utils.find_sl_patterns).
# - every pattern gets the chunk_id of its hit chunk, later stages combine rows per chunk_id
def dt_hits_to_sl_patterns(dt_hits_file, sl_patterns_file, *, step_size=root_utils.DEFAULT_STEP_SIZE, apply_dead_time=True,
                           wide_ts_window=True, simulation_only_muon_patterns=False, n_proc=1, verbose=False):
    from analysis_tools.utils import dt_utils
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> sl patterns] START \"{dt_hits_file}\" -> \"{sl_patterns_file}\" (chunks of {step_size}, n_proc={n_proc})")
    totals = {"hits_in": 0, "hits_after_deadtime": 0, "n_patterns": 0}
    n_chunks = 0
    n_chunks_total = root_utils.n_steps(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size)
    with root_utils.TreeWriter(sl_patterns_file) as writer, _optional_pool(n_proc if not verbose else 1) as pool:
        for _, chunk in root_utils.iterate_tree(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size):
            n_chunks += 1
            t_step = time.perf_counter()
            n_hits_in = root_utils.length(chunk)
            totals["hits_in"] += n_hits_in
            if apply_dead_time:
                chunk = _apply_individual_dead_time_chunk(chunk)
            n_hits = root_utils.length(chunk)
            totals["hits_after_deadtime"] += n_hits
            if n_hits == 0:
                log(f"    chunk {n_chunks} / {n_chunks_total}: {n_hits_in} hits in, none left after dead time cut")
                continue
            sl_patterns = dt_utils.find_sl_patterns(
                hits=chunk, verbose=verbose, silent=True,
                simulation_only_muon_patterns=simulation_only_muon_patterns, fit_vd=wide_ts_window, pool=pool,
            )
            del chunk
            n_patterns = root_utils.length(sl_patterns)
            totals["n_patterns"] += n_patterns
            log(f"    chunk {n_chunks} / {n_chunks_total}: {n_hits_in} hits in, {n_hits} after dead time cut ({100 * n_hits / max(1, n_hits_in):.1f}%) "
                f"-> {n_patterns} patterns ({time.perf_counter() - t_step:.2f}s)")
            if n_patterns > 0:
                writer.write(root_utils.set_chunk_id(sl_patterns, n_chunks))
            gc.collect()
    log(f"[dt hits -> sl patterns] DONE. {n_chunks} chunks, hits_in={totals['hits_in']}, "
        f"hits_after_deadtime={totals['hits_after_deadtime']}, patterns={totals['n_patterns']}")
    return totals

# -----------------------------------------
# stage: sl patterns -> sl fits
# -----------------------------------------

### fit a track segment to every pattern (all lateralities, best one is kept)
# - fit_vd=False: fixed drift velocity, fit_vd=True: drift velocity is a free fit parameter
# - cuts: optional, applied to the input rows before fitting
# - suffix: appended to the names of the fit result branches (e.g. "_refit"), so that a refit of an
#   sl fits file keeps the first fit results next to the new ones
# rows are independent, so the result does not depend on step_size or n_proc
def fit_sl_patterns_file(input_file, output_file, *, fit_vd=False, suffix="", cuts=None, step_size=root_utils.DEFAULT_STEP_SIZE,
                         n_proc=1, verbose=False, label="sl patterns -> sl fits"):
    from analysis_tools.utils import dt_utils
    root_utils.check_input_file(input_file)
    cuts = cuts or []
    log(f"[{label}] START \"{input_file}\" -> \"{output_file}\" (fit_vd={fit_vd}, suffix=\"{suffix}\", cuts={cuts}, n_proc={n_proc})")
    totals = {"rows_in": 0, "rows_after_cuts": 0, "n_fits": 0}
    n_chunks = 0
    n_chunks_total = root_utils.n_steps(input_file, step_size=step_size)
    with root_utils.TreeWriter(output_file) as writer:
        for _, chunk in root_utils.iterate_tree(input_file, step_size=step_size):
            n_chunks += 1
            t_step = time.perf_counter()
            n_in = root_utils.length(chunk)
            totals["rows_in"] += n_in
            if len(cuts) > 0:
                _check_cut_keys(cuts, chunk, input_file)
                chunk = data_utils.cut_data(data=chunk, conditions=cuts, silent=True)
            n_rows = root_utils.length(chunk)
            totals["rows_after_cuts"] += n_rows
            if n_rows == 0:
                log(f"    chunk {n_chunks} / {n_chunks_total}: {n_in} rows in, none left after cuts")
                continue
            sl_fits = _run_rowwise(
                dt_utils.fit_sl_patterns, chunk, "patterns",
                {"verbose": verbose, "silent": True, "fit_vd": fit_vd, "suffix": suffix}, n_proc if not verbose else 1,
            )
            del chunk
            n_fits = root_utils.length(sl_fits)
            totals["n_fits"] += n_fits
            writer.write(sl_fits)
            log(f"    chunk {n_chunks} / {n_chunks_total}: {n_in} rows in, {n_rows} after cuts -> {n_fits} fits ({time.perf_counter() - t_step:.2f}s)")
            del sl_fits
            gc.collect()
    log(f"[{label}] DONE. rows_in={totals['rows_in']}, rows_after_cuts={totals['rows_after_cuts']}, fits={totals['n_fits']}")
    return totals

# -----------------------------------------
# stage: sl fits -> super fits (the two phi superlayers combined)
# -----------------------------------------

### match the sl fits of the two phi superlayers to 8-hit "super patterns" and fit them
# input: sl fits after quality cuts (apply_cuts_file)
# matching is done per chunk_id block (= among the fits which come from the same dt hit chunk)
# - max_alpha: only sl fits with |track angle| below this are combined
# - max_chi2: optional additional chi2/ndf cut on the sl fits which may be combined (None: no additional cut)
# - fit_vd: False = fixed drift velocity (default), True = drift velocity is a free fit parameter
# - super_patterns_file: optionally also store the matched super patterns (input of the super fit)
# output branches "row_sl<n>" (n = the two phi superlayers): rows of the two combined sl fits in sl_fits_file
def sl_fits_to_super_fits(sl_fits_file, super_fits_file, *, super_patterns_file=None, max_chi2=None, max_alpha=np.deg2rad(60),
                          fit_vd=False, suffix=DEFAULT_SUPER_FIT_SUFFIX, step_size=root_utils.DEFAULT_STEP_SIZE, n_proc=1, verbose=False):
    from analysis_tools.utils import dt_utils
    root_utils.check_input_file(sl_fits_file)
    max_chi2ndf = np.inf if max_chi2 is None else max_chi2
    log(f"[sl fits -> super fits] START \"{sl_fits_file}\" -> \"{super_fits_file}\" "
        f"(max_chi2={max_chi2}, max_alpha={max_alpha:.4f} rad, fit_vd={fit_vd}, suffix=\"{suffix}\", n_proc={n_proc})")
    totals = {"n_fits": 0, "n_super_patterns": 0, "n_super_fits": 0}
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]
    patterns_writer = root_utils.TreeWriter(super_patterns_file) if super_patterns_file is not None else None
    n_chunks_total = root_utils.n_blocks(sl_fits_file)
    with root_utils.TreeWriter(super_fits_file) as writer:
        for i_chunk, (chunk_id, entry_start, sl_fits) in enumerate(root_utils.iterate_blocks(sl_fits_file, step_size=step_size), start=1):
            t_step = time.perf_counter()
            n_fits = root_utils.length(sl_fits)
            totals["n_fits"] += n_fits
            try:
                super_patterns = dt_utils.build_phi_super_patterns(
                    sl_fits, silent=True, verbose=verbose, max_chi2ndf=max_chi2ndf, max_alpha=max_alpha,
                )
                n_super_patterns = len(super_patterns.get(f"sl{phi_sls[0]}", []))
            except Exception as e:
                log(f"    chunk {i_chunk} / {n_chunks_total}: build_phi_super_patterns FAILED, block skipped: {e!r}")
                super_patterns = None
                n_super_patterns = 0
            totals["n_super_patterns"] += n_super_patterns
            n_super_fits = 0
            if super_patterns is not None and n_super_patterns > 0:
                # rows of the combined sl fits in the input file: "idx_sl<n>" counts within the fits of sl n which
                # build_phi_super_patterns accepts (same selection as in that function, order of the file)
                max_tan_alpha = np.tan(max_alpha)
                accepted = (sl_fits["impossible"] == 0) & (sl_fits["chi2/ndf"] < max_chi2ndf) & (sl_fits["tan_alpha"] > -max_tan_alpha) & (sl_fits["tan_alpha"] < max_tan_alpha)
                for sl in phi_sls:
                    rows_of_sl = np.flatnonzero(accepted & (sl_fits["sl"] == sl))
                    super_patterns[f"row_sl{sl}"] = (rows_of_sl[super_patterns[f"idx_sl{sl}"]] + entry_start).astype(np.int64)
                if patterns_writer is not None:
                    patterns_writer.write(root_utils.set_chunk_id(dict(super_patterns), chunk_id))
                super_fits = _run_rowwise(
                    dt_utils.fit_super_sl_patterns, super_patterns, "super_patterns",
                    {"silent": True, "verbose": verbose, "fit_vd": fit_vd, "suffix": suffix}, n_proc if not verbose else 1,
                )
                del super_patterns
                n_super_fits = len(super_fits.get("ts0", []))
                if n_super_fits > 0:
                    writer.write(root_utils.set_chunk_id(super_fits, chunk_id))
                del super_fits
            del sl_fits
            totals["n_super_fits"] += n_super_fits
            log(f"    chunk {i_chunk} / {n_chunks_total}: {n_fits} sl fits -> {n_super_patterns} super patterns -> {n_super_fits} super fits "
                f"({time.perf_counter() - t_step:.2f}s)")
            gc.collect()
    if patterns_writer is not None:
        patterns_writer.close()
    log(f"[sl fits -> super fits] DONE. sl_fits={totals['n_fits']}, super_patterns={totals['n_super_patterns']}, super_fits={totals['n_super_fits']}")
    return totals

# -----------------------------------------
# stage: super fits (phi) + sl fits (theta) -> dt muons
# -----------------------------------------

### combine the super fits of the two phi superlayers with the sl fits of the theta superlayer to dt muons
# (dt_utils.reco_muons_from_super_fits), per chunk_id block
# - super_fits_file: super fits after quality cuts
# - sl_fits_file: the cut sl fits file the super fits were made from (its theta sl fits are used here)
# - output branches "super_fit_row" (row in super_fits_file) and "sl<n>_fit_row" (rows in sl_fits_file) point to the
#   fits a muon was built from
def super_fits_to_dt_muons(super_fits_file, sl_fits_file, dt_muons_file, *, suffix=DEFAULT_SUPER_FIT_SUFFIX, tgroup_tolerance=None,
                           step_size=root_utils.DEFAULT_STEP_SIZE, verbose=False):
    from analysis_tools.utils import dt_utils
    root_utils.check_input_file(super_fits_file)
    root_utils.check_input_file(sl_fits_file)
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    log(f"[super fits + theta sl fits -> dt muons] START \"{super_fits_file}\" + \"{sl_fits_file}\" -> \"{dt_muons_file}\" "
        f"(tgroup_tolerance={tgroup_tolerance:.2f} ts units)")
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]
    theta_sl = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "theta"][0]
    fits_blocks = root_utils.iterate_blocks(sl_fits_file, step_size=step_size)
    totals = {"n_super_fits": 0, "n_theta_fits": 0, "n_muons": 0, "n_ambiguous": 0}
    ts_min, ts_max = None, None
    n_chunks_total = root_utils.n_blocks(super_fits_file)
    with root_utils.TreeWriter(dt_muons_file) as writer:
        for i_chunk, (chunk_id, super_start, super_fits) in enumerate(root_utils.iterate_blocks(super_fits_file, step_size=step_size), start=1):
            if "t0" + suffix not in super_fits:
                raise KeyError(f"No super fit results with suffix \"{suffix}\" in {super_fits_file}.")
            if np.any(super_fits["impossible" + suffix] != 0):
                raise RuntimeError(f"{super_fits_file} contains super fits flagged \"impossible\". Apply cuts first "
                                   f"(at least \"impossible{suffix},==,0\"), see scripts/dt_root/apply_cuts.py.")
            for fits_chunk_id, fits_start, sl_fits in fits_blocks:
                if fits_chunk_id == chunk_id:
                    break
            else:
                raise RuntimeError(f"No sl fits with chunk_id {chunk_id} in {sl_fits_file}. Is this the file the super fits were made from?")
            n_fits = root_utils.length(sl_fits)
            # the super fits have to point into this block of sl fits
            for sl in phi_sls:
                local = super_fits[f"row_sl{sl}"] - fits_start
                if np.any(local < 0) or np.any(local >= n_fits) or np.any(sl_fits["sl"][local] != sl):
                    raise RuntimeError(f"The sl fit rows stored in {super_fits_file} do not match {sl_fits_file}. Is this the file the super fits were made from?")
            theta_rows = np.flatnonzero(sl_fits["sl"] == theta_sl)
            theta_fits = {k: v[theta_rows] for k, v in sl_fits.items()}
            n_super, n_theta_fits = root_utils.length(super_fits), len(theta_rows)
            totals["n_super_fits"] += n_super
            totals["n_theta_fits"] += n_theta_fits
            dt_muons = dt_utils.reco_muons_from_super_fits(super_fits, theta_fits, suffix=suffix, tgroup_tolerance=tgroup_tolerance, silent=True, verbose=verbose)
            n_muons = root_utils.length(dt_muons)
            if n_muons > 0:
                super_idx, theta_idx = dt_muons.pop("super_fit_idx"), dt_muons.pop("theta_fit_idx")
                dt_muons["super_fit_row"] = (super_idx + super_start).astype(np.int64)
                for sl in phi_sls:
                    dt_muons[f"sl{sl}_fit_row"] = super_fits[f"row_sl{sl}"][super_idx].astype(np.int64)
                dt_muons[f"sl{theta_sl}_fit_row"] = (theta_rows[theta_idx] + fits_start).astype(np.int64)
                dt_muons = data_utils.sort_by_key(data=dt_muons, sort_key="ts", silent=True)
                writer.write(root_utils.set_chunk_id(dt_muons, chunk_id))
                totals["n_ambiguous"] += int(np.sum(dt_muons["n_theta_candidates"] > 1))
                ts_min = dt_muons["ts"].min() if ts_min is None else min(ts_min, dt_muons["ts"].min())
                ts_max = dt_muons["ts"].max() if ts_max is None else max(ts_max, dt_muons["ts"].max())
            totals["n_muons"] += n_muons
            log(f"    chunk {i_chunk} / {n_chunks_total}: {n_super} super fits, {n_theta_fits} theta sl fits -> {n_muons} dt muons")
    if totals["n_muons"] == 0:
        log(f"[super fits + theta sl fits -> dt muons] WARNING: no dt muons reconstructed, no output file was written.")
    duration = (ts_max - ts_min) * 0.78e-9 if ts_min is not None else 0
    rate = f", rate {totals['n_muons'] / duration:.2f} Hz over {duration:.2f} s" if duration > 0 else ""
    log(f"[super fits + theta sl fits -> dt muons] DONE. super_fits={totals['n_super_fits']}, theta_sl_fits={totals['n_theta_fits']}, "
        f"dt_muons={totals['n_muons']} ({totals['n_ambiguous']} with more than one theta candidate){rate}")
    return totals

# -----------------------------------------
# stage: apply cuts (any ROOT file of this workflow)
# -----------------------------------------

### keep only the rows which pass all cuts, prints the cut flow
def apply_cuts_file(input_file, output_file, cuts, *, tree=None, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(input_file)
    if len(cuts) == 0:
        raise ValueError("No cuts given.")
    tree = root_utils.resolve_tree_name(input_file, tree)
    log(f"[apply cuts] START \"{input_file}\" -> \"{output_file}\" (tree \"{tree}\")")
    n_in = 0
    n_after = [0 for _ in cuts]
    n_chunks_total = root_utils.n_steps(input_file, tree, step_size=step_size)
    with root_utils.TreeWriter(output_file, tree) as writer:
        for i_chunk, (_, chunk) in enumerate(root_utils.iterate_tree(input_file, tree, step_size=step_size), start=1):
            n_rows_chunk = root_utils.length(chunk)
            n_in += n_rows_chunk
            _check_cut_keys(cuts, chunk, input_file)
            for i, cut in enumerate(cuts):
                chunk = data_utils.cut_data(data=chunk, conditions=[cut], silent=True)
                n_after[i] += root_utils.length(chunk)
            writer.write(chunk)
            log(f"    chunk {i_chunk} / {n_chunks_total}: {root_utils.length(chunk)} of {n_rows_chunk} rows pass the cuts")
    log(f"[apply cuts] cut flow w.r.t. the {n_in} input rows:")
    for cut, n in zip(cuts, n_after):
        log(f"    {cut[0]} {cut[1]} {cut[2]}: {n} / {n_in} = {n / max(1, n_in):.4f}")
    if n_after[-1] == 0:
        log(f"[apply cuts] WARNING: no rows pass the cuts, no output file was written.")
    log(f"[apply cuts] DONE.")
    return n_in, n_after


#################################################################
### ROOT-STREAMING PIPELINE v3
### Same logic as v2 (chunked Phase 0 + Phase 1+, TTree fix), with
### much more logging so you can tell what a running job is doing and
### how far it's gotten, especially useful under condor where you can't
### attach a debugger and only see stdout/stderr after the fact.
#################################################################
#
# WHAT'S NEW vs. v2 (logging only -- no algorithmic changes):
#
# 1. A single _log() helper used everywhere: prints with an
#    elapsed-time-since-start prefix and flush=True on every call. Your
#    earlier "nothing prints for ages" issue was output buffering when
#    stdout is redirected to a file (as condor does) -- flush=True on
#    every print sidesteps that regardless of whether PYTHONUNBUFFERED
#    is set in the job environment, so this works even if that
#    environment variable doesn't make it through your condor setup.
#
# 2. Phase 0 now logs every block (not just every 10th), each with the
#    block's hit count, running total, elapsed time, and blocks/sec.
#
# 3. Phase 1+ now logs, PER CHUNK: hits in, hits after dead-time cut,
#    patterns found, fits found, fits surviving the chi2/impossible cut,
#    refits found, super-patterns found, super-fits found, and the
#    wall-clock time that chunk took. Running totals across all chunks
#    are printed too, and a final summary at the end.
#
# 4. main() logs every configuration value at startup (dataset name,
#    block/chunk sizes, resolved file paths) before any real work
#    starts, so a job that dies early still tells you what it was
#    trying to do.
#
#################################################################

import os
import io
import itertools
import numpy as np
import uproot
import gc
import time
import argparse

from analysis_tools.utils import data_utils, timestamp_utils, hist_utils
from analysis_tools.params import params, derived_params
from analysis_tools.utils import ba_justus_utils


# =================================================================
# LOGGING
# =================================================================

_SCRIPT_START = time.perf_counter()


def _log(msg):
    """Print with elapsed-time-since-script-start prefix, always
    flushed. Use this everywhere instead of bare print() so output
    shows up immediately in a redirected/condor stdout log rather than
    sitting in a buffer until it fills or the process exits."""
    elapsed = time.perf_counter() - _SCRIPT_START
    print(f"[{elapsed:9.1f}s] {msg}", flush=True)


# =================================================================
# PHASE 0: chunked raw dumpfile -> DT-only ROOT conversion
# =================================================================

def _read_raw_blocks(file_name, block_n_lines):
    """Yield successive blocks of `block_n_lines` raw text lines from the
    dumpfile, without ever holding the whole file's lines in memory at
    once (unlike the original import_raw's readlines())."""
    with open(file_name) as f:
        while True:
            block = list(itertools.islice(f, block_n_lines))
            if not block:
                return
            yield block


def _decode_hits_block(lines):
    """Vectorized equivalent of import_raw's per-line bitmask/shift
    decode loop, applied to one block of raw text lines."""
    n = len(lines)
    raw = np.array([int(l) for l in lines], dtype=np.uint64)
    hits = {}
    for k, dtype in params._htg_keys.items():
        mask = params._htg_shifted_mask[k]
        shift = params._htg_bitshift[k]
        hits[k] = ((raw & mask) >> np.uint64(shift)).astype(dtype)
    return hits


class _TimestampState:
    """Carries the two scalars add_timestamp's oc-overflow detection
    needs across chunk boundaries -- exact, not lossy (see prior chat)."""
    def __init__(self):
        self.oc_overflow = 0
        self.last_oc = None


def _assign_timestamps_block(hits_block, state: _TimestampState):
    """Exact chunked equivalent of timestamp_utils.add_timestamp's
    active algorithm, with (oc_overflow, last_oc) carried across blocks."""
    n = len(hits_block["oc"])
    ts = np.empty(n, dtype=params._ts_type)
    err_ts = np.full(n, 1 / np.sqrt(12), dtype=np.float64)

    oc_overflow = state.oc_overflow
    last_oc = state.last_oc

    tdc_arr, bx_arr, oc_arr = hits_block["tdc"], hits_block["bx"], hits_block["oc"]

    n_overflows_this_block = 0
    for i in range(n):
        tdc, bx, oc = tdc_arr[i], bx_arr[i], oc_arr[i]
        if last_oc is None:
            last_oc = oc
        if int(last_oc) - int(oc) > params._oc_difference_for_overflow:
            oc_overflow += 1
            n_overflows_this_block += 1
            _log(f"    Detected OC overflow -- block-local i={i} -- last_oc={last_oc} -- oc={oc} "
                 f"(total overflows so far: {oc_overflow})")
        ts[i] = (tdc * derived_params._tdc_to_timestamp
                 + bx * derived_params._bx_to_timestamp
                 + oc * derived_params._orbit_to_timestamp
                 + oc_overflow * derived_params._orbit_overflow_to_timestamp)
        last_oc = oc

    state.oc_overflow = oc_overflow
    state.last_oc = last_oc

    hits_block["ts"] = ts
    hits_block["err_ts"] = err_ts
    return hits_block


def _dt_filter_block(hits_block):
    """Row-independent part of extract_dt_hits: DT mask, dt-key remap,
    wire-range/mask filter. No cross-row state, safe to apply per block."""
    n_hits = len(hits_block["ch"])
    if n_hits == 0:
        return None

    dt_mask = np.full(n_hits, False, dtype=np.bool_)
    for ro_ch in derived_params._dt_ro_chs:
        tmp_mask = np.ma.isin(hits_block["ro_ch"], [ro_ch])
        tmp_mask &= np.ma.isin(hits_block["ch"], derived_params._dt_chs_by_ro_ch[ro_ch])
        dt_mask |= tmp_mask
    tmp_hits = {k: v[dt_mask] for k, v in hits_block.items()}
    n_dt_hits = len(tmp_hits["ch"])
    if n_dt_hits == 0:
        return None

    tmp_hits |= {k: np.full(n_dt_hits, 0, dtype=v) for k, v in params._dt_mapping_keys.items()} | \
                {k: np.full(n_dt_hits, 0, dtype=v) for k, v in params._dt_other_keys.items()
                 if k not in ("ts", "err_ts")}
    for i in range(n_dt_hits):
        ro_ch = tmp_hits["ro_ch"][i]
        ch = tmp_hits["ch"][i]
        if ch not in derived_params._dt_remap_table[ro_ch].keys():
            continue
        for k in params._dt_mapping_keys.keys():
            tmp_hits[k][i] = derived_params._dt_remap_table[ro_ch][ch][k]

    err_ts_com = np.sqrt((1 / np.sqrt(12)) ** 2 + params.dt_hit_add_ts_unc ** 2)
    tmp_hits["err_ts"][:] = err_ts_com

    n_cur = len(tmp_hits["sl"])
    keep_mask = np.zeros(n_cur, dtype=bool)
    for sl in params._dt_chamber["sls"].keys():
        for ly in params._dt_chamber["sls"][sl]["lys"].keys():
            min_wi = params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]
            max_wi = params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]
            this_group = (
                (tmp_hits["sl"] == sl) & (tmp_hits["ly"] == ly)
                & (tmp_hits["wi"] >= min_wi) & (tmp_hits["wi"] <= max_wi)
            )
            for wi_to_mask in params._dt_wire_mask[sl][ly]:
                this_group &= (tmp_hits["wi"] != wi_to_mask)
            keep_mask |= this_group
    tmp_hits = {k: v[keep_mask] for k, v in tmp_hits.items()}
    if len(tmp_hits["sl"]) == 0:
        return None

    return tmp_hits


def convert_dumpfile_to_dt_root_chunked(input_dumpfile, output_root_path, *,
                                         n_lines_to_skip=999, block_n_lines=500_000):
    """Fully chunked Phase 0. Logs every block's progress."""
    _log(f"[Phase 0] START converting \"{input_dumpfile}\" -> \"{output_root_path}\"")
    _log(f"[Phase 0] block_n_lines={block_n_lines}, n_lines_to_skip={n_lines_to_skip}")

    file_size_bytes = os.path.getsize(input_dumpfile) if os.path.exists(input_dumpfile) else None
    if file_size_bytes is not None:
        _log(f"[Phase 0] input file size: {file_size_bytes / 1e6:.1f} MB")

    ts_state = _TimestampState()
    out_file = None
    n_written = 0
    n_raw_lines_seen = 0
    n_blocks = 0
    phase0_t0 = time.perf_counter()

    block_iter = _read_raw_blocks(input_dumpfile, block_n_lines)

    lines_to_skip = n_lines_to_skip
    for block in block_iter:
        n_blocks += 1
        block_t0 = time.perf_counter()
        n_raw_lines_seen += len(block)

        if lines_to_skip > 0:
            if lines_to_skip >= len(block):
                lines_to_skip -= len(block)
                _log(f"[Phase 0] block {n_blocks}: entirely skipped ({len(block)} lines, "
                     f"{lines_to_skip} still to skip)")
                continue
            block = block[lines_to_skip:]
            _log(f"[Phase 0] block {n_blocks}: skipped first {n_lines_to_skip - lines_to_skip} lines, "
                 f"{len(block)} lines remain in this block")
            lines_to_skip = 0

        raw_block = _decode_hits_block(block)
        raw_block = _assign_timestamps_block(raw_block, ts_state)
        dt_block = _dt_filter_block(raw_block)
        n_raw_in_block = len(raw_block["ch"])
        del raw_block

        n_dt_in_block = 0
        if dt_block is not None:
            n_dt_in_block = len(dt_block["sl"])
            if out_file is None:
                out_file = uproot.recreate(output_root_path)
                branch_types = {k: v.dtype for k, v in dt_block.items()}
                out_file.mktree("dt_hits", branch_types)
                out_file["dt_hits"].extend(dt_block)
                _log(f"[Phase 0] created output tree \"dt_hits\" in {output_root_path}")
            else:
                out_file["dt_hits"].extend(dt_block)
            n_written += n_dt_in_block
            del dt_block

        block_dt = time.perf_counter() - block_t0
        total_dt = time.perf_counter() - phase0_t0
        rate = n_raw_lines_seen / total_dt if total_dt > 0 else 0
        dt_fraction = n_dt_in_block / max(1, n_raw_in_block)
        _log(f"[Phase 0] block {n_blocks}: {n_raw_in_block} raw hits -> {n_dt_in_block} DT hits "
             f"({100*dt_fraction:.1f}%) | block took {block_dt:.2f}s | "
             f"running totals: {n_raw_lines_seen} raw, {n_written} DT | "
             f"{rate:.0f} raw lines/s")

        gc.collect()

    if out_file is not None:
        out_file.close()

    total_dt = time.perf_counter() - phase0_t0
    _log(f"[Phase 0] DONE. {n_blocks} blocks, {n_raw_lines_seen} raw lines read, "
         f"{n_written} DT hits written, took {total_dt:.1f}s ({total_dt/60:.1f} min)")

    if n_written == 0:
        raise RuntimeError(f"No DT hits found in {input_dumpfile} -- check block_n_lines / masks / mapping.")


# =================================================================
# PHASE 1+: fully chunked processing from the DT-only ROOT file
# =================================================================

def _apply_individual_dead_time_chunk(hits_chunk):
    """Dead-time cut applied to ONE CHUNK only -- state resets at every
    chunk boundary (no carry-over)."""
    if params._dt_ts_individual_dead_time <= 0:
        return hits_chunk
    dead_time = params._dt_ts_individual_dead_time
    n_cur = len(hits_chunk["ts"])
    if n_cur == 0:
        return hits_chunk

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
    """Fold ONE CHUNK's hit-diff histogram into running totals."""
    n_cur = len(hits_chunk["ts"])
    if n_cur == 0:
        return
    combo = np.stack([hits_chunk["sl"], hits_chunk["ly"], hits_chunk["wi"]], axis=1)
    _, inverse = np.unique(combo, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    group_bounds = np.flatnonzero(np.diff(sorted_inverse)) + 1
    group_starts = np.concatenate(([0], group_bounds))
    group_ends = np.concatenate((group_bounds, [n_cur]))

    for g in range(len(group_starts)):
        start, end = group_starts[g], group_ends[g]
        idx = order[start:end]
        if end - start < 2:
            continue
        ts_list = hits_chunk["ts"][idx]
        err_ts_list = hits_chunk["err_ts"][idx]
        diffs = np.asarray(ts_list[1:], dtype=np.float64) - np.asarray(ts_list[:-1], dtype=np.float64)
        err_diffs = np.sqrt(np.asarray(err_ts_list[1:], dtype=np.float64) ** 2
                             + np.asarray(err_ts_list[:-1], dtype=np.float64) ** 2)
        hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = \
            hist_utils.calculate_histogram_and_shifted_histograms(
                data=diffs, edges=running["edges"], err_data=err_diffs
            )
        running["hist"] += hist_
        running["entries"] += entries_
        running["underflow"] += underflow_
        running["overflow"] += overflow_
        running["hist_err_right"] += hist_err_right_
        running["hist_err_left"] += hist_err_left_


def _extend_or_create(out_files, key, path, data_dict):
    """Create a ROOT tree (classic TTree via mktree) on first write,
    extend() on subsequent writes."""
    if len(data_dict) == 0 or all(len(v) == 0 for v in data_dict.values()):
        return
    if key not in out_files:
        f = uproot.recreate(path)
        branch_types = {}
        for k, v in data_dict.items():
            if v.ndim == 1:
                branch_types[k] = v.dtype
            else:
                branch_types[k] = (v.dtype, v.shape[1:])
        f.mktree("tree", branch_types)
        f["tree"].extend(data_dict)
        out_files[key] = f
        _log(f"    created output tree \"{key}\" -> {path}")
    else:
        out_files[key]["tree"].extend(data_dict)


def run_streamed_pipeline(dt_hits_root_path, dataset_folder_pcls, dataset_name, *,
                           chunk_step_size="200 MB",
                           max_chi2=20, max_alpha=np.deg2rad(60),
                           verbose=False):
    dt_hit_diff_hist_file = dataset_folder_pcls + dataset_name + "_hit_diff.pcl"
    sl_patterns_root = dataset_folder_pcls + dataset_name + "_sl_patterns.root"
    sl_fits_root = dataset_folder_pcls + dataset_name + "_sl_fits.root"
    sl_refits_root = dataset_folder_pcls + dataset_name + "_sl_refits.root"
    super_fits_root = dataset_folder_pcls + dataset_name + "_super_fits.root"

    n_bins = 2500
    edges = np.linspace(0, 5000, n_bins + 1)
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = \
        hist_utils.create_empty_histogram(edges=edges)
    running_hist = {
        "edges": edges, "hist": hist, "entries": entries, "underflow": underflow,
        "overflow": overflow, "hist_err_right": hist_err_right, "hist_err_left": hist_err_left,
    }

    out_files = {}
    n_chunks = 0
    # running totals across all chunks, for the final summary
    totals = {
        "hits_in": 0, "hits_after_deadtime": 0, "n_patterns": 0,
        "n_fits": 0, "n_cut_fits": 0, "n_refits": 0,
        "n_super_patterns": 0, "n_super_fits": 0,
    }

    _log(f"[Phase 1+] START streaming \"{dt_hits_root_path}\" in chunks of {chunk_step_size}")
    phase1_t0 = time.perf_counter()


    cell_counts = {
        sl: {
            ly: {
                wi: 0
                for wi in range(
                    params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                    params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
                )
            }
            for ly in range(0, 4)
        }
        for sl in range(1, 4)
    }
    ts_min, ts_max = None, None

    for chunk in uproot.iterate(f"{dt_hits_root_path}:dt_hits", step_size=chunk_step_size, library="np"):
        n_chunks += 1
        chunk_t0 = time.perf_counter()
        n_hits_in = len(chunk["sl"])
        totals["hits_in"] += n_hits_in
        _log(f"--- chunk {n_chunks}: {n_hits_in} DT hits in ---")

        _fold_histogram_chunk(chunk, running_hist)

        chunk_deadtime = _apply_individual_dead_time_chunk(chunk)

        # --- occupancy/duration accumulation (vectorized, no python loop over hits) ---
        combo = np.stack([chunk["sl"], chunk["ly"], chunk["wi"]], axis=1)
        uniq_cells, uniq_counts = np.unique(combo, axis=0, return_counts=True)
        for (sl_u, ly_u, wi_u), c in zip(uniq_cells, uniq_counts):
            cell_counts[int(sl_u)][int(ly_u)][int(wi_u)] += int(c)

        chunk_ts_min = chunk["ts"].min()
        chunk_ts_max = chunk["ts"].max()
        ts_min = chunk_ts_min if ts_min is None else min(ts_min, chunk_ts_min)
        ts_max = chunk_ts_max if ts_max is None else max(ts_max, chunk_ts_max)

        del chunk
        n_hits_after_deadtime = len(chunk_deadtime["sl"])
        totals["hits_after_deadtime"] += n_hits_after_deadtime
        _log(f"    chunk {n_chunks}: {n_hits_after_deadtime} hits survive dead-time cut "
             f"({100*n_hits_after_deadtime/max(1,n_hits_in):.1f}%)")
        if n_hits_after_deadtime == 0:
            _log(f"    chunk {n_chunks}: empty after dead-time cut, skipping to next chunk")
            continue

        t_step = time.perf_counter()
        sl_patterns_chunk = ba_justus_utils.find_sl_patterns(
            hits=chunk_deadtime, verbose=verbose, silent=True,
            simulation_only_muon_patterns=False, fit_vd=True,
        )
        del chunk_deadtime
        n_patterns = len(sl_patterns_chunk.get("sl", []))
        totals["n_patterns"] += n_patterns
        _log(f"    chunk {n_chunks}: find_sl_patterns -> {n_patterns} patterns "
             f"({time.perf_counter()-t_step:.2f}s)")
        if n_patterns == 0:
            _log(f"    chunk {n_chunks}: no patterns found, skipping rest of chunk")
            gc.collect()
            continue
        _extend_or_create(out_files, "sl_patterns", sl_patterns_root, sl_patterns_chunk)

        t_step = time.perf_counter()
        sl_fits_chunk = ba_justus_utils.fit_sl_patterns(
            patterns=sl_patterns_chunk, verbose=verbose, silent=True, fit_vd=False, suffix="",
        )
        del sl_patterns_chunk
        n_fits = len(sl_fits_chunk.get("sl", []))
        totals["n_fits"] += n_fits
        _log(f"    chunk {n_chunks}: fit_sl_patterns (fixed vd) -> {n_fits} fits "
             f"({time.perf_counter()-t_step:.2f}s)")

        sl_cut_fits_chunk = data_utils.cut_data(
            data=sl_fits_chunk,
            conditions=[("impossible", "==", 0), ("chi2/ndf", "<", max_chi2)],
            silent=True,
        )
        n_cut_fits = len(sl_cut_fits_chunk.get("sl", []))
        totals["n_cut_fits"] += n_cut_fits
        _log(f"    chunk {n_chunks}: after chi2/ndf, impossible cut -> {n_cut_fits} fits "
             f"({100*n_cut_fits/max(1,n_fits):.1f}% survive)")

        t_step = time.perf_counter()
        sl_refits_chunk = ba_justus_utils.fit_sl_patterns(
            patterns=sl_cut_fits_chunk, verbose=verbose, silent=True, fit_vd=True, suffix="_refit",
        )
        del sl_cut_fits_chunk
        n_refits = len(sl_refits_chunk.get("sl", []))
        totals["n_refits"] += n_refits
        _log(f"    chunk {n_chunks}: fit_sl_patterns (vd floated, refit) -> {n_refits} refits "
             f"({time.perf_counter()-t_step:.2f}s)")

        if n_fits > 0:
            _extend_or_create(out_files, "sl_fits", sl_fits_root, sl_fits_chunk)
        if n_refits > 0:
            _extend_or_create(out_files, "sl_refits", sl_refits_root, sl_refits_chunk)
        del sl_refits_chunk

        t_step = time.perf_counter()
        try:
            super_patterns_chunk = ba_justus_utils.build_phi_super_patterns(
                sl_fits_chunk, silent=True, verbose=verbose, max_chi2ndf=max_chi2, max_alpha=max_alpha,
            )
            n_super_patterns = len(super_patterns_chunk.get("sl1", []))
        except Exception as e:
            _log(f"    chunk {n_chunks}: build_phi_super_patterns FAILED/skipped: {e}")
            super_patterns_chunk = None
            n_super_patterns = 0
        del sl_fits_chunk
        totals["n_super_patterns"] += n_super_patterns
        _log(f"    chunk {n_chunks}: build_phi_super_patterns -> {n_super_patterns} super patterns "
             f"({time.perf_counter()-t_step:.2f}s)")

        n_super_fits = 0
        if super_patterns_chunk is not None and n_super_patterns > 0:
            t_step = time.perf_counter()
            suffix = "_free_vd_super_fit"
            super_fits_chunk = ba_justus_utils.fit_super_sl_patterns(
                super_patterns_chunk, silent=True, verbose=verbose, fit_vd=True, suffix=suffix,
            )
            del super_patterns_chunk
            n_super_fits = len(super_fits_chunk.get("ts0", []))
            _log(f"    chunk {n_chunks}: fit_super_sl_patterns -> {n_super_fits} super fits "
                 f"({time.perf_counter()-t_step:.2f}s)")
            if n_super_fits > 0:
                _extend_or_create(out_files, "super_fits", super_fits_root, super_fits_chunk)
            del super_fits_chunk
        totals["n_super_fits"] += n_super_fits

        gc.collect()

        chunk_dt = time.perf_counter() - chunk_t0
        elapsed_total = time.perf_counter() - phase1_t0
        _log(f"--- chunk {n_chunks} done in {chunk_dt:.1f}s | "
             f"cumulative: {n_chunks} chunks, {elapsed_total:.1f}s elapsed, "
             f"{totals['hits_in']} hits processed, {totals['n_super_fits']} super fits so far ---")

    for f in out_files.values():
        f.close()

    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
        hist=running_hist["hist"], hist_err_right=running_hist["hist_err_right"],
        hist_err_left=running_hist["hist_err_left"], do_stat_err=True,
    )
    err_hist_stat = np.sqrt(running_hist["hist"])
    duration_seconds = float(ts_max - ts_min) * 0.78 * 1e-9
    cell_counts_file = dataset_folder_pcls + dataset_name + "_cell_counts.pcl"
    data_utils.store_pickle(
        data={"cell_counts": cell_counts, "duration_seconds": duration_seconds,
              "ts_min": ts_min, "ts_max": ts_max},
        file=cell_counts_file,
    )
    print("Cell counts file saved")
    _log(f"[Phase 1+] cell_counts -> {cell_counts_file}")
    specific_data = {
        "edges": running_hist["edges"], "centers": centers, "hist": running_hist["hist"],
        "err_hist": err_hist, "err_hist_stat": err_hist_stat, "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up, "entries": running_hist["entries"],
        "underflow": running_hist["underflow"], "overflow": running_hist["overflow"],
    }
    data_utils.store_pickle(data=specific_data, file=dt_hit_diff_hist_file)

    total_dt = time.perf_counter() - phase1_t0
    _log(f"[Phase 1+] DONE. {n_chunks} chunks in {total_dt:.1f}s ({total_dt/60:.1f} min)")
    _log(f"[Phase 1+] TOTALS: hits_in={totals['hits_in']}, "
         f"hits_after_deadtime={totals['hits_after_deadtime']}, "
         f"patterns={totals['n_patterns']}, fits={totals['n_fits']}, "
         f"cut_fits={totals['n_cut_fits']}, refits={totals['n_refits']}, "
         f"super_patterns={totals['n_super_patterns']}, super_fits={totals['n_super_fits']}")
    _log(f"[Phase 1+] outputs:")
    _log(f"    sl_patterns -> {sl_patterns_root}")
    _log(f"    sl_fits     -> {sl_fits_root}")
    _log(f"    sl_refits   -> {sl_refits_root}")
    _log(f"    super_fits  -> {super_fits_root}")
    _log(f"    hit_diff hist -> {dt_hit_diff_hist_file}")


# =================================================================
# main
# =================================================================

def find_input_dumpfile(base_path, dataset_name):
    candidate = base_path + "data_tests_cuts/" + dataset_name + ".txt"
    if os.path.exists(candidate):
        _log(f"input dumpfile resolved to (data_tests_cuts): {candidate}")
        return candidate
    candidate = base_path + "data_runs/" + dataset_name + ".txt"
    _log(f"input dumpfile resolved to (data_runs): {candidate}")
    return candidate


def main():
    _log("###### root_streaming_pipeline_v3.py: main() entered")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--phase0_block_lines", type=int, default=500_000,
                         help="raw dumpfile lines per Phase 0 block")
    parser.add_argument("--chunk_size", type=str, default="200 MB",
                         help="uproot step_size for Phase 1+ streaming")
    parser.add_argument("--skip_conversion", action="store_true")
    args = parser.parse_args()

    _log(f"args: dataset_name={args.dataset_name!r}, "
         f"phase0_block_lines={args.phase0_block_lines}, "
         f"chunk_size={args.chunk_size!r}, skip_conversion={args.skip_conversion}")

    start = time.perf_counter()
    base_path = "data_ba/"
    pcls_path = "pcls/"
    dataset_folder_pcls = base_path + pcls_path + args.dataset_name + "/"
    _log(f"dataset_folder_pcls = {dataset_folder_pcls}")

    os.makedirs(dataset_folder_pcls, exist_ok=True)

    dt_hits_root_path = dataset_folder_pcls + args.dataset_name + "_dt_hits.root"
    _log(f"dt_hits_root_path = {dt_hits_root_path}")

    if args.skip_conversion and os.path.exists(dt_hits_root_path):
        _log(f"[Phase 0] SKIPPED -- --skip_conversion set and {dt_hits_root_path} already exists")
    else:
        input_dumpfile = find_input_dumpfile(base_path, args.dataset_name)
        convert_dumpfile_to_dt_root_chunked(
            input_dumpfile, dt_hits_root_path, block_n_lines=args.phase0_block_lines,
        )

    run_streamed_pipeline(
        dt_hits_root_path, dataset_folder_pcls, args.dataset_name,
        chunk_step_size=args.chunk_size,
    )

    stop = time.perf_counter()
    _log(f"###### This script ran for: {(stop-start)/60/60:.4f} hours "
         f"({(stop-start)/60:.1f} minutes)")


if __name__ == "__main__":
    _log("###### Starting root_streaming_pipeline_v3.py")
    main()
    _log("###### Done.")
    os._exit(0)   # force process termination, bypass any hung thread-pool teardown
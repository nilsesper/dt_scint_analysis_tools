###########################################
### DT WORKFLOW STAGES ON ROOT FILES
###########################################
# One function per stage, with the name of its script in scripts/. A stage reads its input file(s) in chunks,
# applies one reconstruction step (see the dt_*_utils modules) and writes one ROOT file.
#
#   dumpfile (.txt)
#     -> dt hits                        dumpfile_to_dt_hits             (optionally with testpulse timing calibration)
#          -> hit diff hist             dt_hits_to_hit_diff_hist        (side product)
#          -> cell counts               dt_hits_to_cell_counts          (side product)
#          -> sl patterns               dt_hits_to_sl_patterns          (applies the dead time)
#               -> sl fits              sl_patterns_to_sl_fits
#                    -> cut sl fits     apply_cuts
#                         -> super fits             sl_fits_to_super_fits   (the two phi superlayers together)
#                              -> cut super fits    apply_cuts
#                                   -> dt muons     super_fits_to_dt_muons  (cut super fits + theta sl fits)
#   testpulse dumpfile (.txt)
#     -> timing calibration             dumpfile_to_dt_tp_corrections
#
# Chunks: the dt hits are processed in chunks of step_size rows. Patterns are only searched inside a chunk, and every
# later row carries the "chunk_id" of its dt hit chunk; the later stages combine only rows of the same chunk_id.

import gc
import time
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import (cut_utils, data_utils, dt_calibration_utils, dt_chamber_utils, dt_dumpfile_utils, dt_hit_utils,
                                  dt_muon_reco_utils, dt_pattern_utils, dt_sl_fit_utils, dt_super_fit_utils, parallel_utils, root_utils)
from analysis_tools.utils.root_utils import log

# -----------------------------------------

# suffix of the super fit result branches (e.g. "t0_super_fits", "chi2/ndf_super_fits")
DEFAULT_SUPER_FIT_SUFFIX = "_super_fits"

TS_UNIT_SECONDS = 0.78e-9

# -----------------------------------------
# dumpfile -> dt hits
# -----------------------------------------

### decode the dumpfile, keep the hits of the analysed dt cells (not masked / dead), write them to tree "dt_hits"
# dt_tp_corrections_file (optional): testpulse timing calibration applied to every hit
def dumpfile_to_dt_hits(input_dumpfile, dt_hits_file, *, n_lines_to_skip=999, block_n_lines=500_000, n_proc=1, dt_tp_corrections_file=None):
    label = "dumpfile -> dt hits"
    root_utils.check_input_file(input_dumpfile)
    log(f"[{label}] START \"{input_dumpfile}\" -> \"{dt_hits_file}\"")
    dt_tp_corrections = None
    if dt_tp_corrections_file is not None:
        dt_tp_corrections = dt_calibration_utils.read_tp_corrections(dt_tp_corrections_file, label)
        log(f"[{label}] testpulse timing corrections from \"{dt_tp_corrections_file}\" are applied to the hits")
    else:
        log(f"[{label}] no testpulse timing corrections given, timestamps are not corrected")
    cells_without_correction = set()
    t_start = time.perf_counter()

    dumpfile, n_blocks = dt_dumpfile_utils.open_dumpfile(input_dumpfile, n_lines_to_skip, block_n_lines, n_proc, label)
    pool = parallel_utils.open_pool(n_proc)
    n_blocks_at_once = 1
    if pool is not None:
        n_blocks_at_once = 2 * n_proc
    overflow_counter = dt_dumpfile_utils.new_overflow_counter()
    writer = root_utils.TreeWriter(dt_hits_file, root_utils.DT_HITS_TREE)
    n_words, i_block = 0, 0
    while True:
        blocks = dt_dumpfile_utils.decode_next_blocks(dumpfile, block_n_lines, n_blocks_at_once, False, pool)
        if len(blocks) == 0:
            break
        for block in blocks:
            i_block += 1
            n_words += block["n_words"]
            dt_hits = dt_dumpfile_utils.set_timestamps(block, overflow_counter)
            n_dt_hits = 0
            if dt_hits is not None:
                n_dt_hits = root_utils.length(dt_hits)
                if dt_tp_corrections is not None:
                    dt_hits = dt_calibration_utils.apply_tp_corrections(dt_hits, dt_tp_corrections, cells_without_correction)
                writer.write(dt_hits)
            log(f"[{label}] block {i_block:,} / {n_blocks:,}: {block['n_words']:,} raw hits -> {n_dt_hits:,} dt hits "
                f"({100 * n_dt_hits / max(1, block['n_words']):.1f}%) | totals: {n_words:,} raw, {writer.n_written:,} dt | "
                f"{n_words / max(time.perf_counter() - t_start, 1e-9):,.0f} raw hits/s")
    dumpfile.close()
    parallel_utils.close_pool(pool)
    if overflow_counter["n_overflows"] > 0:
        log(f"[{label}] orbit counter overflows found: {overflow_counter['n_overflows']:,}")
    writer.close()

    if dt_tp_corrections is not None:
        dt_calibration_utils.warn_about_cells_without_correction(cells_without_correction, label)
    corrected_text = ""
    if dt_tp_corrections is not None:
        corrected_text = " (timing corrected)"
    log(f"[{label}] DONE. {i_block:,} blocks, {n_words:,} raw hits read, {writer.n_written:,} dt hits written{corrected_text}, "
        f"took {time.perf_counter() - t_start:.1f}s")
    if writer.n_written == 0:
        raise RuntimeError(f"No dt hits found in {input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")
    return writer.n_written

### apply a testpulse timing calibration to an existing dt hits file
def dt_hits_timing_correction(dt_hits_file, dt_tp_corrections_file, corr_dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    label = "dt hits -> corrected dt hits"
    root_utils.check_input_file(dt_hits_file)
    log(f"[{label}] START \"{dt_hits_file}\" + \"{dt_tp_corrections_file}\" -> \"{corr_dt_hits_file}\"")
    dt_tp_corrections = dt_calibration_utils.read_tp_corrections(dt_tp_corrections_file, label)
    cells_without_correction = set()
    n_hits = 0
    writer = root_utils.TreeWriter(corr_dt_hits_file, root_utils.DT_HITS_TREE)
    chunks = root_utils.chunk_ranges(dt_hits_file, root_utils.DT_HITS_TREE, step_size)
    for i_chunk in range(len(chunks)):
        start, stop = chunks[i_chunk]
        hits = root_utils.read_entries(dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        n_hits += root_utils.length(hits)
        writer.write(dt_calibration_utils.apply_tp_corrections(hits, dt_tp_corrections, cells_without_correction))
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(hits):,} hits corrected")
    writer.close()
    dt_calibration_utils.warn_about_cells_without_correction(cells_without_correction, label)
    log(f"[{label}] DONE. {n_hits:,} hits")
    return n_hits

# -----------------------------------------
# testpulse dumpfile -> timing calibration
# -----------------------------------------

### timing calibration of every cell from a dumpfile recorded with testpulses on all channels (see dt_calibration_utils)
# all cells of the chamber are used (also masked / dead ones), no dead time cut
# output .root: one row per cell + summary + ROOT histograms; .pcl: only the correction dict (format of the old script)
# dt_tp_hits_file (optional): the testpulse dt hits with the extra branch ts_orbit
def dumpfile_to_dt_tp_corrections(input_dumpfile, dt_tp_corrections_file, *, dt_tp_hits_file=None, n_lines_to_skip=None,
                                  block_n_lines=500_000, n_proc=1, rel_thres=0.2, alignment="chamber", correct_for_offsets=True):
    label = "testpulse dumpfile -> dt tp corrections"
    if alignment != "chamber" and alignment != "sl":
        raise ValueError(f"alignment has to be \"chamber\" or \"sl\", not \"{alignment}\"")
    if n_lines_to_skip is None:
        n_lines_to_skip = params._dumpfile_hits_to_skip
    root_utils.check_input_file(input_dumpfile)
    log(f"[{label}] START \"{input_dumpfile}\" -> \"{dt_tp_corrections_file}\"")
    log(f"[{label}] rel_thres={rel_thres}, alignment={alignment}, correct_for_offsets={correct_for_offsets}")
    t_start = time.perf_counter()

    # 1. histograms of the testpulse hit times of every cell
    tp_histograms = dt_calibration_utils.empty_testpulse_histograms()
    hits_writer = None
    if dt_tp_hits_file is not None:
        hits_writer = root_utils.TreeWriter(dt_tp_hits_file, root_utils.DT_HITS_TREE)
    dumpfile, n_blocks = dt_dumpfile_utils.open_dumpfile(input_dumpfile, n_lines_to_skip, block_n_lines, n_proc, label)
    pool = parallel_utils.open_pool(n_proc)
    n_blocks_at_once = 1
    if pool is not None:
        n_blocks_at_once = 2 * n_proc
    overflow_counter = dt_dumpfile_utils.new_overflow_counter()
    n_words, n_dt_hits, i_block = 0, 0, 0
    while True:
        blocks = dt_dumpfile_utils.decode_next_blocks(dumpfile, block_n_lines, n_blocks_at_once, True, pool)
        if len(blocks) == 0:
            break
        for block in blocks:
            i_block += 1
            n_words += block["n_words"]
            dt_hits = dt_dumpfile_utils.set_timestamps(block, overflow_counter)
            n_block_dt_hits = 0
            if dt_hits is not None:
                n_block_dt_hits = root_utils.length(dt_hits)
                n_dt_hits += n_block_dt_hits
                ts_orbit = dt_calibration_utils.ts_in_orbit(dt_hits)
                dt_calibration_utils.fill_testpulse_histograms(tp_histograms, dt_hits, ts_orbit)
                if hits_writer is not None:
                    dt_hits["ts_orbit"] = ts_orbit.astype(params._ts_type)
                    hits_writer.write(dt_hits)
            log(f"[{label}] block {i_block:,} / {n_blocks:,}: {block['n_words']:,} raw hits -> {n_block_dt_hits:,} dt hits | "
                f"totals: {n_words:,} raw, {n_dt_hits:,} dt")
    dumpfile.close()
    parallel_utils.close_pool(pool)
    if overflow_counter["n_overflows"] > 0:
        log(f"[{label}] orbit counter overflows found: {overflow_counter['n_overflows']:,}")
    if hits_writer is not None:
        hits_writer.close()
        log(f"[{label}] testpulse dt hits written to \"{dt_tp_hits_file}\"")
    if n_dt_hits == 0:
        raise RuntimeError(f"No dt hits found in {input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")

    # 2. calibration of every cell
    calib = dt_calibration_utils.calibrate_cells(tp_histograms, rel_thres=rel_thres, alignment=alignment, correct_for_offsets=correct_for_offsets, label=label)
    valid = (calib["valid"] == 1)
    n_valid = int(np.sum(valid))
    log(f"[{label}] {n_valid:,} / {len(valid):,} cells with testpulse peak; corrections between {np.amin(calib['ts_corr'][valid]):.2f} and "
        f"{np.amax(calib['ts_corr'][valid]):.2f} ts units (rms {np.std(calib['ts_corr'][valid]):.2f}), mean uncertainty {np.mean(calib['err_ts_corr'][valid]):.2f}")
    cells_without_peak = []
    for i in range(len(valid)):
        if not valid[i]:
            cells_without_peak.append((int(calib["sl"][i]), int(calib["ly"][i]), int(calib["wi"][i])))
    if len(cells_without_peak) > 0:
        message = f"[{label}] cells without testpulse peak (ts_corr = 0): {cells_without_peak[:20]}"
        if len(cells_without_peak) > 20:
            message += " ..."
        log(message)

    # 3. output file
    if dt_tp_corrections_file.endswith(".pcl"):
        root_utils.prepare_output_file(dt_tp_corrections_file)
        data_utils.store_pickle(data=dt_calibration_utils.tp_corrections_dict(calib), file=dt_tp_corrections_file, silent=True)
    else:
        summary = {"n_cells": len(valid), "n_valid": n_valid, "rel_thres": float(rel_thres), "chamber_alignment": int(alignment == "chamber"),
                   "correct_for_offsets": int(correct_for_offsets), "n_raw_hits": n_words, "n_dt_hits": n_dt_hits, "n_lines_to_skip": int(n_lines_to_skip)}
        root_utils.write_tree(dt_tp_corrections_file, calib, tree=root_utils.DEFAULT_TREE, summary=summary,
                              histograms=dt_calibration_utils.calibration_histograms(calib, tp_histograms))
    log(f"[{label}] DONE. {len(valid):,} cells written to \"{dt_tp_corrections_file}\", took {time.perf_counter() - t_start:.1f}s")
    return calib

# -----------------------------------------
# dt hits -> side products
# -----------------------------------------

### histogram of the time between consecutive hits of the same cell, summed over all cells (hits before the dead time cut)
# output .root: one row per bin + summary + TH1D "hit_diff_hist"; .pcl: the histogram dict (format of the old pipeline)
def dt_hits_to_hit_diff_hist(dt_hits_file, hit_diff_hist_file, *, step_size=root_utils.DEFAULT_STEP_SIZE, n_bins=5000, ts_max=5000):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> hit diff hist] START \"{dt_hits_file}\" -> \"{hit_diff_hist_file}\"")
    hit_diff_hist = dt_hit_utils.empty_hit_diff_histogram(n_bins, ts_max)
    n_hits = 0
    chunks = root_utils.chunk_ranges(dt_hits_file, root_utils.DT_HITS_TREE, step_size)
    for i_chunk in range(len(chunks)):
        start, stop = chunks[i_chunk]
        hits = root_utils.read_entries(dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        n_hits += root_utils.length(hits)
        dt_hit_utils.fill_hit_diff_histogram(hit_diff_hist, hits)
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(hits):,} hits, {int(hit_diff_hist['entries']):,} entries in histogram so far")
    hist_data = dt_hit_utils.finish_hit_diff_histogram(hit_diff_hist)

    root_utils.prepare_output_file(hit_diff_hist_file)
    if hit_diff_hist_file.endswith(".pcl"):
        data_utils.store_pickle(data=hist_data, file=hit_diff_hist_file, silent=True)
    else:
        edges = hist_data["edges"]
        bins = {"edge_low": np.asarray(edges[:-1], dtype=np.float64), "edge_high": np.asarray(edges[1:], dtype=np.float64),
                "center": np.asarray(hist_data["centers"], dtype=np.float64)}
        for key in ["hist", "err_hist", "err_hist_stat", "err_hist_down", "err_hist_up"]:
            bins[key] = np.asarray(hist_data[key], dtype=np.float64)
        summary = {}
        for key in ["entries", "underflow", "overflow"]:
            summary[key] = np.float64(hist_data[key])
        summary["n_hits"] = np.int64(n_hits)
        root_utils.write_tree(hit_diff_hist_file, bins, summary=summary, histograms={"hit_diff_hist": (hist_data["hist"], edges)})
    log(f"[dt hits -> hit diff hist] DONE. {n_hits:,} hits, {int(hist_data['entries']):,} entries in histogram")
    return hist_data

### number of hits per cell and duration of the run (hits before the dead time cut)
# output .root: one row per cell + summary + TH2D "cell_counts" (x = wire, y = 4 * (sl - 1) + ly); .pcl: dict (old format)
def dt_hits_to_cell_counts(dt_hits_file, cell_counts_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> cell counts] START \"{dt_hits_file}\" -> \"{cell_counts_file}\"")
    cell_counts, ts_min, ts_max, n_hits = dt_hit_utils.count_hits_per_cell(dt_hits_file, step_size=step_size)
    if n_hits == 0:
        raise RuntimeError(f"No dt hits in {dt_hits_file}.")
    duration_seconds = float(ts_max - ts_min) * 0.78 * 1e-9
    counts_data = {"cell_counts": cell_counts, "duration_seconds": duration_seconds, "ts_min": ts_min, "ts_max": ts_max}

    root_utils.prepare_output_file(cell_counts_file)
    if cell_counts_file.endswith(".pcl"):
        data_utils.store_pickle(data=counts_data, file=cell_counts_file, silent=True)
    else:
        cells = dt_chamber_utils.chamber_cells()
        rows = {"sl": np.zeros(len(cells), dtype=np.uint8), "ly": np.zeros(len(cells), dtype=np.uint8),
                "wi": np.zeros(len(cells), dtype=np.uint8), "count": np.zeros(len(cells), dtype=np.int64)}
        for i in range(len(cells)):
            sl, ly, wi = cells[i]
            rows["sl"][i], rows["ly"][i], rows["wi"][i] = sl, ly, wi
            rows["count"][i] = cell_counts[sl][ly][wi]
        summary = {"duration_seconds": np.float64(duration_seconds), "ts_min": np.float64(ts_min), "ts_max": np.float64(ts_max), "n_hits": np.int64(n_hits)}
        # 2d histogram: x = wire, y = 4 * (sl - 1) + ly
        n_wires = int(np.amax(rows["wi"])) + 1
        counts_2d = np.zeros((n_wires, 12))
        for i in range(len(cells)):
            sl, ly, wi = cells[i]
            counts_2d[wi][4 * (sl - 1) + ly] = rows["count"][i]
        root_utils.write_tree(cell_counts_file, rows, summary=summary,
                              histograms={"cell_counts": (counts_2d, np.arange(n_wires + 1) - 0.5, np.arange(13) - 0.5)})
    log(f"[dt hits -> cell counts] DONE. {n_hits:,} hits, duration {duration_seconds:.3f} s")
    return counts_data

# -----------------------------------------
# dt hits -> sl patterns -> sl fits
# -----------------------------------------

### per chunk of dt hits: dead time cut (unless apply_dead_time=False), then the pattern search in each superlayer
# wide_ts_window: hit time window for a free drift velocity (params._dt_sl_patterns_ts_window_fit_vd), else for the fixed one
# n_proc > 1: superlayers and pieces in time are searched in parallel, with the same result
def dt_hits_to_sl_patterns(dt_hits_file, sl_patterns_file, *, step_size=root_utils.DEFAULT_STEP_SIZE, apply_dead_time=True,
                           wide_ts_window=True, simulation_only_muon_patterns=False, n_proc=1, verbose=False):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> sl patterns] START \"{dt_hits_file}\" -> \"{sl_patterns_file}\" (chunks of {step_size}, n_proc={n_proc})")
    n_hits_in_total, n_hits_after_dead_time_total, n_patterns_total = 0, 0, 0
    pool = None
    if not verbose:
        pool = parallel_utils.open_pool(n_proc)
    writer = root_utils.TreeWriter(sl_patterns_file)
    chunks = root_utils.chunk_ranges(dt_hits_file, root_utils.DT_HITS_TREE, step_size)
    for i_chunk in range(len(chunks)):
        t_step = time.perf_counter()
        start, stop = chunks[i_chunk]
        chunk_id = i_chunk + 1
        hits = root_utils.read_entries(dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        n_hits_in = root_utils.length(hits)
        if apply_dead_time:
            hits = dt_hit_utils.apply_dead_time(hits)
        n_hits = root_utils.length(hits)
        n_hits_in_total += n_hits_in
        n_hits_after_dead_time_total += n_hits
        if n_hits == 0:
            log(f"    chunk {chunk_id:,} / {len(chunks):,}: {n_hits_in:,} hits in, none left after dead time cut")
            continue

        sl_patterns = dt_pattern_utils.find_sl_patterns(hits, wide_ts_window=wide_ts_window, only_single_muon_patterns=simulation_only_muon_patterns,
                                                        pool=pool, n_proc=n_proc, verbose=verbose)
        n_patterns = root_utils.length(sl_patterns)
        n_patterns_total += n_patterns
        log(f"    chunk {chunk_id:,} / {len(chunks):,}: {n_hits_in:,} hits in, {n_hits:,} after dead time cut ({100 * n_hits / max(1, n_hits_in):.1f}%) "
            f"-> {n_patterns:,} patterns ({time.perf_counter() - t_step:.2f}s)")
        if n_patterns > 0:
            writer.write(root_utils.set_chunk_id(sl_patterns, chunk_id))
        del hits, sl_patterns
        gc.collect()  # free the memory of this chunk before reading the next one
    writer.close()
    parallel_utils.close_pool(pool)
    log(f"[dt hits -> sl patterns] DONE. {len(chunks):,} chunks, hits_in={n_hits_in_total:,}, "
        f"hits_after_deadtime={n_hits_after_dead_time_total:,}, patterns={n_patterns_total:,}")
    return n_patterns_total

### fit a straight track to every pattern (all lateralities, the best one is kept)
# fit_vd: drift velocity as free fit parameter (default: fixed); rows are independent, so step_size and n_proc do not matter
def sl_patterns_to_sl_fits(sl_patterns_file, sl_fits_file, *, fit_vd=False, step_size=root_utils.DEFAULT_STEP_SIZE, n_proc=1, verbose=False):
    label = "sl patterns -> sl fits"
    root_utils.check_input_file(sl_patterns_file)
    log(f"[{label}] START \"{sl_patterns_file}\" -> \"{sl_fits_file}\" (fit_vd={fit_vd}, n_proc={n_proc})")
    if verbose:
        n_proc = 1
    n_patterns, n_fits = 0, 0
    writer = root_utils.TreeWriter(sl_fits_file)
    chunks = root_utils.chunk_ranges(sl_patterns_file, root_utils.DEFAULT_TREE, step_size)
    for i_chunk in range(len(chunks)):
        t_step = time.perf_counter()
        start, stop = chunks[i_chunk]
        patterns = root_utils.read_entries(sl_patterns_file, root_utils.DEFAULT_TREE, start, stop)
        n_patterns += root_utils.length(patterns)
        sl_fits = parallel_utils.run_fits_in_parallel(dt_sl_fit_utils.fit_sl_patterns, patterns, fit_vd, "", verbose, n_proc)
        n_fits += root_utils.length(sl_fits)
        writer.write(sl_fits)
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(patterns):,} patterns -> {root_utils.length(sl_fits):,} fits "
            f"({time.perf_counter() - t_step:.2f}s)")
        del patterns, sl_fits
        gc.collect()  # free the memory of this chunk before reading the next one
    writer.close()
    log(f"[{label}] DONE. patterns={n_patterns:,}, fits={n_fits:,}")
    return n_fits

# -----------------------------------------
# sl fits -> super fits -> dt muons
# -----------------------------------------

### pair the sl fits of the two phi superlayers to super patterns (per chunk_id) and fit one track through their 8 hits
# input: sl fits after quality cuts; max_alpha: max |track angle| of sl fits which are paired; max_chi2: optional
# additional chi2/ndf cut on them; fit_vd: drift velocity as free fit parameter (default: fixed)
# super_patterns_file: optionally also store the super patterns
# branches "row_sl<n>" (n = phi superlayers): rows of the two paired sl fits in sl_fits_file
def sl_fits_to_super_fits(sl_fits_file, super_fits_file, *, super_patterns_file=None, max_chi2=None, max_alpha=np.deg2rad(60),
                          fit_vd=False, suffix=DEFAULT_SUPER_FIT_SUFFIX, n_proc=1, verbose=False):
    root_utils.check_input_file(sl_fits_file)
    max_chi2ndf = np.inf
    if max_chi2 is not None:
        max_chi2ndf = max_chi2
    log(f"[sl fits -> super fits] START \"{sl_fits_file}\" -> \"{super_fits_file}\" "
        f"(max_chi2={max_chi2}, max_alpha={max_alpha:.4f} rad, fit_vd={fit_vd}, suffix=\"{suffix}\", n_proc={n_proc})")
    if verbose:
        n_proc = 1
    n_fits_total, n_super_patterns_total, n_super_fits_total = 0, 0, 0
    patterns_writer = None
    if super_patterns_file is not None:
        patterns_writer = root_utils.TreeWriter(super_patterns_file)
    writer = root_utils.TreeWriter(super_fits_file)
    chunks = root_utils.chunk_id_ranges(sl_fits_file, root_utils.DEFAULT_TREE)
    for i_chunk in range(len(chunks)):
        t_step = time.perf_counter()
        chunk_id, first_row, stop = chunks[i_chunk]
        sl_fits = root_utils.read_entries(sl_fits_file, root_utils.DEFAULT_TREE, first_row, stop)
        super_patterns = dt_super_fit_utils.build_phi_super_patterns(sl_fits, max_chi2ndf=max_chi2ndf, max_alpha=max_alpha)
        n_super_patterns = root_utils.length(super_patterns)
        n_super_fits = 0
        if n_super_patterns > 0:
            # rows of the paired sl fits in the whole file instead of in this chunk
            for sl in dt_chamber_utils.phi_superlayers():
                super_patterns[f"row_sl{sl}"] += first_row
            if patterns_writer is not None:
                patterns_to_store = {}
                for key in super_patterns:
                    patterns_to_store[key] = super_patterns[key]
                patterns_writer.write(root_utils.set_chunk_id(patterns_to_store, chunk_id))
            super_fits = parallel_utils.run_fits_in_parallel(dt_super_fit_utils.fit_super_sl_patterns, super_patterns, fit_vd, suffix, verbose, n_proc)
            n_super_fits = root_utils.length(super_fits)
            writer.write(root_utils.set_chunk_id(super_fits, chunk_id))
        n_fits_total += root_utils.length(sl_fits)
        n_super_patterns_total += n_super_patterns
        n_super_fits_total += n_super_fits
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(sl_fits):,} sl fits -> {n_super_patterns:,} super patterns -> "
            f"{n_super_fits:,} super fits ({time.perf_counter() - t_step:.2f}s)")
        gc.collect()  # free the memory of this chunk before reading the next one
    writer.close()
    if patterns_writer is not None:
        patterns_writer.close()
    log(f"[sl fits -> super fits] DONE. sl_fits={n_fits_total:,}, super_patterns={n_super_patterns_total:,}, super_fits={n_super_fits_total:,}")
    return n_super_fits_total

### the sl fits with the given chunk_id, and the number of their first row in the file
# checks that the super fits of this chunk point into them
def read_sl_fits_of_chunk(sl_fits_file, sl_fit_chunks, chunk_id, super_fits, super_fits_file):
    first_row, stop = None, None
    for fits_chunk_id, fits_start, fits_stop in sl_fit_chunks:
        if fits_chunk_id == chunk_id:
            first_row, stop = fits_start, fits_stop
            break
    if first_row is None:
        raise RuntimeError(f"No sl fits with chunk_id {chunk_id} in {sl_fits_file}. Is this the file the super fits were made from?")
    sl_fits = root_utils.read_entries(sl_fits_file, root_utils.DEFAULT_TREE, first_row, stop)
    for sl in dt_chamber_utils.phi_superlayers():
        for row in super_fits[f"row_sl{sl}"]:
            row_in_chunk = row - first_row
            if row_in_chunk < 0 or row_in_chunk >= root_utils.length(sl_fits) or sl_fits["sl"][row_in_chunk] != sl:
                raise RuntimeError(f"The sl fit rows stored in {super_fits_file} do not match {sl_fits_file}. Is this the file the super fits were made from?")
    return first_row, sl_fits

### combine the super fits (phi view) with the sl fits of the theta superlayer to dt muons, per chunk_id
# super_fits_file: super fits after quality cuts; sl_fits_file: the cut sl fits the super fits were made from
# branches "super_fit_row" (row in super_fits_file) and "sl<n>_fit_row" (rows in sl_fits_file): the fits of a muon
def super_fits_to_dt_muons(super_fits_file, sl_fits_file, dt_muons_file, *, suffix=DEFAULT_SUPER_FIT_SUFFIX, tgroup_tolerance=None, verbose=False):
    label = "super fits + theta sl fits -> dt muons"
    root_utils.check_input_file(super_fits_file)
    root_utils.check_input_file(sl_fits_file)
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    log(f"[{label}] START \"{super_fits_file}\" + \"{sl_fits_file}\" -> \"{dt_muons_file}\" (tgroup_tolerance={tgroup_tolerance:.2f} ts units)")
    theta_sl = dt_chamber_utils.theta_superlayer()
    sl_fit_chunks = root_utils.chunk_id_ranges(sl_fits_file, root_utils.DEFAULT_TREE)
    n_super_fits_total, n_theta_fits_total, n_muons_total, n_ambiguous_total = 0, 0, 0, 0
    ts_min, ts_max = None, None
    writer = root_utils.TreeWriter(dt_muons_file)
    chunks = root_utils.chunk_id_ranges(super_fits_file, root_utils.DEFAULT_TREE)
    for i_chunk in range(len(chunks)):
        chunk_id, first_super_row, stop = chunks[i_chunk]
        super_fits = root_utils.read_entries(super_fits_file, root_utils.DEFAULT_TREE, first_super_row, stop)
        if "t0" + suffix not in super_fits:
            raise KeyError(f"No super fit results with suffix \"{suffix}\" in {super_fits_file}.")
        if np.any(super_fits["impossible" + suffix] != 0):
            raise RuntimeError(f"{super_fits_file} contains super fits flagged \"impossible\". Apply cuts first "
                               f"(at least \"impossible{suffix},==,0\"), see scripts/apply_cuts.py.")

        # the sl fits of the theta superlayer of this chunk
        first_fit_row, sl_fits = read_sl_fits_of_chunk(sl_fits_file, sl_fit_chunks, chunk_id, super_fits, super_fits_file)
        theta_rows = []
        for row in range(root_utils.length(sl_fits)):
            if sl_fits["sl"][row] == theta_sl:
                theta_rows.append(row)
        theta_fits = {}
        for key in sl_fits:
            theta_fits[key] = sl_fits[key][theta_rows]

        dt_muons = dt_muon_reco_utils.reco_muons_from_super_fits(super_fits, theta_fits, suffix=suffix, tgroup_tolerance=tgroup_tolerance, verbose=verbose)
        n_muons = root_utils.length(dt_muons)
        if n_muons > 0:
            # rows of the fits of every muon, in the input files
            super_idx = dt_muons.pop("super_fit_idx")
            theta_idx = dt_muons.pop("theta_fit_idx")
            dt_muons["super_fit_row"] = np.zeros(n_muons, dtype=np.int64)
            for sl in dt_chamber_utils.phi_superlayers():
                dt_muons[f"sl{sl}_fit_row"] = np.zeros(n_muons, dtype=np.int64)
            dt_muons[f"sl{theta_sl}_fit_row"] = np.zeros(n_muons, dtype=np.int64)
            for m in range(n_muons):
                dt_muons["super_fit_row"][m] = super_idx[m] + first_super_row
                for sl in dt_chamber_utils.phi_superlayers():
                    dt_muons[f"sl{sl}_fit_row"][m] = super_fits[f"row_sl{sl}"][super_idx[m]]
                dt_muons[f"sl{theta_sl}_fit_row"][m] = theta_rows[theta_idx[m]] + first_fit_row

            dt_muons = data_utils.sort_by_key(data=dt_muons, sort_key="ts", silent=True)
            writer.write(root_utils.set_chunk_id(dt_muons, chunk_id))
            n_ambiguous_total += int(np.sum(dt_muons["n_theta_candidates"] > 1))
            if ts_min is None or np.amin(dt_muons["ts"]) < ts_min:
                ts_min = np.amin(dt_muons["ts"])
            if ts_max is None or np.amax(dt_muons["ts"]) > ts_max:
                ts_max = np.amax(dt_muons["ts"])
        n_super_fits_total += root_utils.length(super_fits)
        n_theta_fits_total += len(theta_rows)
        n_muons_total += n_muons
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(super_fits):,} super fits, {len(theta_rows):,} theta sl fits -> {n_muons:,} dt muons")
    writer.close()

    if n_muons_total == 0:
        log(f"[{label}] WARNING: no dt muons reconstructed, no output file was written.")
    rate_text = ""
    if ts_min is not None and ts_max > ts_min:
        duration = (ts_max - ts_min) * TS_UNIT_SECONDS
        rate_text = f", rate {n_muons_total / duration:.2f} Hz over {duration:.2f} s"
    log(f"[{label}] DONE. super_fits={n_super_fits_total:,}, theta_sl_fits={n_theta_fits_total:,}, "
        f"dt_muons={n_muons_total:,} ({n_ambiguous_total:,} with more than one theta candidate){rate_text}")
    return n_muons_total

# -----------------------------------------
# cuts (any ROOT file of this workflow)
# -----------------------------------------

### keep only the rows which pass all cuts [(key, operator, value)], prints the cut flow
def apply_cuts(input_file, output_file, cuts, *, tree=None, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(input_file)
    if len(cuts) == 0:
        raise ValueError("No cuts given.")
    tree = root_utils.resolve_tree_name(input_file, tree)
    log(f"[apply cuts] START \"{input_file}\" -> \"{output_file}\" (tree \"{tree}\")")
    n_in = 0
    n_after_cut = []  # number of rows which pass the first 1, 2, ... cuts
    for cut in cuts:
        n_after_cut.append(0)
    writer = root_utils.TreeWriter(output_file, tree)
    chunks = root_utils.chunk_ranges(input_file, tree, step_size)
    for i_chunk in range(len(chunks)):
        start, stop = chunks[i_chunk]
        rows = root_utils.read_entries(input_file, tree, start, stop)
        n_rows = root_utils.length(rows)
        n_in += n_rows
        cut_utils.check_cut_keys(cuts, rows, input_file)
        for i_cut in range(len(cuts)):
            rows = data_utils.cut_data(data=rows, conditions=[cuts[i_cut]], silent=True)
            n_after_cut[i_cut] += root_utils.length(rows)
        writer.write(rows)
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {root_utils.length(rows):,} of {n_rows:,} rows pass the cuts")
    writer.close()
    log(f"[apply cuts] cut flow w.r.t. the {n_in:,} input rows:")
    for i_cut in range(len(cuts)):
        key, operator, value = cuts[i_cut]
        n = n_after_cut[i_cut]
        log(f"    {key} {operator} {value}: {n:,} / {n_in:,} = {n / max(1, n_in):.4f}")
    if n_after_cut[-1] == 0:
        log("[apply cuts] WARNING: no rows pass the cuts, no output file was written.")
    log("[apply cuts] DONE.")
    return n_in, n_after_cut

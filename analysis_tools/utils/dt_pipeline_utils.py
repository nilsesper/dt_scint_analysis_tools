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
    timing_correction = None
    if dt_tp_corrections_file is not None:
        timing_correction = dt_calibration_utils.TimingCorrection.from_file(dt_tp_corrections_file, label=label)
        log(f"[{label}] testpulse timing corrections from \"{dt_tp_corrections_file}\" are applied to the hits")
    else:
        log(f"[{label}] no testpulse timing corrections given, timestamps are not corrected")
    t_start = time.perf_counter()
    n_words, n_blocks_done = 0, 0
    with root_utils.TreeWriter(dt_hits_file, root_utils.DT_HITS_TREE) as writer:
        for dt_hits, n_block_words, i_block, n_blocks in dt_dumpfile_utils.iterate_dt_hits(
                input_dumpfile, n_lines_to_skip=n_lines_to_skip, block_n_lines=block_n_lines, n_proc=n_proc, label=label):
            n_words += n_block_words
            n_blocks_done += 1
            n_dt_hits = 0
            if dt_hits is not None:
                n_dt_hits = root_utils.length(dt_hits)
                writer.write(timing_correction.apply(dt_hits) if timing_correction is not None else dt_hits)
            log(f"[{label}] block {i_block:,} / {n_blocks:,}: {n_block_words:,} raw hits -> {n_dt_hits:,} dt hits "
                f"({100 * n_dt_hits / max(1, n_block_words):.1f}%) | totals: {n_words:,} raw, {writer.n_written:,} dt | "
                f"{n_words / max(time.perf_counter() - t_start, 1e-9):,.0f} raw hits/s")
    if timing_correction is not None:
        timing_correction.warn_about_cells_without_correction(label)
    log(f"[{label}] DONE. {n_blocks_done:,} blocks, {n_words:,} raw hits read, {writer.n_written:,} dt hits written"
        f"{' (timing corrected)' if timing_correction is not None else ''}, took {time.perf_counter() - t_start:.1f}s")
    if writer.n_written == 0:
        raise RuntimeError(f"No dt hits found in {input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")
    return writer.n_written

### apply a testpulse timing calibration to an existing dt hits file
def dt_hits_timing_correction(dt_hits_file, dt_tp_corrections_file, corr_dt_hits_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    label = "dt hits -> corrected dt hits"
    root_utils.check_input_file(dt_hits_file)
    log(f"[{label}] START \"{dt_hits_file}\" + \"{dt_tp_corrections_file}\" -> \"{corr_dt_hits_file}\"")
    timing_correction = dt_calibration_utils.TimingCorrection.from_file(dt_tp_corrections_file, label=label)
    n_hits = 0
    with root_utils.TreeWriter(corr_dt_hits_file, root_utils.DT_HITS_TREE) as writer:
        for i_chunk, n_chunks, hits in root_utils.iterate_chunks(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size):
            n_hits += root_utils.length(hits)
            writer.write(timing_correction.apply(hits))
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(hits):,} hits corrected")
    timing_correction.warn_about_cells_without_correction(label)
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
    if alignment not in ("chamber", "sl"):
        raise ValueError(f"alignment has to be \"chamber\" or \"sl\", not \"{alignment}\"")
    if n_lines_to_skip is None:
        n_lines_to_skip = params._dumpfile_hits_to_skip
    root_utils.check_input_file(input_dumpfile)
    log(f"[{label}] START \"{input_dumpfile}\" -> \"{dt_tp_corrections_file}\"")
    log(f"[{label}] rel_thres={rel_thres}, alignment={alignment}, correct_for_offsets={correct_for_offsets}")
    t_start = time.perf_counter()
    histograms = dt_calibration_utils.TestpulseTimeHistograms()
    n_words, n_dt_hits = 0, 0
    hits_writer = root_utils.TreeWriter(dt_tp_hits_file, root_utils.DT_HITS_TREE) if dt_tp_hits_file is not None else None
    for dt_hits, n_block_words, i_block, n_blocks in dt_dumpfile_utils.iterate_dt_hits(
            input_dumpfile, n_lines_to_skip=n_lines_to_skip, block_n_lines=block_n_lines, n_proc=n_proc, all_cells=True, label=label):
        n_words += n_block_words
        n_block_dt_hits = 0
        if dt_hits is not None:
            n_block_dt_hits = root_utils.length(dt_hits)
            n_dt_hits += n_block_dt_hits
            ts_orbit = dt_calibration_utils.ts_in_orbit(dt_hits)
            histograms.add(dt_hits, ts_orbit)
            if hits_writer is not None:
                hits_writer.write(dt_hits | {"ts_orbit": ts_orbit.astype(params._ts_type)})
        log(f"[{label}] block {i_block:,} / {n_blocks:,}: {n_block_words:,} raw hits -> {n_block_dt_hits:,} dt hits | totals: {n_words:,} raw, {n_dt_hits:,} dt")
    if hits_writer is not None:
        hits_writer.close()
        log(f"[{label}] testpulse dt hits written to \"{dt_tp_hits_file}\"")
    if n_dt_hits == 0:
        raise RuntimeError(f"No dt hits found in {input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")

    calib = dt_calibration_utils.calibrate_cells(histograms, rel_thres=rel_thres, alignment=alignment, correct_for_offsets=correct_for_offsets, label=label)
    valid = calib["valid"] == 1
    log(f"[{label}] {int(valid.sum()):,} / {len(valid):,} cells with testpulse peak; corrections between {calib['ts_corr'][valid].min():.2f} and "
        f"{calib['ts_corr'][valid].max():.2f} ts units (rms {np.std(calib['ts_corr'][valid]):.2f}), mean uncertainty {calib['err_ts_corr'][valid].mean():.2f}")
    no_peak = [(int(calib["sl"][i]), int(calib["ly"][i]), int(calib["wi"][i])) for i in np.flatnonzero(~valid)]
    if len(no_peak) > 0:
        log(f"[{label}] cells without testpulse peak (ts_corr = 0): {no_peak[:20]}{' ...' if len(no_peak) > 20 else ''}")

    if dt_tp_corrections_file.endswith(".pcl"):
        root_utils.prepare_output_file(dt_tp_corrections_file)
        data_utils.store_pickle(data=dt_calibration_utils.tp_corrections_dict(calib), file=dt_tp_corrections_file, silent=True)
    else:
        summary = {"n_cells": len(valid), "n_valid": int(valid.sum()), "rel_thres": float(rel_thres), "chamber_alignment": int(alignment == "chamber"),
                   "correct_for_offsets": int(correct_for_offsets), "n_raw_hits": n_words, "n_dt_hits": n_dt_hits, "n_lines_to_skip": int(n_lines_to_skip)}
        root_utils.write_tree(dt_tp_corrections_file, calib, tree=root_utils.DEFAULT_TREE, summary=summary,
                              histograms=dt_calibration_utils.calibration_histograms(calib, histograms))
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
    histogram = dt_hit_utils.HitTimeDifferenceHistogram(n_bins, ts_max)
    n_hits = 0
    for i_chunk, n_chunks, hits in root_utils.iterate_chunks(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size):
        n_hits += root_utils.length(hits)
        histogram.fill(hits)
        log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(hits):,} hits, {int(histogram.entries):,} entries in histogram so far")
    hist_data = histogram.result()
    root_utils.prepare_output_file(hit_diff_hist_file)
    if hit_diff_hist_file.endswith(".pcl"):
        data_utils.store_pickle(data=hist_data, file=hit_diff_hist_file, silent=True)
    else:
        edges = hist_data["edges"]
        bins = {"edge_low": np.asarray(edges[:-1], dtype=np.float64), "edge_high": np.asarray(edges[1:], dtype=np.float64),
                "center": np.asarray(hist_data["centers"], dtype=np.float64)}
        bins |= {k: np.asarray(hist_data[k], dtype=np.float64) for k in ["hist", "err_hist", "err_hist_stat", "err_hist_down", "err_hist_up"]}
        summary = {k: np.float64(hist_data[k]) for k in ["entries", "underflow", "overflow"]} | {"n_hits": np.int64(n_hits)}
        root_utils.write_tree(hit_diff_hist_file, bins, summary=summary, histograms={"hit_diff_hist": (hist_data["hist"], edges)})
    log(f"[dt hits -> hit diff hist] DONE. {n_hits:,} hits, {int(histogram.entries):,} entries in histogram")
    return hist_data

### number of hits per cell and duration of the run (hits before the dead time cut)
# output .root: one row per cell + summary + TH2D "cell_counts" (x = wire, y = 4 * (sl - 1) + ly); .pcl: dict (old format)
def dt_hits_to_cell_counts(dt_hits_file, cell_counts_file, *, step_size=root_utils.DEFAULT_STEP_SIZE):
    root_utils.check_input_file(dt_hits_file)
    log(f"[dt hits -> cell counts] START \"{dt_hits_file}\" -> \"{cell_counts_file}\"")
    counts, ts_min, ts_max, n_hits = dt_hit_utils.count_hits_per_cell(dt_hits_file, step_size=step_size)
    if n_hits == 0:
        raise RuntimeError(f"No dt hits in {dt_hits_file}.")
    duration_seconds = float(ts_max - ts_min) * 0.78 * 1e-9
    counts_data = {"cell_counts": counts, "duration_seconds": duration_seconds, "ts_min": ts_min, "ts_max": ts_max}
    root_utils.prepare_output_file(cell_counts_file)
    if cell_counts_file.endswith(".pcl"):
        data_utils.store_pickle(data=counts_data, file=cell_counts_file, silent=True)
    else:
        rows = [(sl, ly, wi, count) for sl in counts for ly in counts[sl] for wi, count in counts[sl][ly].items()]
        cells = {key: np.array([row[i] for row in rows], dtype=np.uint8) for i, key in enumerate(["sl", "ly", "wi"])}
        cells["count"] = np.array([row[3] for row in rows], dtype=np.int64)
        summary = {"duration_seconds": np.float64(duration_seconds), "ts_min": np.float64(ts_min), "ts_max": np.float64(ts_max), "n_hits": np.int64(n_hits)}
        n_wires = int(cells["wi"].max()) + 1
        counts_2d = np.zeros((n_wires, 12))
        counts_2d[cells["wi"].astype(int), 4 * (cells["sl"].astype(int) - 1) + cells["ly"].astype(int)] = cells["count"]
        root_utils.write_tree(cell_counts_file, cells, summary=summary,
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
    totals = {"chunks": 0, "hits_in": 0, "hits_after_deadtime": 0, "n_patterns": 0}
    with root_utils.TreeWriter(sl_patterns_file) as writer, parallel_utils.optional_pool(n_proc if not verbose else 1) as pool:
        for i_chunk, n_chunks, hits in root_utils.iterate_chunks(dt_hits_file, root_utils.DT_HITS_TREE, step_size=step_size):
            t_step = time.perf_counter()
            totals["chunks"] += 1
            n_hits_in = root_utils.length(hits)
            if apply_dead_time:
                hits = dt_hit_utils.apply_dead_time(hits)
            n_hits = root_utils.length(hits)
            totals["hits_in"] += n_hits_in
            totals["hits_after_deadtime"] += n_hits
            if n_hits == 0:
                log(f"    chunk {i_chunk:,} / {n_chunks:,}: {n_hits_in:,} hits in, none left after dead time cut")
                continue
            sl_patterns = dt_pattern_utils.find_sl_patterns(hits, wide_ts_window=wide_ts_window, only_single_muon_patterns=simulation_only_muon_patterns,
                                                            pool=pool, verbose=verbose)
            n_patterns = root_utils.length(sl_patterns)
            totals["n_patterns"] += n_patterns
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {n_hits_in:,} hits in, {n_hits:,} after dead time cut ({100 * n_hits / max(1, n_hits_in):.1f}%) "
                f"-> {n_patterns:,} patterns ({time.perf_counter() - t_step:.2f}s)")
            if n_patterns > 0:
                writer.write(root_utils.set_chunk_id(sl_patterns, i_chunk))
            del hits, sl_patterns
            gc.collect()
    log(f"[dt hits -> sl patterns] DONE. {totals['chunks']:,} chunks, hits_in={totals['hits_in']:,}, "
        f"hits_after_deadtime={totals['hits_after_deadtime']:,}, patterns={totals['n_patterns']:,}")
    return totals

### fit a straight track to every pattern (all lateralities, the best one is kept)
# fit_vd: drift velocity as free fit parameter (default: fixed); rows are independent, so step_size and n_proc do not matter
def sl_patterns_to_sl_fits(sl_patterns_file, sl_fits_file, *, fit_vd=False, step_size=root_utils.DEFAULT_STEP_SIZE, n_proc=1, verbose=False):
    label = "sl patterns -> sl fits"
    root_utils.check_input_file(sl_patterns_file)
    log(f"[{label}] START \"{sl_patterns_file}\" -> \"{sl_fits_file}\" (fit_vd={fit_vd}, n_proc={n_proc})")
    n_patterns, n_fits = 0, 0
    with root_utils.TreeWriter(sl_fits_file) as writer:
        for i_chunk, n_chunks, patterns in root_utils.iterate_chunks(sl_patterns_file, step_size=step_size):
            t_step = time.perf_counter()
            n_patterns += root_utils.length(patterns)
            sl_fits = parallel_utils.run_rowwise(dt_sl_fit_utils.fit_sl_patterns, patterns, "patterns", {"fit_vd": fit_vd, "verbose": verbose},
                                                 n_proc if not verbose else 1)
            n_fits += root_utils.length(sl_fits)
            writer.write(sl_fits)
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(patterns):,} patterns -> {root_utils.length(sl_fits):,} fits "
                f"({time.perf_counter() - t_step:.2f}s)")
            del patterns, sl_fits
            gc.collect()
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
                          fit_vd=False, suffix=DEFAULT_SUPER_FIT_SUFFIX, step_size=root_utils.DEFAULT_STEP_SIZE, n_proc=1, verbose=False):
    root_utils.check_input_file(sl_fits_file)
    max_chi2ndf = np.inf if max_chi2 is None else max_chi2
    log(f"[sl fits -> super fits] START \"{sl_fits_file}\" -> \"{super_fits_file}\" "
        f"(max_chi2={max_chi2}, max_alpha={max_alpha:.4f} rad, fit_vd={fit_vd}, suffix=\"{suffix}\", n_proc={n_proc})")
    totals = {"n_fits": 0, "n_super_patterns": 0, "n_super_fits": 0}
    patterns_writer = root_utils.TreeWriter(super_patterns_file) if super_patterns_file is not None else None
    n_chunks = root_utils.n_blocks(sl_fits_file)
    with root_utils.TreeWriter(super_fits_file) as writer:
        for i_chunk, (chunk_id, first_row, sl_fits) in enumerate(root_utils.iterate_blocks(sl_fits_file, step_size=step_size), start=1):
            t_step = time.perf_counter()
            super_patterns = dt_super_fit_utils.build_phi_super_patterns(sl_fits, max_chi2ndf=max_chi2ndf, max_alpha=max_alpha)
            n_super_patterns = root_utils.length(super_patterns)
            n_super_fits = 0
            if n_super_patterns > 0:
                for sl in dt_chamber_utils.phi_superlayers():
                    super_patterns[f"row_sl{sl}"] += first_row
                if patterns_writer is not None:
                    patterns_writer.write(root_utils.set_chunk_id(dict(super_patterns), chunk_id))
                super_fits = parallel_utils.run_rowwise(dt_super_fit_utils.fit_super_sl_patterns, super_patterns, "super_patterns",
                                                        {"fit_vd": fit_vd, "suffix": suffix, "verbose": verbose}, n_proc if not verbose else 1)
                n_super_fits = root_utils.length(super_fits)
                writer.write(root_utils.set_chunk_id(super_fits, chunk_id))
            totals["n_fits"] += root_utils.length(sl_fits)
            totals["n_super_patterns"] += n_super_patterns
            totals["n_super_fits"] += n_super_fits
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(sl_fits):,} sl fits -> {n_super_patterns:,} super patterns -> {n_super_fits:,} super fits "
                f"({time.perf_counter() - t_step:.2f}s)")
            gc.collect()
    if patterns_writer is not None:
        patterns_writer.close()
    log(f"[sl fits -> super fits] DONE. sl_fits={totals['n_fits']:,}, super_patterns={totals['n_super_patterns']:,}, super_fits={totals['n_super_fits']:,}")
    return totals

### the sl fits of a chunk_id; checks that the super fits of this chunk point into them
def _sl_fits_of_chunk(sl_fits_blocks, chunk_id, super_fits, sl_fits_file, super_fits_file):
    for fits_chunk_id, first_row, sl_fits in sl_fits_blocks:
        if fits_chunk_id == chunk_id:
            break
    else:
        raise RuntimeError(f"No sl fits with chunk_id {chunk_id} in {sl_fits_file}. Is this the file the super fits were made from?")
    for sl in dt_chamber_utils.phi_superlayers():
        rows = super_fits[f"row_sl{sl}"] - first_row
        if np.any(rows < 0) or np.any(rows >= root_utils.length(sl_fits)) or np.any(sl_fits["sl"][rows] != sl):
            raise RuntimeError(f"The sl fit rows stored in {super_fits_file} do not match {sl_fits_file}. Is this the file the super fits were made from?")
    return first_row, sl_fits

### combine the super fits (phi view) with the sl fits of the theta superlayer to dt muons, per chunk_id
# super_fits_file: super fits after quality cuts; sl_fits_file: the cut sl fits the super fits were made from
# branches "super_fit_row" (row in super_fits_file) and "sl<n>_fit_row" (rows in sl_fits_file): the fits of a muon
def super_fits_to_dt_muons(super_fits_file, sl_fits_file, dt_muons_file, *, suffix=DEFAULT_SUPER_FIT_SUFFIX, tgroup_tolerance=None,
                           step_size=root_utils.DEFAULT_STEP_SIZE, verbose=False):
    label = "super fits + theta sl fits -> dt muons"
    root_utils.check_input_file(super_fits_file)
    root_utils.check_input_file(sl_fits_file)
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    log(f"[{label}] START \"{super_fits_file}\" + \"{sl_fits_file}\" -> \"{dt_muons_file}\" (tgroup_tolerance={tgroup_tolerance:.2f} ts units)")
    theta_sl = dt_chamber_utils.theta_superlayer()
    sl_fits_blocks = root_utils.iterate_blocks(sl_fits_file, step_size=step_size)
    totals = {"n_super_fits": 0, "n_theta_fits": 0, "n_muons": 0, "n_ambiguous": 0}
    ts_min, ts_max = None, None
    n_chunks = root_utils.n_blocks(super_fits_file)
    with root_utils.TreeWriter(dt_muons_file) as writer:
        for i_chunk, (chunk_id, first_super_row, super_fits) in enumerate(root_utils.iterate_blocks(super_fits_file, step_size=step_size), start=1):
            if "t0" + suffix not in super_fits:
                raise KeyError(f"No super fit results with suffix \"{suffix}\" in {super_fits_file}.")
            if np.any(super_fits["impossible" + suffix] != 0):
                raise RuntimeError(f"{super_fits_file} contains super fits flagged \"impossible\". Apply cuts first "
                                   f"(at least \"impossible{suffix},==,0\"), see scripts/apply_cuts.py.")
            first_fit_row, sl_fits = _sl_fits_of_chunk(sl_fits_blocks, chunk_id, super_fits, sl_fits_file, super_fits_file)
            theta_rows = np.flatnonzero(sl_fits["sl"] == theta_sl)
            theta_fits = {k: v[theta_rows] for k, v in sl_fits.items()}
            dt_muons = dt_muon_reco_utils.reco_muons_from_super_fits(super_fits, theta_fits, suffix=suffix, tgroup_tolerance=tgroup_tolerance, verbose=verbose)
            n_muons = root_utils.length(dt_muons)
            if n_muons > 0:
                super_idx, theta_idx = dt_muons.pop("super_fit_idx"), dt_muons.pop("theta_fit_idx")
                dt_muons["super_fit_row"] = (super_idx + first_super_row).astype(np.int64)
                for sl in dt_chamber_utils.phi_superlayers():
                    dt_muons[f"sl{sl}_fit_row"] = super_fits[f"row_sl{sl}"][super_idx].astype(np.int64)
                dt_muons[f"sl{theta_sl}_fit_row"] = (theta_rows[theta_idx] + first_fit_row).astype(np.int64)
                dt_muons = data_utils.sort_by_key(data=dt_muons, sort_key="ts", silent=True)
                writer.write(root_utils.set_chunk_id(dt_muons, chunk_id))
                totals["n_ambiguous"] += int(np.sum(dt_muons["n_theta_candidates"] > 1))
                ts_min = dt_muons["ts"].min() if ts_min is None else min(ts_min, dt_muons["ts"].min())
                ts_max = dt_muons["ts"].max() if ts_max is None else max(ts_max, dt_muons["ts"].max())
            totals["n_super_fits"] += root_utils.length(super_fits)
            totals["n_theta_fits"] += len(theta_rows)
            totals["n_muons"] += n_muons
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(super_fits):,} super fits, {len(theta_rows):,} theta sl fits -> {n_muons:,} dt muons")
    if totals["n_muons"] == 0:
        log(f"[{label}] WARNING: no dt muons reconstructed, no output file was written.")
    duration = (ts_max - ts_min) * TS_UNIT_SECONDS if ts_min is not None else 0
    rate = f", rate {totals['n_muons'] / duration:.2f} Hz over {duration:.2f} s" if duration > 0 else ""
    log(f"[{label}] DONE. super_fits={totals['n_super_fits']:,}, theta_sl_fits={totals['n_theta_fits']:,}, "
        f"dt_muons={totals['n_muons']:,} ({totals['n_ambiguous']:,} with more than one theta candidate){rate}")
    return totals

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
    n_in, n_after = 0, [0 for _ in cuts]
    with root_utils.TreeWriter(output_file, tree) as writer:
        for i_chunk, n_chunks, rows in root_utils.iterate_chunks(input_file, tree, step_size=step_size):
            n_rows = root_utils.length(rows)
            n_in += n_rows
            cut_utils.check_cut_keys(cuts, rows, input_file)
            for i, cut in enumerate(cuts):
                rows = data_utils.cut_data(data=rows, conditions=[cut], silent=True)
                n_after[i] += root_utils.length(rows)
            writer.write(rows)
            log(f"    chunk {i_chunk:,} / {n_chunks:,}: {root_utils.length(rows):,} of {n_rows:,} rows pass the cuts")
    log(f"[apply cuts] cut flow w.r.t. the {n_in:,} input rows:")
    for cut, n in zip(cuts, n_after):
        log(f"    {cut[0]} {cut[1]} {cut[2]}: {n:,} / {n_in:,} = {n / max(1, n_in):.4f}")
    if n_after[-1] == 0:
        log(f"[apply cuts] WARNING: no rows pass the cuts, no output file was written.")
    log(f"[apply cuts] DONE.")
    return n_in, n_after

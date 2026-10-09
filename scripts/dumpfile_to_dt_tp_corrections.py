#################################################################
### testpulse run: raw dumpfile (.txt) -> timing calibration of all dt cells (ROOT file)
# For a dumpfile recorded with simultaneous testpulses on all channels:
#   - dt hits of all cells of the chamber (masked / dead cells included, no dead time cut)
#   - per cell: histogram of the time inside the orbit (ts_orbit = tdc + 32 * bx, bins of 1 ts unit), position of
#     the first peak (weighted mean); later peaks are ringing of the testpulse circuit and are ignored
#   - the known testpulse delays per frontend connector (params._tp_time_offset) are subtracted
#   - correction per cell: ts_corr = <mean testpulse time of the chamber (or SL)> - <testpulse time of the cell>,
#     the mean is taken over the cells which are not masked / dead,
#     applied with a plus sign: ts_corrected = ts + ts_corr
#
# output (.root): one row per cell, see OUTPUT_FILES.md. Output ending .pcl: only the correction dict of the old script.
#
# Use the result with
#   python scripts/dumpfile_to_dt_hits.py ... --dt_tp_corrections_file <this output>
#   python scripts/run_dt_pipeline.py ... --dt_tp_corrections_file <this output>
#
# examples:
#   python scripts/dumpfile_to_dt_tp_corrections.py --input_dumpfile tp_run.txt --dt_tp_corrections_file calib/tp_corrections.root
#   python scripts/dumpfile_to_dt_tp_corrections.py --input_dumpfile tp_run.txt --dt_tp_corrections_file calib/tp_corrections.root \
#          --dt_tp_hits_file calib/tp_dt_hits.root --alignment sl
#################################################################

import argparse
import time
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import data_utils, dt_calibration_utils, dt_dumpfile_utils, dt_hit_utils, root_utils, timestamp_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calculate the timing calibration of all dt cells from a testpulse dumpfile.")
    parser.add_argument("--input_dumpfile", type=str, required=True, help="input file path: raw dumpfile (.txt) recorded with testpulses")
    parser.add_argument("--dt_tp_corrections_file", type=str, required=True,
                        help="output file path: timing calibration (.root; .pcl for the format of the old script)")
    parser.add_argument("--dt_tp_hits_file", type=str, default=None,
                        help="optional output file path: the testpulse dt hits (.root, tree \"dt_hits\", with branch ts_orbit) for inspection")
    parser.add_argument("--alignment", type=str, choices=["chamber", "sl"], default="chamber",
                        help="align all cells to the mean of the full chamber (default, needs aligned testpulses of all SLs) "
                             "or each SL to its own mean (a time offset between the SLs remains)")
    parser.add_argument("--rel_thres", type=float, default=0.2,
                        help="threshold for the peak search, relative to the highest bin of the cell histogram")
    parser.add_argument("--no_offset_correction", action="store_true",
                        help="do not subtract the known testpulse delays per frontend connector (params._tp_time_offset)")
    parser.add_argument("--n_lines_to_skip", type=int, default=params._dumpfile_hits_to_skip,
                        help=f"number of lines at the start of the dumpfile to ignore (default: params._dumpfile_hits_to_skip = {params._dumpfile_hits_to_skip})")
    parser.add_argument("--block_lines", type=int, default=500_000, help="number of dumpfile lines processed at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "testpulse dumpfile -> dt tp corrections"
    correct_for_offsets = not args.no_offset_correction
    root_utils.check_input_file(args.input_dumpfile)
    log(f"[{label}] START \"{args.input_dumpfile}\" -> \"{args.dt_tp_corrections_file}\"")
    log(f"[{label}] rel_thres={args.rel_thres}, alignment={args.alignment}, correct_for_offsets={correct_for_offsets}")
    t_start = time.perf_counter()

    ### 1. histograms of the testpulse hit times of every cell (all cells, also masked / dead ones)
    tp_histograms = dt_calibration_utils.empty_testpulse_histograms()
    hits_file = None
    if args.dt_tp_hits_file is not None:
        hits_file = root_utils.create_file(args.dt_tp_hits_file)
    dumpfile, n_blocks = dt_dumpfile_utils.open_dumpfile(args.input_dumpfile, args.n_lines_to_skip, args.block_lines, label)
    overflow_state = {"oc_overflow": 0, "last_oc": None}  # orbit counter overflows, counted from the start of the file
    n_words_total, n_dt_hits_total = 0, 0
    for i_block in range(n_blocks):
        lines = dt_dumpfile_utils.read_lines(dumpfile, args.block_lines)
        hits = data_utils.import_raw_lines(lines, silent=True)
        n_words = len(lines)
        dt_hits = dt_hit_utils.extract_dt_hits(hits, all_cells=True, overflow_state=overflow_state)
        if root_utils.length(dt_hits) == 0:
            dt_hits = None
        n_dt_hits = 0
        if dt_hits is not None:
            n_dt_hits = root_utils.length(dt_hits)
            ts_orbit = timestamp_utils.add_timestamp_this_orbit(hits=dt_hits, silent=True)["ts_orbit"]
            dt_calibration_utils.fill_testpulse_histograms(tp_histograms, dt_hits, ts_orbit)
            if hits_file is not None:
                dt_hits["ts_orbit"] = ts_orbit
                root_utils.write_rows(hits_file, root_utils.DT_HITS_TREE, dt_hits)
        n_words_total += n_words
        n_dt_hits_total += n_dt_hits
        log(f"[{label}] block {i_block + 1:,} / {n_blocks:,}: {n_words:,} raw hits -> {n_dt_hits:,} dt hits | "
            f"totals: {n_words_total:,} raw, {n_dt_hits_total:,} dt")
    dumpfile.close()
    if overflow_state["oc_overflow"] > 0:
        log(f"[{label}] orbit counter overflows found: {overflow_state['oc_overflow']:,}")
    if hits_file is not None:
        hits_file.close()
        log(f"[{label}] testpulse dt hits written to \"{args.dt_tp_hits_file}\"")
    if n_dt_hits_total == 0:
        raise RuntimeError(f"No dt hits found in {args.input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")

    ### 2. calibration of every cell
    calib = dt_calibration_utils.calibrate_cells(tp_histograms, rel_thres=args.rel_thres, alignment=args.alignment,
                                                 correct_for_offsets=correct_for_offsets, label=label)
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

    ### 3. output file
    if args.dt_tp_corrections_file.endswith(".pcl"):
        data_utils.store_pickle(data=dt_calibration_utils.tp_corrections_dict(calib), file=args.dt_tp_corrections_file, silent=True)
    else:
        output_file = root_utils.create_file(args.dt_tp_corrections_file)
        root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, calib)
        summary = {"n_cells": len(valid), "n_valid": n_valid, "rel_thres": float(args.rel_thres), "chamber_alignment": int(args.alignment == "chamber"),
                   "correct_for_offsets": int(correct_for_offsets), "n_raw_hits": n_words_total, "n_dt_hits": n_dt_hits_total,
                   "n_lines_to_skip": int(args.n_lines_to_skip)}
        root_utils.write_summary(output_file, summary)
        histograms = dt_calibration_utils.calibration_histograms(calib, tp_histograms)
        for name in histograms:
            root_utils.write_histogram(output_file, name, histograms[name])
        output_file.close()
    log(f"[{label}] DONE. {len(valid):,} cells written to \"{args.dt_tp_corrections_file}\", took {time.perf_counter() - t_start:.1f}s")

if __name__ == "__main__":
    main()
    log("###### Done.")

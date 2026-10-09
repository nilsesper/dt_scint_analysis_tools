#################################################################
### raw dumpfile (.txt) -> dt hits (ROOT file, tree "dt_hits")
# decodes the data words, adds timestamps, keeps the dt channels, adds the chamber mapping (sl, ly, wi)
# and removes masked / dead wires (see params.py). No dead time cut is applied here.
# Optionally the testpulse timing calibration is applied to every hit (--dt_tp_corrections_file, made by
# dumpfile_to_dt_tp_corrections.py): ts -> ts + ts_corr(sl, ly, wi).
# The dumpfile is read in blocks of --block_lines lines, one after the other.
#
# examples:
#   python scripts/dumpfile_to_dt_hits.py --input_dumpfile run.txt --dt_hits_file out/run_dt_hits.root
#   python scripts/dumpfile_to_dt_hits.py --input_dumpfile run.txt --dt_hits_file out/run_dt_hits.root \
#          --dt_tp_corrections_file calib/tp_corrections.root
#################################################################

import argparse
import time

from analysis_tools.utils import data_utils, dt_calibration_utils, dt_dumpfile_utils, dt_hit_utils, root_utils, timestamp_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert a raw dumpfile into a ROOT file of dt hits.")
    parser.add_argument("--input_dumpfile", type=str, required=True, help="input file path: raw dumpfile (.txt)")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="output file path: dt hits (.root)")
    parser.add_argument("--n_lines_to_skip", type=int, default=999,
                        help="number of lines at the start of the dumpfile to ignore (old hits still in the readout buffer)")
    parser.add_argument("--dt_tp_corrections_file", type=str, default=None,
                        help="optional: testpulse timing calibration (.root from dumpfile_to_dt_tp_corrections.py, or old .pcl); "
                             "if given, the hit timestamps are corrected")
    parser.add_argument("--block_lines", type=int, default=500_000, help="number of dumpfile lines processed at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dumpfile -> dt hits"
    root_utils.check_input_file(args.input_dumpfile)
    log(f"[{label}] START \"{args.input_dumpfile}\" -> \"{args.dt_hits_file}\"")

    ### timing calibration
    dt_tp_corrections = None
    if args.dt_tp_corrections_file is not None:
        dt_tp_corrections = dt_calibration_utils.read_tp_corrections(args.dt_tp_corrections_file, label)
        log(f"[{label}] testpulse timing corrections from \"{args.dt_tp_corrections_file}\" are applied to the hits")
    else:
        log(f"[{label}] no testpulse timing corrections given, timestamps are not corrected")
    cells_without_correction = set()

    ### decode the dumpfile block by block
    t_start = time.perf_counter()
    dumpfile, n_blocks = dt_dumpfile_utils.open_dumpfile(args.input_dumpfile, args.n_lines_to_skip, args.block_lines, label)
    overflow_state = {"oc_overflow": 0, "last_oc": None}  # orbit counter overflows, counted from the start of the file
    output_file = root_utils.create_file(args.dt_hits_file)
    n_words_total, n_dt_hits_total = 0, 0
    for i_block in range(n_blocks):
        lines = dt_dumpfile_utils.read_lines(dumpfile, args.block_lines)
        hits = data_utils.import_raw_lines(lines, silent=True)
        n_words = len(lines)
        dt_hits = dt_hit_utils.extract_dt_hits(hits, all_cells=False, overflow_state=overflow_state)
        if root_utils.length(dt_hits) == 0:
            dt_hits = None
        n_dt_hits = 0
        if dt_hits is not None:
            if dt_tp_corrections is not None:
                dt_hits = dt_calibration_utils.apply_timing_calibration(dt_hits, dt_tp_corrections=dt_tp_corrections, cells_without_correction=cells_without_correction)
                dt_hits = timestamp_utils.sort_by_timestamp(hits=dt_hits, silent=True)  # the corrections change the order a little
            root_utils.write_rows(output_file, root_utils.DT_HITS_TREE, dt_hits)
            n_dt_hits = root_utils.length(dt_hits)
        n_words_total += n_words
        n_dt_hits_total += n_dt_hits
        log(f"[{label}] block {i_block + 1:,} / {n_blocks:,}: {n_words:,} raw hits -> {n_dt_hits:,} dt hits "
            f"({100 * n_dt_hits / max(1, n_words):.1f}%) | totals: {n_words_total:,} raw, {n_dt_hits_total:,} dt | "
            f"{n_words_total / max(time.perf_counter() - t_start, 1e-9):,.0f} raw hits/s")
    dumpfile.close()
    output_file.close()

    if overflow_state["oc_overflow"] > 0:
        log(f"[{label}] orbit counter overflows found: {overflow_state['oc_overflow']:,}")
    if dt_tp_corrections is not None:
        dt_calibration_utils.warn_about_cells_without_correction(cells_without_correction, label)
    log(f"[{label}] DONE. {n_blocks:,} blocks, {n_words_total:,} raw hits read, {n_dt_hits_total:,} dt hits written, "
        f"took {time.perf_counter() - t_start:.1f}s")
    if n_dt_hits_total == 0:
        raise RuntimeError(f"No dt hits found in {args.input_dumpfile} -- check n_lines_to_skip / bit masks / channel mapping in params.py.")

if __name__ == "__main__":
    main()
    log("###### Done.")

#################################################################
### dt hits -> timing corrected dt hits
# adds the time offset per wire from a testpulse run to the hit timestamps (ts, err_ts, oc, bx, tdc)
# the corrected file has the same format as the input and is used in its place by the following stages
# (the correction can also be applied directly in dumpfile_to_dt_hits.py with --dt_tp_corrections_file)
#################################################################

import argparse

from analysis_tools.utils import dt_calibration_utils, root_utils, timestamp_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply the testpulse timing calibration to dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--dt_tp_corrections_file", type=str, required=True,
                        help="input file path: timing corrections from a testpulse run (.root from dumpfile_to_dt_tp_corrections.py, or old .pcl)")
    parser.add_argument("--corr_dt_hits_file", type=str, required=True, help="output file path: timing corrected dt hits (.root)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of hits read at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dt hits -> corrected dt hits"
    root_utils.check_input_file(args.dt_hits_file)
    log(f"[{label}] START \"{args.dt_hits_file}\" + \"{args.dt_tp_corrections_file}\" -> \"{args.corr_dt_hits_file}\"")
    dt_tp_corrections = dt_calibration_utils.read_tp_corrections(args.dt_tp_corrections_file, label)
    cells_without_correction = set()

    n_hits = root_utils.number_of_rows(args.dt_hits_file, root_utils.DT_HITS_TREE)
    n_chunks = (n_hits + args.chunk_size - 1) // args.chunk_size
    output_file = root_utils.create_file(args.corr_dt_hits_file)
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_hits)
        hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        corrected_hits = dt_calibration_utils.apply_timing_calibration(hits, dt_tp_corrections=dt_tp_corrections, cells_without_correction=cells_without_correction)
        corrected_hits = timestamp_utils.sort_by_timestamp(hits=corrected_hits, silent=True)  # the corrections change the order a little
        root_utils.write_rows(output_file, root_utils.DT_HITS_TREE, corrected_hits)
        log(f"    chunk {i_chunk + 1:,} / {n_chunks:,}: {root_utils.length(hits):,} hits corrected")
    output_file.close()
    dt_calibration_utils.warn_about_cells_without_correction(cells_without_correction, label)
    log(f"[{label}] DONE. {n_hits:,} hits")

if __name__ == "__main__":
    main()
    log("###### Done.")

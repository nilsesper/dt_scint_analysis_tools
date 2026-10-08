#################################################################
### raw dumpfile (.txt) -> dt hits (ROOT file, tree "dt_hits")
# decodes the data words, adds timestamps, keeps the dt channels, adds the chamber mapping (sl, ly, wi)
# and removes masked / dead wires (see params.py). No dead time cut is applied here.
# Optionally the testpulse timing calibration is applied to every hit (--dt_tp_corrections_file, made by
# dumpfile_to_dt_tp_corrections.py): ts -> ts + ts_corr(sl, ly, wi).
#
# examples:
#   python scripts/dt_root/dumpfile_to_dt_hits.py --input_dumpfile run.txt --dt_hits_file out/run_dt_hits.root --n_proc 4
#   python scripts/dt_root/dumpfile_to_dt_hits.py --input_dumpfile run.txt --dt_hits_file out/run_dt_hits.root \
#          --dt_tp_corrections_file calib/tp_corrections.root
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a raw dumpfile into a ROOT file of dt hits.")
    parser.add_argument("--input_dumpfile", type=str, required=True, help="input file path: raw dumpfile (.txt)")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="output file path: dt hits (.root)")
    parser.add_argument("--n_lines_to_skip", type=int, default=999,
                        help="number of lines at the start of the dumpfile to ignore (old hits still in the readout buffer)")
    parser.add_argument("--dt_tp_corrections_file", type=str, default=None,
                        help="optional: testpulse timing calibration (.root from dumpfile_to_dt_tp_corrections.py, or old .pcl); "
                             "if given, the hit timestamps are corrected")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes to run in parallel (does not change the result)")
    parser.add_argument("--block_lines", type=int, default=500_000, help="number of dumpfile lines processed at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.convert_dumpfile_to_dt_hits(
        args.input_dumpfile, args.dt_hits_file, n_lines_to_skip=args.n_lines_to_skip, block_n_lines=args.block_lines, n_proc=args.n_proc,
        dt_tp_corrections_file=args.dt_tp_corrections_file,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

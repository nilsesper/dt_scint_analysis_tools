#################################################################
### dt hits -> number of hits per cell and duration of the run
# before any dead time cut
# output: .root (histogram object "cell_counts" (TH2D, x = wire, y = 4 * (sl - 1) + ly) to draw directly in ROOT,
#                tree "tree" with one row per cell, tree "summary" with duration_seconds / ts_min / ts_max)
#         or .pcl (format of the old pipeline), chosen by the file ending
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def main(argv=None):
    parser = argparse.ArgumentParser(description="Count the hits per cell and determine the duration of the run.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--cell_counts_file", type=str, required=True, help="output file path: cell counts (.root or .pcl)")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.dt_hits_to_cell_counts(args.dt_hits_file, args.cell_counts_file, step_size=_step_size(args.step_size))

if __name__ == "__main__":
    main()
    log("###### Done.")

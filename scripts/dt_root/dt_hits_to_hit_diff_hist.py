#################################################################
### dt hits -> histogram of the time difference between consecutive hits of the same cell
# for every cell separately the hits are ordered in time and the differences between neighbours are filled;
# the histograms of all cells are summed. Uses the hits before any dead time cut.
# output .root: - histogram object "hit_diff_hist" (TH1D): draw this one in ROOT / TBrowser
#               - tree "tree" with one row per bin (branches edge_low, edge_high, center, hist = bin content, errors);
#                 note: drawing the branch "hist" of this tree in ROOT shows how often each bin content occurs,
#                 not the time difference distribution. Use the TH1D, or tree->Draw("hist:center")
#               - tree "summary" with entries / underflow / overflow
# output .pcl: format of the old pipeline
# default range: 0 .. 5000 timestamp units (3.9 us) in 5000 bins, larger differences are counted as overflow;
# change it with --ts_max and --n_bins
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def main(argv=None):
    parser = argparse.ArgumentParser(description="Histogram the time difference between consecutive hits of the same cell.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--hit_diff_hist_file", type=str, required=True, help="output file path: histogram (.root or .pcl)")
    parser.add_argument("--n_bins", type=int, default=5000, help="number of bins")
    parser.add_argument("--ts_max", type=float, default=5000, help="upper edge of the histogram in timestamp units (lower edge is 0)")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.dt_hits_to_hit_diff_hist(
        args.dt_hits_file, args.hit_diff_hist_file, step_size=_step_size(args.step_size), n_bins=args.n_bins, ts_max=args.ts_max,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

#################################################################
### simulation: add secondary hits to dt hits
# with the given probability a hit gets a second hit in the same cell, uniformly distributed in the given time
# window after the first one
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, dt_utils, muon_utils, plot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Add secondary hits to simulated dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--dt_hits_file_with_secondaries", type=str, required=True, help="output file path: dt hits with secondary hits (.root)")
    parser.add_argument("--window", type=str, default="0,500", help="time window after the first hit as \"start,stop\" in timestamp units")
    parser.add_argument("--probability", type=float, default=0.05, help="probability of a hit to get a secondary hit")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    root_utils.check_input_file(args.dt_hits_file)
    window = [float(x) for x in args.window.split(",")]
    if len(window) != 2 or window[1] <= window[0]:
        parser.error("--window has to be \"start,stop\" with stop > start")
    dt_hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE)
    n_hits = root_utils.length(dt_hits)
    log(f"###### Adding secondary hits with probability {args.probability:g} in the window {window} TU to {n_hits:,} hits...")
    dt_hits = dt_utils.add_secondary_hits(hits=dt_hits, secondary_hit_window=window, secondary_hit_probability=args.probability)
    n_hits_new = root_utils.length(dt_hits)
    root_utils.write_tree(args.dt_hits_file_with_secondaries, dt_hits, tree=root_utils.DT_HITS_TREE)
    log(f"###### {n_hits:,} hits before, {n_hits_new:,} after: added {n_hits_new - n_hits:,} secondary hits, stored in {args.dt_hits_file_with_secondaries}")

if __name__ == "__main__":
    main()
    log("###### Done.")

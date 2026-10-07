#################################################################
### simulation: add random noise hits to dt hits
# every cell gets poisson distributed noise hits with the given rate over the time range of the input hits
# noise hits have muon_id = 0
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, dt_utils, muon_utils, plot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Add random noise hits to simulated dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--dt_hits_file_with_noise", type=str, required=True, help="output file path: dt hits with added noise (.root)")
    parser.add_argument("--noise_rate_hz", type=float, default=15, help="noise rate per cell in Hz")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    root_utils.check_input_file(args.dt_hits_file)
    dt_hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE)
    n_hits = root_utils.length(dt_hits)
    log(f"###### Adding dt noise of {args.noise_rate_hz:g} Hz per cell to {n_hits} hits...")
    t_start = np.amin(dt_hits["ts"]) - params._dt_max_drift_time
    t_stop = np.amax(dt_hits["ts"]) + params._dt_max_drift_time
    dt_hits = dt_utils.add_noise(hits=dt_hits, ts_range=[t_start, t_stop], ref_cell_noise_rate=args.noise_rate_hz)
    n_hits_new = root_utils.length(dt_hits)
    root_utils.write_tree(args.dt_hits_file_with_noise, dt_hits, tree=root_utils.DT_HITS_TREE)
    log(f"###### {n_hits} hits before, {n_hits_new} after: added {n_hits_new - n_hits} noise hits, stored in {args.dt_hits_file_with_noise}")

if __name__ == "__main__":
    main()
    log("###### Done.")

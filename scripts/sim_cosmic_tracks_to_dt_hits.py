#################################################################
### simulation: cosmic muon tracks -> dt hits (ROOT file, tree "dt_hits")
# propagates the muons through the chamber and creates one hit per crossed cell (with params._dt_cell_efficiency)
# optional: gaussian noise on the drift times, constant random time offset per wire (mis-calibration)
# the output has the format of dumpfile_to_dt_hits.py and can be given to dt_hits_to_sl_patterns.py
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, dt_utils, muon_utils, plot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Create simulated dt hits from cosmic muon tracks.")
    parser.add_argument("--cosmic_muons_file", type=str, required=True, help="input file path: cosmic muon tracks (.root)")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="output file path: simulated dt hits (.root)")
    parser.add_argument("--ts_noise_amplitude", type=float, default=0, help="sigma of gaussian noise on the hit timestamps in timestamp units")
    parser.add_argument("--sys_miscalib_ampl", type=float, default=0, help="sigma of a constant random time offset per wire in timestamp units")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    root_utils.check_input_file(args.cosmic_muons_file)
    cosmic_muons = root_utils.read_tree(args.cosmic_muons_file)
    n_muons = root_utils.length(cosmic_muons)
    log(f"###### Propagating {n_muons:,} cosmic muons through the dt chamber...")
    dt_hits = dt_utils.hits_from_muons(muons=cosmic_muons, noise_ampl=args.ts_noise_amplitude, sys_miscalib_ampl=args.sys_miscalib_ampl)
    n_dt_hits = root_utils.length(dt_hits)
    if n_dt_hits == 0:
        raise RuntimeError("No dt hits were created.")
    root_utils.write_tree(args.dt_hits_file, dt_hits, tree=root_utils.DT_HITS_TREE)
    n_muons_with_hits = len(np.unique(dt_hits["muon_id"]))
    ts_min, ts_max = int(np.amin(dt_hits["ts"])), int(np.amax(dt_hits["ts"]))
    log(f"###### Stored {n_dt_hits:,} dt hits of {n_muons_with_hits:,} muons in {args.dt_hits_file} (ts range {ts_min} .. {ts_max})")

if __name__ == "__main__":
    main()
    log("###### Done.")

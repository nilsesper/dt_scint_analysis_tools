#################################################################
### dt hits -> sl patterns (4-layer hit patterns inside one superlayer)
# applies the individual dead time cut (params._dt_ts_individual_dead_time) before the pattern search
# hits are processed in chunks of --step_size; every pattern is stamped with the "chunk_id" of its chunk
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def main(argv=None):
    parser = argparse.ArgumentParser(description="Find superlayer hit patterns in dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--sl_patterns_file", type=str, required=True, help="output file path: sl patterns (.root)")
    parser.add_argument("--ts_window", type=str, choices=["free_vd", "fixed_vd"], default="free_vd",
                        help="max time difference between the hits of a pattern: \"free_vd\" = wide window for fits with free "
                             "drift velocity (params._dt_sl_patterns_ts_window_fit_vd), \"fixed_vd\" = params._dt_sl_patterns_ts_window")
    parser.add_argument("--no_dead_time", action="store_true", help="do not apply the dead time cut")
    parser.add_argument("--simulation_only_muon_patterns", action="store_true",
                        help="simulation only: keep only patterns whose four hits come from the same simulated muon")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes to run in parallel (does not change the result)")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--verbose", action="store_true", help="print info for every hit")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.dt_hits_to_sl_patterns(
        args.dt_hits_file, args.sl_patterns_file, step_size=_step_size(args.step_size), apply_dead_time=not args.no_dead_time,
        wide_ts_window=(args.ts_window == "free_vd"), simulation_only_muon_patterns=args.simulation_only_muon_patterns,
        n_proc=args.n_proc, verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

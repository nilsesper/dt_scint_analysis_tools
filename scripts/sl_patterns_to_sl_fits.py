#################################################################
### sl patterns -> sl fits (track segment fit per pattern, best laterality is kept)
# default: fixed drift velocity. The output keeps all pattern branches and adds the fit results.
#################################################################

import argparse

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit a track segment to every superlayer pattern.")
    parser.add_argument("--sl_patterns_file", type=str, required=True, help="input file path: sl patterns (.root)")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="output file path: sl fits (.root)")
    parser.add_argument("--fit_vd", action="store_true", help="fit the drift velocity as free parameter (default: fixed)")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes to run in parallel")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--verbose", action="store_true", help="print info for every fit (switches off parallel processing)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.sl_patterns_to_sl_fits(
        args.sl_patterns_file, args.sl_fits_file, fit_vd=args.fit_vd, step_size=root_utils.parse_step_size(args.step_size),
        n_proc=args.n_proc, verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

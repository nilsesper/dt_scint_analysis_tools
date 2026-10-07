#################################################################
### cut sl fits -> super fits
# matches the sl fits of the two phi superlayers to 8-hit "super patterns" and fits them together
# (default: fixed drift velocity, --free_vd makes it a fit parameter)
# input: sl fits after quality cuts (apply_cuts.py), at least "impossible,==,0"
# output branches "row_sl1", "row_sl3": rows of the two combined sl fits in the input file
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine the fits of the two phi superlayers and fit the 8 hits together.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: sl fits after cuts (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="output file path: super fits (.root)")
    parser.add_argument("--super_patterns_file", type=str, default=None,
                        help="optional output file path: matched super patterns before the fit (.root)")
    parser.add_argument("--max_chi2", type=float, default=None,
                        help="additional max chi2/ndf of sl fits which may be combined (default: no additional cut)")
    parser.add_argument("--max_alpha_deg", type=float, default=60, help="max |track angle| in degrees of sl fits which may be combined")
    parser.add_argument("--free_vd", action="store_true", help="fit the drift velocity as free parameter in the super fit (default: fixed)")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes to run in parallel")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--verbose", action="store_true", help="print info for every fit (switches off parallel processing)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.sl_fits_to_super_fits(
        args.sl_fits_file, args.super_fits_file, super_patterns_file=args.super_patterns_file, max_chi2=args.max_chi2,
        max_alpha=np.deg2rad(args.max_alpha_deg), fit_vd=args.free_vd, suffix=args.suffix,
        step_size=_step_size(args.step_size), n_proc=args.n_proc, verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

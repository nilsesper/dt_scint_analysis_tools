#################################################################
### cut super fits (phi superlayers) + cut sl fits (theta superlayer) -> dt muons
# every super fit is combined with the theta sl fit closest in time (within --tgroup_tolerance)
#   phi view of the muon:   position and slope of the super fit
#   theta view of the muon: position and slope of the theta sl fit
# inputs:
#   --super_fits_file: super fits after quality cuts (apply_cuts.py), at least "impossible<suffix>,==,0"
#   --sl_fits_file:    the cut sl fits file the super fits were made from
# output branches "super_fit_row" (row in the super fits file) and "sl1_fit_row", "sl2_fit_row", "sl3_fit_row"
# (rows in the sl fits file) point to the fits of each muon; "n_theta_candidates" > 1 marks ambiguous matches
#################################################################

import argparse

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine phi super fits and theta sl fits to dt muons.")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: super fits after cuts (.root)")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the super fits were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="output file path: dt muons (.root)")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--tgroup_tolerance", type=float, default=None,
                        help="max |t0 difference| between super fit and theta sl fit in timestamp units (default: params._muon_tgroup_tolerance)")
    parser.add_argument("--verbose", action="store_true", help="print info for every muon")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.super_fits_to_dt_muons(
        args.super_fits_file, args.sl_fits_file, args.dt_muons_file, suffix=args.suffix, tgroup_tolerance=args.tgroup_tolerance,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

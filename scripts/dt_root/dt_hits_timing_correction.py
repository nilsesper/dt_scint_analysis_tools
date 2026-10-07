#################################################################
### dt hits -> timing corrected dt hits
# adds the time offset per wire from a testpulse run to the hit timestamps (ts, err_ts, oc, bx, tdc)
# the corrected file has the same format as the input and is used in its place by the following stages
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply the testpulse timing calibration to dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--dt_tp_corrections_file", type=str, required=True, help="input file path: timing corrections from a testpulse run (.pcl)")
    parser.add_argument("--corr_dt_hits_file", type=str, required=True, help="output file path: timing corrected dt hits (.root)")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.apply_timing_correction(
        args.dt_hits_file, args.dt_tp_corrections_file, args.corr_dt_hits_file, step_size=_step_size(args.step_size),
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

#################################################################
### apply cuts to any ROOT file of this workflow, keep the rows which pass all cuts
# example: --cuts "impossible,==,0;chi2/ndf,<,10;x0,>=,-21;x0,<=,21;dt0,>=,0;dt0,<=,params._dt_max_drift_time"
#################################################################

import argparse

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import cut_utils, dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Keep only the rows of a ROOT file which pass all cuts.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path (.root)")
    parser.add_argument("--output_file", type=str, required=True, help="output file path: rows which pass the cuts (.root)")
    parser.add_argument("--cuts", type=str, required=True,
                        help="cuts in format \"key1,operator1,value1;key2,operator2,value2;...\", operators: == != > < >= <=, "
                             "value: a number or a parameter written as params._name")
    parser.add_argument("--tree", type=str, default=None, help="name of the tree (default: found automatically)")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.apply_cuts(
        args.input_file, args.output_file, cut_utils.parse_cuts(args.cuts), tree=args.tree, step_size=root_utils.parse_step_size(args.step_size),
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

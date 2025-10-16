#################################################################
### align t0, ts0,1,2,3 of sl fits globally in time
# respect the timing alignment given in params._sl_time
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sl_fits_file",
        type     = str,
        help     = "input file path: sl fits (pcl file)",
    )
    parser.add_argument(
        "--sl_fits_file_aligned",
        type     = str,
        help     = "output file path: sl fits, time aligned (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    sl_fits_file_aligned = args.sl_fits_file_aligned
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ### data import
    print(f"###### Importing sl fits...")
    sl_fits = data_utils.load_pickle(file=sl_fits_file)

    #### superlayer-level time alignment
    print(f"Align timing on superlayer level, using time_offset[sl] = {params._sl_time_offset} TU...")
    corrected_sl_fits_to_merge = []
    for sl in params._dt_chamber["sls"].keys():
        sl_fits_cut = data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl)], silent=True)
        n_sl_fits_cut = data_utils.length(sl_fits_cut)
        for i in range(n_sl_fits_cut):
            sl_fits_cut[f"t0"][i] = int(sl_fits_cut[f"t0"][i]) + int(params._sl_time_offset[sl])
            for j in range(0,4):
                sl_fits_cut[f"ts{j}"][i] = int(sl_fits_cut[f"ts{j}"][i]) + int(params._sl_time_offset[sl])
        corrected_sl_fits_to_merge.append(sl_fits_cut)
    sl_fits = data_utils.merge_dataset(split_data=corrected_sl_fits_to_merge)
    sl_fits = data_utils.sort_by_key(data=sl_fits, sort_key="t0")

    ### store to pcl file
    print(f"###### Storing SL-level fits to file \"{sl_fits_file_aligned}\"...")
    data_utils.store_pickle(data=sl_fits, file=sl_fits_file_aligned)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

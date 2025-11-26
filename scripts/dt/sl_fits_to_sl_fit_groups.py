#################################################################
### group sl fits to dt sl fit groups
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
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
        "--sl_fit_groups_file",
        type     = str,
        help     = "output file path: dt fit groups (pcl file)",
    )
    ###
    parser.add_argument(
        "--n_proc",
        type     = int,
        help     = "number of processes to run in parallel",
        default = 16,
    )
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    sl_fit_groups_file = args.sl_fit_groups_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ### multiprocessing setup
    n_processes = args.n_proc # no of processes running in parallel
    n_batches = 50000 # batch size for sl fitting of hit clusters
    do_multiprocessing = not verbose

    ### data import
    print(f"###### Importing sl fits...")
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    print(f"sl_fits =",sl_fits)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"])) # secs
    print(f"measurement duration = {duration} s")

    ### group sl fits inside one sl
    print(f"### Group SL fits within same SL...")
    if do_multiprocessing: # with multiprocessing
        sl_fit_groups = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches, function=dt_utils.group_sl_fits_of_one_sl, data=sl_fits, data_key="sl_fits", kwargs={}, mute=True, give_idx_offset=True)
    else: # without multiprocessing
        sl_fit_groups = dt_utils.group_sl_fits_of_one_sl(sl_fits=sl_fits)

    ### sort sl_fit_groups by tgroup
    sl_fit_groups = data_utils.sort_by_key(data=sl_fit_groups, sort_key="tgroup")
    
    #print(f"sl_fit_groups =",sl_fit_groups)

    ### store to pcl file
    print(f"###### Storing SL fit groups to file \"{sl_fit_groups_file}\"...")
    data_utils.store_pickle(data=sl_fit_groups, file=sl_fit_groups_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

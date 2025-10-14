#################################################################
### use sl clusters to perform sl-level track fits
# store sl fits as pcl file
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
        "--sl_patterns_file",
        type     = str,
        help     = "input file path: sl patterns (pcl file)",
    )
    parser.add_argument(
        "--sl_fits_file",
        type     = str,
        help     = "output file path: sl fits (pcl file)",
    )
    parser.add_argument(
        "--fit_vd",
        action   = "store_true",
        help     = "print info",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--n_proc",
        type     = int,
        help     = "number of processes to run in parallel",
        default = 16,
    )
    # ---
    args = parser.parse_args()
    sl_patterns_file = args.sl_patterns_file
    sl_fits_file = args.sl_fits_file
    verbose = False
    if args.verbose:
        verbose = True
    fit_vd = False
    if args.fit_vd:
        fit_vd = True

    #################

    ### multiprocessing setup
    n_processes = args.n_proc # no of processes running in parallel
    n_batches_sl_fitting = 10000 # batch size for sl fitting of hit clusters
    do_multiprocessing = not verbose

    ### data import
    print(f"###### Importing dt hits...")
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)

    ### dt reco
    # fit sl patterns
    print(f"### Fitting of separate SL clusters...")
    if do_multiprocessing: # with multiprocessing
        sl_fits = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_sl_fitting, function=dt_utils.fit_sl_patterns, data=sl_patterns, data_key="patterns", kwargs={"fit_vd": fit_vd}, mute=True)
    else: # without multiprocessing
        sl_fits = dt_utils.fit_sl_patterns(patterns=sl_patterns, verbose=verbose, fit_vd=fit_vd)
    # sort by t0 (muon arrival time)
    sl_patterns = data_utils.sort_by_key(data=sl_fits, sort_key="t0")
    print("sl_fits =",sl_fits)

    ### store to pcl file
    print(f"###### Storing SL-level fits to file \"{sl_fits_file}\"...")
    data_utils.store_pickle(data=sl_fits, file=sl_fits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

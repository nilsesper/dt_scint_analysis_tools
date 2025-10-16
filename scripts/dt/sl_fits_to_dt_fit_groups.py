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
        "--dt_fit_groups_file",
        type     = str,
        help     = "output file path: dt fit groups (pcl file)",
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
    dt_fit_groups_file = args.dt_fit_groups_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ts_tolerance = params._dt_max_drift_time
    ts_tolerance_2 = 100

    ### data import
    print(f"###### Importing sl fits...")
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    print(f"sl_fits =",sl_fits)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"])) # secs
    print(f"measurement duration = {duration} s")

    ### sl fit grouping separately for each superlayer
    sl_fits_sl = {} # sl fits of one sl
    idx_grouped = {} # {sl: [group idx: [idx list of patterns in pattern list which belong to group]]}
    ts_group = {} # {sl: [group idx: timestamp of group (mean of group member timestamps)]}
    n_groups = {}
    group_rate = {}
    for sl in params._dt_chamber["sls"].keys():
        print(f"grouping sl fits of sl = {sl}...")
        sl_fits_sl[sl] = data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl)], silent=True)
        idx_grouped[sl], ts_group[sl] = combination_utils.time_grouping_indices_2(data=sl_fits_sl[sl], ts_tolerance=ts_tolerance, data_ts_key="t0")
        n_groups[sl] = len(idx_grouped[sl])
        group_rate[sl] = n_groups[sl] / duration
    print(f"n_fit_groups per sl = {n_groups}")
    print(f"fit group rate per sl = {group_rate} Hz")

    ### group together sl fit groups of different superlayers
    time_grouping_list = combination_utils.time_grouping_indices_3(data1=ts_group[1], data2=ts_group[2], data3=ts_group[3], ts_tolerance=ts_tolerance_2)
    print(time_grouping_list)

    ### store to pcl file
    print(f"###### Storing DT fit groups to file \"{dt_fit_groups_file}\"...")
    data_utils.store_pickle(data=dt_fit_groups, file=dt_fit_groups_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

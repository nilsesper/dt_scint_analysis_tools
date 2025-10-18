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
        idx_grouped[sl], ts_group[sl] = combination_utils.time_grouping_indices_2(data=sl_fits_sl[sl], ts_tolerance=params._sl_fit_group_ts_tolerance, data_ts_key="t0")
        n_groups[sl] = len(idx_grouped[sl])
        group_rate[sl] = n_groups[sl] / duration
    n_groups_sum = np.sum([n_groups[sl] for sl in params._dt_chamber["sls"].keys()])
    print(f"n_fit_groups per sl = {n_groups}")
    print(f"fit group rate per sl = {group_rate} Hz")

    # sl_fit_groups = { "sl": superlayer, "tgroup": mean t0 of fits in group, "idcs": [indices of group member sl_fits], "n_fits": no of sl fits in group }

    ### translate idcs of sl_fits_sl (only one sl) back to idcs of sl_fits (all sls together)
    sl_fit_groups = {
        "sl": np.zeros(n_groups_sum),
        "tgroup": [0 for i in range(n_groups_sum)],
        "idcs": [[] for i in range(n_groups_sum)],
        "n_fits": [0 for i in range(n_groups_sum)],
    }
    for sl in params._dt_chamber["sls"].keys():
        print(f"translating back indices of groups in sl = {sl}...")
        for i in range(n_groups[sl]):
            j = int( i + np.sum([n_groups[sl_i] for sl_i in params._dt_chamber["sls"].keys() if sl_i < sl]) )
            glob_idcs = []
            for loc_idx in idx_grouped[sl][i]:
                glob_idx = np.where((sl_fits["t0"] == sl_fits_sl[sl]["t0"][loc_idx]) & (sl_fits["sl"] == sl))[0][0]
                glob_idcs.append(glob_idx)
            sl_fit_groups["idcs"][j] = glob_idcs
            sl_fit_groups["tgroup"][j] = ts_group[sl][i]
            sl_fit_groups["sl"][j] = sl
            sl_fit_groups["n_fits"][j] = len(glob_idcs)
    
    ### sort sl_fit_groups by tgroup
    sl_fit_groups = data_utils.sort_by_key(data=sl_fit_groups, sort_key="tgroup")
    
    #print(f"sl_fit_groups =",sl_fit_groups)

    ### store to pcl file
    print(f"###### Storing SL fit groups to file \"{sl_fit_groups_file}\"...")
    data_utils.store_pickle(data=sl_fit_groups, file=sl_fit_groups_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

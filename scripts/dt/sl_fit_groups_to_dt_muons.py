#################################################################
### build muon tracks from sl fit groups
# need to pass sl_fits & sl_fit_groups
#   since sl_fit_groups only contain indices of sl_fits to consider and not the full information
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
        "--sl_fit_groups_file",
        type     = str,
        help     = "input file path: sl fit groups (pcl file)",
    )
    parser.add_argument(
        "--dt_muons_file",
        type     = str,
        help     = "output file path: dt muons (pcl file)",
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
    dt_muons_file = args.dt_muons_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ### data import
    print(f"###### Importing sl fits...")
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    #print(f"sl_fits =",sl_fits)
    sl_fit_groups = data_utils.load_pickle(file=sl_fit_groups_file)
    #print(f"sl_fit_groups =",sl_fit_groups)

    ### dt reco
    # apply clustering algorithm
    print(f"### DT muon track reco from sl fits...")
    dt_muons = dt_utils.reco_muons_from_sl_fit_groups(fits=sl_fits, fit_groups=sl_fit_groups, verbose=verbose)
    # sort by ts3 (reference timestamp)
    dt_muons = data_utils.sort_by_key(data=dt_muons, sort_key="ts")
    print("dt_muons =",dt_muons)

    ### store to pcl file
    print(f"###### Storing DT muon tracks to file \"{dt_muons_file}\"...")
    data_utils.store_pickle(data=dt_muons, file=dt_muons_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

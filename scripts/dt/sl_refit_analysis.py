#################################################################
### analyze refit results of SL patterns and create plots
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import matplotlib.patches as mpatches
from functools import partial
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refits_file",
        type     = str,
        help     = "input file path: dt fits_refits (pcl file)",
    )

    parser.add_argument(
        "--store_plots",
        type     = str,
        help     = "output directory: give argument if plots should be stores, specify output path for plots here",
    )

    # -
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
    )
    parser.add_argument(
        "--store_path",
        type     = str,
        help     = "path to store pdf plot (if desired)",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="show plots",
    )

    parser.add_argument(
        "--simulation",
        action="store_true",
        help="use simulation data"
    )
    # ---
    args = parser.parse_args()
    refits_file = args.refits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True
    # other 
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))
    #################

    ### data import
    print(f"###### Importing all data...")
    arg = "vd_refit"
    # dt
    keylist = ["chi2/ndf", "chi2/ndf_refit","vd", "vd_refit", "tan_alpha", "tan_alpha_refit", "x0", "x0_refit", "t0", "t0_refit", "dt1", "dt1_refit", "dt2", "dt2_refit", "dt2", "dt2_refit"]
    refits = data_utils.load_pickle(file=refits_file)
    print("### imported refits data from file: ", refits_file)
    print(refits.keys())
    
    for key in keylist:
        if "vd" in key:
            factor = 1 / derived_params._drift_velocity_conversion
        else:
            factor = 1

        # hist of vd distribution
        n_refits = data_utils.length(refits)
        plt.figure(figsize=fig_size)
        plt.hist(refits[key]*factor, bins=100, histtype="step", color="black")
        plt.xlabel(key + "value")
        plt.xlim(min(refits[key] * factor), max(refits[key]) * factor)
        plt.ylabel("counts")
        plt.title("distribution of " + key)
        safe_key = key.replace("/", "_")
        plt.savefig(f"{store_plots}/{safe_key}.png", bbox_inches="tight")
        print(f"### saved plot to {store_plots}/{key}.png")
        plt.close()

        #comparison of laterality of fit and refit
        lat_fit = refits["laterality"]
        lat_refit = refits["laterality_refit"]
        change_of_lat = []
        n_lats = len(lat_refit)
        for idx in range(n_lats):
            if lat_fit[idx] != lat_refit[idx]:
                change_of_lat.append(1)
        
    n_changes = len(change_of_lat)

    print(f"{n_changes} changes of laterality in {n_lats} refits: {round(n_changes/n_lats*100, 2)}%")


    
    


if __name__ == "__main__":
    main()
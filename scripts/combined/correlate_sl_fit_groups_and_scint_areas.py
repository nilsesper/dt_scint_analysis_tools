#################################################################
### correlate dt patterns and scint areas in time
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
        "--sl_fit_groups_file",
        type     = str,
        help     = "input file path: sl fit groups (pcl file)",
    )
    parser.add_argument(
        "--scint_areas_file",
        type     = str,
        help     = "input file path: scint areas (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    sl_fit_groups_file = args.sl_fit_groups_file
    scint_areas_file = args.scint_areas_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ts_tolerance = 1000 # in ts units

    ### data import
    print(f"###### Importing scint & dt data...")
    sl_fit_groups = data_utils.load_pickle(file=sl_fit_groups_file)
    #print(f"sl_fit_groups =",sl_fit_groups)
    scint_areas = data_utils.load_pickle(file=scint_areas_file)
    #print(f"scint_areas =",scint_areas)

    ### temporal correlation
    time_grouping_list = combination_utils.time_grouping_indices(data1=scint_areas, data2=sl_fit_groups, data2_ts_tolerance=ts_tolerance, data1_ts_key="ts", data2_ts_key="tgroup")

    correlation_counter = 0
    delta_ts_corr = []

    for scint_idx, dt_idcs in enumerate(time_grouping_list):

        if len(dt_idcs) < 1:
            continue

        correlation_counter += 1

        scint_ts = scint_areas["ts"][scint_idx]
        scint_pixel = scint_areas["pixel"][scint_idx]
        print(f"scint_idx = {scint_idx} -- scint_ts = {scint_ts} -- pixel = {scint_pixel}")
        for dt_idx in dt_idcs:
            dt_tgroup = sl_fit_groups["tgroup"][dt_idx]
            dt_sl = sl_fit_groups["sl"][dt_idx]
            print(f"   dt_idcs = {dt_idcs} -- dt_tgroup = {dt_tgroup} -- sl = {dt_sl}")

        delta_ts_corr.append(scint_ts - dt_tgroup)

    #### time difference between dt & scint correlated hits
    additional_data = {}
    print("Plotting time differences between dt muons...")
    k = f"scint_ts - dt_tgroup"
    additional_data[k] = np.array(delta_ts_corr)
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=True, title=f"", scale="norm")

    #### rates
    duration = 0.78e-9 * (np.amax(sl_fit_groups["tgroup"]) - np.amin(sl_fit_groups["tgroup"])) # secs
    print(f"duration = {duration} s")
    scint_rate = data_utils.length(scint_areas) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(sl_fit_groups) / duration
    print(f"dt total sl fit group rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")





    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

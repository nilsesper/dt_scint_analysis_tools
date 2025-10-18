#################################################################
### analysis plots
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
        "--sl_fit_groups_file",
        type     = str,
        help     = "input file path: sl fit groups (pcl file)",
    )
    # plotting / store plot
    parser.add_argument(
        "--show_plots",
        action = "store_true",
        help     = "show plots flag",
    )
    parser.add_argument(
        "--store_plots",
        type     = str,
        help     = "output directory: give argument if plots should be stores, specify output path for plots here",
    )
    parser.add_argument(
        "--simulation",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    # ---
    args = parser.parse_args()
    sl_fit_groups_file = args.sl_fit_groups_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True
    cuts_list = []
    if args.cuts:
        for cuts_str in args.cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            cuts_list.append((key, operator, value))

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    sl_fit_groups = data_utils.load_pickle(file=sl_fit_groups_file)
    
    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    sl_fit_groups = data_utils.cut_data(data=sl_fit_groups, conditions=cuts_list)

    n_sl_fit_groups = data_utils.length(sl_fit_groups)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(sl_fit_groups["tgroup"]) - np.amin(sl_fit_groups["tgroup"])) # secs
    print(f"measurement duration = {duration} s")

    ### sl fit groups
    print(f"### sl fit groups")
    hist_bins = {
        "sl": np.arange(1, 3+1),
        "tgroup": "auto200",
        "n_fits": "step1",
    }
    for k in hist_bins.keys():
        if k not in sl_fit_groups.keys(): continue
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_fit_groups, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(sl_fit_groups)} underflow={underflow}, overflow={overflow}")
        if len(hists) == 0: continue
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/sl_fit_groups_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    #"""
    #### time difference between sl fits for all sls together
    additional_data = {}
    print("Plotting time differences between sl fit groups for all sls together...")
    k = f"delta_tgroup"
    additional_data[k] = np.zeros(n_sl_fit_groups)
    for i in range(1,n_sl_fit_groups):
        additional_data[k][i] = int(sl_fit_groups[f"tgroup"][i]) - int(sl_fit_groups["tgroup"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e3,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    #"""
    #### time difference between sl fits separately for each sl
    for sl in range(1,4):
        additional_data = {}
        print("Plotting time differences between sl fit groups for sl = {sl} only...")
        k = f"delta_tgroup_sl{sl}"
        sl_fit_groups_cut = data_utils.cut_data(data=sl_fit_groups, conditions=[("sl","==",sl)])
        n_sl_fit_groups_cut = data_utils.length(sl_fit_groups_cut)
        additional_data[k] = np.zeros(n_sl_fit_groups_cut)
        for i in range(1,n_sl_fit_groups_cut):
            additional_data[k][i] = int(sl_fit_groups_cut[f"tgroup"][i]) - int(sl_fit_groups_cut["tgroup"][i-1]) 
        # plot
        hist_bins = np.linspace(0,1e3,500) #"auto500" 
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
        xlabel = f"{k} [TU]"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
        # plot
        hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
        xlabel = f"{k} [TU]"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    #"""
    ### rate of fit groups
    group_count = data_utils.length(sl_fit_groups)
    group_rate = group_count / duration
    print(f"total sl fit group rate: {group_rate:.03f} Hz")
    for sl in range(1,4):
        sl_fit_groups_cut = data_utils.cut_data(data=sl_fit_groups, conditions=[("sl","==",sl)], silent=True)
        group_count = data_utils.length(sl_fit_groups_cut)
        group_rate = group_count / duration
        print(f"sl={sl} sl fit group rate: {group_rate:.03f} Hz")
    #"""

    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

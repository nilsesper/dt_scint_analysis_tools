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
        "--sl_patterns_file",
        type     = str,
        help     = "input file path: sl patterns (pcl file)",
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
    sl_patterns_file = args.sl_patterns_file
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
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)
    
    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    sl_patterns = data_utils.cut_data(data=sl_patterns, conditions=cuts_list)

    ## cut if desired
    #sl_patterns = data_utils.cut_data(data=sl_patterns, conditions=[("pat_type","in",[0,1]), ])

    n_sl_patterns = data_utils.length(data=sl_patterns)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(sl_patterns["ts0"]) - np.amin(sl_patterns["ts0"])) # secs
    print(f"measurement duration = {duration} s")

    ### sl patterns
    print(f"### sl patterns")
    hist_bins = {
        "sl": np.arange(1, 3+1),
        "pat_type": np.arange(10), # index of string name of pattern (index of key of _dt_sl_patterns)
        "wi3": np.arange(0, 80+1),
        "ts3": "auto200",
    }
    if simulation:
        hist_bins |= {
            "muon_ts": "auto200",
            "muon_lat_id": "step1",
            "muon_x0_loc": "auto200",
            "muon_tan_alpha": "auto200",
            "muon_vd": "auto200",
            "muon_id": "auto200",
            "muon_dt0": "auto200",
            "muon_dt1": "auto200",
            "muon_dt2": "auto200",
            "muon_dt3": "auto200",
            "muon_dd0": "auto200",
            "muon_dd1": "auto200",
            "muon_dd2": "auto200",
            "muon_dd3": "auto200",
            "muon_lat0": "step1",
            "muon_lat1": "step1",
            "muon_lat2": "step1",
            "muon_lat3": "step1",
        }
    for k in hist_bins.keys():
        if k == "pat_type":
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_patterns, key=k, bin_centers=hist_bins[k], silent=True)
            rate_hist = hists / duration
            print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{DT})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/sl_patterns_{k}.png"
            hist_utils.plot_1hist(hist=rate_hist, centers=centers, xlabel=xlabel, ylabel="Rate [Hz]", round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
        else:
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_patterns, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{DT})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/sl_patterns_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    """
    ### time difference between hits
    additional_data = {}
    for wi in [2,1,0]:
        k = f"diff_ts_wi{wi}-wi3"
        n_sl_patterns = data_utils.length(sl_patterns)
        additional_data[k] = np.zeros(n_sl_patterns)
        for i in range(n_sl_patterns):
            additional_data[k][i] = int(sl_patterns[f"ts{wi}"][i]) - int(sl_patterns["ts3"][i])
        hist_bins = "auto200"
        # plot
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
        xlabel = f"{k} [TU]"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    #"""
        
    """
    # meantimer testing
    additional_data = {}
    n_sl_patterns = data_utils.length(sl_patterns)

    # Tmuon = 1/4 T3 + 1/4 T1 + 1/2 T2 - 1/2 tmax
    k = f"meantimer(123)_t0"
    additional_data[k] = 1/4*np.float64(sl_patterns[f"ts3"]) + 1/4*np.float64(sl_patterns[f"ts1"]) + 1/2*np.float64(sl_patterns[f"ts2"]) - 1/2*params._dt_max_drift_time
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    
    # Tmuon = 1/4 T2 + 1/4 T0 + 1/2 T1 - 1/2 tmax
    k = f"meantimer(012)_t0"
    additional_data[k] = 1/4*np.float64(sl_patterns[f"ts2"]) + 1/4*np.float64(sl_patterns[f"ts0"]) + 1/2*np.float64(sl_patterns[f"ts1"]) - 1/2*params._dt_max_drift_time
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # difference between Tmuon meantimers
    k = f"meantimer(123)_t0 - meantimer(012)_t0"
    additional_data[k] = additional_data["meantimer(123)_t0"] - additional_data["meantimer(012)_t0"]
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # tan(alpha) = (T1 – T3) vd / 2h
    k = f"meantimer(13)_tan_alpha"
    additional_data[k] = (np.float64(sl_patterns[f"ts1"]) - np.float64(sl_patterns[f"ts3"]))*0.78e-9 * params._drift_velocity*1e3 / (2*params._cell_height*1e-3) #- int(sl_patterns[f"tan_alpha"][i])
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # tan(alpha) = (T0 – T2) vd / 2h
    k = f"meantimer(02)_tan_alpha"
    additional_data[k] = (np.float64(sl_patterns[f"ts0"]) - np.float64(sl_patterns[f"ts2"]))*0.78e-9 * params._drift_velocity*1e3 / (2*params._cell_height*1e-3)
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # difference between tan_alpha meantimers
    k = f"meantimer(13)_tan_alpha - meantimer(02)_tan_alpha"
    additional_data[k] = additional_data["meantimer(13)_tan_alpha"] - additional_data["meantimer(02)_tan_alpha"]
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # x0 = (T3 – Tmuon) vd sgn(alpha)
    k = f"meantimer(3)_x0"
    additional_data[k] = (np.float64(sl_patterns[f"ts3"]) - additional_data[f"meantimer(123)_t0"])*0.78e-9 * params._drift_velocity*1e3 * np.sign(additional_data[f"meantimer(13)_tan_alpha"]) *1e3
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    #"""
    
    
    """
    ### plots of superlayers
    for sl in range(1,4):
        hist_bins = {
            "pat_type": np.arange(10), # index of string name of pattern (index of key of _dt_sl_patterns)
            "wi3": np.arange(0, 80+1),
            "ts3": "auto200",
        }
        sl_patterns_cut = data_utils.cut_data(data=sl_patterns, conditions=[("sl","==",sl)])
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_patterns_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(sl_patterns_cut)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{DT})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/sl_patterns_sl{sl}_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"sl {sl}") # scale="log"
    #"""
            
    """
    #### mean timer plots
    k = f"mt123"
    for pat_type in [0, 1]: 
        for sl in [1, 2, 3]:
            mt_list = []
            i_offset = 0
            sl_patterns_cut = data_utils.cut_data(data=sl_patterns, conditions=[("sl","==",sl), ("pat_type","==",pat_type)], silent=True)
            n_sl_patterns_cut = data_utils.length(sl_patterns_cut)
            sub_list = {k: []}
            for i in range(1,n_sl_patterns_cut):
                sub_list[k].append( ((int(sl_patterns_cut[f"ts1"][i])-int(sl_patterns_cut[f"muon_ts"][i])+int(sl_patterns_cut[f"ts3"][i])-int(sl_patterns_cut[f"muon_ts"][i]) )/2 + int(sl_patterns_cut[f"ts2"][i])-int(sl_patterns_cut[f"muon_ts"][i])) )
                #print( int(sl_patterns_cut[f"ts1"][i]), int(sl_patterns_cut[f"ts3"][i]),  int(sl_patterns_cut[f"ts2"][i]),  int(sl_patterns_cut[f"muon_ts"][i]), sub_list[k][-1] )
            sub_list[k] = np.array(sub_list[k])
            additional_data = sub_list
            # plot
            hist_bins = "auto200"
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
            print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
            xlabel = f"{k} [TU]"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"sl {sl} pat_type {pat_type}") # scale="log"
    #"""

    #"""
    ### rate of patterns per superlayer
    for sl in range(1,4):
        sl_patterns_cut = data_utils.cut_data(data=sl_patterns, conditions=[("sl","==",sl)], silent=True)
        pattern_count = data_utils.length(sl_patterns_cut)
        pattern_rate = pattern_count / duration
        print(f"sl={sl} pattern rate: {pattern_rate:.03f} Hz")
    #"""

    """
    #### on-the-fly superlayer-level time alignment
    print(f"Align timing on superlayer level, using time_offset[sl] = {params._sl_time_offset} TU...")
    corrected_sl_patterns_merge = []
    for sl in range(1,4):
        k = f"delta_ts3_sl{sl}"
        sl_patterns_cut = data_utils.cut_data(data=sl_patterns, conditions=[("sl","==",sl)], silent=True)
        n_sl_patterns_cut = data_utils.length(sl_patterns_cut)
        for i in range(n_sl_patterns_cut):
            for j in range(0,4):
                sl_patterns_cut[f"ts{j}"][i] = int(sl_patterns_cut[f"ts{j}"][i]) + int(params._sl_time_offset[sl])
        corrected_sl_patterns_merge.append(sl_patterns_cut)
    sl_patterns = data_utils.merge_dataset(split_data=corrected_sl_patterns_merge)
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="ts3")
    #"""

    #"""
    #### time difference between sl fits
    additional_data = {}
    print("Plotting time differences between sl fits...")
    k = f"delta_ts3"
    additional_data[k] = np.zeros(n_sl_patterns)
    for i in range(1,n_sl_patterns):
        additional_data[k][i] = int(sl_patterns[f"ts3"][i]) - int(sl_patterns["ts3"][i-1]) 
    # plot
    hist_bins = np.linspace(0,2e3,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k}[TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    #"""
    #### time difference between sl fits within same superlayer
    additional_data = {}
    print("Plotting time differences between sl fits within same sl...")
    for sl in range(1,4):
        k = f"delta_ts3_sl{sl}"
        sl_patterns_cut = data_utils.cut_data(data=sl_patterns, conditions=[("sl","==",sl)], silent=True)
        n_sl_patterns_cut = data_utils.length(sl_patterns_cut)
        additional_data[k] = np.zeros(n_sl_patterns_cut)
        for i in range(1,n_sl_patterns_cut):
            additional_data[k][i] = int(sl_patterns_cut[f"ts3"][i]) - int(sl_patterns_cut["ts3"][i-1]) 
        # plot
        hist_bins = np.linspace(0,2e3,500) #"auto500" 
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
        xlabel = f"{k} [TU]"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
        # plot
        hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
        xlabel = f"{k}[TU]"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""


    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

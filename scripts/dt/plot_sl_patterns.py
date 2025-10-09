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
    # ---
    args = parser.parse_args()
    sl_patterns_file = args.sl_patterns_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)
    
    ## cut if desired
    #sl_patterns = data_utils.cut_data(data=sl_patterns, conditions=[("pat_type","in",[2])])

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
        #"muon_ts": "auto200",
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
    # time difference between hits
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
    
    # meantimer testing
    additional_data = {}
    n_sl_patterns = data_utils.length(sl_patterns)

    # t0 = - 1/2*td_max + 1/4*ts3 + 1/4*ts1 + 1/2 ts2
    k = f"meantimer_t0 - muon_ts"
    additional_data[k] = np.zeros(n_sl_patterns)
    for i in range(n_sl_patterns):
        additional_data[k][i] = 1/4*int(sl_patterns[f"ts3"][i]) + 1/4*int(sl_patterns["ts1"][i]) + 1/2*int(sl_patterns["ts2"][i]) - 1/2*params._dt_max_drift_time - int(sl_patterns["muon_ts"][i])
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    
    # tan(alpha) = (ts1 - ts3) * vd / (2*h_cell)
    k = f"meantimer_tan_alpha"
    additional_data[k] = np.zeros(n_sl_patterns)
    for i in range(n_sl_patterns):
        additional_data[k][i] = (int(sl_patterns[f"ts1"][i]) - int(sl_patterns[f"ts3"][i]))*0.78e-9 * params._drift_velocity*1e3 / (2*params._cell_height*1e-3) #- int(sl_patterns[f"tan_alpha"][i])
    print(additional_data[k])
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_patterns)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"


    
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



    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

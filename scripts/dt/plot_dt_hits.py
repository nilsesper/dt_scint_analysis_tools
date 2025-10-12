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
        "--dt_hits_file",
        type     = str,
        help     = "input file path: dt hits (pcl file)",
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
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    n_dt_hits = data_utils.length(dt_hits)

    ### dt hits
    print(f"### dt hits")
    n_hist_bins = 100
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
        "oc": "auto200", #"step1", #np.linspace(0, params._lhc_orbit_count, n_hist_bins),
        "sl": np.arange(1, 3+1),
        "ly": np.arange(0, 3+1),
        "wi": np.arange(0, 100+1),
        "ts": "auto200",
    }
    if simulation:
        hist_bins |= {
            "muon_ts": "auto200",
            "muon_dt": "auto200",
            "muon_dd": "auto200",
            "muon_lat": "auto200",
            "muon_tan_alpha": "auto200",
        }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dt_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(dt_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/dt_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    
    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_hits["ts"]) - np.amin(dt_hits["ts"])) # secs
    print(f"measurement duration = {duration} s")

    #"""
    ### plots of superlayers & layers
    for sl in range(1,4):
        fig, ax = plt.subplots(4, 1, figsize=(12,8), sharex=True)
        # put all layers in one plot
        for ly in range(0,4):
            hist_bins = {
                "wi": np.arange(0, params._dt_chamber["sls"][sl]["n_wis"]),
            }
            dt_hits_cut = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",sl), ("ly","==",ly)], silent=True)
            for k in hist_bins.keys():
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dt_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
                print(f"key \"{k}\": entries={data_utils.length(dt_hits_cut)} underflow={underflow}, overflow={overflow}")
                if k == "wi":
                    # calculate rate
                    rate_hists = hists / duration
                    # plot hist
                    plotname = False
                    if store_plots != None:
                        plotname = store_plots+f"/dt_hits_sl{sl}_ly{ly}_rate.png"
                    rel_spacing = 0
                    barwidth = np.mean(np.diff(centers))*(1-rel_spacing) # relative spacing between bins
                    ax[ly].bar(centers, rate_hists, width=barwidth, align="center")
                    ax[ly].set_ylim(bottom=0, top=np.amax(rate_hists)*1.1)
                    ax[ly].set_xlabel("Wire")
                    ax[ly].set_ylabel("Rate [Hz]")
                    ax[ly].set_title(f"Superlayer {sl}, Layer {ly}")

        # show plot
        fig.tight_layout()
        fig.show()
    #"""

    """
    ### plots of ro_chs & chs
    for ro_ch in [8, 10, 14]:
        hist_bins = {
            "ch": np.arange(0, 255+1),
        }
        dt_hits_cut = data_utils.cut_data(data=dt_hits, conditions=[("ro_ch","==",ro_ch)], silent=False)
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dt_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(dt_hits_cut)} underflow={underflow}, overflow={overflow}")
            if k == "ch":
                # # occupancy
                # round_digits = 0
                # xlabel = params._key_symbols[k]+"$(\\text{DT})$"
                # xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
                # plotname = False
                # if store_plots != None:
                #     plotname = store_plots+f"/dt_hits_roch{ro_ch}.png"
                # hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"ro_ch {ro_ch}") # scale="log"

                # rate
                rate_hists = hists / duration
                xlabel = params._key_symbols[k]+"$(\\text{DT})$"
                xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
                plotname = False
                if store_plots != None:
                    plotname = store_plots+f"/dt_hits_roch{ro_ch}_rate.png"
                hist_utils.plot_1hist(hist=rate_hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"ro_ch {ro_ch} (rate [Hz])") # scale="log"
    #"""

    """
    #### time difference between hits of same channel
    print("Plotting time differences between hits of same wire...")
    k = f"delta_ts"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(0, 60):
                dt_hits_cut = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","==",wi)], silent=True)
                dt_hits_cut = timestamp_utils.sort_by_timestamp(hits=dt_hits_cut, silent=True)
                n_dt_hits_cut = data_utils.length(dt_hits_cut)
                sub_list = {k: []}
                for i in range(1,n_dt_hits_cut):
                    sub_list[k].append( int(dt_hits_cut[f"ts"][i]) - int(dt_hits_cut["ts"][i-1]) )
                sub_list[k] = np.array(sub_list[k])
                ch_list.append(sub_list)
    additional_data = data_utils.merge_dataset(split_data=ch_list, silent=True)

    # plot

    hist_bins = "auto500" #np.linspace(0, 1e4, 1000) #"auto200"
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"

    hist_bins = np.linspace(0, 1e4, 1000)
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""



    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

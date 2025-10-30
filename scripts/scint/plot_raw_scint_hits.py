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
        "--raw_scint_hits_file",
        type     = str,
        help     = "input file path: raw scintillator hits (pcl file)",
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
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    # ---
    args = parser.parse_args()
    raw_scint_hits_file = args.raw_scint_hits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
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
    # scint
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    raw_scint_hits = data_utils.cut_data(data=raw_scint_hits, conditions=cuts_list)

    n_raw_scint_hits = data_utils.length(raw_scint_hits)

    ### raw scintillator hits
    print(f"### raw scintillator hits")
    n_hist_bins = 100
    hist_bins = [
        { "ro_ch": np.arange(0, 32) },
        { "ch": np.arange(0, 255) },
        { "tdc": np.arange(0, params._lhc_tdc_count+1) },
        { "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins) },
        { "oc": "auto200" },
        { "ly": np.arange(0, 1+1) },
        { "st": np.arange(0, 16+1) },
        { "ts": "auto1000" },
    ]
    for hist_bin in hist_bins:
        k, b = list(hist_bin.keys())[0], list(hist_bin.values())[0]
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=raw_scint_hits, key=k, bin_centers=b, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(raw_scint_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/scint_raw_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots)
    
    """
    ## separate for both scintillator layers
    ly_ch_list = { # hardcoded
        0: list(range(0, 16)),
        1: list(range(16, 32))
    }
    for ly in [0,1]:
        n_hist_bins = 100
        hist_bins = {
            "ro_ch": np.arange(0, 32),
            "ch": np.arange(0, 255),
            "tdc": np.arange(0, params._lhc_tdc_count+1),
            "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
            "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
            "ts": "auto200",
        }
        dumpfile_hits_cut = data_utils.cut_data(data=dumpfile_hits, conditions=[("ch","in",ly_ch_list[ly])])
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dumpfile_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(dumpfile_hits_cut)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{scint})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/scint_raw_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"Layer {ly}")
    """
            
    ### measurement duration
    duration = 0.78e-9 * (np.amax(raw_scint_hits["ts"]) - np.amin(raw_scint_hits["ts"])) # secs
    print(f"measurement duration = {duration} s")
    
    ### estimate rates
    ch_hit_list = np.array([
        data_utils.length( data_utils.cut_data(data=raw_scint_hits, conditions=[("ch","==",ch)], silent=True ) ) for ch in range(32)
    ])
    ch_rate_list = ch_hit_list / duration # hz
    print("channel rates in Hz:")
    for ch in range(32):
        print(f"  ch {ch:2d}: {ch_rate_list[ch]:6.2f} Hz")

    #"""
    ## plot hit differences
    # for each sipm separately
    other_data_dict = {}
    for ly in range(2):
        for st in range(16):
            for sipm in range(2):
                if not (ly in [0] and st in [7, 15] and sipm in [0]): continue
                raw_scint_hits_cut = data_utils.cut_data(data=raw_scint_hits, conditions=[("ly","==",ly), ("st","==",st), ("sipm","==",sipm)])
                n_hits_cut = data_utils.length(data=raw_scint_hits_cut)
                # calculate ts difference
                last_ts = None
                k = f"delta_ts_ly{ly}_st{st}_sipm{sipm}"
                other_data_dict[k] = np.zeros(n_hits_cut)
                for i in range(1,n_hits_cut):
                    cur_ts = raw_scint_hits_cut["ts"][i]
                    last_ts = raw_scint_hits_cut["ts"][i-1]
                    other_data_dict[k][i] = np.uint64(int(cur_ts)-int(last_ts))
                print(other_data_dict)

                # plot
                b = np.linspace(0,1000,1000) #"auto500"
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=other_data_dict, key=k, bin_centers=b, silent=True) # bin_edges
                print(f"key \"{k}\": entries={data_utils.length(raw_scint_hits_cut)} underflow={underflow}, overflow={overflow}")
                round_digits = 0 if k in ["ts"] else 2
                xlabel = k + " [TU]"
                plotname = False
                if store_plots != None:
                    plotname = store_plots+f"/scint_raw_{k}.png"
                hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots)
    #"""

    #"""
    ### plots of sipm rates
    fig, ax = plt.subplots(4, 1, figsize=(12,8), sharex=True)
    for ly in range(0,2):
        for sipm in range(0,2):
            hist_bins = {
                "st": np.arange(0, 16),
            }
            raw_scint_hits_cut = data_utils.cut_data(data=raw_scint_hits, conditions=[("ly","==",ly), ("sipm","==",sipm)], silent=True)
            for k in hist_bins.keys():
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=raw_scint_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
                print(f"key \"{k}\": entries={data_utils.length(raw_scint_hits_cut)} underflow={underflow}, overflow={overflow}")
                if k == "st":
                    # calculate rate
                    rate_hists = hists / duration
                    if len(rate_hists) == 0:
                        continue
                    # plot hist
                    rel_spacing = 0
                    barwidth = np.mean(np.diff(centers))*(1-rel_spacing) # relative spacing between bins
                    ax[2*ly+sipm].bar(centers, rate_hists, width=barwidth, align="center")
                    ax[2*ly+sipm].set_ylim(bottom=0, top=np.amax(rate_hists)*1.1)
                    ax[2*ly+sipm].set_xlabel("Strip")
                    ax[2*ly+sipm].set_ylabel("Rate [Hz]")
                    ax[2*ly+sipm].set_title(f"Layer {ly}, SiPM {sipm}")

    # show plot
    fig.tight_layout()
    fig.show()
    #"""

    #"""
    #### time difference between scint hits
    additional_data = {}
    print("Plotting time differences between scint hits...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_raw_scint_hits)
    for i in range(1,n_raw_scint_hits):
        additional_data[k][i] = int(raw_scint_hits[f"ts"][i]) - int(raw_scint_hits["ts"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e3,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    #"""
    #### time difference between scint hits of different ro_chs
    additional_data = {}
    print("Plotting time differences between scint hits...")
    k = f"delta_ts_hits_1-hits_2"
    hits_1 = data_utils.cut_data(data=raw_scint_hits, conditions=[("ly","==",0), ("st","==",7), ("sipm","==",0)], silent=True)
    hits_2 = data_utils.cut_data(data=raw_scint_hits, conditions=[("ly","==",0), ("st","==",15), ("sipm","==",0)], silent=True)
    n_delta_lys_hits = np.amin([data_utils.length(hits_1), data_utils.length(hits_2)])
    additional_data[k] = np.zeros(n_delta_lys_hits)
    for i in range(1,n_delta_lys_hits):
        additional_data[k][i] = np.clip(a=int(hits_1[f"ts"][i]) - int(hits_2["ts"][i]), a_min=None, a_max=None )
    # plot
    hist_bins = np.linspace(-1e4,1e4,500) #"auto500" #np.linspace(-1e3,1e3,500) #"auto500" 
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





    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

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
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scintillator hits (pcl file)",
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
    scint_hits_file = args.scint_hits_file
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
    scint_hits = data_utils.load_pickle(file=scint_hits_file)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    scint_hits = data_utils.cut_data(data=scint_hits, conditions=cuts_list)

    n_scint_hits = data_utils.length(scint_hits)

    ### scintillator hits
    print(f"### scintillator hits")
    n_hist_bins = 100
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
        "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
        "ly": np.arange(0, 1+1),
        "st": np.arange(0, 16+1),
        "ts": "auto200",
        "sipm_delta_ts": np.linspace(0, 1000, 500), #"auto1000",
        "st_delta_last_ts0": "auto1000",
        "st_delta_last_ts1": "auto1000",
        "st_delta_last_ts": "auto1000", #np.arange(0, 1000+1),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(scint_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/scint_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    
    ### 2d plots of layers
    for ly in range(2):
        hist_bins = {
            "st": np.arange(0, 16+1),
        }
        scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)])
        if data_utils.length(scint_hits_cut) == 0:
            continue
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(scint_hits_cut)} underflow={underflow}, overflow={overflow}")
            if k == "st":
                # occupancy
                px_matrix = np.zeros((16, 16))
                fig, ax = plt.subplots(1, 1, figsize=(10,8))
                for st0 in range(16):
                    for st1 in range(16):
                        px = derived_params._scint_pixel_mapping[(st0, st1)]
                        px_matrix[st0][st1] = hists[st0] if (ly == 0) else hists[st1]
                imshow_obj = ax.imshow(px_matrix)
                ax.invert_yaxis()
                ax.set_xlabel("Strip (Layer 1)")
                ax.set_ylabel("Strip (Layer 0)")
                cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
                fig.tight_layout()
                if show_plots:
                    fig.show()
                # rate (in hits / min)
                duration = 0.78e-9 * (np.amax(scint_hits["ts"]) - np.amin(scint_hits["ts"])) # secs
                px_matrix = np.zeros((16, 16))
                fig, ax = plt.subplots(1, 1, figsize=(10,8))
                for st0 in range(16):
                    for st1 in range(16):
                        px = derived_params._scint_pixel_mapping[(st0, st1)]
                        px_matrix[st0][st1] = hists[st0] / duration if (ly == 0) else hists[st1] / duration
                imshow_obj = ax.imshow(px_matrix)
                ax.invert_yaxis()
                ax.set_xlabel("Strip (Layer 1)")
                ax.set_ylabel("Strip (Layer 0)")
                cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
                cbar.set_label("Hz")
                fig.tight_layout()
                if show_plots:
                    fig.show()


    """
    ## separate for both scintillator layers
    for ly in [0,1]:
        n_hist_bins = 100
        hist_bins = {
            #"ro_ch": np.arange(0, 32),
            #"ch": np.arange(0, 255),
            #"tdc": np.arange(0, params._lhc_tdc_count+1),
            #"bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
            #"oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
            #"ly": np.arange(0, 1+1),
            "st": np.arange(0, 16+1),
            #"ts": "auto200",
        }
        scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)])
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(scint_hits_cut)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{scint})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/scint_hits_ly{ly}_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"Layer {ly}")
    #"""
            
    """
    ### for each channel separately
    for ly in [0,1]:
        for st in range(8):
            print(f"### scintillator hits ly{ly} st{st}")
            n_hist_bins = 100
            hist_bins = {
                #"ts": "auto200",
                #"sipm_delta_ts": "step1",
                #"st_delta_last_ts0": np.arange(0, 1000+1),
                #"st_delta_last_ts1": np.arange(0, 1000+1),
                "st_delta_last_ts": np.linspace(0, 1000, 100), # "auto5000",
            }
            cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly), ("st","==",st)])
            for k in hist_bins.keys():
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cut_scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
                print(f"key \"{k}\": entries={data_utils.length(scint_hits)} underflow={underflow}, overflow={overflow}")
                round_digits = 0 if k in ["ts"] else 2
                xlabel = params._key_symbols[k]+"$(\\text{scint})$"
                xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
                plotname = False
                if store_plots != None:
                    plotname = store_plots+f"/scint_hits_{k}.png"
                hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"ly{ly} st{st}")
    #"""

    #"""
    ### plots of strip rates
    fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    for ly in range(0,2):
        hist_bins = {
            "st": np.arange(0, 16),
        }
        scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)], silent=True)
        if data_utils.length(scint_hits_cut) == 0:
            continue
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(scint_hits_cut)} underflow={underflow}, overflow={overflow}")
            if k == "st":
                # calculate rate
                rate_hists = hists / duration
                # plot hist
                rel_spacing = 0
                barwidth = np.mean(np.diff(centers))*(1-rel_spacing) # relative spacing between bins
                ax[ly].bar(centers, rate_hists, width=barwidth, align="center")
                ax[ly].set_ylim(bottom=0, top=np.amax(rate_hists)*1.1)
                ax[ly].set_xlabel("Strip")
                ax[ly].set_ylabel("Rate [Hz]")
                ax[ly].set_title(f"Layer {ly}")
    # show plot
    fig.tight_layout()
    fig.show()
    #"""

    #"""
    #### time difference between scint hits
    additional_data = {}
    print("Plotting time differences between scint hits...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_scint_hits)
    for i in range(1,n_scint_hits):
        additional_data[k][i] = int(scint_hits[f"ts"][i]) - int(scint_hits["ts"][i-1]) 
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

    """
    #### time difference between scint hits of different ro_chs
    additional_data = {}
    print("Plotting time differences between scint hits...")
    k = f"delta_ts_ly0-ly1"
    hits_ly0 = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",0)], silent=True)
    hits_ly1 = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",1)], silent=True)
    n_delta_lys_hits = np.amin([data_utils.length(hits_ly0), data_utils.length(hits_ly1)])
    additional_data[k] = np.zeros(n_delta_lys_hits)
    for i in range(1,n_delta_lys_hits):
        additional_data[k][i] = np.clip(a=int(hits_ly0[f"ts"][i]) - int(hits_ly1["ts"][i]), a_min=None, a_max=None )
    # plot
    hist_bins = np.linspace(-1e3,1e3,500) #"auto500" 
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
    #### time difference between hits of same channel
    print("Plotting time differences between hits of same strip...")
    k = f"delta_ts_same_st"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for ly in range(0,2):
        for st in range(0,16):
            print(f"  calculating for ly={ly}, st={st}...")
            for wi in range(0, 60):
                scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly), ("st","==",st)], silent=True)
                scint_hits_cut = timestamp_utils.sort_by_timestamp(hits=scint_hits_cut, silent=True)
                n_scint_hits_cut = data_utils.length(scint_hits_cut)
                sub_list = {k: []}
                for i in range(1,n_scint_hits_cut):
                    sub_list[k].append( int(scint_hits_cut[f"ts"][i]) - int(scint_hits_cut["ts"][i-1]) )
                sub_list[k] = np.array(sub_list[k])
                ch_list.append(sub_list)
    additional_data = data_utils.merge_dataset(split_data=ch_list, silent=True)

    # plot

    hist_bins = "auto500" #np.linspace(0, 1e4, 1000) #"auto200"
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"

    hist_bins = np.linspace(0, 5e3, 500)
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

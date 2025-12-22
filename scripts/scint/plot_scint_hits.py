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
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
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
    # other 
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    scint_hits = data_utils.load_pickle(file=scint_hits_file)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    scint_hits = data_utils.cut_data(data=scint_hits, conditions=cuts_list)

    n_scint_hits = data_utils.length(scint_hits)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(scint_hits["ts"]) - np.amin(scint_hits["ts"])) # secs
    print(f"measurement duration = {duration} s")

    print(f"total rate = {n_scint_hits/duration} +- {np.sqrt(n_scint_hits)/duration} Hz")

    """
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
    #"""
        
    """
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
    #"""

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

    """
    ### plots of strip occupancies
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
                ## calculate rate
                #rate_hists = hists / duration
                # plot hist
                rel_spacing = 0
                barwidth = np.mean(np.diff(centers))*(1-rel_spacing) # relative spacing between bins
                ax[ly].bar(centers, hists, width=barwidth, align="center")
                ax[ly].set_ylim(bottom=0, top=np.amax(hists)*1.1)
                ax[ly].set_xlabel("Strip")
                #ax[ly].set_ylabel("Rate [Hz]")
                ax[ly].set_title(f"Layer {ly}")
    # show plot
    fig.tight_layout()
    fig.show()
    #"""

    """
    #### time difference between scint hits
    additional_data = {}
    print("Plotting time differences between scint hits...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_scint_hits)
    for i in range(1,n_scint_hits):
        additional_data[k][i] = int(scint_hits[f"ts"][i]) - int(scint_hits["ts"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e3,int(1e3)) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
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

    
    """
    #### time difference between hits of same channel
    print("Plotting time differences between hits of same strip...")
    k = f"delta_ts_same_st"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for ly in range(0,2):
        for st in range(0,16):
            print(f"  calculating for ly={ly}, st={st}...")
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
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
    #
    hist_bins = np.linspace(0, 1e3,int(1e3))
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    """
    #### time difference between hits of same layer
    print("Plotting time differences between hits of same layer...")
    k = f"delta_ts_same_ly"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for ly in range(0,2):
        print(f"  calculating for ly={ly}...")
        scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)], silent=True)
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
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
    #
    hist_bins = np.linspace(0, 1e3,int(1e3))
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    ### count hits
    # prepare data frame
    raw_counts = {}
    for ly in range(0,2):
        raw_counts[ly] = {}
        for st in range(0,16):
            raw_counts[ly][st] = 0
    # fill count map
    for i in range(n_scint_hits):
        ly, st = scint_hits["ly"][i], scint_hits["st"][i]
        raw_counts[ly][st] += 1

    ########################
    ####### rate plot (multiple bar plots)
    ############ occupancy
    fig, ax = plt.subplots(2, 1, figsize=(16,6), sharex=True)
    # put both layers in one plot
    for ly in range(0,2):
        strips = np.arange(0,16)
        strip_hits = np.array([raw_counts[ly][st] for st in strips])
        err_strip_hits = np.sqrt(strip_hits)
        strip_rates = strip_hits/duration
        err_strip_rates = np.sqrt(strip_hits)/duration
        ax[ly].bar(strips, strip_hits, width=1, align="center")
        ax[ly].bar(strips, bottom=strip_hits-err_strip_hits, height=2*err_strip_hits, width=1, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
        if ly == 1:
            ax[ly].set_xlabel("Strip")
        ax[ly].set_ylabel("Counts")
        ax[ly].set_title(f"Layer {ly}", fontsize=20)
        # ax limits
        ax[ly].set_ylim(bottom=0, top=np.amax(strip_hits+err_strip_hits)*1.1)
        ax[ly].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[ly].yaxis.get_major_formatter().set_powerlimits([-3, 3]) # 10^X power limits for prescale
        #ax[2*ly+sipm].set_yscale("log")
        #ax[2*ly+sipm].set_ylim(bottom=5000, top=np.amax(strip_hits+err_strip_hits)*np.exp(1.1))
        # info box
        info_str = f"entries = {int(np.sum(strip_hits))}"
        ax[ly] = hist_utils.add_infobox(ax=ax[ly], info_str=info_str, info_loc="top right")
        fig.tight_layout()
        fig.show()
    ## store plot
    if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_OCCUPANCY.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
    ############# rate
    fig, ax = plt.subplots(2, 1, figsize=(16,6), sharex=True)
    # put both layers in one plot
    for ly in range(0,2):
        strips = np.arange(0,16)
        strip_hits = np.array([raw_counts[ly][st] for st in strips])
        err_strip_hits = np.sqrt(strip_hits)
        strip_rates = strip_hits/duration
        err_strip_rates = np.sqrt(strip_hits)/duration
        ax[ly].bar(strips, strip_rates, width=1, align="center")
        ax[ly].bar(strips, bottom=strip_rates-err_strip_rates, height=2*err_strip_rates, width=1, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
        if ly == 1:
            ax[ly].set_xlabel("Strip")
        ax[ly].set_ylabel("Rate [Hz]")
        ax[ly].set_title(f"Layer {ly}", fontsize=20)
        # ax limits
        ax[ly].set_ylim(bottom=0, top=np.amax(strip_rates+err_strip_rates)*1.1)
        ax[ly].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[ly].yaxis.get_major_formatter().set_powerlimits([-3, 3]) # 10^X power limits for prescale
        #ax[2*ly+sipm].set_yscale("log")
        #ax[2*ly+sipm].set_ylim(bottom=5000, top=np.amax(strip_hits+err_strip_hits)*np.exp(1.1))
        # info box
        info_str = f"entries = {int(np.sum(strip_hits))}"
        ax[ly] = hist_utils.add_infobox(ax=ax[ly], info_str=info_str, info_loc="top right")
        fig.tight_layout()
        fig.show()
    ## store plot
    if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_RATE.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ######################
    ### HIT TIMESTAMPS

    ## import data
    #ts_list = scint_hits["ts"]
    ## calculate hist
    #edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=f"linear,0,{np.amax(ts_list)},100")
    #hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_list, edges=edges)
    #err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    ## tu to ns
    #centers = centers*0.78
    ## plot
    #fig, ax = plt.subplots(1, 1, figsize=(7,6))
    #ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="ns", info_loc="bottom center")
    #xlabel = "$T$ [ns]"
    #ax.set_xlabel(xlabel)
    #fig.tight_layout()
    #fig.show()
    ### store plot
    #if args.store_path:
    #    hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_TS.pdf"
    #    print(f"store histogram plot as {hist_plot_file}.")
    #    fig.savefig(hist_plot_file)

    # import data
    ts_list = scint_hits["ts"]

    binnings = [ # (binning name, binning arg, new unit name, new unit conversion)
        ( "fullrange", f"linear,0,{np.amax(ts_list)},100", "s", 0.78e-9 ),
    ]
    for binning_name, binning_arg, new_unit_name, new_unit_conversion in binnings:
        # calculate hist
        edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=binning_arg)
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_list, edges=edges)
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
        # unit conversion
        unit_name = "TU"
        if new_unit_conversion != None:
            centers = centers*new_unit_conversion
            unit_name = new_unit_name
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
        ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=new_unit_name, power_limits=[-3, 3], info_loc="bottom center")
        xlabel = "$T$(same layer) ["+unit_name+"]"
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_TS.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ######################
    ### COINCIDENCE: TIME DIFFERENCE OF SIPMS OF THIS STRIP HIT

    # import data
    delta_ts_sipm = scint_hits["sipm_delta_ts"]
    # calculate hist
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg="step1", data_min_val=np.amin(delta_ts_sipm), data_max_val=np.amax(delta_ts_sipm))
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=delta_ts_sipm, edges=edges)
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    # tu to ns
    centers = centers*0.78
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="ns")
    xlabel = "$\\Delta T_\\text{SiPMs}$ [ns]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_SIPM-TS-DIFF.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ######################
    ### COINCIDENCE: SIGNED TIME DIFFERENCE OF SIPMS OF THIS STRIP HIT

    # import data
    delta_ts_sipm = scint_hits["sipm_delta_ts_signed"]
    # calculate hist
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg="step1", data_min_val=np.amin(delta_ts_sipm), data_max_val=np.amax(delta_ts_sipm))
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=delta_ts_sipm, edges=edges)
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    # tu to ns
    centers = centers*0.78
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="ns")
    xlabel = "$\\Delta T_\\text{SiPMs}$ [ns]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_SIPM-TS-DIFF_SIGNED.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)
    
    ######################
    ### COINCIDENCE: TIME DIFFERENCE OF SIPMS OF THIS STRIP HIT
    ### FOR SINGLE STRIP

    strip = 8
    layer = 1

    # calculate ts difference of consecutive scint hits
    cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",layer),("st","==",strip)], silent=True)
    cut_scint_hits = timestamp_utils.sort_by_timestamp(hits=cut_scint_hits, silent=True)
    n_cut_scint_hits = data_utils.length(cut_scint_hits)
    ts_diff_list = np.array(cut_scint_hits["sipm_delta_ts_signed"])

    # calculate hist
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg="step1", data_min_val=np.amin(ts_diff_list), data_max_val=np.amax(ts_diff_list))
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff_list, edges=edges)
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    # tu to ns
    centers = centers*0.78
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="ns")
    xlabel = "$\\Delta T_\\text{SiPMs}$ [ns]"
    ax.set_xlabel(xlabel)
    ax.set_title(f"Ly {layer}, St {strip}")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_SIPM-TS-DIFF_SIGNED_SINGLE_ly{layer}_st{strip}.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ######################
    ### COINCIDENCE: TIME DIFFERENCE OF SIPMS OF THIS STRIP HIT
    ### SEPARATELY PLOTTED FOR ALL STRIPS

    # calculate ts difference of consecutive scint hits
    ts_diffs = {} # (ly,st): ts_diffs
    for ly in range(0,2):
        for st in range(0,16):
            cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly),("st","==",st)], silent=True)
            cut_scint_hits = timestamp_utils.sort_by_timestamp(hits=cut_scint_hits, silent=True)
            n_cut_scint_hits = data_utils.length(cut_scint_hits)
            ts_diff_list = cut_scint_hits["sipm_delta_ts"]
            ts_diffs[(ly,st)] = np.array(ts_diff_list)

    fig, ax = plt.subplots(4, 8, figsize=(17,10), sharex=True, sharey=True)
    max_hist_val = 0
    for ly in range(0,2):
        for st in range(0,16):
            if ly == 1 and st//8 == 1:
                ax[2*ly+st//8][st%8].set_xlabel(f"$\\Delta T_\\text{{SiPMs}}$ [ns]")
            ax[2*ly+st//8][st%8].set_title(f"Ly {ly}, St {st}", fontsize=20)
            # calculate hist
            ts_diff = ts_diffs[(ly,st)]
            if len(ts_diff) == 0:
                # info box
                info_str = f"entries = 0"
                ax[2*ly+st//8][st%8] = hist_utils.add_infobox(ax=ax[2*ly+st//8][st%8], info_str=info_str, info_loc="top center")
                continue
            edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=f"linear,-0.5,31.5,32")
            hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
            max_hist_val = np.amax([max_hist_val, np.amax(hist)])
            err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
            # tu to ns
            centers = centers*0.78
            # plot
            ax[2*ly+st//8][st%8] = hist_utils.plot_histogram(ax=ax[2*ly+st//8][st%8], hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=False)
            # info box
            info_str = f"entries = {entries}"
            ax[2*ly+st//8][st%8] = hist_utils.add_infobox(ax=ax[2*ly+st//8][st%8], info_str=info_str, info_loc="top center")
    ax[0][0].set_ylim(0, max_hist_val*1.2)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_DELTA-TS_SEPARATE.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ######################
    ### COINCIDENCE: SIGNED TIME DIFFERENCE OF SIPMS OF THIS STRIP HIT
    ### SEPARATELY PLOTTED FOR ALL STRIPS

    # calculate ts difference of consecutive scint hits
    ts_diffs = {} # (ly,st): ts_diffs
    for ly in range(0,2):
        for st in range(0,16):
            cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly),("st","==",st)], silent=True)
            cut_scint_hits = timestamp_utils.sort_by_timestamp(hits=cut_scint_hits, silent=True)
            n_cut_scint_hits = data_utils.length(cut_scint_hits)
            ts_diff_list = cut_scint_hits["sipm_delta_ts_signed"]
            ts_diffs[(ly,st)] = np.array(ts_diff_list)

    fig, ax = plt.subplots(4, 8, figsize=(17,10), sharex=True, sharey=True) # constrained_layout=True
    max_hist_val = 0
    for ly in range(0,2):
        for st in range(0,16):
            if ly == 1 and st//8 == 1:
                ax[2*ly+st//8][st%8].set_xlabel(f"$\\Delta T_\\text{{SiPMs}}$ [ns]")
            ax[2*ly+st//8][st%8].set_title(f"Ly {ly}, St {st}", fontsize=20)
            # calculate hist
            ts_diff = ts_diffs[(ly,st)]
            if len(ts_diff) == 0:
                # info box
                info_str = f"entries = 0"
                ax[2*ly+st//8][st%8] = hist_utils.add_infobox(ax=ax[2*ly+st//8][st%8], info_str=info_str, info_loc="top center")
                continue
            edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=f"linear,-31.5,31.5,63")
            hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
            max_hist_val = np.amax([max_hist_val, np.amax(hist)])
            err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
            # tu to ns
            centers = centers*0.78
            # plot
            ax[2*ly+st//8][st%8] = hist_utils.plot_histogram(ax=ax[2*ly+st//8][st%8], hist=hist, centers=centers, err_hist=err_hist, log_scale=False, add_info=False)
            # info box
            info_str = f"entries = {entries}"
            ax[2*ly+st//8][st%8] = hist_utils.add_infobox(ax=ax[2*ly+st//8][st%8], info_str=info_str, info_loc="top center")
    ax[0][0].set_ylim(0, max_hist_val*1.2)
    #fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_DELTA-TS_SIGNED_SEPARATE.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ######################
    ### TIME DIFFERENCE OF ARRIVAL TIMES
    ### FOR ALL SCINT HITS

    # calculate ts difference of consecutive scint hits
    scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits, silent=True)
    n_scint_hits = data_utils.length(scint_hits)
    ts_diff_list = []
    for i in range(1,n_scint_hits):
        ts_diff_list.append(scint_hits["ts"][i] - scint_hits["ts"][i-1])
    ts_diff = np.array(ts_diff_list)

    binnings = [ # (binning name, binning arg, new unit name, new unit conversion)
        ( "fullrange", f"linear,0,{np.amax(ts_diff)},100", "ms", 0.78e-6 ),
        ( "closeup", f"linear,0,200,200", "ns", 0.78 ),
    ]
    for binning_name, binning_arg, new_unit_name, new_unit_conversion in binnings:
        # calculate hist
        edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=binning_arg)
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
        # unit conversion
        unit_name = "TU"
        if new_unit_conversion != None:
            centers = centers*new_unit_conversion
            unit_name = new_unit_name
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
        ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=new_unit_name, power_limits=[-3, 3])
        xlabel = "$\\Delta T$(all hits) ["+unit_name+"]"
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_DELTA-TS_ALL_{binning_name}.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ######################
    ### TIME DIFFERENCE OF ARRIVAL TIMES
    ### FOR ALL SCINT HITS OF SAME LAYER

    # calculate ts difference of consecutive scint hits
    ts_diff_list = []
    for ly in range(0,2):
        cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)], silent=True)
        cut_scint_hits = timestamp_utils.sort_by_timestamp(hits=cut_scint_hits, silent=True)
        n_cut_scint_hits = data_utils.length(cut_scint_hits)
        for i in range(1,n_cut_scint_hits):
            ts_diff_list.append(cut_scint_hits["ts"][i] - cut_scint_hits["ts"][i-1])
    ts_diff = np.array(ts_diff_list)

    binnings = [ # (binning name, binning arg, new unit name, new unit conversion)
        ( "fullrange", f"linear,0,{np.amax(ts_diff)},100", "ms", 0.78e-6 ),
        ( "closeup", f"linear,0,200,200", "ns", 0.78 ),
    ]
    for binning_name, binning_arg, new_unit_name, new_unit_conversion in binnings:
        # calculate hist
        edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=binning_arg)
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
        # unit conversion
        unit_name = "TU"
        if new_unit_conversion != None:
            centers = centers*new_unit_conversion
            unit_name = new_unit_name
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
        ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=new_unit_name, power_limits=[-3, 3])
        xlabel = "$\\Delta T$(same layer) ["+unit_name+"]"
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_DELTA-TS_SAME-LY_{binning_name}.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ######################
    ### TIME DIFFERENCE OF ARRIVAL TIMES
    ### FOR ALL SCINT HITS OF SAME STRIP

    # calculate ts difference of consecutive scint hits
    ts_diff_list = []
    for ly in range(0,2):
        for st in range(0,16):
            cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly),("st","==",st)], silent=True)
            cut_scint_hits = timestamp_utils.sort_by_timestamp(hits=cut_scint_hits, silent=True)
            n_cut_scint_hits = data_utils.length(cut_scint_hits)
            for i in range(1,n_cut_scint_hits):
                ts_diff_list.append(cut_scint_hits["ts"][i] - cut_scint_hits["ts"][i-1])
    ts_diff = np.array(ts_diff_list)

    binnings = [ # (binning name, binning arg, new unit name, new unit conversion)
        ( "fullrange", f"linear,0,{np.amax(ts_diff)},100", "ms", 0.78e-6 ),
        ( "closeup", f"linear,0,1300,100", "ns", 0.78 ),
    ]
    for binning_name, binning_arg, new_unit_name, new_unit_conversion in binnings:
        # calculate hist
        edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=binning_arg)
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
        # unit conversion
        unit_name = "TU"
        if new_unit_conversion != None:
            centers = centers*new_unit_conversion
            unit_name = new_unit_name
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
        ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=new_unit_name, power_limits=[-3, 3])
        xlabel = "$\\Delta T$(same strip) ["+unit_name+"]"
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"SCINT_HITS_SPECIFIC_DELTA-TS_SAME-ST_{binning_name}.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ########################
    ####### find dead & noisy cells

    # mean rate all cells (incl dead and noisy ones)
    total_count_all_cells = 0
    n_cells = 0
    for ly in range(0, 2):
        for st in range(0, 16):
            total_count_all_cells += raw_counts[ly][st]
            n_cells += 1
    duration_seconds = duration
    print(f"total count all strips: {total_count_all_cells} +- {np.sqrt(total_count_all_cells)}")
    print(f"mean count all strips: {total_count_all_cells/n_cells} +- {np.sqrt(total_count_all_cells)/n_cells}")
    print(f"mean rate all strips: {total_count_all_cells/n_cells/duration_seconds} +- {np.sqrt(total_count_all_cells)/n_cells/duration_seconds} Hz")

    # find dead and noisy cells
    print("dead and noisy chs:")
    count_thres = total_count_all_cells/n_cells
    dead_cells = [] # list of (ly, st) with low rates - considered "dead" and are not considered in rate averaging
    noisy_cells = [] # list of (ly, st) with high rates - considered "noisy" and are not considered in rate averaging
    thres_fac = 50
    for ly in range(0, 2):
        for st in range(0, 16):
            if raw_counts[ly][st] < 1/thres_fac*count_thres:
                print(f"  low occupancy in  ly={ly:1}, st={st:2}")
                dead_cells.append((ly,st))
            if raw_counts[ly][st] > thres_fac*count_thres:
                print(f"  high occupancy in ly={ly:1}, st={st:2}")
                noisy_cells.append((ly,st))

    #"""
    ########################
    ####### average scint rates (without dead channels)

    ly0_total_count, ly1_total_count = 0, 0
    n_ly0, n_ly1 = 0, 0
    for ly in range(0, 2):
        for st in range(0, 16):
            if (ly,st) not in dead_cells:
                if ly == 0:
                    ly0_total_count += raw_counts[ly][st]
                    n_ly0 += 1
                if ly == 1:
                    ly1_total_count += raw_counts[ly][st]
                    n_ly1 += 1
    print(f"* = dead or noisy cells not considered")
    print(f"average ly 0 strip rate *  : {ly0_total_count/n_ly0/duration_seconds} +- {np.sqrt(ly0_total_count)/n_ly0/duration_seconds} Hz")
    print(f"average ly 1 strip rate *  : {ly1_total_count/n_ly1/duration_seconds} +- {np.sqrt(ly1_total_count)/n_ly1/duration_seconds} Hz")
    print(f"average strip rate *       : {(ly0_total_count+ly1_total_count)/(n_ly0+n_ly1)/duration_seconds} +- {np.sqrt(ly0_total_count+ly1_total_count)/(n_ly0+n_ly1)/duration_seconds} Hz")
    #"""






    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

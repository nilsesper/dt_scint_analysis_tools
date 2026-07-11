#######################
### plot stored histogram by dataset & key
#######################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import subprocess
import atexit
import sys
import time
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# allowed datasets
allowed_datasets = [
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS", "DT_HITS_SIM",
]

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main(argv=None):
    # NOTE: argv=None -> read sys.argv (unchanged CLI behaviour).
    # Pass a list of strings to call this programmatically instead.

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type     = str,
        help     = "base / working directory",
        required=True,
    )
    parser.add_argument(
        "--data_config_file",
        type     = str,
        help     = "path to data config file which stores the data file names to be considered for the analysis",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type     = str,
        help     = "data set to create histogram from",
        required=True,
    )
    parser.add_argument(
        "--store_path",
        type     = str,
        help     = "path to store pdf plot (if desired)",
    )
    # ---
    args = parser.parse_args(argv)
    # base file path
    base_path = args.base_path
    # dataset to be merged
    dataset = args.dataset
    if dataset not in allowed_datasets:
        raise Exception(f"Forbidden dataset {dataset}.")
    # list of data files to be used
    dump_files = [] # list of dumpfile names
    file_prefixes = [] # list of data file prefixes to be used
    data_config_file = args.data_config_file
    with open(data_config_file) as f:
        lines = f.readlines()
        for line in lines:
            dump_file, file_prefix = line.split(",")
            file_prefixes.append(file_prefix.replace("\n","").replace("\r","").replace("\t",""))
            dump_files.append(dump_file.replace("\n","").replace("\r","").replace("\t",""))
    n_data = len(dump_files)
    common_file_prefix = os.path.commonprefix(file_prefixes)

    ####################

    ### import calculated hist
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC.pcl"
    print(f"open specific data from file \"{specific_data_file}\"...")
    specific_data = data_utils.load_pickle(file=specific_data_file, silent=True)
    # read data
    duration = specific_data["duration"]
    cell_counts = specific_data["cell_counts"]

    duration_seconds = duration*0.78*1e-9
    print(f"duration = {duration_seconds} s")

    ########################
    ####### occupancy plot (2d matrix)

    # generate chamber matrix
    chamber_matrix = np.full((12,58), np.nan) # -1: invalid cell
    cell_hits = 0
    # fill chamber matrix
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                chamber_matrix[4*(sl-1)+ly][wi] = cell_counts[sl][ly][wi]
                cell_hits += cell_counts[sl][ly][wi]
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    im_obj = ax.imshow(X=chamber_matrix, origin="lower", extent=[0-0.5, 57+0.5, 0-0.5, 11+0.5], vmin=0)
    ax.set_xlabel("Wire")
    layer_labels = {
         0: "SL 1, Ly 0",
         1: "SL 1, Ly 1",
         2: "SL 1, Ly 2",
         3: "SL 1, Ly 3",
         4: "SL 2, Ly 0",
         5: "SL 2, Ly 1",
         6: "SL 2, Ly 2",
         7: "SL 2, Ly 3",
         8: "SL 3, Ly 0",
         9: "SL 3, Ly 1",
        10: "SL 3, Ly 2",
        11: "SL 3, Ly 3",
    }
    ax.set_yticks(list(layer_labels.keys()))
    ax.set_yticklabels(list(layer_labels.values()))
    ax.set_aspect("auto")
    cmap = plt.get_cmap('viridis')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3]) # 10^X power limits for prescale
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap, format=formatter)
    cbar.set_label("Count")
    # info box
    entries = int(cell_hits)
    info_str = f"entries = {entries}"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
    # show plot
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_"+"OCCUPANCY"+".pdf"
        print(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ########################
    ####### rate plot (2d matrix)

    # generate chamber matrix
    chamber_matrix = np.full((12,58), np.nan) # -1: invalid cell
    cell_hits = 0
    # fill chamber matrix
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                chamber_matrix[4*(sl-1)+ly][wi] = cell_counts[sl][ly][wi]/duration_seconds
                cell_hits += cell_counts[sl][ly][wi]
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    im_obj = ax.imshow(X=chamber_matrix, origin="lower", extent=[0-0.5, 57+0.5, 0-0.5, 11+0.5], vmin=0)
    ax.set_xlabel("Wire")
    layer_labels = {
         0: "SL 1, Ly 0",
         1: "SL 1, Ly 1",
         2: "SL 1, Ly 2",
         3: "SL 1, Ly 3",
         4: "SL 2, Ly 0",
         5: "SL 2, Ly 1",
         6: "SL 2, Ly 2",
         7: "SL 2, Ly 3",
         8: "SL 3, Ly 0",
         9: "SL 3, Ly 1",
        10: "SL 3, Ly 2",
        11: "SL 3, Ly 3",
    }
    ax.set_yticks(list(layer_labels.keys()))
    ax.set_yticklabels(list(layer_labels.values()))
    ax.set_aspect("auto")
    cmap = plt.get_cmap('viridis')
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap)
    cbar.set_label("Rate [Hz]")
    # info box
    entries = int(cell_hits)
    info_str = f"entries = {entries}"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
    # show plot
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_"+"RATE"+".pdf"
        print(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ########################
    ####### find dead & noisy cells

    # mean rate all cells (incl dead and noisy ones)
    total_count_all_cells = 0
    n_cells = 0
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                total_count_all_cells += cell_counts[sl][ly][wi]
                n_cells += 1
    print(f"total count all cells: {total_count_all_cells} +- {np.sqrt(total_count_all_cells)}")
    print(f"mean count all cells: {total_count_all_cells/n_cells} +- {np.sqrt(total_count_all_cells)/n_cells}")
    print(f"mean rate all cells: {total_count_all_cells/n_cells/duration_seconds} +- {np.sqrt(total_count_all_cells)/n_cells/duration_seconds} Hz")

    # find dead and noisy cells
    print("dead and noisy cells:")
    count_thres = total_count_all_cells/n_cells
    dead_cells = [] # list of (sl, ly, wi) with low rates - considered "dead" and are not considered in rate averaging
    noisy_cells = [] # list of (sl, ly, wi) with high rates - considered "noisy" and are not considered in rate averaging
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                if cell_counts[sl][ly][wi] < 0.5*count_thres:
                    ro_ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ro_ch"]
                    ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ch"]
                    print(f"  low occupancy in  sl={sl:1}, ly={ly:1}, wi={wi:2} (ro_ch={ro_ch:2}, ch={ch:3})")
                    dead_cells.append((sl,ly,wi))
                if cell_counts[sl][ly][wi] > 1.5*count_thres:
                    ro_ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ro_ch"]
                    ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ch"]
                    print(f"  high occupancy in sl={sl:1}, ly={ly:1}, wi={wi:2} (ro_ch={ro_ch:2}, ch={ch:3})")
                    noisy_cells.append((sl,ly,wi))

    ########################
    ####### average phi and theta rates (without dead channels)

    phi1_total_count, phi3_total_count, theta_total_count = 0, 0, 0
    n_phi1, n_phi3, n_theta = 0, 0, 0
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                if (sl,ly,wi) not in dead_cells:
                    if sl in [1]:
                        phi1_total_count += cell_counts[sl][ly][wi]
                        n_phi1 += 1
                    elif sl in [3]:
                        phi3_total_count += cell_counts[sl][ly][wi]
                        n_phi3 += 1
                    elif sl in [2]:
                        theta_total_count += cell_counts[sl][ly][wi]
                        n_theta += 1
    print(f"* = dead or noisy cells not considered")
    print(f"average sl 1 phi cell rate *    : {phi1_total_count/n_phi1/duration_seconds} +- {np.sqrt(phi1_total_count)/n_phi1/duration_seconds} Hz")
    print(f"average sl 2 theta cell rate *  : {theta_total_count/n_theta/duration_seconds} +- {np.sqrt(theta_total_count)/n_theta/duration_seconds} Hz")
    print(f"average sl 3 phi cell rate *    : {phi3_total_count/(n_phi3 + 1)/duration_seconds} +- {np.sqrt(phi3_total_count)/(n_phi3 + 1)/duration_seconds} Hz")
    print(f"average sl 1 & 3 phi cell rate *: {(phi1_total_count+phi3_total_count)/(n_phi1+n_phi3)/duration_seconds} +- {np.sqrt(phi1_total_count+phi3_total_count)/(n_phi1+n_phi3)/duration_seconds} Hz")
    print(f"average chamber cell rate *     : {(phi1_total_count+phi3_total_count+theta_total_count)/(n_phi1+n_phi3+n_theta)/duration_seconds} +- {np.sqrt(phi1_total_count+phi3_total_count+theta_total_count)/(n_phi1+n_phi3+n_theta)/duration_seconds} Hz")

    ########################
    ####### rate plot (multiple bar plots)
    ### plots of superlayers & layers
    for sl in range(1,4):
        fig, ax = plt.subplots(4, 1, figsize=(16,8), sharex=True)
        # put all layers in one plot
        for ly in range(0,4):
            wires = np.array(list(range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1)))
            wire_hits = np.array([cell_counts[sl][ly][wi] for wi in wires])
            wire_rates = wire_hits/duration_seconds
            err_wire_rates = np.sqrt(wire_hits)/duration_seconds
            ax[ly].bar(wires, wire_rates, width=1, align="center")
            ax[ly].bar(wires, bottom=wire_rates-err_wire_rates, height=2*err_wire_rates, width=1, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
            ax[ly].set_ylim(bottom=0, top=np.amax(wire_rates+err_wire_rates)*1.1)
            if ly == 3:
                ax[ly].set_xlabel("Wire")
            ax[ly].set_ylabel("Rate [Hz]")
            ax[ly].set_title(f"SL {sl}, Ly {ly}")
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_"+f"SL{sl}_RATE"+".pdf"
            print(f"store plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)


    #########################

    #calculate dt hit diff histogram
    ### import TS_RANGE files. expect it to have name [file_prefix]_TS_RANGE.txt in the given base_path
    ts_min = []
    ts_max = []
    print(f"open ts_range files...")
    for data_idx in range(n_data):
        ts_range_file = base_path+"/"+file_prefixes[data_idx]+"_TS_RANGE.txt"
        with open(ts_range_file) as f:
            lines = f.readlines()
            ts_min_, ts_max_ = lines[0].split(",")
            ts_min.append(int(ts_min_))
            ts_max.append(int(ts_max_))
    ts_min = np.array(ts_min)
    ts_max = np.array(ts_max)
    ts_len = ts_max - ts_min
    print(f"   ts_min = {ts_min}")
    print(f"   ts_max = {ts_max}")
    print(f"   ts_len = {ts_len}")

    ### calculate ts offsets to be applied to the data
    print(f"calculate ts offsets to apply to sub-datasets to merge data in time...")
    ts_offset = []
    ts_starting_point = 10000 # in tu
    ts_distance_between_datafiles = 0 # in tu
    # ts_merged = ts_data[i] + ts_offset[i]
    # ts_merged has range (ts_starting_point, ...)
    cum_ts_length = 0 # cumulated length of dataset
    for data_idx in range(n_data):
        ts_length = ts_max[data_idx] - ts_min[data_idx]
        ts_offset_ = -ts_min[data_idx] + ts_starting_point + cum_ts_length
        ts_offset.append(ts_offset_)
        cum_ts_length += ts_length + ts_distance_between_datafiles
    ts_offset = np.array(ts_offset)
    print(f"   ts_offset = {ts_offset}")

    ### fixed bins
    n_bins = 2500
    edges = np.linspace(0, 5000, n_bins+1) # in tu

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"open {n_data} data files, apply timing offset and extract data for histogram...")
    print(f"CALCULATING DT HIT TIME DIFFERENCE HISTOGRAM...")
    data_to_merge = []
    for data_idx in tqdm(range(n_data)):
        sub_data_file = base_path+"/"+file_prefixes[data_idx]+"_"+dataset+".pcl"
        # pcl file import
        sub_data = data_utils.load_pickle(file=sub_data_file, silent=True)
        ## apply ts shift
        #for ts_key in ts_keys:
        #    if ts_key in sub_data.keys():
        #        sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
        ### do something with data
        ## calculate time difference between hits
        ch_list = []
        err_ch_list = []
        cut_layers = True # cut layers to calculate time difference only for hits in the same layer
        
        for sl in range(1,4):
            for ly in range(0,4):
                print(f"   sub_data_idx={data_idx}, sl={sl}, ly={ly}...")
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    sub_data_cut = data_utils.cut_data(data=sub_data, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","==",wi)], silent=True)
                    sub_data_cut = timestamp_utils.sort_by_timestamp(hits=sub_data_cut, silent=True)
                    n_sub_data_cut = data_utils.length(sub_data_cut)
                    ts_diff_list = []
                    err_ts_diff_list = []
                    for i in range(1,n_sub_data_cut):
                        ts_diff_list.append(sub_data_cut["ts"][i] - sub_data_cut["ts"][i-1])
                        err_ts_diff_list.append( np.sqrt(sub_data_cut["err_ts"][i]**2 + sub_data_cut["err_ts"][i]**2) )
                    ts_diff_list = np.array(ts_diff_list)
                    err_ts_diff_list = np.array(err_ts_diff_list)
                    ch_list.append({"key": ts_diff_list})
                    err_ch_list.append({"key": err_ts_diff_list})
        merged_ts_diff = data_utils.merge_dataset(split_data=ch_list, silent=True)["key"]
        merged_err_ts_diff = data_utils.merge_dataset(split_data=err_ch_list, silent=True)["key"]
        
        # create histogram of specified key and shifted hists to respect data error
        hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = hist_utils.calculate_histogram_and_shifted_histograms(data=merged_ts_diff, edges=edges, err_data=merged_err_ts_diff)
        # add to combined histogram
        hist += hist_
        entries += entries_
        underflow += underflow_
        overflow += overflow_
        hist_err_right += hist_err_right_
        hist_err_left += hist_err_left_

    duration = cum_ts_length
    print(f"duration = {duration*0.78*1e-9} s")

    ### error calculation for full hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    ### calculate once only stat unc
    err_hist_stat = np.sqrt(hist)

    print(f"created histogram:")
    print(f"  dataset = {dataset}")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    ### store histogram into file
    specific_data_to_store = {
        "edges": edges,
        "centers": centers,
        "hist": hist,
        "err_hist": err_hist,
        "err_hist_stat": err_hist_stat,
        "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up,
        "entries": entries,
        "underflow": underflow,
        "overflow": overflow,
    }
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC.pcl"
    print(f"storing specific data as {specific_data_file}...")
    data_utils.store_pickle(data=specific_data_to_store, file=specific_data_file)

    cell_half_width = 21000 # um
    err_cell_half_width = 100 # um

    legend_font_size = 13

    ### import calculated hist
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC.pcl"
    print(f"open specific data from file \"{specific_data_file}\"...")
    specific_data = data_utils.load_pickle(file=specific_data_file, silent=True)
    # read data
    start_idx = 0
    hist = np.array(specific_data["hist"])[start_idx:]
    err_hist = np.array(specific_data["err_hist"])[start_idx:]
    err_hist_down = np.array(specific_data["err_hist_down"])[start_idx:]
    err_hist_up = np.array(specific_data["err_hist_up"])[start_idx:]
    edges = np.array(specific_data["edges"])[start_idx:]*0.78 # convert from tu to ns
    centers = hist_utils.centers_from_edges(edges)
    bins = centers
    overflow = specific_data["overflow"]
    underflow = specific_data["underflow"]

    ######################
    ##### poisson bg subtraction

    ### plot dt hit differences
    # plot hist
    fig_size = (8, 6)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
    ax.set_xlim(0,np.amax(bins))
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_ALL.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ### remove exponential "poisson" background
    boarder = 2000 # ns
    fit_index_range = (bins > boarder) # > 1000 ns
    extrapol_index_range = (bins <= boarder)
    fit_bins = bins[fit_index_range]
    fit_hist = hist[fit_index_range]
    err_fit_hist = err_hist[fit_index_range]
    def f_bg_fit(x, a, b):
        return a*np.exp(-x/b)
    def err_f_bg_fit(x, a, b, err_a, err_b):
        return np.sqrt( (err_a*np.exp(-x/b))**2 + (-1/b*a*np.exp(-x/b)*err_b)**2 )
    p0 = (1000, 100)
    popt, pcov, infodict, mesg, _ = curve_fit(f=f_bg_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True)
    a_fit, b_fit = popt
    err_a_fit = np.sqrt(pcov[0][0])
    err_b_fit = np.sqrt(pcov[1][1])
    chi2 = np.sum((fit_hist - f_bg_fit(x=fit_bins, a=a_fit, b=b_fit))**2/err_fit_hist**2)
    ndf = len(fit_hist)-2
    chi2ndf = chi2/ndf
    print(f"exp fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) ns")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    ## plot fit, with residual plot
    fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5,1))
    rel_spacing = 0
    # main plot
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing) # relative spacing between bins
    ax[0] = hist_utils.plot_histogram(ax[0], hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
    fit_label = f"""Exponential fit:
$f(\\Delta T) = a \\cdot e^{{-x/b}}$
$a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$
$b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
$\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
    ax[0].plot(fit_bins, f_bg_fit(fit_bins, a=a_fit, b=b_fit), color="tab:red", label=fit_label)
    ax[0].fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
    ax[0].plot(bins[extrapol_index_range], f_bg_fit(bins[extrapol_index_range], a=a_fit, b=b_fit), color="tab:red", linestyle="--", label="Extrapolated fit")
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
    ax[0].legend(loc="lower right", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
    # residual plot
    residuals = fit_hist - f_bg_fit(fit_bins, a=a_fit, b=b_fit)
    err_residuals = err_fit_hist
    ax[1].axhline(y=0, color="gray", linewidth=1)
    ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1, linestyle="")
    # show plot
    ax[1].set_xlim(0,np.amax(bins))
    ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_BGFIT.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ### subtract exp bg
    hist_nobg = hist - f_bg_fit(bins, a=a_fit, b=b_fit)
    err_hist_nobg = np.sqrt( err_hist**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    err_hist_nobg_down = np.sqrt( err_hist_down**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    err_hist_nobg_up = np.sqrt( err_hist_up**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    bins_nobg = bins

    # plot wo bg
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing) # relative spacing between bins
    ax = hist_utils.plot_histogram(ax, hist=hist_nobg, centers=bins_nobg, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
    info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")
    #ax.set_yscale("log")
    #ax.set_ylim(bottom=0.5, top=np.amax(hist_nobg)*np.exp(1.1))
    #ax.set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax.set_xlim(0,600)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    #ax.legend(prop={'size': 14})
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_NOBG.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    # plot wo bg -- IN TDC UNITS
    fig, ax = plt.subplots(1, 1, figsize=(7,6.5))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing) # relative spacing between bins
    ax = hist_utils.plot_histogram(ax, hist=hist_nobg, centers=bins_nobg/0.78, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
    info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg/0.78)):.3g} TU"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")
    #ax.set_yscale("log")
    #ax.set_ylim(bottom=0.5, top=np.amax(hist_nobg)*np.exp(1.1))
    #ax.set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax.set_xlim(0,600/0.78)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [TU]")
    #ax.legend(prop={'size': 14})
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_NOBG_tdc.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    """
    input("Press enter to exit.")
    exit()
    #"""

    ######################
    ##### fit peak position

    ### fit parabola photopeak to determine position
    fit_index_range = (bins_nobg >= 400) & (bins_nobg <= 440) # fit range in ns
    fit_bins = bins_nobg[fit_index_range]
    fit_hist = hist_nobg[fit_index_range]
    err_fit_hist = err_hist_nobg[fit_index_range]
    def f_peak_fit(x, a, b, c):
        return a*(x-b)**2+c
    def err_f_peak_fit(x, a, b, c, err_a, err_b, err_c):
        return np.sqrt( ( err_a*(x-b)**2 )**2 + ( -2*a*(x-b)*err_b )**2 + ( err_c )**2 )
    p0 = (-1, 415, 1000)
    popt, pcov, infodict, mesg, _ = curve_fit(f=f_peak_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True, )
    a_fit, b_fit, c_fit = popt
    err_a_fit = np.sqrt(pcov[0][0])
    err_b_fit = np.sqrt(pcov[1][1])
    err_c_fit = np.sqrt(pcov[2][2])
    chi2 = np.sum((fit_hist - f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit))**2/err_fit_hist**2)
    ndf = len(fit_hist)-2
    chi2ndf = chi2/ndf
    print(f"parabola fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) TU")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  c = {c_fit} +- {err_c_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    # plot fit
    fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5,1))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing) # relative spacing between bins
    ax[0] = hist_utils.plot_histogram(ax[0], hist=hist_nobg, centers=bins_nobg, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
    info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
    ax[0] = hist_utils.add_infobox(ax=ax[0], info_str=info_str, info_loc="top left")

    fit_label = f"""Parabolic fit:
$f(\\Delta T) = a \\cdot (\\Delta T-b)^2+c$
$a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$ 1/ns${{}}^2$
$b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
$c=({np.round(c_fit,0):.0f}\\pm{np.round(err_c_fit,0):.0f})$
$\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
    ax[0].plot(fit_bins, f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit), color="tab:red", label=fit_label)
    ax[0].fill_between(fit_bins, y1=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)-err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), y2=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)+err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), color="tab:red", alpha=0.1)
    ax[0].axvline(x=b_fit, color="tab:red", linestyle="--", label="Peak position $b$")
    ax[0].axvspan(xmin=b_fit-err_b_fit, xmax=b_fit+err_b_fit, color="tab:red", alpha=0.1)
    #ax.set_yscale("log")
    ax[0].set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax[0].legend(loc="lower left", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
    # residual plot
    residuals = fit_hist - f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit)
    err_residuals = err_fit_hist
    ax[1].axhline(y=0, color="gray", linewidth=1)
    ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1, linestyle="")
    # show plot
    ax[1].set_xlim(0,600)
    ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_PEAKFIT.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ### estimate drift velocity
    v_drift = cell_half_width / b_fit # um/ns
    err_v_drift = np.sqrt(
          (-cell_half_width/b_fit**2)**2 * err_b_fit**2 # peak position error
        + (1/b_fit)**2 * err_cell_half_width**2 # drift space error
    )
    print(f"v_drift = {v_drift} +- {err_v_drift} um/ns")

if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
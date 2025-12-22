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
def main():

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
    args = parser.parse_args()
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
    print(f"average sl 3 phi cell rate *    : {phi3_total_count/n_phi3/duration_seconds} +- {np.sqrt(phi3_total_count)/n_phi3/duration_seconds} Hz")
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

if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
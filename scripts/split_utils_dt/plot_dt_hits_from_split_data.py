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

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# allowed datasets
allowed_datasets = [
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS", "SL_PATTERNS", "SL_FITS", "SL_FITS_AFTERCUTS", "SL_FIT_GROUPS", "DT_MUONS",
    "RAW_SCINT_HITS", "RAW_SCINT_GROUPS", "SCINT_HITS", "SCINT_AREAS",
    "DT_HIT_DIFFERENCES",
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
    ####### occupancy plot

    # generate chamber matrix
    chamber_matrix = np.full((12,58), np.nan) # -1: invalid cell
    # fill chamber matrix
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                chamber_matrix[4*(sl-1)+ly][wi] = cell_counts[sl][ly][wi]
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
    #cbar.set_label("Rate [Hz]")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_"+"OCCUPANCY"+".pdf"
        print(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ########################
    ####### occupancy plot

    # generate chamber matrix
    chamber_matrix = np.full((12,58), np.nan) # -1: invalid cell
    # fill chamber matrix
    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                chamber_matrix[4*(sl-1)+ly][wi] = cell_counts[sl][ly][wi]/duration_seconds
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
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_"+"RATE"+".pdf"
        print(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)


    #########################

if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
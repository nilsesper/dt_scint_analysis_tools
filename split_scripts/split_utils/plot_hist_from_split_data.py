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
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS", "SL_PATTERNS", "SL_FAKE_PATTERNS", "SL_FITS", "SL_FITS_AFTERCUTS", "SL_FIT_GROUPS", "SL_FIT_GROUPS_AFTERCUTS", "DT_MUONS",
    "RAW_SCINT_HITS", "RAW_SCINT_HITS_DEADTIME", "RAW_SCINT_GROUPS", "SCINT_HITS", "SCINT_AREAS",
    "DT_HIT_DIFFERENCES",
    "SIM_MUONS", "DT_HITS_SIM",
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
        "--key",
        type     = str,
        help     = "key of dataset to create histogram from",
        required=True,
    )
    parser.add_argument( # if this flag is given: print hist & err_hist when finished
        "--print_hist",
        action = "store_true",
    )
    parser.add_argument( # if this flag is given: use log y axis
        "--log_y_scale",
        action = "store_true",
    )
    parser.add_argument( # if this flag is given: use log y axis
        "--use_asymm_err",
        action = "store_true",
    )
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
    )
    parser.add_argument(
        "--info_box_loc",
        type     = str,
        default = "top right",
        help     = "custom location of info box (if desired)",
    )
    parser.add_argument( # if this flag is given: do not write full info box but only give sum of entries
        "--info_box_only_entries",
        action = "store_true",
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
    # hist data key
    hist_key = args.key
    # other params
    log_y_scale = False
    if args.log_y_scale:
        log_y_scale = True
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))
    use_asym_err = False
    if args.use_asymm_err:
        use_asym_err = True

    ####################

    ### import calculated hist
    hist_key_str = hist_key.replace("/","-")
    hist_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_HIST_"+hist_key_str+".pcl"
    print(f"open histogram from file \"{hist_data_file}\"...")
    hist_data = data_utils.load_pickle(file=hist_data_file, silent=True)
    # read data
    entries = hist_data["entries"]
    underflow = hist_data["underflow"]
    overflow = hist_data["overflow"]
    hist = hist_data["hist"]
    err_hist = hist_data["err_hist"]
    err_hist_down = hist_data["err_hist_down"]
    err_hist_up = hist_data["err_hist_up"]
    edges = hist_data["edges"]
    centers = hist_data["centers"]

    print(f"imported histogram:")
    print(f"  dataset = {dataset}  ,  key = {hist_key}")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")
    if args.print_hist:
        print(f"  edges     =  {edges}")
        print(f"  centers   =  {centers}")
        print(f"  hist      =  {hist}")
        print(f"  err_hist  =  {err_hist}")
        print(f"  err_hist_down  =  {err_hist_down}")
        print(f"  err_hist_up    =  {err_hist_up}")

    ## plot
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    if not use_asym_err: # symm err
        if not args.info_box_only_entries:
            ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=log_y_scale, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=params._key_units[hist_key], info_loc=args.info_box_loc)
        else:
            ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=log_y_scale, add_info=False)
            # info box
            info_str = f"entries = {entries}"
            ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc=args.info_box_loc)
    else: # asymm err
        if not args.info_box_only_entries:
            ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=log_y_scale, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit=params._key_units[hist_key], info_loc=args.info_box_loc)
        else:
            ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=log_y_scale)
            # info box
            info_str = f"entries = {entries}"
            ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc=args.info_box_loc)
    xlabel = (params._key_symbols[hist_key]) if (params._key_units[hist_key] == "") else (params._key_symbols[hist_key]+" ["+ params._key_units[hist_key]+"]")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()

    ## store plot
    if args.store_path:
        hist_key_str = hist_key.replace("/","-")
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_HIST_"+hist_key_str+".pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ## calculate weighted mean of plotted histogram (ignoring the overflow & underflow bins)
    mean, err_mean = hist_utils.weighted_mean_peak_position(hist=hist, centers=centers, err_hist=err_hist, err_centers=np.zeros(len(centers)))
    print(f"histogram weighted mean = ( {mean} +- {err_mean} ) {params._key_units[hist_key]}")


if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
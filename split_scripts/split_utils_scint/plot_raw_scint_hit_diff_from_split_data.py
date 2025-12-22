#################################################################
### analyze raw scint hit differences
#################################################################

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
from scipy.optimize import curve_fit

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# allowed datasets
allowed_datasets = [
    "RAW_SCINT_HITS", "RAW_SCINT_HITS_DEADTIME",
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
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
    )
    parser.add_argument(
        "--suffix",
        type     = str,
        default="",
        help     = "add suffix to output file",
    )
    ###
    parser.add_argument(
        "--new_unit_name",
        type     = str,
        help     = "name of new unit to be used for hist plotting (if desired)",
    )
    parser.add_argument(
        "--new_unit_conversion",
        type     = str,
        help     = "conversion factor to multiply with x-axis in order to convert to new unit (if desired)",
    )
    parser.add_argument(
        "--n_x_ticks",
        type     = str,
        help     = "specify count of x ticks to be plotted (if desired)",
    )
    parser.add_argument(
        "--x_tick_minmax",
        type     = str,
        help     = "specify min,max x ticks to be plotted (if desired)",
    )
    parser.add_argument(
        "--new_x_label",
        type     = str,
        help     = "overwrite x label (if desired)",
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
    # other args
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))
    # new unit
    new_unit = False
    if args.new_unit_name and args.new_unit_conversion:
        new_unit = True
        new_unit_name = args.new_unit_name
        new_unit_conversion = float(args.new_unit_conversion)
    custom_x_ticks = False
    x_tick_minmax = (None, None)
    if args.n_x_ticks:
        custom_x_ticks = True
        n_x_ticks = int(args.n_x_ticks)
        if args.x_tick_minmax:
            x_tick_minmax = [float(s) for s in args.x_tick_minmax.split(",")]
    new_x_label = None
    if args.new_x_label:
        new_x_label = args.new_x_label

    ####################

    legend_font_size = 13

    ### import calculated hist
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_"+args.suffix+".pcl"
    print(f"open specific data from file \"{specific_data_file}\"...")
    specific_data = data_utils.load_pickle(file=specific_data_file, silent=True)
    # read data
    hist = np.array(specific_data["hist"])
    err_hist = np.array(specific_data["err_hist"])
    err_hist_down = np.array(specific_data["err_hist_down"])
    err_hist_up = np.array(specific_data["err_hist_up"])
    edges = np.array(specific_data["edges"]) # convert from tu to ns
    centers = hist_utils.centers_from_edges(edges)
    overflow = np.array(specific_data["overflow"])
    underflow = np.array(specific_data["underflow"])
    entries = int(np.sum(hist))

    ## unit conversion if necessary
    unit_name = "TU"
    label_name = "$\\Delta T_\\text{SiPM}$"
    if new_unit:
        edges = edges*new_unit_conversion
        centers = centers*new_unit_conversion
        unit_name = new_unit_name
    if new_x_label != None:
        label_name = new_x_label


    ##################
    ##### plot raw scint hit differences
    bins = centers
    
    # plot hist
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="ns", info_loc="top right")
    ax.set_xlim(0,np.amax(bins))
    xlabel = (label_name) if (unit_name == "") else (label_name+" ["+ unit_name+"]")
    ax.set_xlabel(xlabel)
    if custom_x_ticks:
        if x_tick_minmax  == (None, None):
            ax.set_xticks(np.linspace(np.amin(edges), np.amax(edges), n_x_ticks))
        else:
            ax.set_xticks(np.linspace(x_tick_minmax[0], x_tick_minmax[1], n_x_ticks))
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC_SAMESIPM_"+args.suffix+".pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)





    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

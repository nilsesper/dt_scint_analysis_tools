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
    "SL_FITS", "SL_FITS_AFTERCUTS",
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
    parser.add_argument( # if this flag is given: use log y axis
        "--log_y_scale",
        action = "store_true",
    )
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
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
    # other params
    log_y_scale = False
    if args.log_y_scale:
        log_y_scale = True
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))

    ####################

    ### import calculated hist
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC.pcl"
    print(f"open specific data from file \"{specific_data_file}\"...")
    specific_data = data_utils.load_pickle(file=specific_data_file, silent=True)
    # read data
    duration = specific_data["duration"]
    pattern_count = specific_data["pattern_count"]
    pattern_rate = specific_data["pattern_rate"]
    err_pattern_rate = specific_data["err_pattern_rate"]
    hist = specific_data["hist"]
    err_hist = specific_data["err_hist"]
    err_hist_down = specific_data["err_hist_down"]
    err_hist_up = specific_data["err_hist_up"]
    edges = specific_data["edges"]
    centers = hist_utils.centers_from_edges(edges)

    duration_seconds = duration*0.78*1e-9
    print(f"duration = {duration_seconds} s")

    ########################
    ####### calculate pattern rates
    # and output them as tex table

    float_precision = 2

    tex_table = f"""\\begin{{tabular}}{{|c|c|c|c|}} 
        \\hline
        Pattern & \\ac{{SL}} 1 (phi) & \\ac{{SL}} 2 (theta) & \\ac{{SL}} 3 (phi) \\\\ \\hline
        $0$ & $({np.round(pattern_rate[1][0],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][0],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][0],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][0],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][0],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][0],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\
        $1$ & $({np.round(pattern_rate[1][1],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][1],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][1],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][1],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][1],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][1],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\
        $2$ & $({np.round(pattern_rate[1][2],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][2],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][2],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][2],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][2],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][2],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\
        $3$ & $({np.round(pattern_rate[1][3],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][3],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][3],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][3],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][3],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][3],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\
        $4$ & $({np.round(pattern_rate[1][4],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][4],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][4],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][4],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][4],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][4],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\
        $5$ & $({np.round(pattern_rate[1][5],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1][5],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2][5],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2][5],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3][5],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3][5],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\ \\hline
        Cumulative & $({np.round(pattern_rate[1]['com'],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[1]['com'],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[2]['com'],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[2]['com'],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ & $({np.round(pattern_rate[3]['com'],float_precision):.{float_precision}f} \\pm {np.round(err_pattern_rate[3]['com'],float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$ \\\\ \\hline
    \\end{{tabular}}"""
    print(tex_table)

    ########################
    ####### calculate total no of patterns
    total_pattern_count = 0
    for sl in range(1,4):
        total_pattern_count += pattern_count[sl]["com"]
    print(f"total pattern count: {total_pattern_count} +- {np.sqrt(total_pattern_count)}")

    #########################
    ####### plot drift time hist

    ## plot in tu
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=log_y_scale, power_limits=[-3, 3])
    hist_key = "dt"
    xlabel = (params._key_symbols[hist_key]) if (params._key_units[hist_key] == "") else (params._key_symbols[hist_key]+" ["+ params._key_units[hist_key]+"]")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_HIST_TIMEBOX.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ## plot in ns
    centers_ns = centers*0.78
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers_ns, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=log_y_scale, power_limits=[-3, 3])
    hist_key = "dt"
    xlabel = params._key_symbols[hist_key] + " [ns]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_HIST_TIMEBOX_ns.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)


    #########################

if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
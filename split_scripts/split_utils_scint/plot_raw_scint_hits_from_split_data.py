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
    raw_counts = specific_data["raw_counts"]

    duration_seconds = duration*0.78*1e-9
    print(f"duration = {duration_seconds} s")


    ########################
    ####### rate plot (multiple bar plots)
    ### plots of superlayers & layers
    fig, ax = plt.subplots(4, 1, figsize=(16,8), sharex=True)
    # put both layers in one plot
    for ly in range(0,2):
        for sipm in range(0,2):
            strips = np.arange(0,16)
            strip_hits = np.array([raw_counts[ly][st][sipm] for st in strips])
            err_strip_hits = np.sqrt(strip_hits)
            strip_rates = strip_hits/duration_seconds
            err_strip_rates = np.sqrt(strip_hits)/duration_seconds
            ax[2*ly+sipm].bar(strips, strip_hits, width=1, align="center")
            ax[2*ly+sipm].bar(strips, bottom=strip_hits-err_strip_hits, height=2*err_strip_hits, width=1, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
            if ly == 1 and sipm == 1:
                ax[2*ly+sipm].set_xlabel("Strip")
            ax[2*ly+sipm].set_title(f"Layer {ly}, SiPM {sipm}", fontsize=20)
            # ax limits
            #ax[2*ly+sipm].set_ylim(bottom=0, top=np.amax(strip_hits+err_strip_hits)*1.1)
            ax[2*ly+sipm].set_yscale("log")
            ax[2*ly+sipm].set_ylim(bottom=5000, top=np.amax(strip_hits+err_strip_hits)*np.exp(1.1))
            fig.tight_layout()
            fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC_OCCUPANCY.pdf"
        print(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)


    ########################
    ####### find dead & noisy cells

    # mean rate all cells (incl dead and noisy ones)
    total_count_all_cells = 0
    n_cells = 0
    for ly in range(0, 2):
        for st in range(0, 16):
            for sipm in range(0, 2):
                total_count_all_cells += raw_counts[ly][st][sipm]
                n_cells += 1
    print(f"total count all sipms: {total_count_all_cells} +- {np.sqrt(total_count_all_cells)}")
    print(f"mean count all sipms: {total_count_all_cells/n_cells} +- {np.sqrt(total_count_all_cells)/n_cells}")
    print(f"mean rate all sipms: {total_count_all_cells/n_cells/duration_seconds} +- {np.sqrt(total_count_all_cells)/n_cells/duration_seconds} Hz")

    # find dead and noisy cells
    print("dead and noisy chs:")
    count_thres = total_count_all_cells/n_cells
    dead_cells = [] # list of (ly, st, sipm) with low rates - considered "dead" and are not considered in rate averaging
    noisy_cells = [] # list of (ly, st, sipm) with high rates - considered "noisy" and are not considered in rate averaging
    for ly in range(0, 2):
        for st in range(0, 16):
            for sipm in range(0, 2):
                if raw_counts[ly][st][sipm] < 1/10*count_thres:
                    ro_ch = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]["ro_ch"]
                    ch = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]["ch"]
                    print(f"  low occupancy in  ly={ly:1}, st={st:2}, sipm={sipm:2} (ro_ch={ro_ch:2}, ch={ch:3})")
                    dead_cells.append((ly,st,sipm))
                if raw_counts[ly][st][sipm] > 10*count_thres:
                    ro_ch = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]["ro_ch"]
                    ch = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]["ch"]
                    print(f"  high occupancy in ly={ly:1}, st={st:2}, sipm={sipm:2} (ro_ch={ro_ch:2}, ch={ch:3})")
                    noisy_cells.append((ly,st,sipm))

    """
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
    #"""


    #########################

if __name__ == "__main__":
    main()
    input("press [enter] to exit.")
#################################################################
### plot raw scint hit groups
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
from tqdm import tqdm
from itertools import combinations

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_scint_groups_file",
        type     = str,
        help     = "input file path: raw scintillator groups (pcl file)",
    )
    #parser.add_argument(
    #    "--cuts",
    #    type     = str,
    #    help     = "cuts to apply to data in format \"key1,operator1,value1;key2,operator2,value;...\"",
    #)
    parser.add_argument(
        "--show_plots",
        action = "store_true",
        help     = "show plots flag",
    )
    # ---
    args = parser.parse_args()
    raw_scint_groups_file = args.raw_scint_groups_file
    #cuts_list = []
    #if args.cuts:
    #    for cuts_str in args.cuts.split(";"):
    #        key, operator, value = cuts_str.split(",")
    #        if "params." in value:
    #            value = getattr(params, value.split("params.")[1])
    #        else:
    #            value = float(value)
    #        cuts_list.append((key, operator, value))
    show_plots = False
    if args.show_plots:
        show_plots = True

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    raw_scint_groups = data_utils.load_pickle(file=raw_scint_groups_file)

    #### cut data
    #print(f"###### Applying data cuts: {cuts_list}...")
    #raw_scint_groups = data_utils.cut_data(data=raw_scint_groups, conditions=cuts_list)
    n_raw_scint_groups = data_utils.length(raw_scint_groups)

    ########## general plots

    ### measurement duration
    duration = 0.78e-9 * (np.amax(raw_scint_groups["tgroup"]) - np.amin(raw_scint_groups["tgroup"])) # secs
    print(f"measurement duration = {duration} s")

    ### raw scint hit groups
    print(f"### raw scint hit groups")
    hist_bins = {
        "tgroup": "auto200",
        "n_hits": "step1",
        "n_hits_nodupl": "step1",
    }
    for k in hist_bins.keys():
        if k not in raw_scint_groups.keys(): continue
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=raw_scint_groups, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(raw_scint_groups)} underflow={underflow}, overflow={overflow}")
        if len(hists) == 0: continue
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    #"""
    #### time difference between tgroup values
    additional_data = {}
    print("Plotting time differences between groups...")
    k = f"delta_tgroup"
    additional_data[k] = np.zeros(n_raw_scint_groups)
    for i in range(1,n_raw_scint_groups):
        additional_data[k][i] = int(raw_scint_groups[f"tgroup"][i]) - int(raw_scint_groups["tgroup"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e3,500) #"auto500" 
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


    ########## grouping matrix

    def _group_matrix_index(hit_tuple):
        (ly, st, sipm) = hit_tuple
        idx = 32*ly + 2*st + sipm
        return idx
    def _inv_group_matrix_index(idx):
        ly = idx//(32)
        st = idx%(32)//2
        sipm = idx%2 
        return ly, st, sipm

    ### construct matrix
    group_matrix = np.zeros((64,64))
    # fill grouping matrix
    # use "_nodupl" i.e. removed double hits of same channel
    for i in range(n_raw_scint_groups):
        n_hit_tuples = raw_scint_groups["n_hits_nodupl"][i]
        tuples = raw_scint_groups["tuples_nodupl"][i]
        for tuple0, tuple1 in combinations(tuples, 2):  # 2 for pairs, 3 for triplets, etc
            gm_idx0 = _group_matrix_index(tuple0)
            gm_idx1 = _group_matrix_index(tuple1)
            group_matrix[gm_idx0][gm_idx1] += 1
        # count each hit also once on diagonal
        for tuple0 in tuples:
            gm_idx0 = _group_matrix_index(tuple0)
            group_matrix[gm_idx0][gm_idx0] += 1

    ### plot matrix

    # plot only half matrix
    for i in range(64):
        for j in range(64):
            if i>j:
                group_matrix[i][j] = None

    # index map
    group_matrix_idx_map = ["" for i in range(64)]
    for i in range(64):
        ly, st, sipm = _inv_group_matrix_index(idx=i)
        group_matrix_idx_map[i] = f"ly{ly} st{st} sipm{sipm}"

    ### plot ctm
    print(f"plotting crosstalk matrix...")
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(group_matrix)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(group_matrix_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(group_matrix_idx_map, rotation=0)
    ax.set_title("Grouping matrix")
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()

    ### plot ctm without diagonal
    group_matrix_nodiag = copy.deepcopy(group_matrix)
    # remove diagonal
    for i in range(64):
        group_matrix_nodiag[i][i] = 0
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(group_matrix_nodiag)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(group_matrix_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(group_matrix_idx_map, rotation=0)
    ax.set_title("Grouping matrix: Removed diagonal")
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()

    ### plot ctm without diagonal and without expected coincidences
    group_matrix_nocoinc = copy.deepcopy(group_matrix)
    # remove diagonal
    for i in range(64):
        group_matrix_nocoinc[i][i] = 0
    # remove expected coicidences
    for i in range(64):
        for j in range(i,64):
            ly0, st0, sipm0 = _inv_group_matrix_index(idx=i)
            ly1, st1, sipm1 = _inv_group_matrix_index(idx=j)
            if (ly0 == ly1) and (st0 == st1):
                group_matrix_nocoinc[i][j] = 0
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(group_matrix_nocoinc)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(group_matrix_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(group_matrix_idx_map, rotation=0)
    ax.set_title("Grouping matrix: Removed coincidences")
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()







    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")





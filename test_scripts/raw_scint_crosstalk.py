#################################################################
### calculate crosstalk hits between raw scintillator hits (i.e. input channels)
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

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_scint_hits_file",
        type     = str,
        help     = "input file path: raw scintillator hits (pcl file)",
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
    raw_scint_hits_file = args.raw_scint_hits_file
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

    # timestamp difference of hits to be counted as crosstalk
    crosstalk_time_window = 20 # in tu

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    raw_scint_hits = data_utils.cut_data(data=raw_scint_hits, conditions=cuts_list)
    n_raw_scint_hits = data_utils.length(raw_scint_hits)

    ### generate crosstalk matrix
    # sort hits by timestamp
    raw_scint_hits = data_utils.sort_by_key(data=raw_scint_hits, sort_key="ts")
    # create empty crosstalk matrix
    #   ctm index = 16*2*ly + 2*st + sipm = 32*ly + 2*st + sipm
    ctm = np.zeros((64,64))
    ctm_idx_map = ["" for i in range(64)]
    for ly in range(0,2):
        for st in range(0,16):
            for sipm in range(0,2):
                ctm_idx = 32*ly + 2*st + sipm
                ctm_idx_map[ctm_idx] = f"ly{ly} st{st} sipm{sipm}"
    # create empty last hit matrix
    last_hits = {ly: {st: {sipm: 0 for sipm in range(0,2)} for st in range(0,16)} for ly in range(0,2)}
    # go through all hits
    print(f"calculating crosstalk matrix...")
    for i in tqdm(range(n_raw_scint_hits)):
        # fill in last hit storage
        ly, st, sipm, ts = raw_scint_hits["ly"][i], raw_scint_hits["st"][i], raw_scint_hits["sipm"][i], raw_scint_hits["ts"][i]
        last_hits[ly][st][sipm] = ts
        # check for crosstalk
        ctm_idx = 32*ly + 2*st + sipm
        for ly0 in range(0,2):
            for st0 in range(0,16):
                for sipm0 in range(0,2):
                    delta_ts = ts - last_hits[ly0][st0][sipm0]
                    if np.abs(delta_ts) < crosstalk_time_window:
                        ctm_idx0 = 32*ly0 + 2*st0 + sipm0
                        ctm[ctm_idx][ctm_idx0] += 1

    ### plot ctm
    print(f"plotting crosstalk matrix...")
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(ctm)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(ctm_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(ctm_idx_map, rotation=0)
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()

    ### plot ctm without diagonal
    ctm_nodiag = copy.deepcopy(ctm)
    # remove diagonal
    for i in range(64):
        ctm_nodiag[i][i] = 0
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(ctm_nodiag)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(ctm_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(ctm_idx_map, rotation=0)
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()

    ### plot ctm without diagonal & without sipm coincidences of same strip 
    ctm_nostripcoinc = copy.deepcopy(ctm)
    # remove diagonal
    for i in range(64):
        ctm_nostripcoinc[i][i] = 0
    # remove same strip sipm0-sipm1 coincidence
    for ly in range(0,2):
        for st in range(0,16):
            idx0 = 32*ly + 2*st + 0
            idx1 = 32*ly + 2*st + 1
            ctm_nostripcoinc[idx0][idx1] = 0
            ctm_nostripcoinc[idx1][idx0] = 0
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    imshow_obj = ax.imshow(ctm_nostripcoinc)
    ax.invert_yaxis()
    ax.set_xticks(list(range(64)))
    ax.set_xticklabels(ctm_idx_map, rotation=90)
    ax.set_yticks(list(range(64)))
    ax.set_yticklabels(ctm_idx_map, rotation=0)
    cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.show()







    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")





#################################################################
### group raw scint hits and store these groups
# grouping based on closely lying timestamps
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
        "--raw_scint_hits_file",
        type     = str,
        help     = "input file path: raw scintillator hits (pcl file)",
    )
    parser.add_argument(
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    parser.add_argument(
        "--raw_scint_groups_file",
        type     = str,
        help     = "output file path: raw scintillator groups (pcl file)",
    )
    parser.add_argument(
        "--n_proc",
        type     = int,
        help     = "number of processes to run in parallel",
        default = 16,
    )
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    raw_scint_hits_file = args.raw_scint_hits_file
    raw_scint_groups_file = args.raw_scint_groups_file
    cuts_list = []
    if args.cuts:
        for cuts_str in args.cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            cuts_list.append((key, operator, value))
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ### multiprocessing setup
    n_processes = args.n_proc # no of processes running in parallel
    n_batches = 50000 # batch size for sl fitting of hit clusters
    do_multiprocessing = not verbose

    ### data import
    print(f"###### Importing all data...")
    # scint
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    raw_scint_hits = data_utils.cut_data(data=raw_scint_hits, conditions=cuts_list)
    n_raw_scint_hits = data_utils.length(raw_scint_hits)

    ########## grouping calculation
    # grouping in time with ts tolerance given by:  params._raw_scint_hits_grouping_ts_tolerance
    print(f"### Group raw scint hits lying close in time...")
    if do_multiprocessing: # with multiprocessing
        raw_scint_groups = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches, function=scint_utils.group_raw_scint_hits, data=raw_scint_hits, data_key="hits", kwargs={"ts_tolerance": params._raw_scint_hits_grouping_ts_tolerance}, mute=True, give_idx_offset=True)
    else: # without multiprocessing
        raw_scint_groups = scint_utils.group_raw_scint_hits(hits=raw_scint_hits, ts_tolerance=params._raw_scint_hits_grouping_ts_tolerance, silent=False)

    ### sort sl_fit_groups by tgroup
    raw_scint_groups = data_utils.sort_by_key(data=raw_scint_groups, sort_key="tgroup")

    ### store to pcl file
    print(f"###### Storing raw scint groups groups to file \"{raw_scint_groups_file}\"...")
    data_utils.store_pickle(data=raw_scint_groups, file=raw_scint_groups_file)




if __name__ == "__main__":
    main()
    print(f"###### Done.")





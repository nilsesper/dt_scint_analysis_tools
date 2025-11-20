#################################################################
### cluster dt hits to sl clusters
# store sl clusters pcl file
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_hits_file",
        type     = str,
        help     = "input file path: dt hits (pcl file)",
    )
    parser.add_argument(
        "--sl_patterns_file",
        type     = str,
        help     = "output file path: sl patterns (pcl file)",
    )
    #parser.add_argument(
    #    "--dt_tp_corrections_file",
    #    type     = str,
    #    help     = "[optional] file path to timing correction file from tp run (pcl file)",
    #)
    ###
    parser.add_argument(
        "--sl_cut",
        type     = str,
        help     = "cut data to given sl id",
    )
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--simulation_only_muon_patterns",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--fit_vd",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--n_proc",
        type     = int,
        help     = "number of processes to run in parallel",
        default = 16,
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    sl_patterns_file = args.sl_patterns_file
    verbose = False
    if args.verbose:
        verbose = True
    simulation_only_muon_patterns = False
    if args.simulation_only_muon_patterns:
        simulation_only_muon_patterns = True
    #do_timing_correction = False
    #if args.dt_tp_corrections_file:
    #    do_timing_correction = True
    #    dt_tp_corrections_file = args.dt_tp_corrections_file
    fit_vd = False
    if args.fit_vd:
        fit_vd = True

    #################

    ### multiprocessing setup
    n_processes = args.n_proc # no of processes running in parallel
    n_batches_clustering = 50000 # batch size for hit clustering
    do_multiprocessing = not verbose

    ### data import
    print(f"###### Importing dt hits...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    #if do_timing_correction:
    #    dt_tp_corrections = data_utils.load_pickle(file=dt_tp_corrections_file)

    ### optional data cut
    if args.sl_cut:
        print(f"### Applying data cut to SL = {args.sl_cut}...")
        dt_hits = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",int(args.sl_cut))])

    #### optionally apply timing corrections
    #if do_timing_correction:
    #    print(f"### Applying timing correction from file \"{dt_tp_corrections_file}\"...")
    #    dt_hits = dt_utils.apply_timing_calibration(hits=dt_hits, dt_tp_corrections=dt_tp_corrections)

    print("dt_hits =",dt_hits)

    ### dt reco
    # apply clustering algorithm
    print(f"### DT hit clustering for each superlayer...")
    if do_multiprocessing:
        sl_patterns = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_clustering, function=dt_utils.find_sl_patterns, data=dt_hits, data_key="hits", kwargs={"verbose": verbose, "simulation_only_muon_patterns": simulation_only_muon_patterns, "fit_vd": fit_vd}, mute=True)
    else:
        sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits, verbose=verbose, simulation_only_muon_patterns=simulation_only_muon_patterns)
    # sort by ts3 (reference timestamp)
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="ts3")
    print("sl_patterns =",sl_patterns)

    ### store to pcl file
    print(f"###### Storing SL patterns to file \"{sl_patterns_file}\"...")
    data_utils.store_pickle(data=sl_patterns, file=sl_patterns_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

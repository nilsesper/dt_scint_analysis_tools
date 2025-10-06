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
    ###
    parser.add_argument(
        "--sl_cut",
        type     = str,
        help     = "cut data to given sl id",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    sl_patterns_file = args.sl_patterns_file

    #################

    ### multiprocessing setup
    n_processes = 8 *2 # no of processes running in parallel
    n_batches_clustering = 10000 # batch size for hit clustering

    ### data import
    print(f"###### Importing dt hits...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)

    ### optional data cut
    if args.sl_cut:
        print(f"### Applying data cut to SL = {args.sl_cut}...")
        dt_hits = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",int(args.sl_cut))])

    ### dt reco
    # apply clustering algorithm
    print(f"### DT hit clustering for each superlayer...")
    sl_patterns = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_clustering, function=dt_utils.find_sl_patterns, data=dt_hits, data_key="hits", kwargs={}, mute=True)
    print("sl_patterns =",sl_patterns)

    ### store to pcl file
    print(f"###### Storing SL patterns to file \"{sl_patterns_file}\"...")
    data_utils.store_pickle(data=sl_patterns, file=sl_patterns_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

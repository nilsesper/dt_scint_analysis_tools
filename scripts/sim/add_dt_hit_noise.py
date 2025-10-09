#################################################################
### add noise to (simulated) dt hits
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
        "--dt_hits_file_with_noise",
        type     = str,
        help     = "output file path: dt hits with added noise (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    dt_hits_file_with_noise = args.dt_hits_file_with_noise
    
    #################

    ###### additional options for dt hit generation
    ### noise generation
    add_noise = True
    ref_cell_noise_rate = 15 # Hz

    ### data import
    print(f"###### Importing cosmic muon tracks...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    n_hits = data_utils.length(dt_hits)

    ### add noise to dt chamber
    if add_noise:
        print(f"###### Adding DT noise of {ref_cell_noise_rate} Hz per cell...")
        # use time range where cosmic muons are
        t_start = dt_hits["ts"][0] - params._dt_max_drift_time
        t_sim = dt_hits["ts"][-1] - dt_hits["ts"][0] + params._dt_max_drift_time
        dt_hits = dt_utils.add_noise(hits=dt_hits, ts_range=[t_start, t_start+t_sim], ref_cell_noise_rate=ref_cell_noise_rate)
        print("dt_hits =",dt_hits)
    n_hits_new = data_utils.length(dt_hits)
    n_noise_hits = n_hits_new - n_hits
    print(f"Before adding noise: {n_hits} hits.")
    print(f"After adding noise: {n_hits_new} hits.")
    print(f"Added noise hits {n_noise_hits} hits.")

    ### store to pcl file
    print(f"###### Storing DT hits with noise to file \"{dt_hits_file_with_noise}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file_with_noise)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

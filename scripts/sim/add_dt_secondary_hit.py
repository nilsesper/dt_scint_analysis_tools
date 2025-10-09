#################################################################
### add secondary hits (simulated) dt hits
# should be applied before applying noise
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
        "--dt_hits_file_with_secondaries",
        type     = str,
        help     = "output file path: dt hits with secondary hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    dt_hits_file_with_secondaries = args.dt_hits_file_with_secondaries
    
    #################

    ###### additional options for dt hit generation
    ### noise generation
    add_secondary_hit = True
    secondary_hit_window = [0, 500] # tu, time distance between initial dt hit and secondary hit, within this time window, the secondary hit is assumed to be equally/uniformly distributed
    secondary_hit_probability = 0.05 # probability for dt hit to get secondary hit

    ### data import
    print(f"###### Importing cosmic muon tracks...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    n_hits = data_utils.length(dt_hits)

    ### add noise to dt chamber
    if add_secondary_hit:
        print(f"###### Adding DT secondary hits for all cell hits p={secondary_hit_probability} probability, ts_window={secondary_hit_window} TU...")
        # use time range where cosmic muons are
        t_start = dt_hits["ts"][0] - params._dt_max_drift_time
        t_sim = dt_hits["ts"][-1] - dt_hits["ts"][0] + params._dt_max_drift_time
        dt_hits = dt_utils.add_secondary_hits(hits=dt_hits, secondary_hit_window=secondary_hit_window, secondary_hit_probability=secondary_hit_probability)
        print("dt_hits =",dt_hits)
    n_hits_new = data_utils.length(dt_hits)
    n_secondary_hits = n_hits_new - n_hits
    print(f"Before adding secondary hits: {n_hits} hits.")
    print(f"After adding secondary hits: {n_hits_new} hits.")
    print(f"Added  secondary hits {n_secondary_hits} hits.")

    ### store to pcl file
    print(f"###### Storing DT hits with secondary hits to file \"{dt_hits_file_with_secondaries}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file_with_secondaries)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

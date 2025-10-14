#################################################################
### correlate dt patterns and scint areas in time
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sl_patterns_file",
        type     = str,
        help     = "input file path: sl patterns (pcl file)",
    )
    parser.add_argument(
        "--scint_areas_file",
        type     = str,
        help     = "input file path: scint areas (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    sl_patterns_file = args.sl_patterns_file
    scint_areas_file = args.scint_areas_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ts_tolerance = 1000 # in ts units

    ### data import
    print(f"###### Importing scint & dt data...")
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)
    print(f"sl_fits =",sl_patterns)
    scint_areas = data_utils.load_pickle(file=scint_areas_file)
    print(f"scint_areas =",scint_areas)

    ### temporal correlation
    time_grouping_list = combination_utils.time_grouping_indices(data1=scint_areas, data2=sl_patterns, data2_ts_tolerance=ts_tolerance, data1_ts_key="ts", data2_ts_key="ts3")

    correlation_counter = 0
    for scint_idx, dt_idcs in enumerate(time_grouping_list):
        do_print = True
        
        scint_ts = scint_areas["ts"][scint_idx]
        scint_pixel = scint_areas["pixel"][scint_idx]

        dt_ts3 = sl_patterns["ts3"][dt_idcs]
        dt_sl = sl_patterns["sl"][dt_idcs]
        dt_pat_type = sl_patterns["pat_type"][dt_idcs]
        dt_wi3 = sl_patterns["wi3"][dt_idcs]

        if not len(dt_idcs) > 0: continue
        if not 1 in dt_sl: continue
        if not 2 in dt_sl: continue
        if not 3 in dt_sl: continue

        correlation_counter += 1

        print(f"scint_idx = {scint_idx} -- scint_ts = {scint_ts} -- pixel = {scint_pixel}")
        print(f"   dt_idcs = {dt_idcs} -- dt_ts3 = {dt_ts3} -- sl = {dt_sl}, wi3 = {dt_wi3}, pat_type = {dt_pat_type}")

    duration = 0.78e-9 * (np.amax(sl_patterns["ts3"]) - np.amin(sl_patterns["ts3"])) # secs
    print(f"duration = {duration} s")
    scint_rate = data_utils.length(scint_areas) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(sl_patterns) / duration
    print(f"dt total pattern rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")


if __name__ == "__main__":
    main()
    print(f"###### Done.")

#################################################################
### correlate dt muons and scint hits in time
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
        "--dt_muons_file",
        type     = str,
        help     = "input file path: dt reco muons (pcl file)",
    )
    parser.add_argument(
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scint areas (pcl file)",
    )
    parser.add_argument(
        "--corr_hits_file",
        type     = str,
        help     = "output file path: indices of correlated hits (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--dt_cuts",
        type     = str,
        help     = "cuts to apply to dt in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    parser.add_argument(
        "--scint_cuts",
        type     = str,
        help     = "cuts to apply to scint in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    # ---
    args = parser.parse_args()
    dt_muons_file = args.dt_muons_file
    scint_hits_file = args.scint_hits_file
    corr_hits_file = args.corr_hits_file
    verbose = False
    if args.verbose:
        verbose = True
    dt_cuts_list = []
    if args.dt_cuts:
        for cuts_str in args.dt_cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            dt_cuts_list.append((key, operator, value))
    scint_cuts_list = []
    if args.scint_cuts:
        for cuts_str in args.scint_cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            scint_cuts_list.append((key, operator, value))

    #################

    ts_tolerance = 1000 # in ts units

    ### data import
    print(f"###### Importing scint & dt data...")
    dt_muons = data_utils.load_pickle(file=dt_muons_file)
    #print(f"sl_fit_groups =",sl_fit_groups)
    scint_hits = data_utils.load_pickle(file=scint_hits_file)
    #print(f"scint_areas =",scint_areas)

    ### cut data
    print(f"###### Applying dt cuts: {dt_cuts_list}...")
    dt_muons = data_utils.cut_data(data=dt_muons, conditions=dt_cuts_list)
    n_dt_muons = data_utils.length(data=dt_muons)
    print(f"###### Applying scint cuts: {scint_cuts_list}...")
    scint_hits = data_utils.cut_data(data=scint_hits, conditions=scint_cuts_list)
    n_scint_hits = data_utils.length(data=scint_hits)



    ### temporal correlation
    time_grouping_list = combination_utils.time_grouping_indices(data1=scint_hits, data2=dt_muons, data2_ts_tolerance=ts_tolerance, data1_ts_key="ts", data2_ts_key="ts")

    correlation_counter = 0
    delta_ts_corr = []

    corr_list = [] # list of (scint_area_idx, dt_muon_idx)
    corr_dt_idcs = []

    for scint_idx, dt_idcs in enumerate(time_grouping_list):

        if len(dt_idcs) != 1: # <
            continue

        dt_idx = dt_idcs[0]
        # prevent double assignment of same muon
        if dt_idx in corr_dt_idcs:
            continue

        correlation_counter += 1

        scint_ts = scint_hits["ts"][scint_idx]
        scint_ly = scint_hits["ly"][scint_idx]
        scint_st = scint_hits["st"][scint_idx]
        #print(f"scint_idx = {scint_idx} -- scint_ts = {scint_ts} -- pixel = {scint_pixel}")

        dt_ts = dt_muons["ts"][dt_idx]
        dt_theta = dt_muons["theta"][dt_idx]
        dt_phi = dt_muons["phi"][dt_idx]
        #print(f"   dt_ts = {dt_ts} -- dt_theta = {dt_theta} -- dt_phi = {dt_phi}")

        delta_ts_corr.append(np.float64(scint_ts) - np.float64(dt_ts))
        corr_list.append((scint_idx, dt_idx))
        corr_dt_idcs.append(dt_idx)
    


    #### rates
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"duration = {duration} s")
    scint_rate = data_utils.length(scint_hits) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(dt_muons) / duration
    print(f"dt total sl fit group rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")




    ### store to pcl file
    print(f"###### Storing data to file \"{corr_hits_file}\"...")
    data_utils.store_pickle(data=corr_list, file=corr_hits_file)






if __name__ == "__main__":
    main()
    print(f"###### Done.")

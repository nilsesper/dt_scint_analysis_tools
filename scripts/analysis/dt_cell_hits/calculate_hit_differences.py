#################################################################
### calculate time differences between dt hits of same cell and store them
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
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_hits_file",
        type     = str,
        help     = "input file path: dt hits (pcl file)",
    )
    parser.add_argument(
        "--dt_hit_differences_file",
        type     = str,
        help     = "output file path: dt hit timestamp differences (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    dt_hit_differences_file = args.dt_hit_differences_file

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_hits = data_utils.load_pickle(file=dt_hits_file)


    #### time difference between hits of same channel
    print("Plotting time differences between hits of same wire...")
    k = f"delta_ts"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for sl in range(1,4):
        for ly in range(0,4):
            print(f"  calculating for sl={sl}, ly={ly}...")
            for wi in range(0, 60):
                dt_hits_cut = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","==",wi)], silent=True)
                dt_hits_cut = timestamp_utils.sort_by_timestamp(hits=dt_hits_cut, silent=True)
                n_dt_hits_cut = data_utils.length(dt_hits_cut)
                sub_list = {k: []}
                for i in range(1,n_dt_hits_cut):
                    sub_list[k].append( int(dt_hits_cut[f"ts"][i]) - int(dt_hits_cut["ts"][i-1]) )
                sub_list[k] = np.array(sub_list[k])
                ch_list.append(sub_list)
    additional_data = data_utils.merge_dataset(split_data=ch_list, silent=True)

    # plot

    hist_bins = np.arange(0,10000)
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")

    dt_hit_differences = {
        "hist": hists,
        "err_hist": np.sqrt(hists),
        "bins": hist_bins,
        "entries": data_utils.length(additional_data),
        "underflow": underflow,
        "overflow": overflow,
    }


    ### store to pcl file
    print(f"###### Storing data to file \"{dt_hit_differences_file}\"...")
    data_utils.store_pickle(data=dt_hit_differences, file=dt_hit_differences_file)



    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

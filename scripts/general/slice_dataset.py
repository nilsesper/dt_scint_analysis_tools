#################################################################
### slice dataset into subset
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
        "--input_data_file",
        type     = str,
        help     = "input file path: data to be sliced (pcl file)",
    )
    parser.add_argument(
        "--sliced_data_file",
        type     = str,
        help     = "output file path: sliced data subset (pcl file)",
    )
    parser.add_argument(
        "--slice_index",
        type     = str,
        help     = "indices to slice from input data in format \"min_idx:max_idx\"",
    )
    # ---
    args = parser.parse_args()
    input_data_file = args.input_data_file
    sliced_data_file = args.sliced_data_file
    slice_index = np.int64(np.array(args.slice_index.split(":")))

    #################

    ### data import
    print(f"###### Importing data from \"{input_data_file}\"...")
    input_data = data_utils.load_pickle(file=input_data_file)
    print("input_data =",input_data)

    ### slice data
    print(f"###### Extracting dt hits...")
    sliced_data = copy.deepcopy(input_data)
    for k in input_data.keys():
        sliced_data[k] = sliced_data[k][slice_index[0] : slice_index[1]+1]
    print("sliced_data =",sliced_data)

    ### store to pcl file
    print(f"###### Storing sliced data to file \"{sliced_data_file}\"...")
    data_utils.store_pickle(data=sliced_data, file=sliced_data_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

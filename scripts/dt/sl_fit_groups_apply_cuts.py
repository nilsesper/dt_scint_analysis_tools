#################################################################
### apply cuts to sl fit groups
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
        "--cut_data_file",
        type     = str,
        help     = "output file path: cut data subset (pcl file)",
    )
    # ---
    args = parser.parse_args()
    input_data_file = args.input_data_file
    cut_data_file = args.cut_data_file

    #################

    ### data import
    print(f"###### Importing data from \"{input_data_file}\"...")
    input_data = data_utils.load_pickle(file=input_data_file)
    n_input_data = data_utils.length(input_data)
    #print("input_data =",input_data)

    ### cut data
    print(f"###### Applying data cuts...")
    last_data = copy.deepcopy(input_data)
    masked_data = {}
    mask = np.full(data_utils.length(last_data), True)
    ## calculate cuts
    # nfits cut
    mask &= (last_data["n_fits"] == 1)
    
    ## apply mask to data
    masked_data = {}
    for name in input_data.keys():
        # if python list at this key
        masked_data[name] = []
        if isinstance(last_data[name], list):
            for i in range(len(last_data[name])):
                if mask[i]:
                    masked_data[name].append(last_data[name][i])
        # if numpy array at this key
        elif isinstance(last_data[name], np.ndarray):
            masked_data[name] = copy.deepcopy(last_data[name][mask])
        else:
            raise Exception(f"CUT_DATA ERROR: data[{name}] is of unsupported type {type(last_data[name])}. can only cut lists or numpy arrays")

    ## print cut flow
    n_masked_data = data_utils.length(masked_data)
    print(f"Cut flow: {n_masked_data} / {n_input_data} = {n_masked_data/n_input_data}")

    ### store to pcl file
    print(f"###### Storing sliced data to file \"{cut_data_file}\"...")
    data_utils.store_pickle(data=masked_data, file=cut_data_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

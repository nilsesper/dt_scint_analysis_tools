#################################################################
### apply cuts to sim dt hits
# to simulate dead cells
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

    ### manual input of low occupancy wires / dead cells
    # all hits of those cells will be removed
    low_occ_wires = { # sl: (ly, wi)
        1: [  ] + [ (1,49), ],
        2: [ (0,2), (0,5), (0,13), (0,35), (0,50), (0,51), (1,4), (1,5), (1,44), (1,45), ] + [ (1,57), ],
        3: [ (0,10), (1,41), (2,26), ] + [ (1,49), ],
    }

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
    # dead cell cut
    for sl in [1,2,3]:
        for ly, wi in low_occ_wires[sl]:
            mask &= (~((last_data["sl"] == sl) & (last_data["ly"] == ly) & (last_data["wi"] == wi)))
    ## apply cuts
    for name in last_data.keys():
        masked_data[name] = copy.deepcopy(last_data[name][mask])
    ## print cut flow
    n_masked_data = data_utils.length(masked_data)
    print(f"Cut flow: {n_masked_data} / {n_input_data} = {n_masked_data/n_input_data}")

    ### store to pcl file
    print(f"###### Storing sliced data to file \"{cut_data_file}\"...")
    data_utils.store_pickle(data=masked_data, file=cut_data_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

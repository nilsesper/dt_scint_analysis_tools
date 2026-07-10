#################################################################
### apply cuts to data
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
    parser.add_argument(
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1+key2,operator2,value;...\"",
    )
    # ---
    args = parser.parse_args()
    input_data_file = args.input_data_file
    cut_data_file = args.cut_data_file
    cuts_list = []
    for cuts_str in args.cuts.split("+"):
        key, operator, value = cuts_str.split(",")
        if "params." in value:
            value = getattr(params, value.split("params.")[1])
        else:
            value = float(value)
        cuts_list.append((key, operator, value))

    #################

    ### data import
    print(f"###### Importing data from \"{input_data_file}\"...")
    input_data = data_utils.load_pickle(file=input_data_file)
    n_input_data = data_utils.length(input_data)
    #print("input_data =",input_data)

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    cut_data = copy.deepcopy(input_data)
    for i in range(len(cuts_list)):
        print("***")
        cut_data = data_utils.cut_data(data=cut_data, conditions=[cuts_list[i]])
        n_cut_data = data_utils.length(cut_data)
        print(f"cut flow w.r.t. initial dataset: {n_cut_data} / {n_input_data} = {n_cut_data/n_input_data}")
    print("***")
    #print("cut_data =",cut_data)

    ### store to pcl file
    print(f"###### Storing sliced data to file \"{cut_data_file}\"...")
    data_utils.store_pickle(data=cut_data, file=cut_data_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

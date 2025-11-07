#################################################################
### import dumpfile and extract dt hits
# store dt hits as pkl file
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
        "--input_dumpfile",
        type     = str,
        help     = "input file path: dumpfile recorded by htg box (txt file)",
    )
    parser.add_argument(
        "--dt_hits_file",
        type     = str,
        help     = "output file path: dt hits (pcl file)",
    )
    parser.add_argument(
        "--nodeadtime",
        action   = "store_true",
        help     = "do not apply dead time",
    )
    # ---
    args = parser.parse_args()
    input_dumpfile = args.input_dumpfile
    dt_hits_file = args.dt_hits_file
    nodeadtime = False
    if args.nodeadtime:
        nodeadtime = True

    #################

    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)
    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hit
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits, ignore_deadtime=nodeadtime)
    print("dt_hits =",dt_hits)

    ### store to pcl file
    print(f"###### Storing data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

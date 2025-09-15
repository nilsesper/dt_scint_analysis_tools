#################################################################
### import dumpfile and extract raw scintillator hits
# raw hits = recorded hits without coincidence on fpga
# store scintillator hits as pkl file
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
        "--raw_scint_hits_file",
        type     = str,
        help     = "output file path: raw scintillator hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    input_dumpfile = args.input_dumpfile
    raw_scint_hits_file = args.raw_scint_hits_file

    #################

    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    print("dumpfile_hits =",dumpfile_hits)

    ### extract scintillator hit
    print(f"###### Extracting raw scintillator hits...")
    raw_scint_hits = scint_utils.extract_raw_scint_hits(hits=dumpfile_hits)
    print("raw_scint_hits =",raw_scint_hits)
    # add timestamp and sort by timestamp
    raw_scint_hits = timestamp_utils.add_timestamp(hits=raw_scint_hits)
    raw_scint_hits = timestamp_utils.sort_by_timestamp(hits=raw_scint_hits)

    ### store to pcl file
    print(f"###### Storing data to file \"{raw_scint_hits_file}\"...")
    data_utils.store_pickle(data=raw_scint_hits, file=raw_scint_hits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

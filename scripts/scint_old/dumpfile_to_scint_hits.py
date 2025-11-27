#################################################################
### import dumpfile and extract scintillator hits
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
        "--scint_hits_file",
        type     = str,
        help     = "output file path: scintillator hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    input_dumpfile = args.input_dumpfile
    scint_hits_file = args.scint_hits_file

    #n_cut = params._dumpfile_hits_to_skip
    n_cut = 2000

    #################

    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=n_cut)
    print("dumpfile_hits =",dumpfile_hits)

    ### extract scintillator hit
    print(f"###### Extracting scintillator hits...")
    scint_hits = scint_utils.extract_scint_hits(hits=dumpfile_hits)
    print("scint_hits =",scint_hits)

    ### store to pcl file
    print(f"###### Storing data to file \"{scint_hits_file}\"...")
    data_utils.store_pickle(data=scint_hits, file=scint_hits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

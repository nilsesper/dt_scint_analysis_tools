#################################################################
### import dumpfile and extract dt hits and scint hits
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
        "--scint_hits_file",
        type     = str,
        help     = "output file path: scint hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    input_dumpfile = args.input_dumpfile
    dt_hits_file = args.dt_hits_file
    scint_hits_file = args.scint_hits_file

    #################

    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)

    ### globally assign timestamp on order to (hopefully) keep "synchronization" of detectors
    dumpfile_hits = timestamp_utils.add_timestamp(hits=dumpfile_hits)
    dumpfile_hits = timestamp_utils.sort_by_timestamp(hits=dumpfile_hits)

    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hit
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits, has_timestamp=True)
    print("dt_hits =",dt_hits)

    ### extract scintillator hit
    print(f"###### Extracting raw scintillator hits...")
    raw_scint_hits = scint_utils.extract_scint_hits(hits=dumpfile_hits, has_timestamp=True)
    print("raw_scint_hits =",raw_scint_hits)

    ### store to pcl file
    print(f"###### Storing dt hit data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)
    print(f"###### Storing scint hit data to file \"{scint_hits_file}\"...")
    data_utils.store_pickle(data=raw_scint_hits, file=scint_hits_file)


if __name__ == "__main__":
    main()
    print(f"###### Done.")

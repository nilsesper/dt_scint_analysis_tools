#################################################################
### import dumpfile and extract dt hits and raw scint hits
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
        "--raw_scint_hits_file",
        type     = str,
        help     = "output file path: raw scint hits (pcl file)",
    )
    # optional: store txt file with min and max ts of this dumpfile
    parser.add_argument(
        "--ts_range_file",
        type     = str,
        help     = "optional output file path: timestamp range (txt file)",
    )
    # ---
    args = parser.parse_args()
    input_dumpfile = args.input_dumpfile
    dt_hits_file = args.dt_hits_file
    raw_scint_hits_file = args.raw_scint_hits_file
    create_ts_file = False
    if args.ts_range_file:
        create_ts_file = True

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
    raw_scint_hits = scint_utils.extract_raw_scint_hits(hits=dumpfile_hits, has_timestamp=True)
    print("raw_scint_hits =",raw_scint_hits)

    ### store to pcl file
    print(f"###### Storing dt hit data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)
    print(f"###### Storing raw scint hit data to file \"{raw_scint_hits_file}\"...")
    data_utils.store_pickle(data=raw_scint_hits, file=raw_scint_hits_file)

    ### optionally create ts file
    if create_ts_file:
        ts_min = np.amin(dumpfile_hits["ts"])
        ts_max = np.amax(dumpfile_hits["ts"])
        print(f"store ts range = [{ts_min}, {ts_max}] in file \"{args.ts_range_file}\".")
        ts_file_string = f"{int(ts_min)},{int(ts_max)}"
        with open(args.ts_range_file, 'w') as f:
            f.write(ts_file_string)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

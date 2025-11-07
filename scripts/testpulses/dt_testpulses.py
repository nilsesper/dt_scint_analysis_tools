#################################################################
### timing analysis of the dt readout system
# with simultaneous testpulses sent to all input channels
# plot timing by fe connector
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from datetime import datetime
import json
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
dumpfile_path = REPO_PATH+"/dumpfiles"
calib_path = REPO_PATH+"/calibration_files"

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### constants
    dump = True
    plot_hists = False
    granularity = "wi" #"fec" # select tp analysis per fe connector ("fec") or per wire ("wi")
    correct_for_offsets = True
    all_sls_aligned = True # flag whether calibration should be calculated on sl level or chamber level (last only possible of tps of all sls are aligned)

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type     = str,
        help     = "input dumpfile path (with recorded testpulses)",
    )
    parser.add_argument(
        "--dt_tp_timing_file",
        type     = str,
        help     = "output: path of generated testpulse timestamps for all chamber channels",
    )
    parser.add_argument(
        "--dt_tp_corrections_file",
        type     = str,
        help     = "output: path of generated testpulse timing corrections for all chamber channels",
    )
    # ---
    args = parser.parse_args()
    tp_dumpfile_name = args.inputfile
    tp_timing_file = args.dt_tp_timing_file
    dt_tp_corrections_file = args.dt_tp_corrections_file

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    tp_dumpfile_hits = data_utils.import_raw(file_name=tp_dumpfile_name) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    tp_dumpfile_hits = data_utils.cut_first_entries(data=tp_dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)
    print("tp_dumpfile_hits =",tp_dumpfile_hits)

    ### extract dt hits
    tp_hits = dt_utils.extract_dt_hits(hits=tp_dumpfile_hits, ignore_deadtime=True)
    tp_hits = timestamp_utils.sort_by_timestamp(hits=tp_hits)
    tp_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_hits)
    print("tp_hits =",tp_hits)

    ### analyze timing of all fe connectors of all superlayers individually
    # if desired: correct testpulse timing for offsets (due to phi/theta difference and tp cable lengths)
    print(f"###### Analyzing testpulse hits for all frontend connectors of all superlayers...")
    rel_thres = 0.2
    if granularity == "fec": # fe conn granularity
        tp_timing = dt_utils.analyze_testpulses(tp_hits, rel_thres=rel_thres, plot_hists=plot_hists, correct_for_offsets=correct_for_offsets)
    elif granularity == "wi": # wire granularity
        tp_timing = dt_utils.analyze_testpulses_per_wire(tp_hits, rel_thres=rel_thres, plot_hists=plot_hists, correct_for_offsets=correct_for_offsets)
    print("tp_timing =",tp_timing)

    ### convert channel timing corrections from tp timing object
    if all_sls_aligned == False:
        # each sl will be treated separately
        # --> only the channels within the sl are aligned after applying this correction, BUT A TIMING OFFSET BETWEEN THE SUPERLAYERS REMAINS (to be corrected in a later step) !!!
        dt_tp_corrections = dt_utils.calculate_sl_tp_corrections(tp_timing=tp_timing)
    else:
        # calibrate with mean of full chamber, only if tps of all obdts are aligned !!!
        dt_tp_corrections = dt_utils.calculate_chamber_tp_corrections(tp_timing=tp_timing)
    print("dt_tp_corrections =",dt_tp_corrections)

    ### store to pcl file
    print(f"###### Storing extracted testpulse timestamps to \"{tp_timing_file}\"...")
    data_utils.store_pickle(data=tp_timing, file=tp_timing_file)
    print(f"###### Storing extracted testpulse timing corrections to \"{dt_tp_corrections_file}\"...")
    data_utils.store_pickle(data=dt_tp_corrections, file=dt_tp_corrections_file)



if __name__ == "__main__":
    main()
    input("Press [Enter] to exit.")
    print(f"###### Done.")









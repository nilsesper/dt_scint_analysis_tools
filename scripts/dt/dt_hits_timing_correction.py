#################################################################
### dt hits apply timing correction
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
        "--dt_hits_file",
        type     = str,
        help     = "input file path: dt hits (pcl file)",
    )
    parser.add_argument(
        "--dt_tp_corrections_file",
        type     = str,
        help     = "[optional] file path to timing correction file from tp run (pcl file)",
    )
    parser.add_argument(
        "--corr_dt_hits_file",
        type     = str,
        help     = "output file path: timing corrected dt hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    dt_tp_corrections_file = args.dt_tp_corrections_file
    corr_dt_hits_file = args.corr_dt_hits_file

    #################

    ### data import
    print(f"###### Importing dt hits...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    dt_tp_corrections = data_utils.load_pickle(file=dt_tp_corrections_file)

    ### do timing correction
    print(f"### Applying timing correction from file \"{dt_tp_corrections_file}\"...")
    corr_dt_hits = dt_utils.apply_timing_calibration(hits=dt_hits, dt_tp_corrections=dt_tp_corrections)

    print("corr_dt_hits =",dt_hits)

    ### store to pcl file
    print(f"###### Storing corr DT hits to file \"{corr_dt_hits_file}\"...")
    data_utils.store_pickle(data=corr_dt_hits, file=corr_dt_hits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

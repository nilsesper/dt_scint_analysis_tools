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

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type     = str,
        help     = "input dumpfile path (with recorded testpulses)",
    )
    # ---
    args = parser.parse_args()
    tp_dumpfile_name = args.inputfile

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    

    


if __name__ == "__main__":
    main()
    input("Press [Enter] to exit.")
    print(f"###### Done.")









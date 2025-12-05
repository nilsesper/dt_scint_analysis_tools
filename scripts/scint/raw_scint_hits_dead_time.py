#################################################################
### apply individual dead time to scint hits
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_scint_hits_file",
        type     = str,
        help     = "input file path: raw scint hits (pcl file)",
    )
    parser.add_argument(
        "--raw_scint_hits_deadtime_file",
        type     = str,
        help     = "output file path: raw scint hits with applied dead time (pcl file)",
    )
    # ---
    args = parser.parse_args()
    raw_scint_hits_file = args.raw_scint_hits_file
    raw_scint_hits_deadtime_file = args.raw_scint_hits_deadtime_file

    # dead time to apply
    dead_time = 320 # in tu # corresponds to 250 ns

    #################

    ### data import
    print(f"###### Importing all data...")
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)
    n_raw_scint_hits = len(raw_scint_hits["ts"])

    ### scint reco
    print(f"###### Applying dead time to {raw_scint_hits_file} raw scintillator hits...")
    # reco muon areas from scintillator hits (+ assign pixel indices)
    raw_scint_hits_deadtime = scint_utils.deadtime_raw_hits(hits=raw_scint_hits, dead_time=dead_time)
    ## remove crosstalk hits
    #scint_areas = scint_utils.remove_crosstalk_areas(areas=scint_areas)

    print("raw_scint_hits_deadtime =",raw_scint_hits_deadtime)

    ### store to pcl file
    print(f"###### Storing data to file \"{raw_scint_hits_deadtime_file}\"...")
    data_utils.store_pickle(data=raw_scint_hits_deadtime, file=raw_scint_hits_deadtime_file)





if __name__ == "__main__":
    main()
    print(f"###### Done.")

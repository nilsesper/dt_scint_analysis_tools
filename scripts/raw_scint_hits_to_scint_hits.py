#################################################################
### reconstruct scint hits (2 sipm coincidences of strips) from raw scint hits (single sipm hits)
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
        help     = "input file path: raw scintillator hits (pcl file)",
    )
    parser.add_argument(
        "--scint_hits_file",
        type     = str,
        help     = "output file path: matched scintillator hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    raw_scint_hits_file = args.raw_scint_hits_file
    scint_hits_file = args.scint_hits_file

    #################

    ### data import
    print(f"###### Importing all data...")
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)
    n_raw_scint_hits = len(raw_scint_hits["ts"])

    ### scint reco
    print(f"###### Reconstructing {n_raw_scint_hits} raw scintillator hits to scintillator hits...")
    # reco muon areas from scintillator hits (+ assign pixel indices)
    scint_hits = scint_utils.reco_hits_from_raw_hits(hits=raw_scint_hits)
    print("scint_hits =",scint_hits)

    ### store to pcl file
    print(f"###### Storing data to file \"{scint_hits_file}\"...")
    data_utils.store_pickle(data=scint_hits, file=scint_hits_file)





if __name__ == "__main__":
    main()
    print(f"###### Done.")

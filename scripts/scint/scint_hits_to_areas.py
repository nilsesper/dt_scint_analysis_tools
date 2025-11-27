#################################################################
### reconstruct scint areas (muon positions) from scint hits
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
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scintillator hits (pcl file)",
    )
    parser.add_argument(
        "--scint_areas_file",
        type     = str,
        help     = "output file path: reconstructed scintillator areas (pcl file)",
    )
    # ---
    args = parser.parse_args()
    scint_hits_file = args.scint_hits_file
    scint_areas_file = args.scint_areas_file

    #################

    ### data import
    print(f"###### Importing all data...")
    scint_hits = data_utils.load_pickle(file=scint_hits_file)
    n_scint_hits = len(scint_hits["ts"])

    ### scint reco
    print(f"###### Reconstructing {n_scint_hits} scintillator hits to muon areas...")
    # reco muon areas from scintillator hits (+ assign pixel indices)
    scint_areas = scint_utils.reco_muon_area_from_hits(hits=scint_hits)
    ## remove crosstalk hits
    #scint_areas = scint_utils.remove_crosstalk_areas(areas=scint_areas)

    print("scint_areas =",scint_areas)

    ### store to pcl file
    print(f"###### Storing data to file \"{scint_areas_file}\"...")
    data_utils.store_pickle(data=scint_areas, file=scint_areas_file)





if __name__ == "__main__":
    main()
    print(f"###### Done.")

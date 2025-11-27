#################################################################
### raw groups to scint areas
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
from tqdm import tqdm
from itertools import combinations

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------

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
        "--raw_scint_groups_file",
        type     = str,
        help     = "input file path: raw scintillator groups (pcl file)",
    )
    parser.add_argument(
        "--scint_areas_file",
        type     = str,
        help     = "output file path: scintillator areas i.e. pixels (pcl file)",
    )
    # ---
    args = parser.parse_args()
    raw_scint_hits_file = args.raw_scint_hits_file
    scint_areas_file = args.scint_areas_file
    raw_scint_groups_file = args.raw_scint_groups_file

    # coincidence strip isolation
    isolation_criterion = True

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    raw_scint_groups = data_utils.load_pickle(file=raw_scint_groups_file)
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)

    ### coincidence to scint hits (sipm coincidence for scintillator strip hits)
    scint_areas = scint_utils.raw_scint_groups_to_pixels(groups=raw_scint_groups, hits=raw_scint_hits, silent=False, isolation_criterion=isolation_criterion)
        
    ### sort sl_fit_groups by ts
    scint_areas = data_utils.sort_by_key(data=scint_areas, sort_key="ts")

    ### store to pcl file
    print(f"###### Storing scint areas to file \"{scint_areas_file}\"...")
    data_utils.store_pickle(data=scint_areas, file=scint_areas_file)




if __name__ == "__main__":
    main()
    print(f"###### Done.")





#################################################################
### generate simulated dt hits from cosmic muon tracks
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
        "--cosmic_muons_file",
        type     = str,
        help     = "input file path: cosmic_muons (pcl file)",
    )
    parser.add_argument(
        "--dt_hits_file",
        type     = str,
        help     = "output file path: dt hits (pcl file)",
    )
    # ---
    args = parser.parse_args()
    cosmic_muons_file = args.cosmic_muons_file
    dt_hits_file = args.dt_hits_file
    
    #################

    ### data import
    print(f"###### Importing cosmic muon tracks...")
    cosmic_muons = data_utils.load_pickle(file=cosmic_muons_file)
    n_muons = data_utils.length(cosmic_muons)

    ### muon propagation through dt chamber
    print(f"###### Propagating {n_muons} cosmic muons through DT chamber...")
    # determine dt hits from cosmic muons
    dt_hits = dt_utils.hits_from_muons(muons=cosmic_muons, noise_ampl=0)
    print("dt_hits =",dt_hits)
    n_dt_muon_hits = data_utils.length(dt_hits)
    print(f"Generated {n_dt_muon_hits} DT hits from muon tracks.")

    ### store to pcl file
    print(f"###### Storing DT hits to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

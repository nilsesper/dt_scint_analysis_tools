#################################################################
### generate simulated scint hits (strips) from cosmic muon tracks
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
        "--scint_hits_file",
        type     = str,
        help     = "output file path: scint hits (pcl file)",
    )
    # optional: store txt file with min and max ts of this dumpfile
    parser.add_argument(
        "--ts_range_file",
        type     = str,
        help     = "optional output file path: timestamp range (txt file)",
    )
    # ---
    args = parser.parse_args()
    cosmic_muons_file = args.cosmic_muons_file
    scint_hits_file = args.scint_hits_file
    create_ts_file = False
    if args.ts_range_file:
        create_ts_file = True
    
    #################

    ### data import
    print(f"###### Importing cosmic muon tracks...")
    cosmic_muons = data_utils.load_pickle(file=cosmic_muons_file)
    n_muons = data_utils.length(cosmic_muons)

    ### muon propagation through dt chamber
    print(f"###### Propagating {n_muons} cosmic muons through DT chamber...")
    # determine dt hits from cosmic muons
    scint_hits = scint_utils.hits_from_muons(muons=cosmic_muons)
    print("scint_hits =",scint_hits)
    n_scint_hits = data_utils.length(scint_hits)
    print(f"Generated {n_scint_hits} DT hits from muon tracks.")

    ### store to pcl file
    print(f"###### Storing DT hits to file \"{scint_hits_file}\"...")
    data_utils.store_pickle(data=scint_hits, file=scint_hits_file)

    ### optionally create ts file
    if create_ts_file:
        ts_min = np.amin(scint_hits["ts"])
        ts_max = np.amax(scint_hits["ts"])
        print(f"store ts range = [{ts_min}, {ts_max}] in file \"{args.ts_range_file}\".")
        ts_file_string = f"{int(ts_min)},{int(ts_max)}"
        with open(args.ts_range_file, 'w') as f:
            f.write(ts_file_string)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

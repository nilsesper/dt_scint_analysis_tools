#################################################################
### generate expected scintillator hits from reconstructed muons
# generate muon area objects from reco dt muon objects
# to later compare dt and scint hits (as muon area objects)
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
# data input files:
cosmic_muon_dt_muons_file = pcl_path+"/dummy_cosmic_muon_dt_muons.pcl"
# data output files:
cosmic_muon_dt_expected_scint_hits_file = pcl_path+"/cosmic_muon_dt_expected_scint_hits.pcl"
cosmic_muon_dt_expected_muon_areas_file = pcl_path+"/cosmic_muon_dt_expected_muon_areas.pcl"

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### data import
    print(f"###### Importing DT reco muons...")
    cosmic_muon_dt_muons = data_utils.load_pickle(file=cosmic_muon_dt_muons_file)
    n_muons = len(cosmic_muon_dt_muons["ts"])

    ### scint hit gen from dt reco muons
    print(f"###### Propagating {n_muons} DT reco muons through scintillator and storing hits to \"{cosmic_muon_dt_expected_muon_areas_file}\"...")
    # determine scint hits from dt reco muons
    cosmic_muon_dt_expected_scint_hits = scint_utils.hits_from_muons(muons=cosmic_muon_dt_muons)
    print("cosmic_muon_dt_expected_scint_hits =",cosmic_muon_dt_expected_scint_hits)
    # reco scint hits to muon areas
    cosmic_muon_dt_expected_muon_areas = scint_utils.reco_muon_area_from_hits(hits=cosmic_muon_dt_expected_scint_hits)
    print("cosmic_muon_dt_expected_muon_areas =",cosmic_muon_dt_expected_muon_areas)
    # store to pcl file
    data_utils.store_pickle(data=cosmic_muon_dt_expected_scint_hits, file=cosmic_muon_dt_expected_scint_hits_file)
    data_utils.store_pickle(data=cosmic_muon_dt_expected_muon_areas, file=cosmic_muon_dt_expected_muon_areas_file)




if __name__ == "__main__":
    main()
    print(f"###### Done.")

#################################################################
### generate dummy cosmic muons
# and propagate them through scintillator and dt chamber
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
# data input files:
# - none
# data output files:
cosmic_muon_file = pcl_path+"/dummy_cosmic_muons.pcl"
cosmic_muon_dt_hits_file = pcl_path+"/dummy_cosmic_muon_dt_hits.pcl"
cosmic_muon_scint_hits_file = pcl_path+"/dummy_cosmic_muon_scint_hits.pcl"

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ## muon gen setup
    n_muons = 100000 #100000 # no of muons to generate
    t_start = 10000 # timestamp of first muon
    t_step = 1000 # timestamp distance between muons
    # geometrical area where muons should be generated
    xyspacing = 100
    xrange = [ params._scintillator["pos"][0]-xyspacing , params._scintillator["pos"][0]+params._scintillator["size"][0]+xyspacing ]
    yrange = [ params._scintillator["pos"][1]-xyspacing , params._scintillator["pos"][1]+params._scintillator["size"][1]+xyspacing ]
    z0 = params._scintillator["pos"][2]
    phirange = [ 0 , 2*np.pi ]
    thetarange = [ 0 , np.pi/4 ]

    ### generate cosmic muons
    print(f"###### Generating {n_muons} cosmic muons and storing them to \"{cosmic_muon_file}\"...")
    # cosmic muon gen
    cosmic_muons = muon_utils.generate_cosmic_muons(n = n_muons, ts = t_start+t_step*np.arange(0,n_muons), xrange = xrange, yrange = yrange, z0 = z0, phirange = phirange, thetarange = thetarange)
    print("cosmic_muons =",cosmic_muons)
    # store to pcl file
    data_utils.store_pickle(data=cosmic_muons, file=cosmic_muon_file)

    ### dt hit gen
    print(f"###### Propagating {n_muons} cosmic muons through DT chamber and storing hits to \"{cosmic_muon_dt_hits_file}\"...")
    # determine dt hits from cosmic muons
    dt_cosmic_muon_hits = dt_utils.hits_from_muons(muons=cosmic_muons, noise_ampl=0)
    print("dt_cosmic_muon_hits =",dt_cosmic_muon_hits)
    # store to pcl file
    data_utils.store_pickle(data=dt_cosmic_muon_hits, file=cosmic_muon_dt_hits_file)

    ### scint hit gen
    print(f"###### Propagating {n_muons} cosmic muons through scintillator and storing hits to \"{cosmic_muon_scint_hits_file}\"...")
    # determine scint hits from cosmic muons
    scint_cosmic_muon_hits = scint_utils.hits_from_muons(muons=cosmic_muons)
    print("scint_cosmic_muon_hits =",scint_cosmic_muon_hits)
    # store to pcl file
    data_utils.store_pickle(data=scint_cosmic_muon_hits, file=cosmic_muon_scint_hits_file)





if __name__ == "__main__":
    main()
    print(f"###### Done.")

#################################################################
### generate muon objects (tracks) according to cosmic muon distribution
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
        help     = "output file path: cosmic_muons (pcl file)",
    )
    # ---
    args = parser.parse_args()
    cosmic_muons_file = args.cosmic_muons_file
    
    #################

    ###### cosmic muon / track generator setup
    t_sim = int(1000 / 0.78e-9) # in ts units, simulation runtime

    ### geometrical area where muons should be generated
    
    ## in scintillator
    # xyspacing = 100
    # xrange = [ params._scintillator["pos"][0]-xyspacing , params._scintillator["pos"][0]+params._scintillator["size"][0]+xyspacing ]
    # yrange = [ params._scintillator["pos"][1]-xyspacing , params._scintillator["pos"][1]+params._scintillator["size"][1]+xyspacing ]
    # z0 = params._scintillator["pos"][2]
    # phirange = [ 0 , 2*np.pi ]
    # thetarange = [ 0 , np.pi/4 ]
    
    ## in full dt chamber
    xyspacing = 1500 # mm (additional area of muon source beyond chamber coordinates)
    xrange = [ params._dt_chamber["pos"][0]-xyspacing , params._dt_chamber["pos"][0]+params._dt_chamber["size"][0]+xyspacing ]
    yrange = [ params._dt_chamber["pos"][1]-xyspacing , params._dt_chamber["pos"][1]+params._dt_chamber["size"][1]+xyspacing ]
    z0 = params._dt_chamber["pos"][2] # lowest point of chamber (closest to sl 1)
    phirange = [ 0 , 2*np.pi ]
    thetarange = [ 0 , np.pi/2 ]
    theta_weight = params.cosmic_muon_theta_weight
    #theta_weight = params.flat_theta_weight

    ### time distribution of muons
    t_start = 1000 # timestamp of first muon
    
    ## fixed time interval
    # t_step = 1000000 # timestamp distance between muons
    # n_muons = int(t_sim/t_step)
    # ts = t_start+t_step*np.arange(0,n_muons)
    
    ## poisson random
    muon_area = np.abs(xrange[1]-xrange[0]) * np.abs(yrange[1]-yrange[0]) * 1e-6 # m^2 (mm^2 to m^2 -> 100^(1+1+1) = 100^3 = 1e6)
    # vertical muon rate
    muon_ref_rate = 147 * 0.78e-9 # 1/(tu*m^2) ( 0.78e-9 Hz / m^2 = 1/(tu*m^2) )
    muon_rate = muon_area * muon_ref_rate # 1 / timestamp units
    muon_lambda = muon_rate * t_sim # expected muon count in simulation time
    # number of muons in time interval is poisson distributed
    n_muons = np.random.poisson(lam = muon_lambda)
    # simulate: for poisson, the time between events is exponentially distributed (!)
    inter_arrival_times = np.random.exponential(1.0 / muon_rate, n_muons) # in tu
    # generate muon timestamps from time differences between muon events
    ts = t_start +  np.cumsum(inter_arrival_times) # timestamp units

    ### generate cosmic muons
    print(f"###### Generating {n_muons} cosmic muon tracks over the time {t_sim*0.78e-9:.3f} s = {t_sim} TU...")
    # cosmic muon gen
    cosmic_muons = muon_utils.generate_cosmic_muons(n = n_muons, ts = ts, xrange = xrange, yrange = yrange, z0 = z0, phirange = phirange, thetarange = thetarange, theta_weight=theta_weight)
    print("cosmic_muons =",cosmic_muons)

    ### store to pcl file
    print(f"###### Storing cosmic muon tracks to file \"{cosmic_muons_file}\"...")
    data_utils.store_pickle(data=cosmic_muons, file=cosmic_muons_file)



if __name__ == "__main__":
    main()
    print(f"###### Done.")

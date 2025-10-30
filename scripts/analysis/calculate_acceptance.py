#################################################################
### calculate geometric acceptance from simulated flat muon generator and reco dt muons
# to later apply this correction to data
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
        help     = "input file path: simulated cosmic muons (pcl file)",
    )
    #parser.add_argument(
    #    "--dt_hits_file",
    #    type     = str,
    #    help     = "input file path: dt hits from this simulated cosmic muon dataset (pcl file)",
    #)
    parser.add_argument(
        "--dt_muons_file",
        type     = str,
        help     = "input file path: reco dt muons from this simulated cosmic muon dataset (pcl file)",
    )
    parser.add_argument(
        "--geom_acceptance_file",
        type     = str,
        help     = "output file path: calculated geom acceptance factors (pcl file)",
    )
    # ---
    args = parser.parse_args()
    cosmic_muons_file = args.cosmic_muons_file
    #dt_hits_file = args.dt_hits_file
    dt_muons_file = args.dt_muons_file
    geom_acceptance_file = args.geom_acceptance_file
    
    #################

    ### data import
    print(f"###### Importing data...")
    cosmic_muons = data_utils.load_pickle(file=cosmic_muons_file)
    #dt_hits = data_utils.load_pickle(file=dt_hits_file)
    dt_muons = data_utils.load_pickle(file=dt_muons_file)

    """
    ### restrict cosmic muons to the ones which do at least one hit in the chamber
    mask = np.isin(cosmic_muons["muon_id"], dt_hits["muon_id"])
    for k in cosmic_muons.keys():
        cosmic_muons[k] = cosmic_muons[k][mask]
    #"""
    
    """
    #### theta bins used
    theta_bins = np.linspace(0, np.pi/2, 100)

    ### create theta histograms
    cosmic_muons_theta_hists, cosmic_muons_theta_edges, cosmic_muons_theta_centers, cosmic_muons_theta_underflow, cosmic_muons_theta_overflow = hist_utils.calculate_hist(data=cosmic_muons, key="theta", bin_centers=theta_bins, silent=True)
    print(f"cosmic_muons: key \"theta\": entries={data_utils.length(cosmic_muons)} underflow={cosmic_muons_theta_underflow}, overflow={cosmic_muons_theta_overflow}")

    dt_muons_theta_hists, dt_muons_theta_edges, dt_muons_theta_centers, dt_muons_theta_underflow, dt_muons_theta_overflow = hist_utils.calculate_hist(data=dt_muons, key="theta", bin_centers=theta_bins, silent=True)
    print(f"dt_muons: key \"theta\": entries={data_utils.length(dt_muons)} underflow={dt_muons_theta_underflow}, overflow={dt_muons_theta_overflow}")

    ### calculate relative reco efficiency
    eff_theta = dt_muons_theta_hists / cosmic_muons_theta_hists
    err_eff_theta = np.sqrt( (1 / cosmic_muons_theta_hists)**2 * np.sqrt(dt_muons_theta_hists)**2 + (-dt_muons_theta_hists / cosmic_muons_theta_hists**2)**2 * np.sqrt(cosmic_muons_theta_hists)**2 )

    ### plot efficiency
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=eff_theta, where="mid", color="tab:blue")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=eff_theta-err_eff_theta, height=2*err_eff_theta, align="center", color="tab:blue", alpha=0.2)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_{\\text{muons, }\\geq\\text{ 1 hit}}$ / $N_\\text{reco muons}$")
    fig.tight_layout()
    fig.show()
    #"""

    ### calculate hist2d (theta, phi)
    n_theta_bins = 20
    theta_min = 0.1
    theta_max = 1.0
    n_phi_bins = 100
    theta_edges = np.linspace(theta_min, theta_max, n_theta_bins)
    phi_edges = np.linspace(0, 2*np.pi, n_phi_bins)
    theta_bins = np.array([(theta_edges[i]+theta_edges[i+1])/2 for i in range(len(theta_edges)-1)])
    phi_bins = np.array([(phi_edges[i]+phi_edges[i+1])/2 for i in range(len(phi_edges)-1)])
    
    ## for cosmic muons
    cosmic_muons_hist2d, _, _ = np.histogram2d(x=cosmic_muons["theta"], y=cosmic_muons["phi"], bins=(theta_edges, phi_edges))
    # solid angle correction
    for i in range(len(cosmic_muons_hist2d)):
        cosmic_muons_hist2d[i] = cosmic_muons_hist2d[i] / np.sin(theta_bins[i])
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=cosmic_muons_hist2d, origin="lower", extent=[min(phi_bins), max(phi_bins), min(theta_bins), max(theta_bins)])
    ax.set_title("$N_\\text{muons}$")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xlabel("$\\phi$ [rad]")
    #ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()
    
    ## for reco muons
    reco_muons_hist2d, _, _ = np.histogram2d(x=dt_muons["muon_theta"], y=dt_muons["muon_phi"], bins=(theta_edges, phi_edges))
    # solid angle correction
    for i in range(len(reco_muons_hist2d)):
        reco_muons_hist2d[i] = reco_muons_hist2d[i] / np.sin(theta_bins[i])
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=reco_muons_hist2d, origin="lower", extent=[min(phi_bins), max(phi_bins), min(theta_bins), max(theta_bins)])
    ax.set_title("$N_\\text{reco muons}$")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xlabel("$\\phi$ [rad]")
    #ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()

    ## 2d acceptance factors
    ratio_hist2d = reco_muons_hist2d / cosmic_muons_hist2d
    ratio_hist2d = ratio_hist2d / np.amax(ratio_hist2d) # normalize to 1
    err_ratio_hist2d = np.sqrt(
          (1 / cosmic_muons_hist2d)**2 * reco_muons_hist2d
        + (reco_muons_hist2d / cosmic_muons_hist2d**2)**2 * cosmic_muons_hist2d
    )
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=ratio_hist2d, origin="lower", extent=[min(phi_bins), max(phi_bins), min(theta_bins), max(theta_bins)])
    ax.set_title("$N_\\text{reco muons}$ / $N_\\text{muons}$ (normalized)")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xlabel("$\\phi$ [rad]")
    #ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()

    geom_acceptance = {
        "theta_edges": theta_edges,
        "theta_bins": theta_bins,
        "phi_edges": phi_edges,
        "phi_bins": phi_bins,
        "ratio_hist2d": ratio_hist2d,
        "err_ratio_hist2d": err_ratio_hist2d,
    }

    ### store to pcl file
    print(f"###### Storing calculated acceptance \"{geom_acceptance_file}\"...")
    data_utils.store_pickle(data=geom_acceptance, file=geom_acceptance_file)




    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

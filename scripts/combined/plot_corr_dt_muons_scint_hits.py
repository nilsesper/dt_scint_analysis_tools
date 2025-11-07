#################################################################
### analysis plots
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
        "--dt_muons_file",
        type     = str,
        help     = "input file path: dt reco muons (pcl file)",
    )
    parser.add_argument(
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scint hits (pcl file)",
    )
    parser.add_argument(
        "--corr_hits_file",
        type     = str,
        help     = "input file path: indices of correlated hits (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    
    # ---
    args = parser.parse_args()
    dt_muons_file = args.dt_muons_file
    scint_hits_file = args.scint_hits_file
    corr_hits_file = args.corr_hits_file
    verbose = False
    if args.verbose:
        verbose = True
        

    #################

    ts_tolerance = 1000 # in ts units

    ### data import
    print(f"###### Importing scint & dt data...")
    dt_muons = data_utils.load_pickle(file=dt_muons_file)
    #print(f"sl_fit_groups =",sl_fit_groups)
    scint_hits = data_utils.load_pickle(file=scint_hits_file)
    #print(f"scint_areas =",scint_areas)
    corr_list = copy.deepcopy(data_utils.load_pickle(file=corr_hits_file))

    # some basic things
    delta_ts_corr = []
    dt_idcs = []
    scint_idcs = []
    correlation_counter = 0
    for scint_idx, dt_idx in corr_list:
        correlation_counter += 1
        scint_ts = scint_hits["ts"][scint_idx]
        dt_ts = dt_muons["ts"][dt_idx]
        delta_ts_corr.append(np.float64(scint_ts) - np.float64(dt_ts))
        dt_idcs.append(dt_idx)
        scint_idcs.append(scint_idx)
    
    dt_idcs = np.array(dt_idcs, dtype=int)
    scint_idcs = np.array(scint_idcs, dtype=int)

    # collect corr hits
    dt_corr_muons = {}
    for k in dt_muons.keys():
        dt_corr_muons[k] = dt_muons[k][dt_idcs]
    scint_corr_hits = {}
    for k in scint_hits.keys():
        scint_corr_hits[k] = scint_hits[k][scint_idcs]


    ######## time difference between dt & scint correlated hits

    additional_data = {}
    print("Plotting time differences between dt muons...")
    k = f"scint_area_ts - dt_muon_ts"
    additional_data[k] = np.array(delta_ts_corr)
    # plot
    hist_bins = "step1" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=0, bin_labels=False, silent=True, store=False, show=True, title=f"", scale="norm")

    #### rates
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"duration = {duration} s")
    scint_rate = data_utils.length(scint_hits) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(dt_muons) / duration
    print(f"dt total rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")



    ####### geomplot corr muons

    # generate dt cell data
    dt_cell_data = dt_utils._chamber_data()
        
    # generate scintillator cell data
    scint_cell_data = scint_utils._scint_data()

     # actual plotting
    show_wires = True
    for orient in ["phi", "theta"]:
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot chamber geometry
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=show_wires)
        # plot scintillator geometry
        ax = geoplot_utils.scintillator_ax(ax=ax, orient=orient, cell_data=scint_cell_data)
        # plot reconstructed muon
        for scint_idx, dt_idx in corr_list:
            ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=dt_muons, muon_idx=dt_idx, color="red" )
        # # plot reconstructed muon scint hits
        # for i in range(n_reco_muons):
        #     ax = geoplot_utils.scint_hits_ax(ax=ax, orient=orient, scint_hits=scint_reco_muon_hits, muon_id=i, color="tab:blue")
        # # plot reconstructed reconstructed muon area in scintillator
        # ax = geoplot_utils.scint_muon_area_ax(ax=ax, orient=orient, scint_muon_areas=reco_muon_areas, muon_id=i, color="red")
        # axis limits
        ax.margins(x=0.05, y=0.05)
        # text labels
        ax.legend()
        #ax.set_aspect('equal', adjustable='box')
        axbox = ax.get_position()
        x_topleft = axbox.p0[0]
        x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
        ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
        ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
        #description = params._dt_chamber["name"]
        description = ""
        if orient == "theta":
            description += "$y$-$z$-plane (SL-$\\theta$ view)"
            ax.set_xlabel("$y$ [mm]")
            ax.set_ylabel("$z$ [mm]")
        elif orient == "phi":
            description += "$x$-$z$-plane (SL-$\\phi$ view)"
            ax.set_xlabel("$x$ [mm]")
            ax.set_ylabel("$z$ [mm]")
        ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
        # show/store figure
        fig.show()


    ####### 2d projections

    xy_marigin = 400
    n_xy_bins = 60
    x_edges = np.linspace(derived_params.scint_x_min-xy_marigin, derived_params.scint_x_max+xy_marigin, n_xy_bins)
    y_edges = np.linspace(derived_params.scint_y_min-xy_marigin, derived_params.scint_y_max+xy_marigin, n_xy_bins)
    x_bins = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(len(x_edges)-1)])
    y_bins = np.array([(y_edges[i]+y_edges[i+1])/2 for i in range(len(y_edges)-1)])
    x_binwidth = x_edges[1]-x_edges[0]
    y_binwidth = y_edges[1]-y_edges[0]

    ### corr muon x,y position plot (for z = mean_scint_z)

    # project muons onto scintillator z pos
    dt_corr_muons_scint = muon_utils.change_muon_base_point(muons=dt_corr_muons, z_new=derived_params.scint_z_center)

    pos_corr_muons_hist2d, _, _ = np.histogram2d(x=dt_corr_muons_scint["y0"], y=dt_corr_muons_scint["x0"], bins=(y_edges, x_edges))
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=pos_corr_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)])
    # draw scint into plot
    patches = []
    patches.append( pat.Rectangle(
        (derived_params.scint_x_min, derived_params.scint_y_min),
        width=(derived_params.scint_x_max - derived_params.scint_x_min),
        height=(derived_params.scint_y_max - derived_params.scint_y_min),
        edgecolor="white", facecolor="None",
        label="Scintillator position")
    )
    for patch in patches:
        ax.add_patch(patch)
    # plot setup
    ax.set_title(f"$N_\\text{{DT muons, correlated}}$ ($z={np.round(derived_params.scint_z_center,0):.0f}$mm)")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()

    ### all muon x,y position plot (for z = mean_scint_z)

    # project muons onto scintillator z pos
    dt_muons_scint = muon_utils.change_muon_base_point(muons=dt_muons, z_new=derived_params.scint_z_center)

    pos_dt_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_scint["y0"], y=dt_muons_scint["x0"], bins=(y_edges, x_edges))
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=pos_dt_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)])
    # draw scint into plot
    patches = []
    patches.append( pat.Rectangle(
        (derived_params.scint_x_min, derived_params.scint_y_min),
        width=(derived_params.scint_x_max - derived_params.scint_x_min),
        height=(derived_params.scint_y_max - derived_params.scint_y_min),
        edgecolor="white", facecolor="None",
        label="Scintillator position")
    )
    for patch in patches:
        ax.add_patch(patch)
    # plot setup
    ax.set_title(f"$N_\\text{{DT muons}}$ ($z={np.round(derived_params.scint_z_center,0):.0f}$mm)")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()















    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

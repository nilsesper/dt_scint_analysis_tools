#################################################################
### correlate dt muons and scint areas in time
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
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
        "--scint_areas_file",
        type     = str,
        help     = "input file path: scint areas (pcl file)",
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
    scint_areas_file = args.scint_areas_file
    verbose = False
    if args.verbose:
        verbose = True

    #################

    ts_tolerance = 1000 # in ts units

    ### data import
    print(f"###### Importing scint & dt data...")
    dt_muons = data_utils.load_pickle(file=dt_muons_file)
    #print(f"sl_fit_groups =",sl_fit_groups)
    scint_areas = data_utils.load_pickle(file=scint_areas_file)
    #print(f"scint_areas =",scint_areas)

    ### temporal correlation
    time_grouping_list = combination_utils.time_grouping_indices(data1=scint_areas, data2=dt_muons, data2_ts_tolerance=ts_tolerance, data1_ts_key="ts", data2_ts_key="ts")

    correlation_counter = 0
    delta_ts_corr = []

    corr_list = [] # list of (scint_area_idx, dt_muon_idx)

    for scint_idx, dt_idcs in enumerate(time_grouping_list):

        if len(dt_idcs) < 1:
            continue

        correlation_counter += 1

        scint_ts = scint_areas["ts"][scint_idx]
        scint_pixel = scint_areas["pixel"][scint_idx]
        print(f"scint_idx = {scint_idx} -- scint_ts = {scint_ts} -- pixel = {scint_pixel}")
        
        dt_idx = dt_idcs[0]

        dt_ts = dt_muons["ts"][dt_idx]
        dt_theta = dt_muons["theta"][dt_idx]
        dt_phi = dt_muons["phi"][dt_idx]
        print(f"   dt_ts = {dt_ts} -- dt_theta = {dt_theta} -- dt_phi = {dt_phi}")

        delta_ts_corr.append(np.float64(scint_ts) - np.float64(dt_ts))
        corr_list.append((scint_idx, dt_idx))
    
    #### time difference between dt & scint correlated hits
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
    scint_rate = data_utils.length(scint_areas) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(dt_muons) / duration
    print(f"dt total sl fit group rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")




    ### geomplot corr muons

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






    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

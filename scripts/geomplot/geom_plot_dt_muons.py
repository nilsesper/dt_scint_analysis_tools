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
        "--sl_fits_file",
        type     = str,
        help     = "input file path: dt sl fits hits (pcl file)",
    )
    parser.add_argument(
        "--sl_fit_groups_file",
        type     = str,
        help     = "input file path: dt sl fits hits (pcl file)",
    )
    parser.add_argument(
        "--dt_muons_file",
        type     = str,
        help     = "input file path: dt reco muons (pcl file)",
    )
    # plotting / store plot
    parser.add_argument(
        "--show_plots",
        action = "store_true",
        help     = "show plots flag",
    )
    parser.add_argument(
        "--store_plots",
        type     = str,
        help     = "output directory: give argument if plots should be stores, specify output path for plots here",
    )
    parser.add_argument(
        "--simulation",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--dt_muon_idcs",
        type     = str,
        help     = "idcs of dt muons groups to plot \"index1,index2,...\"",
    )
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    sl_fit_groups_file = args.sl_fit_groups_file
    dt_muons_file = args.dt_muons_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True
    dt_muon_idcs = []
    if args.dt_muon_idcs:
        dt_muon_idcs = args.dt_muon_idcs.split(",")
        dt_muon_idcs = np.array(dt_muon_idcs, dtype=int)

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    sl_fit_groups = data_utils.load_pickle(file=sl_fit_groups_file)
    dt_muons = data_utils.load_pickle(file=dt_muons_file)

    # ### select indices
    # if len(sl_fit_group_idcs) > 0:
    #     sl_fit_groups = data_utils.slice_data(data=sl_fit_groups, slice_indices=sl_fit_group_idcs)

    n_dt_muons = data_utils.length(data=dt_muons)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"measurement duration = {duration} s")



    ### plot full geometry

    # generate dt cell data
    dt_cell_data = dt_utils._chamber_data()
    ## illustrate patterns that should be recognized
    #for i, (pat_name, pat_rel_wi) in enumerate(params._dt_sl_patterns.items()):
    #    start_wi = 6*i+3
    #    for ly in range(4):
    #        cell_data[1][ly][start_wi+pat_rel_wi[ly]]["color"] = "tab:red"
    # mark dt hits in chamber
    for i in range(n_dt_muons):
        if len(dt_muon_idcs) > 0:
            if i not in dt_muon_idcs:
                continue
        sl_fit_group_idcs = [dt_muons[f"sl{sl}_fit_group"][i] for sl in range(1,4)]
        for j in sl_fit_group_idcs:
            sl = sl_fit_groups["sl"][j]
            sl_fit_idcs = sl_fit_groups["idcs"][j]
            for k in sl_fit_idcs:
                for ly in range(4):
                    wi = sl_fits[f"wi{ly}"][k]
                    dt_cell_data[sl][ly][wi]["color"] = "aqua"

    #dt_cell_data[1][0][0]["color"] = "red"

    # actual plotting
    show_wires = True
    for orient in ["phi", "theta"]:
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot chamber geometry
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=show_wires)
        # # plot muon simulated track
        # for i in range(n_muons):
        #     ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=cosmic_muons, muon_id=i, color="tab:green")
        # # plot individual simulated muon dt hits
        # for i in range(n_muons):
        #     ax = geoplot_utils.cell_hits_ax(ax=ax, orient=orient, dt_hits=dt_muon_hits, muon_id=i, color="tab:green")
        # plot individual simulated muon dt sl fits
        for i in range(n_dt_muons):
            if len(dt_muon_idcs) > 0:
                if i not in dt_muon_idcs:
                    continue
            sl_fit_group_idcs = [dt_muons[f"sl{sl}_fit_group"][i] for sl in range(1,4)]
            for j in sl_fit_group_idcs:
                sl = sl_fit_groups["sl"][j]
                sl_fit_idcs = sl_fit_groups["idcs"][j]
                for k in sl_fit_idcs:
                    ax = geoplot_utils.chamber_muon_fit_ax(ax=ax, orient=orient, sl_dt_fits=sl_fits, pattern_idx=k, color=derived_params.color_wheel(j), label=f"muon={int(i)} fit_group={int(j)} fit={int(k)} sl={int(sl)} t0={int(sl_fits['t0'][k])} tan={sl_fits['tan_alpha'][k]:.3f}")
        # plot reconstructed muon
        for i in range(n_dt_muons):
            if len(dt_muon_idcs) > 0:
                if i not in dt_muon_idcs:
                    continue
            ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=dt_muons, muon_idx=i, color="black", label=f"muon={int(i)} ts={int(dt_muons['ts'][i])} theta={dt_muons['theta'][i]:.3f}={dt_muons['theta'][i]*180/np.pi:.1f}° phi={dt_muons['phi'][i]:.3f}={dt_muons['phi'][i]*180/np.pi:.1f}°")
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

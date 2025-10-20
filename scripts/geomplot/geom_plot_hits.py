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
        "--dt_hits_file",
        type     = str,
        help     = "input file path: dt hits (pcl file)",
    )
    parser.add_argument(
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scint hits (pcl file)",
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
        "--dt_cuts",
        type     = str,
        help     = "cuts to apply to dt in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    parser.add_argument(
        "--scint_cuts",
        type     = str,
        help     = "cuts to apply to scint in format \"key1,operator1,value1;key2,operator2,value;...\"",
    )
    # ---
    args = parser.parse_args()
    dt_hits_file = args.dt_hits_file
    scint_hits_file = args.scint_hits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True
    dt_cuts_list = []
    if args.dt_cuts:
        for cuts_str in args.dt_cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            dt_cuts_list.append((key, operator, value))
    scint_cuts_list = []
    if args.scint_cuts:
        for cuts_str in args.scint_hits.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            scint_cuts_list.append((key, operator, value))

    #################

    n_dt_hits = 0
    n_scint_hits = 0

    ### data import
    print(f"###### Importing all data...")
    # dt
    if args.dt_hits_file:
        dt_hits = data_utils.load_pickle(file=dt_hits_file)
    # scint
    if args.scint_hits_file:
        scint_hits = data_utils.load_pickle(file=scint_hits_file)
    
    ### cut dt data
    if args.dt_hits_file:
        print(f"###### Applying dt cuts: {dt_cuts_list}...")
        dt_hits = data_utils.cut_data(data=dt_hits, conditions=dt_cuts_list)
        n_dt_hits = data_utils.length(data=dt_hits)
    if args.scint_hits_file:
        print(f"###### Applying scint cuts: {scint_cuts_list}...")
        scint_hits_file = data_utils.cut_data(data=scint_hits, conditions=scint_cuts_list)
        n_scint_hits = data_utils.length(data=scint_hits)


    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_hits["ts"]) - np.amin(dt_hits["ts"])) # secs
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
    for i in range(n_dt_hits):
        sl, ly, wi = dt_hits["sl"][i], dt_hits["ly"][i], dt_hits["wi"][i]
        dt_cell_data[sl][ly][wi]["color"] = "aqua"
        
    # generate scintillator cell data
    scint_cell_data = scint_utils._scint_data()
    # mark scint hits in chamber
    for i in range(n_scint_hits):
        ly, st = scint_hits["ly"][i], scint_hits["st"][i]
        scint_cell_data[ly][st]["color"] = "aqua"
            
    # actual plotting
    show_wires = True
    for orient in ["phi", "theta"]:
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot chamber geometry
        if args.dt_hits_file:
            ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=show_wires)
        # plot scintillator geometry
        if args.scint_hits_file:
            ax = geoplot_utils.scintillator_ax(ax=ax, orient=orient, cell_data=scint_cell_data)
        # # plot muon simulated track
        # for i in range(n_muons):
        #     ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=cosmic_muons, muon_id=i, color="tab:green")
        # # plot individual simulated muon dt hits
        # for i in range(n_muons):
        #     ax = geoplot_utils.cell_hits_ax(ax=ax, orient=orient, dt_hits=dt_muon_hits, muon_id=i, color="tab:green")
        # # plot individual simulated muon dt sl fits
        # for i in range(n_patterns):
        #     ax = geoplot_utils.chamber_muon_fit_ax(ax=ax, orient=orient, sl_dt_fits=sl_dt_fits, pattern_id=i, color="red")
        # # plot reconstructed muon
        # for i in range(n_reco_muons):
        #     ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=reco_muons, muon_id=i, color="tab:blue")
        # # plot reconstructed muon scint hits
        # for i in range(n_reco_muons):
        #     ax = geoplot_utils.scint_hits_ax(ax=ax, orient=orient, scint_hits=scint_reco_muon_hits, muon_id=i, color="tab:blue")
        # # plot reconstructed reconstructed muon area in scintillator
        # ax = geoplot_utils.scint_muon_area_ax(ax=ax, orient=orient, scint_muon_areas=reco_muon_areas, muon_id=i, color="red")
        # axis limits
        ax.margins(x=0.05, y=0.05)
        # text labels
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

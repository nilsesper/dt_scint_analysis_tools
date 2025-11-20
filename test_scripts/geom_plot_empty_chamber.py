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


    #################

    ### plot full geometry

    # generate dt cell data
    dt_cell_data = dt_utils._chamber_data()
    dt_cell_data[1][0][1]["color"] = "aqua"
    dt_cell_data[1][1][1]["color"] = "aqua"
        
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

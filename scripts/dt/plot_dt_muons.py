#################################################################
### plot dt muon reco tracks
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import matplotlib.patches as mpatches

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_muons_file",
        type     = str,
        help     = "input file path: dt muons (pcl file)",
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
    # -
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
    )
    parser.add_argument(
        "--store_path",
        type     = str,
        help     = "path to store pdf plot (if desired)",
    )
    # ---
    args = parser.parse_args()
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
    # other 
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))

    ### manual input of low occupancy wires to be shown in plot
    low_occ_wires = { # sl: (ly, wi)
        1: [  ] + [ (1,49), ],
        2: [ (0,2), (0,5), (0,13), (0,35), (0,50), (0,51), (1,4), (1,5), (1,44), (1,45), ] + [ (1,57), ],
        3: [ (0,10), (1,41), (2,26), ] + [ (1,49), ],
    }

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_muons = data_utils.load_pickle(file=dt_muons_file)

    n_dt_muons = data_utils.length(dt_muons)

    ### dt hits
    """
    print(f"### dt muons")
    n_hist_bins = 100
    hist_bins = {
        "ts": "auto200",
        "x0": "auto200",
        "y0": "auto200",
        "z0": "auto200",
        "phi": "auto200",
        "theta": "auto200",
        "err_ts": "auto200",
        "err_x0": "auto200",
        "err_y0": "auto200",
        "err_z0": "auto200",
        "err_phi": "auto200",
        "err_theta": "auto200",
    }
    if simulation:
        hist_bins |= {
            "muon_id": "auto200",
        }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dt_muons, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(dt_muons)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/dt_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
        if k == "theta":
            hists_solidangle = hists / np.sin(centers)
            ylabel = "Counts / $\\text{sin}\\theta$"
            hist_utils.plot_1hist(hist=hists_solidangle, centers=centers, xlabel=xlabel, ylabel=ylabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    #"""
            
    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"measurement duration = {duration} s")

    ### rate of muons
    muon_count = data_utils.length(dt_muons)
    pattern_rate = muon_count / duration
    print(f"dt muon rate: {pattern_rate:.03f} Hz")

    
    """
    #### time difference between dt muons
    additional_data = {}
    print("Plotting time differences between dt muons...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_dt_muons)
    for i in range(1,n_dt_muons):
        additional_data[k][i] = int(dt_muons[f"ts"][i]) - int(dt_muons["ts"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e4,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    """
    if simulation:
        ### simulation muon difference
        print(f"### sl fit difference to simulation")
        additional_data = {}
        hist_bins = {
            ("ts", "muon_ts"): "step1",
            ("phi", "muon_phi"): "auto200",
            ("theta", "muon_theta"): "auto200",
        }
        for k1,k2 in hist_bins.keys():
            # calculate
            k = f"{k1} - {k2}"
            additional_data[k] = np.zeros(n_dt_muons)
            for i in range(n_dt_muons):
                additional_data[k][i] = dt_muons[k1][i] - dt_muons[k2][i]
                if False and k1 == "theta" and additional_data[k][i] < -0.5:
                    print( dt_muons[k1][i], dt_muons[k2][i])
            # plot
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins[(k1,k2)], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(dt_muons)} underflow={underflow}, overflow={overflow}")
            if len(hists) == 0: continue
            round_digits = 0 if k in ["ts"] else 2
            xlabel = k
            plotname = False
            if store_plots != None: 
                plotname = store_plots+f"/sl_fits_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, scale="log") # scale="log"
    #"""


    ####### X-Y 2d projections
    #"""

    ### for center of each superlayer separately
    for sl in [1,2,3]:

        # plot range
        x_margin = 100
        y_margin = 100
        x_bin_width = 10 # mm
        z_bin_width = 10 # mm
        x_edges = np.arange(start=derived_params.sl_x_min[sl]-x_margin, stop=derived_params.sl_x_max[sl]+x_margin, step=x_bin_width)
        y_edges = np.arange(start=derived_params.sl_y_min[sl]-y_margin, stop=derived_params.sl_y_max[sl]+y_margin, step=z_bin_width)
        x_bins = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(len(x_edges)-1)])
        y_bins = np.array([(y_edges[i]+y_edges[i+1])/2 for i in range(len(y_edges)-1)])
        x_binwidth = x_edges[1]-x_edges[0]
        y_binwidth = y_edges[1]-y_edges[0]
        
        ### muon x,y position plot (for z = mean_scint_z)

        # project muons onto scintillator z pos
        dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=derived_params.sl_z_center[sl])

        pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
        # plot
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)])
        # draw sl into plot
        patches = []
        patches.append( pat.Rectangle(
            (derived_params.sl_x_min[sl], derived_params.sl_y_min[sl]),
            width=(derived_params.sl_x_max[sl]-derived_params.sl_x_min[sl]),
            height=(derived_params.sl_y_max[sl]-derived_params.sl_y_min[sl]),
            edgecolor="white", facecolor="None",
            label="Superlayer position")
        )
        # draw low occ wire positions into plot
        first_label = True
        for ly, wi in low_occ_wires[sl]:
            # derived_params._dt_cell_coordinates = {sl: {ly: {wi: [[xmin, xmax], [ymin, ymax], [zmin, zmax], x_center_pos, y_center_pos, z_center_pos]}}}
            if first_label:
                patches.append( pat.Rectangle(
                    (derived_params._dt_cell_coordinates[sl][ly][wi][0][0], derived_params._dt_cell_coordinates[sl][ly][wi][1][0]),
                    width=(derived_params._dt_cell_coordinates[sl][ly][wi][0][1]-derived_params._dt_cell_coordinates[sl][ly][wi][0][0]),
                    height=(derived_params._dt_cell_coordinates[sl][ly][wi][1][1]-derived_params._dt_cell_coordinates[sl][ly][wi][1][0]),
                    edgecolor="red", facecolor="None",
                    label="Low-occupancy cells")
                )
                first_label = False
            else:
                patches.append( pat.Rectangle(
                    (derived_params._dt_cell_coordinates[sl][ly][wi][0][0], derived_params._dt_cell_coordinates[sl][ly][wi][1][0]),
                    width=(derived_params._dt_cell_coordinates[sl][ly][wi][0][1]-derived_params._dt_cell_coordinates[sl][ly][wi][0][0]),
                    height=(derived_params._dt_cell_coordinates[sl][ly][wi][1][1]-derived_params._dt_cell_coordinates[sl][ly][wi][1][0]),
                    edgecolor="red", facecolor="None",
                    )
                )        
        for patch in patches:
            ax.add_patch(patch)
        # plot setup
        ax.set_title(f"DT tracks in SL {sl} ($z={np.round(derived_params.sl_z_center[sl],0):.0f}$mm)", fontsize=20)
        ax.set_ylabel("$y$ [mm]")
        ax.set_xlabel("$x$ [mm]")
        ax.legend(prop={"size":14}, loc="upper center")
        plt.colorbar(im_obj)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"DT_MUON_SPECIFIC_xy_SL{sl}.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
    #"""

    ####### X-Z and Y-Z 2d projections
    #"""
    for orient in ["phi", "theta"]:

        # orientation
        slice_name = None
        if orient == "phi":
            slice_name = "xz"
        elif orient == "theta":
            slice_name = "yz"

        # chamber coordinates
        sl_z_coord = (np.amin([derived_params.sl_z_min[sl] for sl in [1,2,3]]), np.amax([derived_params.sl_z_max[sl] for sl in [1,2,3]]))
        if slice_name == "xz":
            sl_x_coord = (np.amin([derived_params.sl_x_min[sl] for sl in [1,2,3]]), np.amax([derived_params.sl_x_max[sl] for sl in [1,2,3]]))
        elif slice_name == "yz":
            sl_x_coord = (np.amin([derived_params.sl_y_min[sl] for sl in [1,2,3]]), np.amax([derived_params.sl_y_max[sl] for sl in [1,2,3]]))

        # plot range
        x_margin = 100
        z_margin = 100
        x_bin_width = 5 # mm
        z_bin_width = 5 # mm
        x_edges = np.arange(start=sl_x_coord[0]-x_margin, stop=sl_x_coord[1]+x_margin, step=x_bin_width)
        z_edges = np.arange(start=sl_z_coord[0]-z_margin, stop=sl_z_coord[1]+z_margin, step=z_bin_width)
        x_bins = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(len(x_edges)-1)])
        z_bins = np.array([(z_edges[i]+z_edges[i+1])/2 for i in range(len(z_edges)-1)])
        x_binwidth = x_edges[1]-x_edges[0]
        z_binwidth = z_edges[1]-z_edges[0]
        
        ### muon X-Z or Y-Z position plot

        # project muons onto different z pos
        x_muons = []
        z_muons = []
        for i, z_pos in enumerate(z_bins):
            dt_muons_moved = muon_utils.change_muon_base_point(muons=dt_muons, z_new=z_pos)
            if slice_name == "xz":
                x_muons.extend(dt_muons_moved["x0"])
            elif slice_name == "yz":
                x_muons.extend(dt_muons_moved["y0"])
            z_muons.extend(dt_muons_moved["z0"])
        x_muons = np.array(x_muons)
        z_muons = np.array(z_muons)

        pos_muons_hist2d, _, _ = np.histogram2d(x=z_muons, y=x_muons, bins=(z_edges, x_edges)) 

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12,3)) # fig_size
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(z_bins), max(z_bins)])
        
        # store dead wires
        dead_dt_cell_data = dt_utils._chamber_data()
        for sl in [1,2,3]:
            for ly, wi in low_occ_wires[sl]:
                dead_dt_cell_data[sl][ly][wi]["color"] = "tab:red"
        
        # plot chamber geometry
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dead_dt_cell_data, wire=False, transparent=True)

        # plot setup
        ax.set_title(f"DT tracks", fontsize=20)
        ax.set_ylabel("$z$ [mm]")
        if slice_name == "xz":
            ax.set_xlabel("$x$ [mm]")
        elif slice_name == "yz":
            ax.set_xlabel("$y$ [mm]")
        plt.colorbar(im_obj)
        # plot legend
        #ax.legend(prop={"size":14}, loc="upper center")
        legend_entries = {
            "Chamber geometry": mpatches.Patch(edgecolor="white", facecolor="none"),
            "Low occupancy cells": mpatches.Patch(edgecolor="tab:red", facecolor="none")
        }
        ax.legend(legend_entries.values(), legend_entries.keys(), prop={'size': 14}, loc="upper center")
        # show plot
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"DT_MUON_SPECIFIC_"+slice_name+"_CHAMBER.pdf"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
    #"""

    """
    ### for ceiling of teststand
    z_above_chamber = 1500 # mm
    z_pos_to_plot = derived_params.dt_chamber_z_max + z_above_chamber

    # plot range
    xy_marigin = 400
    n_xy_bins = 60
    x_edges = np.linspace(derived_params.dt_chamber_x_min-xy_marigin, derived_params.dt_chamber_x_max+xy_marigin, n_xy_bins)
    y_edges = np.linspace(derived_params.dt_chamber_y_min-xy_marigin, derived_params.dt_chamber_y_max+xy_marigin, n_xy_bins)
    x_bins = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(len(x_edges)-1)])
    y_bins = np.array([(y_edges[i]+y_edges[i+1])/2 for i in range(len(y_edges)-1)])
    x_binwidth = x_edges[1]-x_edges[0]
    y_binwidth = y_edges[1]-y_edges[0]
    
    ### muon x,y position plot (for z = mean_scint_z)

    # project muons onto scintillator z pos
    dt_muons_proj = muon_utils.change_muon_base_point(muons=dt_muons, z_new=z_pos_to_plot)

    pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_proj["y0"], y=dt_muons_proj["x0"], bins=(y_edges, x_edges))
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)])
    # draw sl into plot
    patches = []
    patches.append( pat.Rectangle(
        (derived_params.sl_x_min[sl], derived_params.sl_y_min[sl]),
        width=(derived_params.sl_x_max[sl]-derived_params.sl_x_min[sl]),
        height=(derived_params.sl_y_max[sl]-derived_params.sl_y_min[sl]),
        edgecolor="white", facecolor="None",
        label="Chamber position")
    )
    for patch in patches:
        ax.add_patch(patch)
    # plot setup
    ax.set_title(f"DT muons ${np.round(z_above_chamber,0):.0f}$mm above chamber ($z={np.round(z_pos_to_plot,0):.0f}$mm)")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()
    #"""






    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

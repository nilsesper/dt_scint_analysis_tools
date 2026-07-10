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
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16}) #'font.sans-serif': 'Arial',
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
    parser.add_argument(
        "--corr_hits_file",
        type     = str,
        help     = "input file path: indices of correlated areas (pcl file)",
    )
    ###
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--store_path",
        type     = str,
        help     = "path to store pdf plot (if desired)",
    )
    parser.add_argument(
        "--also_plot_unmatched",
        action   = "store_true",
        help     = "print info",
    )
    # ---
    args = parser.parse_args()
    dt_muons_file = args.dt_muons_file
    scint_areas_file = args.scint_areas_file
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
    scint_areas = data_utils.load_pickle(file=scint_areas_file)
    #print(f"scint_areas =",scint_areas)
    corr_list = copy.deepcopy(data_utils.load_pickle(file=corr_hits_file))

    # some basic things
    delta_ts_corr = []
    #err_delta_ts_corr = []
    dt_idcs = []
    scint_idcs = []
    correlation_counter = 0
    for scint_idx, dt_idx in corr_list:
        correlation_counter += 1
        scint_ts = scint_areas["ts"][scint_idx]
        #err_scint_ts = scint_areas["err_ts"][scint_idx]
        dt_ts = dt_muons["ts"][dt_idx]
        err_dt_ts = dt_muons["err_ts"][dt_idx]
        delta_ts_corr.append(np.float64(scint_ts) - np.float64(dt_ts))
        #err_delta_ts_corr.append(np.sqrt(err_scint_ts**2 + err_dt_ts**2))
        dt_idcs.append(dt_idx)
        scint_idcs.append(scint_idx)
    delta_ts_corr = np.array(delta_ts_corr)
    #err_delta_ts_corr = np.array(err_delta_ts_corr)
    
    dt_idcs = np.array(dt_idcs, dtype=int)
    scint_idcs = np.array(scint_idcs, dtype=int)

    # collect corr hits
    dt_corr_muons = {}
    for k in dt_muons.keys():
        dt_corr_muons[k] = dt_muons[k][dt_idcs]
    scint_corr_hits = {}
    for k in scint_areas.keys():
        scint_corr_hits[k] = scint_areas[k][scint_idcs]

    ### manual input of low occupancy wires to be shown in plot
    # dt chamber
    low_occ_wires = { # sl: (ly, wi)
        1: [  ] + [ (1,49), ],
        2: [ (0,2), (0,5), (0,13), (0,35), (0,50), (0,51), (1,4), (1,5), (1,44), (1,45), ] + [ (1,57), ],
        3: [ (0,10), (1,41), (2,26), ] + [ (1,49), ],
    }
    # scintillator
    low_occ_strips = [ # (ly, st)
        
    ]





    #### rates
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"duration = {duration} s")
    scint_rate = data_utils.length(scint_areas) / duration
    print(f"scintillator rate = {scint_rate} Hz")
    dt_rate = data_utils.length(dt_muons) / duration
    print(f"dt total rate = {dt_rate} Hz")
    correlation_rate = correlation_counter / duration
    print(f"correlation rate = {correlation_rate} Hz")



    ####### geomplot corr muons

    """
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
    #"""

    ####### 2d X-Y projections

    xy_marigin = 200
    x_bin_width = 10 # mm
    y_bin_width = 10 # mm
    x_edges = np.arange(start=derived_params.scint_x_min-xy_marigin, stop=derived_params.scint_x_max+xy_marigin, step=x_bin_width)
    y_edges = np.arange(start=derived_params.scint_y_min-xy_marigin, stop=derived_params.scint_y_max+xy_marigin, step=y_bin_width)
    x_bins = np.array([(x_edges[i]+x_edges[i+1])/2 for i in range(len(x_edges)-1)])
    y_bins = np.array([(y_edges[i]+y_edges[i+1])/2 for i in range(len(y_edges)-1)])
    x_binwidth = x_edges[1]-x_edges[0]
    y_binwidth = y_edges[1]-y_edges[0]

    ### corr muon x,y position plot (for z = mean_scint_z)

    # project muons onto scintillator z pos
    dt_corr_muons_scint = muon_utils.change_muon_base_point(muons=dt_corr_muons, z_new=derived_params.scint_z_center)

    pos_corr_muons_hist2d, _, _ = np.histogram2d(x=dt_corr_muons_scint["y0"], y=dt_corr_muons_scint["x0"], bins=(y_edges, x_edges))
    ## convert to rate
    #pos_corr_muons_hist2d /= duration
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
    ax.set_title(f"$N_\\text{{DT tracks, correlated}}$ ($z={np.round(derived_params.scint_z_center,0):.0f}$mm)")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.legend()
    cmap = plt.get_cmap('viridis')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3]) # 10^X power limits for prescale
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap, format=formatter)
    #cbar.set_label("Hz")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_xy.png"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    #### all muon x,y position plot (for z = mean_scint_z)

    ## project muons onto scintillator z pos
    #dt_muons_scint = muon_utils.change_muon_base_point(muons=dt_muons, z_new=derived_params.scint_z_center)

    #pos_dt_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_scint["y0"], y=dt_muons_scint["x0"], bins=(y_edges, x_edges))
    ### convert to rate
    ##pos_dt_muons_hist2d /= duration
    ## plot
    #fig, ax = plt.subplots(1, 1, figsize=(12,8))
    #im_obj = ax.imshow(X=pos_dt_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)])
    ## draw scint into plot
    #patches = []
    #patches.append( pat.Rectangle(
    #    (derived_params.scint_x_min, derived_params.scint_y_min),
    #    width=(derived_params.scint_x_max - derived_params.scint_x_min),
    #    height=(derived_params.scint_y_max - derived_params.scint_y_min),
    #    edgecolor="white", facecolor="None",
    #    label="Scintillator position")
    #)
    #for patch in patches:
    #    ax.add_patch(patch)
    ## plot setup
    #ax.set_title(f"$N_\\text{{DT tracks}}$ ($z={np.round(derived_params.scint_z_center,0):.0f}$mm)")
    #ax.set_ylabel("$y$ [mm]")
    #ax.set_xlabel("$x$ [mm]")
    #ax.legend()
    #cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05)
    #cbar.set_label("Hz")
    #fig.tight_layout()
    #fig.show()


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
        sl_z_coord = (np.amin([derived_params.sl_z_min[sl] for sl in [1,2,3]] + [derived_params.scint_z_min]), np.amax([derived_params.sl_z_max[sl] for sl in [1,2,3]] + [derived_params.scint_z_max]))
        if slice_name == "xz":
            sl_x_coord = (np.amin([derived_params.sl_x_min[sl] for sl in [1,2,3]] + [derived_params.scint_x_min]), np.amax([derived_params.sl_x_max[sl] for sl in [1,2,3]] + [derived_params.scint_x_max]))
        elif slice_name == "yz":
            sl_x_coord = (np.amin([derived_params.sl_y_min[sl] for sl in [1,2,3]] + [derived_params.scint_y_min]), np.amax([derived_params.sl_y_max[sl] for sl in [1,2,3]] + [derived_params.scint_y_max]))

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
            dt_muons_moved = muon_utils.change_muon_base_point(muons=dt_corr_muons_scint, z_new=z_pos)
            if slice_name == "xz":
                x_muons.extend(dt_muons_moved["x0"])
            elif slice_name == "yz":
                x_muons.extend(dt_muons_moved["y0"])
            z_muons.extend(dt_muons_moved["z0"])
        x_muons = np.array(x_muons)
        z_muons = np.array(z_muons)

        pos_muons_hist2d, _, _ = np.histogram2d(x=z_muons, y=x_muons, bins=(z_edges, x_edges)) 

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12,6)) # fig_size
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(z_bins), max(z_bins)])

        # store dead wires
        dead_dt_cell_data = dt_utils._chamber_data()
        for sl in [1,2,3]:
            for ly, wi in low_occ_wires[sl]:
                dead_dt_cell_data[sl][ly][wi]["color"] = "tab:red"
        
        # plot chamber geometry
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dead_dt_cell_data, wire=False, transparent=True)

        # store dead strips
        dead_scint_cell_data = scint_utils._scint_data()
        for ly, st in low_occ_strips:
            dead_scint_cell_data[ly][st]["color"] = "tab:red"

        # plot scintillator geometry
        ax = geoplot_utils.scintillator_ax(ax=ax, orient=orient, cell_data=dead_scint_cell_data, transparent=True)

        # plot setup
        ax.set_title(f"Correlated DT tracks", fontsize=20)
        ax.set_ylabel("$z$ [mm]")
        if slice_name == "xz":
            ax.set_xlabel("$x$ [mm]")
        elif slice_name == "yz":
            ax.set_xlabel("$y$ [mm]")
        cmap = plt.get_cmap('viridis')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits([-3, 3]) # 10^X power limits for prescale
        cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap, format=formatter)
        # plot legend
        #ax.legend(prop={"size":14}, loc="upper center")
        legend_entries = {
            "Detector geometry": mpatches.Patch(edgecolor="white", facecolor="none"),
            "Low occupancy channels": mpatches.Patch(edgecolor="tab:red", facecolor="none")
        }
        ax.legend(legend_entries.values(), legend_entries.keys(), prop={'size': 14}, loc="center left")
        # show plot
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_{slice_name}.png"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
    #"""

    ######################
    ### MATCHED MUONS: ARRIVAL TIMES

    ts = dt_corr_muons["ts"]
    # calculate hist
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=f"auto,50", data_min_val=np.amin(ts), data_max_val=np.amax(ts))
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts, edges=edges)
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    # tu to ns
    centers = centers*0.78
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=False)
    xlabel = "$T_\\text{DT} \\text{(matched)}$ [ns]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_MUON_TS.png"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ######################
    ### MATCHED MUONS: TIME DIFFERENCE OF ARRIVAL TIMES

    # calculate ts difference of consecutive muons
    dt_corr_muons = timestamp_utils.sort_by_timestamp(hits=dt_corr_muons, silent=True)
    n_dt_corr_muons = data_utils.length(dt_corr_muons)
    ts_diff_list = []
    for i in range(1,n_dt_corr_muons):
        ts_diff_list.append(dt_corr_muons["ts"][i] - dt_corr_muons["ts"][i-1])
    ts_diff = np.array(ts_diff_list)
    # calculate hist
    binnings = [ # (binning name, binning arg)
        ( "fullrange", f"linear,0,{np.amax(ts_diff)},100" ),
        ( "closeup", f"linear,0,10000,100" ),
    ]
    for binning_name, binning_arg in binnings:
        edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=binning_arg)
        hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=ts_diff, edges=edges)
        err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
        # tu to ns
        centers = centers*0.78
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
        ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True)
        xlabel = "$\\Delta T_\\text{DT} \\text{(matched)}$ [ns]"
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.show()
        ## store plot
        if args.store_path:
            hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_MUON_DELTA-TS_{binning_name}.png"
            print(f"store histogram plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)

    ######################
    ### MATCHED MUONS AND SCINT HITS: TIME DIFFERENCE
    # delta_ts_corr = ts_scint - ts_dt

    # calculate hist
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=f"linear,{np.amin(delta_ts_corr)},{np.amax(delta_ts_corr)},100")
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.calculate_histogram_and_shifted_histograms(data=delta_ts_corr, edges=edges)
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    # tu to ns
    centers = centers*0.78
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist=err_hist, log_scale=True)
    xlabel = "$T_\\text{scint} \\text{(matched)} - T_\\text{DT} \\text{(matched)}$ [ns]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_CORR_DELTA-TS.png"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)





    ################################################

    ####### X-Z and Y-Z 2d projections
    ### SAME PLOT BUT BEFORE MATCHING
    #"""
    if args.also_plot_unmatched:
        for orient in ["phi", "theta"]:

            # orientation
            slice_name = None
            if orient == "phi":
                slice_name = "xz"
            elif orient == "theta":
                slice_name = "yz"

            # chamber coordinates
            sl_z_coord = (np.amin([derived_params.sl_z_min[sl] for sl in [1,2,3]] + [derived_params.scint_z_min]), np.amax([derived_params.sl_z_max[sl] for sl in [1,2,3]] + [derived_params.scint_z_max]))
            if slice_name == "xz":
                sl_x_coord = (np.amin([derived_params.sl_x_min[sl] for sl in [1,2,3]] + [derived_params.scint_x_min]), np.amax([derived_params.sl_x_max[sl] for sl in [1,2,3]] + [derived_params.scint_x_max]))
            elif slice_name == "yz":
                sl_x_coord = (np.amin([derived_params.sl_y_min[sl] for sl in [1,2,3]] + [derived_params.scint_y_min]), np.amax([derived_params.sl_y_max[sl] for sl in [1,2,3]] + [derived_params.scint_y_max]))

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
            fig, ax = plt.subplots(1, 1, figsize=(12,6)) # fig_size
            im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[min(x_bins), max(x_bins), min(z_bins), max(z_bins)])

            # store dead wires
            dead_dt_cell_data = dt_utils._chamber_data()
            for sl in [1,2,3]:
                for ly, wi in low_occ_wires[sl]:
                    dead_dt_cell_data[sl][ly][wi]["color"] = "tab:red"
            
            # plot chamber geometry
            ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dead_dt_cell_data, wire=False, transparent=True)

            # store dead strips
            dead_scint_cell_data = scint_utils._scint_data()
            for ly, st in low_occ_strips:
                dead_scint_cell_data[ly][st]["color"] = "tab:red"

            # plot scintillator geometry
            ax = geoplot_utils.scintillator_ax(ax=ax, orient=orient, cell_data=dead_scint_cell_data, transparent=True)

            # plot setup
            ax.set_title(f"All DT tracks", fontsize=20)
            ax.set_ylabel("$z$ [mm]")
            if slice_name == "xz":
                ax.set_xlabel("$x$ [mm]")
            elif slice_name == "yz":
                ax.set_xlabel("$y$ [mm]")
            cmap = plt.get_cmap('viridis')
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits([-3, 3]) # 10^X power limits for prescale
            cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap, format=formatter)
            # plot legend
            #ax.legend(prop={"size":14}, loc="upper center")
            legend_entries = {
                "Detector geometry": mpatches.Patch(edgecolor="white", facecolor="none"),
                "Low occupancy channels": mpatches.Patch(edgecolor="tab:red", facecolor="none")
            }
            ax.legend(legend_entries.values(), legend_entries.keys(), prop={'size': 14}, loc="center left")
            # show plot
            fig.tight_layout()
            fig.show()
            ## store plot
            if args.store_path:
                hist_plot_file = args.store_path+"/"+f"CORRELATED_AREAS_SPECIFIC_{slice_name}_BEFOREMATCHING.png"
                print(f"store histogram plot as {hist_plot_file}.")
                fig.savefig(hist_plot_file)
    #"""













    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

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
        help     = "input file path: sl fits (pcl file)",
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
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots
    simulation = False
    if args.simulation:
        simulation = True

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    
    # do cut if desired
    #sl_fits = data_utils.cut_data(data=sl_fits, conditions=[("pat_type","in",[0])])
    #sl_fits = data_utils.cut_data(data=sl_fits, conditions=[("laterality","in",[1])])
    sl_fits = data_utils.cut_data(data=sl_fits, conditions=[
        ("chi2/ndf","<",10),
        ("x0","<=",21), ("x0",">=",-21),
        #("tan_alpha","<=",2), ("tan_alpha",">=",-2),
        ("dt0",">=",0), ("dt0","<=",params._dt_max_drift_time),
        ("dt1",">=",0), ("dt1","<=",params._dt_max_drift_time),
        ("dt2",">=",0), ("dt2","<=",params._dt_max_drift_time),
        ("dt3",">=",0), ("dt3","<=",params._dt_max_drift_time),
    ])

    n_sl_fits = data_utils.length(sl_fits)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"])) # secs
    print(f"measurement duration = {duration} s")

    ### sl fits
    print(f"### sl fits")
    hist_bins = {
        "sl": np.arange(1, 3+1),
        "pat_type": "step1",
        "laterality": np.arange(0, 6+1),
        "t0": "auto200",
        "wi3": np.arange(0, 60+1),
        "x0": "auto200",
        "tan_alpha": "auto200",
        "chi2/ndf": "auto1000", #np.arange(0,1000+1),
        "dt0": "auto200",
        "dt1": "auto200",
        "dt2": "auto200",
        "dt3": "auto200",
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_fits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
        if len(hists) == 0: continue
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/sl_fits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    ### fitted drift times
    additional_data = {}
    k = f"dt"
    additional_data[k] = np.zeros(n_sl_fits*4)
    for ly in range(4):
        for i in range(n_sl_fits):    
            additional_data[k][i+ly*n_sl_fits] = sl_fits[f"dt{ly}"][i]
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = params._key_symbols[k]+"$(\\text{DT})$"
    xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    """
    ### plots of superlayers & layers
    for sl in range(1,4):
        hist_bins = {
            "laterality": np.arange(0, 6+1),
            "t0": "auto200",
            "wi3": np.arange(0, 80+1),
            "x0": "auto200",
            "tan_alpha": "auto200",
            "chi2/ndf": "auto200",
            "theta_proj": "auto200",
        }
        sl_fits_cut = data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl), ]) # ("chi2/ndf","<",1000)
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_fits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(sl_fits_cut)} underflow={underflow}, overflow={overflow}")
            if len(hists) == 0: continue
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{DT})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/sl_fits_sl{sl}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"sl {sl}") # scale="log"
    #"""

    """
    ### plots of some sl fits
    # plot all fitted patterns
    for pattern_id in range(n_sl_fits):
        if pattern_id not in [500, 600, 700, 800]:
            continue
        ### print sl fit info
        print(f"SL FIT ID {pattern_id}:")
        for k in sl_fits.keys():
            print(f"  {k}: {sl_fits[k][pattern_id]}")
        ### plot sl pattern
        show_wires = True
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot sl pattern
        ax = geoplot_utils.sl_fit_ax(ax, sl_dt_fits=sl_fits, pattern_id=pattern_id, wire=show_wires)
        # # plot originally simulated muon
        # ax = geoplot_utils.sl_muon_proj_ax(ax, muons=cosmic_muons, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id, color="tab:green")
        # # plot dt hits of simulated muon
        # ax = geoplot_utils.sl_dt_hits_proj_ax(ax, dt_hits=dt_muon_hits, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id, color="tab:green", other_lat=True)
        # plot fitted muon
        ax = geoplot_utils.sl_muon_fit_ax(ax, sl_dt_fits=sl_fits, pattern_id=pattern_id)
        # axis limits
        ax.margins(x=0.05, y=0.05)
        # text labels
        axbox = ax.get_position()
        x_topleft = axbox.p0[0]
        x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
        ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
        ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
        description = f"Pattern #{pattern_id}, SL {sl_fits["sl"][pattern_id]}"
        ax.set_xlabel("$x_\\text{rel}$ [mm]")
        ax.set_ylabel("$z_\\text{rel}$ [mm]")
        ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
        # show/store figure
        fig.show()
    #"""

    #"""
    ### rate of fits
    for sl in range(1,4):
        sl_patterns_cut = data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl)], silent=True)
        pattern_count = data_utils.length(sl_patterns_cut)
        pattern_rate = pattern_count / duration
        print(f"sl={sl} sl fit rate: {pattern_rate:.03f} Hz")
    #"""


    #"""
    if simulation:
        ### simulation keys
        print(f"### sl pattern / fit simulation keys")
        hist_bins = {
            "muon_ts": "auto200",
            "muon_lat_id": "step1",
            "muon_x0": "auto200",
            "muon_tan_alpha": "auto200",
            "muon_id": "auto200",
            "muon_dt0": "auto200",
            "muon_dt1": "auto200",
            "muon_dt2": "auto200",
            "muon_dt3": "auto200",
        }
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=sl_fits, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{DT})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/sl_patterns_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
        ## simulated muon drift times
        additional_data = {}
        k = f"dt"
        additional_data[k] = np.zeros(n_sl_fits*4)
        for ly in range(4):
            for i in range(n_sl_fits):    
                additional_data[k][i+ly*n_sl_fits] = sl_fits[f"muon_dt{ly}"][i]
        hist_bins = "auto200"
        # plot
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
        print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"


        ### simulation fit difference
        print(f"### sl fit difference to simulation")
        additional_data = {}
        hist_bins = {
            ("t0", "muon_ts"): "auto200",
            ("x0", "muon_x0"): "auto200",
            ("tan_alpha", "muon_tan_alpha"): "auto200",
        }
        for k1,k2 in hist_bins.keys():
            # calculate
            k = f"{k1} - {k2}"
            additional_data[k] = np.zeros(n_sl_fits*4)
            for i in range(n_sl_fits):
                if k1 == "t0":
                    additional_data[k][i+ly*n_sl_fits] = int(sl_fits[k1][i]) - int(sl_fits[k2][i])
                else:
                    additional_data[k][i+ly*n_sl_fits] = sl_fits[k1][i] - sl_fits[k2][i]
            # plot
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins[(k1,k2)], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
            if len(hists) == 0: continue
            round_digits = 0 if k in ["ts"] else 2
            xlabel = k
            plotname = False
            if store_plots != None: 
                plotname = store_plots+f"/sl_fits_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    #"""
    

    """
    # meantimer testing
    additional_data = {}

    # Tmuon = 1/4 T3 + 1/4 T1 + 1/2 T2 - 1/2 tmax
    k = f"meantimer(123)_t0 - t0_muon"
    additional_data[k] = 1/4*np.float64(sl_fits[f"ts3"]) + 1/4*np.float64(sl_fits[f"ts1"]) + 1/2*np.float64(sl_fits[f"ts2"]) - 1/2*params._dt_max_drift_time - sl_fits[f"muon_ts"]
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    # Tmuon = 1/4 T2 + 1/4 T0 + 1/2 T1 - 1/2 tmax
    k = f"meantimer(012)_t0 - t0_muon"
    additional_data[k] = 1/4*np.float64(sl_fits[f"ts2"]) + 1/4*np.float64(sl_fits[f"ts0"]) + 1/2*np.float64(sl_fits[f"ts1"]) - 1/2*params._dt_max_drift_time - sl_fits[f"muon_ts"]
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

    k = f"meantimer(123)_t0 - meantimer(012)_t0"
    additional_data[k] = ( 1/4*np.float64(sl_fits[f"ts3"]) + 1/4*np.float64(sl_fits[f"ts1"]) + 1/2*np.float64(sl_fits[f"ts2"]) - 1/2*params._dt_max_drift_time ) - ( 1/4*np.float64(sl_fits[f"ts2"]) + 1/4*np.float64(sl_fits[f"ts0"]) + 1/2*np.float64(sl_fits[f"ts1"]) - 1/2*params._dt_max_drift_time )
    hist_bins = "auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(sl_fits)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    #"""


    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

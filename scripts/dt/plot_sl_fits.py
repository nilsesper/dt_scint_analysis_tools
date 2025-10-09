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
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    
    # do cut if desired
    #sl_fits = data_utils.cut_data(data=sl_fits, conditions=[("pat_type","in",[0,1])])
    #sl_fits = data_utils.cut_data(data=sl_fits, conditions=[("chi2/ndf","<",100), ]) #("sl","==",2)

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
        "theta_proj": "auto200",
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

    ### fitted drift times
    additional_data = {}
    k = f"td"
    additional_data[k] = np.zeros(n_sl_fits*4)
    for ly in range(4):
        for i in range(n_sl_fits):    
            additional_data[k][i+ly*n_sl_fits] = int(sl_fits[f"ts{ly}"][i]) - sl_fits[f"t0"][i]
    hist_bins = np.arange(0,2000) #"auto200"
    # plot
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"

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


    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

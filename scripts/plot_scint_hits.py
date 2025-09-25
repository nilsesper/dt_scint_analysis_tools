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
        "--scint_hits_file",
        type     = str,
        help     = "input file path: scintillator hits (pcl file)",
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
    scint_hits_file = args.scint_hits_file
    show_plots = False
    if args.show_plots:
        show_plots = True
    store_plots = None
    if args.store_plots:
        store_plots = args.store_plots

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    scint_hits = data_utils.load_pickle(file=scint_hits_file)

    ### scintillator hits
    print(f"### scintillator hits")
    n_hist_bins = 100
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
        "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
        "ly": np.arange(0, 1+1),
        "st": np.arange(0, 16+1),
        "ts": "auto200",
        "sipm_delta_ts": np.linspace(0, 1000, 500), #"auto1000",
        "st_delta_last_ts0": "auto1000",
        "st_delta_last_ts1": "auto1000",
        "st_delta_last_ts": "auto1000", #np.arange(0, 1000+1),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(scint_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/scint_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"
    
    """
    ## separate for both scintillator layers
    for ly in [0,1]:
        n_hist_bins = 100
        hist_bins = {
            "ro_ch": np.arange(0, 32),
            "ch": np.arange(0, 255),
            "tdc": np.arange(0, params._lhc_tdc_count+1),
            "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
            "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
            "ly": np.arange(0, 1+1),
            "st": np.arange(0, 16+1),
            "ts": "auto200",
        }
        scint_hits_cut = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly)])
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(scint_hits_cut)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{scint})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/scint_hits_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"Layer {ly}")
    """
            
    """
    ### for each channel separately
    for ly in [0,1]:
        for st in range(8):
            print(f"### scintillator hits ly{ly} st{st}")
            n_hist_bins = 100
            hist_bins = {
                #"ts": "auto200",
                #"sipm_delta_ts": "step1",
                #"st_delta_last_ts0": np.arange(0, 1000+1),
                #"st_delta_last_ts1": np.arange(0, 1000+1),
                "st_delta_last_ts": np.linspace(0, 1000, 100), # "auto5000",
            }
            cut_scint_hits = data_utils.cut_data(data=scint_hits, conditions=[("ly","==",ly), ("st","==",st)])
            for k in hist_bins.keys():
                hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cut_scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
                print(f"key \"{k}\": entries={data_utils.length(scint_hits)} underflow={underflow}, overflow={overflow}")
                round_digits = 0 if k in ["ts"] else 2
                xlabel = params._key_symbols[k]+"$(\\text{scint})$"
                xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
                plotname = False
                if store_plots != None:
                    plotname = store_plots+f"/scint_hits_{k}.png"
                hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"ly{ly} st{st}")
    """

    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

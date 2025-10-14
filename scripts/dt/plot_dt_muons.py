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

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_muons = data_utils.load_pickle(file=dt_muons_file)

    # cuts
    dt_muons = data_utils.cut_data(data=dt_muons, conditions=[
        #("theta",">",0.5),
        ("muon_theta","<",np.pi/2), ("muon_theta",">",-np.pi/2),
    ])

    n_dt_muons = data_utils.length(dt_muons)

    ### dt hits
    print(f"### dt muons")
    n_hist_bins = 100
    hist_bins = {
        "ts": "auto200",
        "x0": "auto200",
        "y0": "auto200",
        "z0": "auto200",
        "phi": "auto200",
        "theta": "auto200",
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
    
    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"measurement duration = {duration} s")

    ### rate of muons
    muon_count = data_utils.length(dt_muons)
    pattern_rate = muon_count / duration
    print(f"dt muon rate: {pattern_rate:.03f} Hz")

    
    #"""
    #### time difference between dt muons
    additional_data = {}
    print("Plotting time differences between dt muons...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_dt_muons)
    for i in range(1,n_dt_muons):
        additional_data[k][i] = int(dt_muons[f"ts"][i]) - int(dt_muons["ts"][i-1]) 
    # plot
    hist_bins = "auto500" #np.linspace(0,1e6,500) #"auto500" 
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"delta_ts [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""

    #"""
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



    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

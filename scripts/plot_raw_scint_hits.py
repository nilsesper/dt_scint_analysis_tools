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
        "--raw_scint_hits_file",
        type     = str,
        help     = "input file path: raw scintillator hits (pcl file)",
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
    raw_scint_hits_file = args.raw_scint_hits_file
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
    raw_scint_hits = data_utils.load_pickle(file=raw_scint_hits_file)

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
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=raw_scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(raw_scint_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/scint_raw_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots)
    
    """
    ## separate for both scintillator layers
    ly_ch_list = { # hardcoded
        0: list(range(0, 16)),
        1: list(range(16, 32))
    }
    for ly in [0,1]:
        n_hist_bins = 100
        hist_bins = {
            "ro_ch": np.arange(0, 32),
            "ch": np.arange(0, 255),
            "tdc": np.arange(0, params._lhc_tdc_count+1),
            "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
            "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
            "ts": "auto200",
        }
        dumpfile_hits_cut = data_utils.cut_data(data=dumpfile_hits, conditions=[("ch","in",ly_ch_list[ly])])
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=dumpfile_hits_cut, key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(dumpfile_hits_cut)} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]+"$(\\text{scint})$"
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            plotname = False
            if store_plots != None:
                plotname = store_plots+f"/scint_raw_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"Layer {ly}")
    """
            
    ## estimate rates
    duration = 0.78e-9 * (np.amax(raw_scint_hits["ts"]) - np.amin(raw_scint_hits["ts"])) # secs
    ch_hit_list = np.array([
        data_utils.length( data_utils.cut_data(data=raw_scint_hits, conditions=[("ch","==",ch)], silent=True ) ) for ch in range(32)
    ])
    ch_rate_list = ch_hit_list / duration # hz
    print("channel rates in Hz:")
    for ch in range(32):
        print(f"  ch {ch:2d}: {ch_rate_list[ch]:6.2f} Hz")

    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

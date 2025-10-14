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
        "--scint_areas_file",
        type     = str,
        help     = "input file path: scintillator areas (pcl file)",
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
    scint_areas_file = args.scint_areas_file
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
    scint_areas = data_utils.load_pickle(file=scint_areas_file)

    ### scint reco muon areas
    print(f"### scint reco muon areas")
    n_hist_bins = 100
    hist_bins = {
        "xmin": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "xmax": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "ymin": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "ymax": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "z0": np.linspace(params._scintillator["pos"][2]-10, params._scintillator["pos"][2]+params._scintillator["size"][2]+10, n_hist_bins),
        "ts": "auto200",
        "pixel": np.arange(0, 255+1),
        "ly_delta_ts": np.linspace(0,40,20), #np.linspace(0, 1000, 500) #"auto100",
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=scint_areas, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(scint_areas)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/scint_reco_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots)
        
        ### 2d plots of pixels
        if k == "pixel":
            # occupancy
            px_matrix = np.zeros((16, 16))
            fig, ax = plt.subplots(1, 1, figsize=(10,8))
            for st0 in range(16):
                for st1 in range(16):
                    px = derived_params._scint_pixel_mapping[(st0, st1)]
                    px_matrix[st0][st1] = hists[px]
            imshow_obj = ax.imshow(px_matrix)
            ax.set_xlabel("Strip (Layer 1)")
            ax.set_ylabel("Strip (Layer 0)")
            cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
            fig.tight_layout()
            if show_plots:
                fig.show()
            # rate (in hits / min)
            duration = 0.78e-9 * (np.amax(scint_areas["ts"]) - np.amin(scint_areas["ts"])) # secs
            px_matrix = np.zeros((16, 16))
            fig, ax = plt.subplots(1, 1, figsize=(10,8))
            for st0 in range(16):
                for st1 in range(16):
                    px = derived_params._scint_pixel_mapping[(st0, st1)]
                    px_matrix[st0][st1] = hists[px] / duration
            imshow_obj = ax.imshow(px_matrix)
            ax.set_xlabel("Strip (Layer 1)")
            ax.set_ylabel("Strip (Layer 0)")
            cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
            cbar.set_label("Hz")
            fig.tight_layout()
            if show_plots:
                fig.show()

    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

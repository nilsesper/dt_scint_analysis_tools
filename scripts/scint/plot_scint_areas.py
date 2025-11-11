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
from tqdm import tqdm

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
    parser.add_argument(
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1;key2,operator2,value;...\"",
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
    cuts_list = []
    if args.cuts:
        for cuts_str in args.cuts.split(";"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            cuts_list.append((key, operator, value))

    #################

    ### data import
    print(f"###### Importing all data...")
    # scint
    scint_areas = data_utils.load_pickle(file=scint_areas_file)

    #cuts_list.append(("pixel","in",
    #    #[  8 +i for i in range(0,8)] +
    #    #[ 24 +i for i in range(0,8)] +
    #    #[ 40 +i for i in range(0,8)] +
    #    #[ 56 +i for i in range(0,8)] +
    #    #[ 72 +i for i in range(0,8)] +
    #    #[ 88 +i for i in range(0,8)] +
    #    #[104 +i for i in range(0,8)] +
    #    #[120 +i for i in range(0,8)] +
    #
    #    #[128 +i for i in range(0,8)] +
    #    #[144 +i for i in range(0,8)] +
    #    #[160 +i for i in range(0,8)] +
    #    #[176 +i for i in range(0,8)] +
    #    #[192 +i for i in range(0,8)] +
    #    #[208 +i for i in range(0,8)] +
    #    #[224 +i for i in range(0,8)] +
    #    #[240 +i for i in range(0,8)] +
    #
    #    #[  0 +i for i in range(0,8)] +
    #    #[ 16 +i for i in range(0,8)] +
    #    #[ 32 +i for i in range(0,8)] +
    #    #[ 48 +i for i in range(0,8)] +
    #    #[ 64 +i for i in range(0,8)] +
    #    #[ 80 +i for i in range(0,8)] +
    #    #[ 96 +i for i in range(0,8)] +
    #    #[112 +i for i in range(0,8)]
    #
    #    #[136 +i for i in range(0,8)] +
    #    #[152 +i for i in range(0,8)] +
    #    #[168 +i for i in range(0,8)] +
    #    #[184 +i for i in range(0,8)] +
    #    #[200 +i for i in range(0,8)] +
    #    #[216 +i for i in range(0,8)] +
    #    #[232 +i for i in range(0,8)] +
    #    #[248 +i for i in range(0,8)]
    #))

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    scint_areas = data_utils.cut_data(data=scint_areas, conditions=cuts_list)
    n_scint_areas = data_utils.length(scint_areas)

    ### measurement duration
    duration = 0.78e-9 * (np.amax(scint_areas["ts"]) - np.amin(scint_areas["ts"])) # secs
    print(f"measurement duration = {duration} s")

    ## global rate
    global_rate = n_scint_areas/duration
    print(f"all pixels together rate: {global_rate} Hz")


    ### scint reco muon areas
    print(f"### scint reco muon areas")
    n_hist_bins = 100
    hist_bins = {
        #"xmin": np.linspace(params._scintillator["pos"][0]+10, params._scintillator["pos"][0]-(params._scintillator["size"][0]+10), n_hist_bins),
        #"xmax": np.linspace(params._scintillator["pos"][0]+10, params._scintillator["pos"][0]-(params._scintillator["size"][0]+10), n_hist_bins),
        #"ymin": np.linspace(params._scintillator["pos"][1]+10, params._scintillator["pos"][1]-(params._scintillator["size"][1]+10), n_hist_bins),
        #"ymax": np.linspace(params._scintillator["pos"][1]+10, params._scintillator["pos"][1]-(params._scintillator["size"][1]+10), n_hist_bins),
        #"z0": np.linspace(params._scintillator["pos"][2]-10, params._scintillator["pos"][2]+params._scintillator["size"][2]+10, n_hist_bins),
        "ts": "auto200",
        "pixel": np.arange(0, 255+1),
        "ly_delta_ts": "step1", #"step1", #np.linspace(0, 1000, 500) #"auto100",
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
            ax.invert_yaxis()
            cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
            fig.tight_layout()
            if show_plots:
                fig.show()
            # rate (in hits / min)
            px_matrix = np.zeros((16, 16))
            fig, ax = plt.subplots(1, 1, figsize=(10,8))
            for st0 in range(16):
                for st1 in range(16):
                    px = derived_params._scint_pixel_mapping[(st0, st1)]
                    px_matrix[st0][st1] = hists[px] / duration
            imshow_obj = ax.imshow(px_matrix)
            ax.set_xlabel("Strip (Layer 1)")
            ax.set_ylabel("Strip (Layer 0)")
            ax.invert_yaxis()
            cbar = fig.colorbar(imshow_obj, ax=ax, fraction=0.05)
            cbar.set_label("Hz")
            fig.tight_layout()
            if show_plots:
                fig.show()


    #"""
    #### time difference between scint areas
    additional_data = {}
    print("Plotting time differences between scint areas...")
    k = f"delta_ts"
    additional_data[k] = np.zeros(n_scint_areas)
    for i in range(1,n_scint_areas):
        additional_data[k][i] = int(scint_areas[f"ts"][i]) - int(scint_areas["ts"][i-1]) 
    # plot
    hist_bins = np.linspace(0,1e3,500) #"auto500" 
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

    #"""
    #### time difference between hits of same channel
    print("Plotting time differences between hits of same pixel...")
    k = f"delta_ts_same_st"
    ch_list = []
    # time difference between hits
    i_offset = 0
    for pixel in tqdm(range(0, 256)):
        scint_areas_cut = data_utils.cut_data(data=scint_areas, conditions=[("pixel","==",pixel)], silent=True)
        scint_areas_cut = timestamp_utils.sort_by_timestamp(hits=scint_areas_cut, silent=True)
        n_scint_areas_cut = data_utils.length(scint_areas_cut)
        sub_list = {k: []}
        for i in range(1,n_scint_areas_cut):
            sub_list[k].append( int(scint_areas_cut[f"ts"][i]) - int(scint_areas_cut["ts"][i-1]) )
        sub_list[k] = np.array(sub_list[k])
        ch_list.append(sub_list)
    additional_data = data_utils.merge_dataset(split_data=ch_list, silent=True)

    # plot

    hist_bins = "auto500" #np.linspace(0, 1e4, 1000) #"auto200"
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"

    hist_bins = np.linspace(0, 5e3, 500)
    hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=additional_data, key=k, bin_centers=hist_bins, silent=True)
    print(f"key \"{k}\": entries={data_utils.length(additional_data)} underflow={underflow}, overflow={overflow}")
    xlabel = f"{k} [TU]"
    hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots, title=f"", scale="log") # scale="log"
    #"""








    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

#################################################################
### inspect dumpfile
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
        "--input_dumpfile",
        type     = str,
        help     = "input file path: data to be sliced (pcl file)",
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
    input_dumpfile = args.input_dumpfile
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
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename

    ### cut data
    print(f"###### Applying data cuts: {cuts_list}...")
    hits = data_utils.cut_data(data=hits, conditions=cuts_list)

    n_hits = data_utils.length(hits)

    ### plot ro_ch in order of dumpfile
    hit_idcs = np.arange(0, n_hits)
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(hit_idcs, hits["ro_ch"], marker=".", ms=3)
    fig.tight_layout()
    fig.show()


    ### plot dumpfile content
    print(f"### dumpfile content")
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count+1),
        "oc": "auto200", #"step1", #np.linspace(0, params._lhc_orbit_count, n_hist_bins),
    }
    for k in hist_bins.keys():
        if k not in hits.keys(): continue
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(hits)} underflow={underflow}, overflow={overflow}")
        if len(hists) == 0: continue
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname = False
        if store_plots != None:
            plotname = store_plots+f"/dumpfile_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname, show=show_plots) # scale="log"




    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

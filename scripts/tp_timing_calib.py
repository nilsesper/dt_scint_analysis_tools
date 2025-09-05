#################################################################
### generate expected scintillator hits from reconstructed muons
# generate muon area objects from reco dt muon objects
# to later compare dt and scint hits (as muon area objects)
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
dumpfile_path = REPO_PATH+"/dumpfiles"
# input files:
tp_dumpfile_name = dumpfile_path+"/sipm_testpulses.txt"
# data output files:
# --

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### constants
    n_chs = 32

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    tp_dumpfile_hits = data_utils.import_raw(file_name=tp_dumpfile_name) # dummy_filename, data_filename
    print("tp_dumpfile_hits =",tp_dumpfile_hits)

    ### assign hit timestamps
    tp_dumpfile_hits = timestamp_utils.add_timestamp(hits=tp_dumpfile_hits)
    ### calculate ts difference to orbit start
    tp_dumpfile_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_dumpfile_hits)

    ### split testpulses by channel
    tp_ch = []
    for ch in range(n_chs):
        tp_ch.append( data_utils.cut_data(data=tp_dumpfile_hits, conditions=[("ch","==",ch)]) )
        hist_utils

    ### plot testpulse timing
    print(f"### tp timing plots")
    for ch in range(n_chs):
        hist_bins = {
            "ts_orbit": "step1",
        }
        for k in hist_bins.keys():
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=tp_ch[ch], key=k, bin_centers=hist_bins[k], silent=True)
            print(f"key \"{k}\": entries={data_utils.length(tp_ch[ch])} underflow={underflow}, overflow={overflow}")
            round_digits = 0 if k in ["ts"] else 2
            xlabel = params._key_symbols[k]
            xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
            #plotname =  plot_path+f"/corr_muons_{k}.png"
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, show=True)#, store=plotname)




if __name__ == "__main__":
    main()
    print(f"###### Done.")
    input()

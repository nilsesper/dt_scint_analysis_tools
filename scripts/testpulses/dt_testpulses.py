#################################################################
### timing analysis of the dt readout system
# with simultaneous testpulses sent to all input channels
# plot timing by fe connector
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from datetime import datetime
import json
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
dumpfile_path = REPO_PATH+"/dumpfiles"
calib_path = REPO_PATH+"/calibration_files"

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### constants
    dump = True
    plot_hists = False
    granularity = "wi" #"fec" # select tp analysis per fe connector ("fec") or per wire ("wi")
    correct_for_offsets = True

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type     = str,
        help     = "input dumpfile path (with recorded testpulses)",
    )
    # ---
    args = parser.parse_args()
    tp_dumpfile_name = args.inputfile

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    tp_dumpfile_hits = data_utils.import_raw(file_name=tp_dumpfile_name) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    tp_dumpfile_hits = data_utils.cut_first_entries(data=tp_dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)
    print("tp_dumpfile_hits =",tp_dumpfile_hits)

    ### extract dt hits
    tp_hits = dt_utils.extract_dt_hits(hits=tp_dumpfile_hits)
    tp_hits = timestamp_utils.sort_by_timestamp(hits=tp_hits)
    tp_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_hits)
    print("tp_hits =",tp_hits)

    ### analyze timing of all fe connectors of all superlayers individually
    # if desired: correct testpulse timing for offsets (due to phi/theta difference and tp cable lengths)
    print(f"###### Analyzing testpulse hits for all frontend connectors of all superlayers...")
    rel_thres = 0.2
    if granularity == "fec": # fe conn granularity
        tp_timing = dt_utils.analyze_testpulses(tp_hits, rel_thres=rel_thres, plot_hists=plot_hists, correct_for_offsets=correct_for_offsets)
        print("tp_timing =",tp_timing)
    elif granularity == "wi": # wire granularity
        tp_timing = dt_utils.analyze_testpulses_per_wire(tp_hits, rel_thres=rel_thres, plot_hists=plot_hists, correct_for_offsets=correct_for_offsets)
        print("tp_timing =",tp_timing)
    
    ### plot distribution of tp timing
    # plot timing as scatter
    # do not plot rejected/dead channels (which have tp_ts_mean = 0)
    if granularity == "fec": # fe conn granularity
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        for sl in params._dt_chamber["sls"].keys():
            fe_id_list_plot, tp_ts_mean, tp_ts_err = [], [], []
            for fe_id in derived_params._dt_fe_id_remap_table[sl]:
                if tp_timing[sl][int(fe_id)]["tp_ts_mean"] > 0:
                    tp_ts_mean.append(tp_timing[sl][int(fe_id)]["tp_ts_mean"])
                    tp_ts_err.append(tp_timing[sl][int(fe_id)]["tp_ts_err"])
                    fe_id_list_plot.append(int(fe_id))
            ax.errorbar(x=np.array(fe_id_list_plot)-0.2+0.1*sl, y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(sl-1), linestyle="", marker="o", markersize=5, label=f"Superlayer {sl}")
            ax.set_xlabel(f"Frontend connector ID")
            ylabel = params._key_symbols["ts_orbit"]
            ylabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
            ax.set_ylabel(ylabel)
            ax.set_title(f"Testpulse timing distribution (DT chamber)")
            ax.legend()
        fig.tight_layout()
        fig.show()
    elif granularity == "wi": # wire granularity
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        for sl in params._dt_chamber["sls"].keys():
            for ly in derived_params._dt_inverted_remap_table[sl].keys():
                wi_list_plot, tp_ts_mean, tp_ts_err = [], [], []
                for wi in derived_params._dt_inverted_remap_table[sl][ly].keys():
                    if tp_timing[sl][ly][wi]["tp_ts_mean"] > 0:
                        tp_ts_mean.append(tp_timing[sl][ly][wi]["tp_ts_mean"])
                        tp_ts_err.append(tp_timing[sl][ly][wi]["tp_ts_err"])
                        wi_list_plot.append(wi)
                ax.errorbar(x=np.array(wi_list_plot)+0.01*(-6+(4*sl+ly)), y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(4*sl+ly-1), linestyle="", marker="o", markersize=5, label=f"SL {sl} LY {ly}")
                ax.set_xlabel(f"Wire")
                ylabel = params._key_symbols["ts_orbit"]
                ylabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
                ax.set_ylabel(ylabel)
                ax.set_title(f"Testpulse timing distribution (DT chamber)")
                ax.legend()
            fig.tight_layout()
            fig.show()


if __name__ == "__main__":
    main()
    input("Press [Enter] to exit.")
    print(f"###### Done.")









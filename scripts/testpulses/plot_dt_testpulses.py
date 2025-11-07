#################################################################
### plot dt testpulses and extracted correction
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
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### constants
    dump = True
    plot_hists = False
    granularity = "wi" #"fec" # select tp analysis per fe connector ("fec") or per wire ("wi")
    correct_for_offsets = True
    all_sls_aligned = True # flag whether calibration should be calculated on sl level or chamber level (last only possible of tps of all sls are aligned)
    tu_to_ns_conversion = 0.78 # tu to ns conversion
    ignore_channel = ( # do not plot (sl,ly,wi)
        (1,1,49),
        (2,1,57),
        (3,1,49),
    )

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_tp_timing_file",
        type     = str,
        help     = "input: calculated tp timing",
    )
    parser.add_argument(
        "--dt_tp_corrections_file",
        type     = str,
        help     = "input: calculated timing correction from testpulses",
    )
    # ---
    args = parser.parse_args()
    tp_timing_file = args.dt_tp_timing_file
    dt_tp_corrections_file = args.dt_tp_corrections_file

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    tp_timing = data_utils.load_pickle(file=tp_timing_file)
    print("tp_timing =",tp_timing)
    dt_tp_corrections = data_utils.load_pickle(file=dt_tp_corrections_file)
    print("dt_tp_corrections =",dt_tp_corrections)

    ########### TP TIMING

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
            ax.errorbar(x=np.array(fe_id_list_plot)-0.2+0.1*sl, y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(sl), linestyle="", marker="o", markersize=5, label=f"SL {sl}")
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
                    if (sl,ly,wi) in ignore_channel:
                        continue
                    if tp_timing[sl][ly][wi]["tp_ts_mean"] > 0:
                        tp_ts_mean.append(tp_timing[sl][ly][wi]["tp_ts_mean"])
                        tp_ts_err.append(tp_timing[sl][ly][wi]["tp_ts_err"])
                        wi_list_plot.append(wi)
                ax.errorbar(x=np.array(wi_list_plot)+0.04*(-6+(4*ly+sl)), y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(sl), linestyle="", marker=derived_params.marker_wheel(ly), markersize=5, label=f"SL{sl} Ly{ly}")
                ax.set_xlabel(f"Wire")
                ax.set_ylabel("$\\left\\langle T_\\text{orbit} \\right\\rangle$ [TU]")
                ax.set_title(f"Testpulse timing distribution (DT chamber)")
                ax.legend()

    #"""
    ### plots of tp timing
    fig, ax = plt.subplots(3, 1, figsize=(16,8), sharex=True)
    for sl in range(1,4):
        for ly in [0,1,2,3]:
            wi_list_plot, tp_ts_mean, tp_ts_err = [], [], []
            for wi in derived_params._dt_inverted_remap_table[sl][ly].keys():
                if (sl,ly,wi) in ignore_channel:
                    continue
                if tp_timing[sl][ly][wi]["tp_ts_mean"] > 0:
                    tp_ts_mean.append(tp_timing[sl][ly][wi]["tp_ts_mean"])
                    tp_ts_err.append(tp_timing[sl][ly][wi]["tp_ts_err"])
                    wi_list_plot.append(wi)
            if sl == 1:
                ax[sl-1].errorbar(x=np.array(wi_list_plot)+0.15*(-1+ly), y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(ly), linestyle="", marker="o", markersize=5, label=f"Ly {ly}")
            else:
                ax[sl-1].errorbar(x=np.array(wi_list_plot)+0.15*(-1+ly), y=tp_ts_mean, yerr=tp_ts_err, color=derived_params.color_wheel(ly), linestyle="", marker="o", markersize=5)
            ax[sl-1].set_ylabel("$\\left\\langle T_\\text{orbit} \\right\\rangle$ [TU]")
            ax[sl-1].set_title(f"SL {sl}")
            #ax[sl-1].legend()
            if sl == 3:
                ax[sl-1].set_xlabel("Wire")
        # global legend
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
        # show plot
        fig.tight_layout()
        fig.show()
    #"""

    ########## TP TIMING CORRECTION

    ### plot channel timing corrections
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    for sl in params._dt_chamber["sls"].keys():
        for ly in [0,1,2,3]:
            ts_corr, err_ts_corr, wi_list = [], [], []
            for wi in derived_params._dt_inverted_remap_table[sl][ly].keys():
                if (sl,ly,wi) in ignore_channel:
                    continue
                ts_corr.append(dt_tp_corrections[sl][ly][wi]["ts_corr"])
                err_ts_corr.append(dt_tp_corrections[sl][ly][wi]["err_ts_corr"])
                wi_list.append(wi)
            ts_corr = np.array(ts_corr)
            err_ts_corr = np.array(err_ts_corr)
            wi_list = np.array(wi_list)
            ax.errorbar(x=wi_list+0.04*(-6+(4*ly+sl)), y=ts_corr*tu_to_ns_conversion, yerr=err_ts_corr*tu_to_ns_conversion, color=derived_params.color_wheel(sl), linestyle="", marker=derived_params.marker_wheel(ly), markersize=5, label=f"SL{sl} Ly{ly}")
            ax.set_xlabel(f"Wire")
            ax.set_ylabel("$T_\\text{corr}$ [ns]")
            ax.set_title(f"Extracted timing correction")
            ax.legend()
        fig.tight_layout()
        fig.show()

    #"""
    ### plots of tp timing correction
    fig, ax = plt.subplots(3, 1, figsize=(16,8), sharex=True)
    for sl in range(1,4):
        for ly in [0,1,2,3]:
            ts_corr, err_ts_corr, wi_list = [], [], []
            for wi in derived_params._dt_inverted_remap_table[sl][ly].keys():
                if (sl,ly,wi) in ignore_channel:
                    continue
                ts_corr.append(dt_tp_corrections[sl][ly][wi]["ts_corr"])
                err_ts_corr.append(dt_tp_corrections[sl][ly][wi]["err_ts_corr"])
                wi_list.append(wi)
            ts_corr = np.array(ts_corr)
            err_ts_corr = np.array(err_ts_corr)
            wi_list = np.array(wi_list)
            if sl == 1:
                ax[sl-1].errorbar(x=np.array(wi_list)+0.15*(-1+ly), y=ts_corr*tu_to_ns_conversion, yerr=err_ts_corr*tu_to_ns_conversion, color=derived_params.color_wheel(ly), linestyle="", marker="o", markersize=5, label=f"Ly {ly}")
            else:
                ax[sl-1].errorbar(x=np.array(wi_list)+0.15*(-1+ly), y=ts_corr*tu_to_ns_conversion, yerr=err_ts_corr*tu_to_ns_conversion, color=derived_params.color_wheel(ly), linestyle="", marker="o", markersize=5)
            ax[sl-1].set_ylabel("$T_\\text{corr}$ [ns]")
            ax[sl-1].set_title(f"SL {sl}")
            #ax[sl-1].legend()
            if sl == 3:
                ax[sl-1].set_xlabel("Wire")
        # global legend
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
        # show plot
        fig.tight_layout()
        fig.show()
    #"""


if __name__ == "__main__":
    main()
    input("Press [Enter] to exit.")
    print(f"###### Done.")









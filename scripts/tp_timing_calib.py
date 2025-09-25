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
# input files:
#tp_dumpfile_name = dumpfile_path+"/sipm_testpulses.txt"
# data output files:
# --

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### constants
    n_chs = 32
    ts_no_digits = 1
    dump = True

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type     = str,
        help     = "input dumpfile path (with recorded testpulses)",
    )
    parser.add_argument(
        "--create_calib",
        action="store_true",
        help     = "calculate timing calibration file from inputfile",
    )
    parser.add_argument(
        "--validationfile",
        type     = str,
        help     = "validation dumpfile path (with recorded testpulses), if given create comparison plots to inputfile",
    )
    # ---
    args = parser.parse_args()
    tp_dumpfile_name = args.inputfile
    if args.validationfile:
        tp_validationfile_name = args.validationfile

    ### data import
    print(f"###### Importing dumpfile of testpulse run...")
    tp_dumpfile_hits = data_utils.import_raw(file_name=tp_dumpfile_name) # dummy_filename, data_filename
    print("tp_dumpfile_hits =",tp_dumpfile_hits)
    ## validationfile
    if args.validationfile:
        tp_validationfile_hits = data_utils.import_raw(file_name=tp_validationfile_name) # dummy_filename, data_filename
        print("tp_validationfile_hits =",tp_validationfile_hits)

    ### assign hit timestamps
    print(f"###### Assigning timestamps...")
    tp_dumpfile_hits = timestamp_utils.add_timestamp(hits=tp_dumpfile_hits)
    # calculate ts difference to orbit start (key "ts_orbit")
    tp_dumpfile_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_dumpfile_hits, silent=True)
    ## validationfile
    if args.validationfile:
        tp_validationfile_hits = timestamp_utils.add_timestamp(hits=tp_validationfile_hits)
        # calculate ts difference to orbit start (key "ts_orbit")
        tp_validationfile_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_validationfile_hits, silent=True)

    ### analyze testpulses channel by channel
    print(f"###### Analyzing testpulse hits for all input channels...")
    tp_ch = [None for ch in range(n_chs)] # hits of each input channel
    tp_ts, err_tp_ts = np.zeros(n_chs), np.zeros(n_chs) # timestamps of tps of each input channel
    for ch in range(n_chs):
        # select hits of one channel
        tp_ch[ch] = data_utils.cut_data(data=tp_dumpfile_hits, conditions=[("ch","==",ch)], silent=True)
        # calculate histogram of hit timing (bin width = 1 ts unit)
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=tp_ch[ch], key="ts_orbit", bin_centers="step1", silent=True)
        # select first peak of histogram (with lowest ts), the higher ts hits are due to ringing of the testpulse circuit
        peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=0.2) # 20% of max amplitude for peak
        sel_peak_indices = peak_indices[0] # first peak
        hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
        err_hists_peak = np.sqrt(hists_peak)
        err_centers_peak = np.full( len(centers_peak), 8/np.sqrt(12) )
        # calculate peak position (weighted mean)
        tp_ts[ch], err_tp_ts[ch] = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
    # calculate mean for fpga banks
    mean_bank, err_mean_bank = np.zeros(len(derived_params.fpga_banks)), np.zeros(len(derived_params.fpga_banks))
    for i, bank in enumerate(derived_params.fpga_banks):
        ch_list = derived_params.mezzanine_input_bank_mapping[bank]
        mean_bank[i], err_mean_bank[i], _ = math_utils.calculate_mean_std(data=tp_ts[ch_list], err_data=err_tp_ts[ch_list])
    # calculate global mean
    mean_ts, err_mean_ts, _ = math_utils.calculate_mean_std(data=tp_ts, err_data=err_tp_ts)
    ## validationfile
    if args.validationfile:
        tp_ch_valid = [None for ch in range(n_chs)] # hits of each input channel
        tp_ts_valid, err_tp_ts_valid = np.zeros(n_chs), np.zeros(n_chs) # timestamps of tps of each input channel
        for ch in range(n_chs):
            # select hits of one channel
            tp_ch_valid[ch] = data_utils.cut_data(data=tp_validationfile_hits, conditions=[("ch","==",ch)], silent=True)
            # calculate histogram of hit timing (bin width = 1 ts unit)
            hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=tp_ch_valid[ch], key="ts_orbit", bin_centers="step1", silent=True)
            # select first peak of histogram (with lowest ts), the higher ts hits are due to ringing of the testpulse circuit
            peak_indices = hist_utils.find_peak_indices(hist=hists, rel_thres=0.2) # 20% of max amplitude for peak
            sel_peak_indices = peak_indices[0] # first peak
            hists_peak, centers_peak = hists[sel_peak_indices], centers[sel_peak_indices]
            err_hists_peak = np.sqrt(hists_peak)
            err_centers_peak = np.full( len(centers_peak), 8/np.sqrt(12) )
            # calculate peak position (weighted mean)
            tp_ts_valid[ch], err_tp_ts_valid[ch] = hist_utils.weighted_mean_peak_position(hist=hists_peak, centers=centers_peak, err_hist=err_hists_peak, err_centers=err_centers_peak)
        # calculate global mean
        mean_ts_valid, err_mean_ts_valid, _ = math_utils.calculate_mean_std(data=tp_ts_valid, err_data=err_tp_ts_valid)

    """
    ### plot testpulse timing for all channels separately
    print(f"### tp timing plots")
    for ch in [1, 13]: #range(n_chs):
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
            hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, show=True, title=f"ch{ch}")#, store=plotname)
    #"""
    
    ### plot testpulse timing for all peak positions
    print(f"###### Plotting testpulse timing...")
    # plot timing as scatter
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    xspacing = np.arange(0, n_chs)
    for i, bank in enumerate(derived_params.fpga_banks):
        ch_list = derived_params.mezzanine_input_bank_mapping[bank]
        ax.errorbar(x=xspacing[ch_list], y=tp_ts[ch_list], yerr=err_tp_ts[ch_list], color=derived_params.color_wheel(i), linestyle="", marker=".", label=f"Bank {bank} channels")
        ax.axhline(y=mean_bank[i], color=derived_params.color_wheel(i), label=f"Bank {bank} mean:\n$T_\\text{{orbit}}={round(mean_bank[i], ts_no_digits)}\\pm{round(err_mean_bank[i], ts_no_digits)}$ TU")
        ax.axhspan(ymin=(mean_bank[i]-err_mean_bank[i]), ymax=(mean_bank[i]+err_mean_bank[i]), color=derived_params.color_wheel(i), alpha=0.1)
    # global mean
    #ax.axhline(y=mean_ts, color="tab:gray", label=f"Global mean:\n$T_\\text{{orbit}}={round(mean_ts, ts_no_digits)}\\pm{round(err_mean_ts, ts_no_digits)}$ TU")
    #ax.axhspan(ymin=(mean_ts-err_mean_ts), ymax=(mean_ts+err_mean_ts), color="tab:gray", alpha=0.1)
    ax.set_xlabel(f"Input channel")
    ylabel = params._key_symbols["ts_orbit"]
    ylabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
    ax.set_ylabel(ylabel)
    ax.set_title(f"Testpulse timing distribution")
    ax.legend()
    fig.tight_layout()
    fig.show()
    ## validationfile
    if args.validationfile:
        # plot timing as scatter
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        xspacing = np.arange(0, n_chs)
        # before
        ax.errorbar(x=xspacing-0.05, y=tp_ts, yerr=err_tp_ts, color="tab:blue", linestyle="", marker=".", label=f"Initial measurement")
        ax.axhline(y=mean_ts, color="tab:blue", label=f"Initial mean:\n$T_\\text{{orbit}}={round(mean_ts, ts_no_digits)}\\pm{round(err_mean_ts, ts_no_digits)}$ TU")
        ax.axhspan(ymin=(mean_ts-err_mean_ts), ymax=(mean_ts+err_mean_ts), color="tab:blue", alpha=0.1)
        # after
        ax.errorbar(x=xspacing+0.05, y=tp_ts_valid, yerr=err_tp_ts_valid, color="tab:red", linestyle="", marker=".", label=f"After calibration")
        ax.axhline(y=mean_ts_valid, color="tab:red", label=f"Mean after calibration:\n$T_\\text{{orbit}}={round(mean_ts_valid, ts_no_digits)}\\pm{round(err_mean_ts_valid, ts_no_digits)}$ TU")
        ax.axhspan(ymin=(mean_ts_valid-err_mean_ts_valid), ymax=(mean_ts_valid+err_mean_ts_valid), color="tab:red", alpha=0.1)
        ax.set_xlabel(f"Input channel")
        ylabel = params._key_symbols["ts_orbit"]
        ylabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
        ax.set_ylabel(ylabel)
        ax.set_title(f"Testpulse timing calibration")
        ax.legend()
        fig.tight_layout()
        fig.show()
    
    #"""
    if args.create_calib:
        ### find new timing target for all channels
        print(f"###### Calculate timing calibration...")
        # use (rounded) maximum value of all tp input channels, since on the fpga one can only delay channels, i.e. increase the ts and not decrease it...
        ts_target_idx = np.argmax(tp_ts)
        err_ts_target = err_tp_ts[ts_target_idx]
        ts_target = round(tp_ts[ts_target_idx], 0)
        # determine difference to target & reformulate in terms of clk_160 clk cycles = 8 ts units
        # this is the granularity with which the calibration can be tuned
        ts_160_bias = (ts_target - tp_ts)/8
        ts_bias = np.round(ts_160_bias, 0) # round to int
        err_ts_bias = np.sqrt( err_ts_target**2 + err_tp_ts**2 )/8
        print("Calibration result:\n input_delays (in 8 ts = clk_160 cycles units) =", ts_bias)

        ### dump into json file with timestamp
        # prepare result dict
        tp_calib_dict = { 
            "mean": [tp_ts[ch] for ch in range(n_chs)] } | {
            "err_mean": [err_tp_ts[ch] for ch in range(n_chs)] } | {
            "target": ts_target } | {
            "bias": [ts_bias[ch] for ch in range(n_chs)] } | {
            "err_bias": [err_ts_bias[ch] for ch in range(n_chs)] } | {
            "bank": [params.mezzanine_input_mapping[ch]["fpga_bank"] for ch in range(n_chs)] } | {
        }
        # store json
        if dump:
            print(f"###### Export calibration data...")
            timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            json_filepath = calib_path+f"/tp_calib_{timestamp_str}.json"
            with open(json_filepath, 'w') as file_obj:
                json.dump(tp_calib_dict, file_obj)
            print(f"* Store calibration data in file \"{json_filepath}\".")
        #"""

if __name__ == "__main__":
    main()
    print(f"###### Done.")
    input()

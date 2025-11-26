#################################################################
### timing calibration of scintillator readout system
# with simultaneous testpulses sent to all input channels
# plot tp timing
# calculate timing correction to be set for the registers in the delay ip in the fpga
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
    if args.validationfile:
        tp_validationfile_hits = data_utils.import_raw(file_name=tp_validationfile_name) # dummy_filename, data_filename
        print("tp_validationfile_hits =",tp_validationfile_hits)

    ### restrict data to last 10000 entries
    n_keep = 10000
    tp_dumpfile_hits = data_utils.restrict_to_last_entries(data=tp_dumpfile_hits, n_keep=n_keep)
    if args.validationfile:
        tp_validationfile_hits = data_utils.restrict_to_last_entries(data=tp_validationfile_hits, n_keep=n_keep)

    ### extract raw scint hits / input channels
    print(f"###### Extracting raw scintillator hits...")
    tp_hits = scint_utils.extract_raw_scint_hits(hits=tp_dumpfile_hits)
    print("tp_hits =",tp_hits)
    if args.validationfile:
        tp_validation_hits = scint_utils.extract_raw_scint_hits(hits=tp_validationfile_hits)
        print("tp_validation_hits =",tp_validation_hits)
    # add timestamp and sort by timestamp, add timestamp relative to orbit
    tp_hits = timestamp_utils.add_timestamp(hits=tp_hits)
    tp_hits = timestamp_utils.sort_by_timestamp(hits=tp_hits)
    tp_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_hits)
    if args.validationfile:
        tp_validation_hits = timestamp_utils.add_timestamp(hits=tp_validation_hits)
        tp_validation_hits = timestamp_utils.sort_by_timestamp(hits=tp_validation_hits)
        tp_validation_hits = timestamp_utils.add_timestamp_this_orbit(hits=tp_validation_hits)

    ### analyze timing of all raw scint channels individually
    print(f"###### Analyzing testpulse hits for all input channels...")
    rel_thres = 0.2
    tp_timing = scint_utils.analyze_testpulses(tp_hits, rel_thres=rel_thres, plot_hists=plot_hists)
    print("tp_timing =",tp_timing)
    if args.validationfile:
        tp_validation_timing = scint_utils.analyze_testpulses(tp_validation_hits, rel_thres=rel_thres, plot_hists=plot_hists)
        print("tp_validation_timing =",tp_validation_timing)

    ### remap tp_timing from (ly,st,sipm) back to (ro_ch,ch) which is more convenient and needed in the end anyway
    # also add fpga_bank identifier
    tp_timing_remap = {}
    for ly in derived_params._scint_inverted_remap_table.keys():
        for st in derived_params._scint_inverted_remap_table[ly].keys():
            for sipm in [0,1]:
                remap = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]
                ro_ch, ch = remap["ro_ch"], remap["ch"]
                if ro_ch not in tp_timing_remap.keys():
                    tp_timing_remap[ro_ch] = {}
                tp_timing_remap[ro_ch][ch] = tp_timing[ly][st][sipm] | {"fpga_bank": params.mezzanine_input_mapping[ch]["fpga_bank"]}
    if args.validationfile:
        tp_validation_timing_remap = {}
        for ly in derived_params._scint_inverted_remap_table.keys():
            for st in derived_params._scint_inverted_remap_table[ly].keys():
                for sipm in [0,1]:
                    remap = derived_params._raw_scint_inverted_remap_table[ly][st][sipm]
                    ro_ch, ch = remap["ro_ch"], remap["ch"]
                    if ro_ch not in tp_validation_timing_remap.keys():
                        tp_validation_timing_remap[ro_ch] = {}
                    tp_validation_timing_remap[ro_ch][ch] = tp_validation_timing[ly][st][sipm] | {"fpga_bank": params.mezzanine_input_mapping[ch]["fpga_bank"]}

    ### plot distribution of tp timing
    # plot timing as scatter
    # do not plot rejected/dead channels (which have tp_ts_mean = 0)
    for ro_ch in derived_params._raw_scint_ro_chs:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        for i, bank in enumerate(derived_params.fpga_banks):
            bank_ch_list = np.array(derived_params.mezzanine_input_bank_mapping[bank])
            bank_ch_list_plot, tp_ts_mean, tp_ts_err, bank_ch_list_validation_plot, tp_ts_validation_mean, tp_ts_validation_err = [], [], [], [], [], []
            for ch in bank_ch_list:
                if tp_timing_remap[ro_ch][ch]["tp_ts_mean"] > 0:
                    tp_ts_mean.append(tp_timing_remap[ro_ch][ch]["tp_ts_mean"])
                    tp_ts_err.append(tp_timing_remap[ro_ch][ch]["tp_ts_err"])
                    bank_ch_list_plot.append(ch)
            ax.errorbar(x=np.array(bank_ch_list_plot)-0.1, y=tp_ts_mean, yerr=tp_ts_err, color="tab:blue", linestyle="", marker=derived_params.marker_wheel(i), markersize=5, label=f"Initial measurement (I/O bank {bank})")
        for i, bank in enumerate(derived_params.fpga_banks):
            if args.validationfile:
                for ch in bank_ch_list:
                    if tp_validation_timing_remap[ro_ch][ch]["tp_ts_mean"] > 0:
                        tp_ts_validation_mean.append(tp_validation_timing_remap[ro_ch][ch]["tp_ts_mean"])
                        tp_ts_validation_err.append(tp_validation_timing_remap[ro_ch][ch]["tp_ts_err"])
                        bank_ch_list_validation_plot.append(ch)
                ax.errorbar(x=np.array(bank_ch_list_validation_plot)+0.1, y=tp_ts_validation_mean, yerr=tp_ts_validation_err, color="tab:red", linestyle="", marker=derived_params.marker_wheel(i), markersize=5, label=f"After calibration (I/O bank {bank})")
        ax.set_xlabel(f"Input channel")
        ylabel = "$\\left\\langle T_{orbit} \\right\\rangle$"
        ylabel += " ["+params._key_units["ts_orbit"]+"]" if (params._key_units["ts_orbit"] != "") else ""
        ax.set_ylabel(ylabel)
        ax.set_title(f"Testpulse timing calibration: Result") #({params._ro_ch_labels[ro_ch]})
        ax.legend(loc="center")
        fig.tight_layout()
        fig.show()

    ### find new timing target for all channels
    if args.create_calib:
        print(f"###### Calculate timing calibration...")
        tp_ts_mean_combined_list = [tp_timing_remap[ro_ch][ch]["tp_ts_mean"] for ch in params.mezzanine_input_mapping.keys() for ro_ch in derived_params._raw_scint_ro_chs]
        tp_ts_err_combined_list = [tp_timing_remap[ro_ch][ch]["tp_ts_err"] for ch in params.mezzanine_input_mapping.keys() for ro_ch in derived_params._raw_scint_ro_chs]
        ## common target ts for all mezzanines / ro_chs
        # use maximum value of all tp input channels, since on the fpga one can only delay channels, i.e. increase the ts and not decrease it
        ts_target_idx = np.argmax(tp_ts_mean_combined_list)
        ts_target = round(tp_ts_mean_combined_list[ts_target_idx], 0)
        err_ts_target = tp_ts_err_combined_list[ts_target_idx]
        print(f"Common target (max. timestamp): {ts_target} +- {err_ts_target} TU")
        
        ## calculate & store results separately for each ro_ch (for each fpga/delay ip core)
        tp_calib_dict = {}
        for ro_ch in derived_params._raw_scint_ro_chs:
            print(f"For ro_ch {ro_ch} ({params._ro_ch_labels[ro_ch]}):")
            # determine difference to target & reformulate in terms of clk_160 clk cycles = 8 ts units
            # this is the granularity with which the calibration can be tuned
            ts_bias, err_ts_bias = [], []
            for ch in params.mezzanine_input_mapping.keys():
                if tp_timing_remap[ro_ch][ch]["tp_ts_mean"]  != 0:
                    ts_bias.append( int(np.round((ts_target - tp_timing_remap[ro_ch][ch]["tp_ts_mean"])/8, 0))) # round to int
                    err_ts_bias.append(np.sqrt( err_ts_target**2 + tp_timing_remap[ro_ch][ch]["tp_ts_err"] **2 )/8)
                else:
                    ts_bias.append(0)
                    err_ts_bias.append(0)
            print("  Calibration result:\n input_delays (in 8 ts = clk_160 cycles units) =", ts_bias)
            ### dump into json file with timestamp
            # prepare result dict
            tp_calib_dict[ro_ch] = { 
                "mean": [tp_timing_remap[ro_ch][ch]["tp_ts_mean"]  for ch in params.mezzanine_input_mapping.keys()] } | {
                "err_mean": [tp_timing_remap[ro_ch][ch]["tp_ts_err"]  for ch in params.mezzanine_input_mapping.keys()] } | {
                "target": ts_target } | {
                "bias": [ts_bias[ch] for ch in params.mezzanine_input_mapping.keys()] } | {
                "err_bias": [err_ts_bias[ch] for ch in params.mezzanine_input_mapping.keys()] } | {
                "bank": [params.mezzanine_input_mapping[ch]["fpga_bank"] for ch in params.mezzanine_input_mapping.keys()] } | {
            }
            # store json
            if dump:
                timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                json_filepath = calib_path+f"/tp_calib_{params._ro_ch_labels[ro_ch]}_{timestamp_str}.json"
                with open(json_filepath, 'w') as file_obj:
                    json.dump(tp_calib_dict[ro_ch], file_obj)
                print(f"  Store calibration data in file \"{json_filepath}\".")


if __name__ == "__main__":
    main()
    input("Press [Enter] to exit.")
    print(f"###### Done.")









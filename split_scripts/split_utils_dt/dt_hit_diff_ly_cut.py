#######################
### calculate dt hit difference information from split data
#######################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import subprocess
import atexit
import sys
import time
from tqdm import tqdm

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# allowed datasets
allowed_datasets = [
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS",
]
# possible ts keys that need to be shifted
ts_keys = [
    "ts", "t0",
]

# maximum number of wires shown in a single subplot-figure
MAX_WIRES_PER_FIG = 10

# ---------------------------------------------------------------

def plot_wire_histograms(wire_ts_diff, sl, ly, edges, base_path, common_file_prefix, dataset, max_wires_per_fig=MAX_WIRES_PER_FIG):
    """
    Create one or more figures with subplots, each subplot showing the ts-diff
    histogram of a single wire.

    wire_ts_diff : dict
        {wi: np.array(ts_diff values)} for the given (sl, ly)
    """
    wire_ids = sorted(wire_ts_diff.keys())
    if len(wire_ids) == 0:
        return

    centers = 0.5 * (edges[1:] + edges[:-1])

    # split wire list into chunks of max_wires_per_fig
    chunks = [wire_ids[i:i + max_wires_per_fig] for i in range(0, len(wire_ids), max_wires_per_fig)]

    for chunk_idx, chunk in enumerate(chunks):
        n_wires = len(chunk)
        # grid layout: up to 5 columns
        ncols = min(5, n_wires)
        nrows = int(np.ceil(n_wires / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for idx, wi in enumerate(chunk):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            data = wire_ts_diff[wi]
            if len(data) > 0:
                hist_, _ = np.histogram(data, bins=edges)
                ax.step(centers, hist_, where="mid", color="tab:blue")
            ax.set_title(f"sl={sl}, ly={ly}, wi={wi}", fontsize=10)
            ax.set_xlabel("ts diff [tu]", fontsize=8)
            ax.set_ylabel("entries", fontsize=8)
            ax.tick_params(labelsize=7)

        # hide unused axes in the grid
        for idx in range(n_wires, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].axis("off")

        fig.suptitle(f"{dataset}  -  sl={sl}, ly={ly}  (wires {chunk[0]}-{chunk[-1]})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_file = base_path + "/plots/split_lys/" + common_file_prefix + "_" + dataset + f"_WIRE_HISTS_sl{sl}_ly{ly}_chunk{chunk_idx}.png"
        print(f"   saving wire-histogram figure: {out_file}")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type     = str,
        help     = "base / working directory",
        required=True,
    )
    parser.add_argument(
        "--data_config_file",
        type     = str,
        help     = "path to data config file which stores the data file names to be considered for the analysis",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type     = str,
        help     = "data set to create histogram from",
        required=True,
    )
    parser.add_argument(
        "--plot_wire_hists",
        action   = "store_true",
        help     = "if set, also create per-wire histogram figures (up to 10 wires per figure)",
    )
    # ---
    args = parser.parse_args()
    # base file path
    base_path = args.base_path
    # dataset to be merged
    dataset = args.dataset
    if dataset not in allowed_datasets:
        raise Exception(f"Forbidden dataset {dataset}.")
    # list of data files to be used
    dump_files = [] # list of dumpfile names
    file_prefixes = [] # list of data file prefixes to be used
    data_config_file = args.data_config_file
    with open(data_config_file) as f:
        lines = f.readlines()
        for line in lines:
            dump_file, file_prefix = line.split(",")
            file_prefixes.append(file_prefix.replace("\n","").replace("\r","").replace("\t",""))
            dump_files.append(dump_file.replace("\n","").replace("\r","").replace("\t",""))
    n_data = len(dump_files)
    common_file_prefix = os.path.commonprefix(file_prefixes)

    ####################

    ### import TS_RANGE files. expect it to have name [file_prefix]_TS_RANGE.txt in the given base_path
    ts_min = []
    ts_max = []
    print(f"open ts_range files...")
    for data_idx in range(n_data):
        ts_range_file = base_path+"/"+file_prefixes[data_idx]+"_TS_RANGE.txt"
        with open(ts_range_file) as f:
            lines = f.readlines()
            ts_min_, ts_max_ = lines[0].split(",")
            ts_min.append(int(ts_min_))
            ts_max.append(int(ts_max_))
    ts_min = np.array(ts_min)
    ts_max = np.array(ts_max)
    ts_len = ts_max - ts_min
    print(f"   ts_min = {ts_min}")
    print(f"   ts_max = {ts_max}")
    print(f"   ts_len = {ts_len}")

    ### calculate ts offsets to be applied to the data
    print(f"calculate ts offsets to apply to sub-datasets to merge data in time...")
    ts_offset = []
    ts_starting_point = 10000 # in tu
    ts_distance_between_datafiles = 0 # in tu
    # ts_merged = ts_data[i] + ts_offset[i]
    # ts_merged has range (ts_starting_point, ...)
    cum_ts_length = 0 # cumulated length of dataset
    for data_idx in range(n_data):
        ts_length = ts_max[data_idx] - ts_min[data_idx]
        ts_offset_ = -ts_min[data_idx] + ts_starting_point + cum_ts_length
        ts_offset.append(ts_offset_)
        cum_ts_length += ts_length + ts_distance_between_datafiles
    ts_offset = np.array(ts_offset)
    print(f"   ts_offset = {ts_offset}")

    ### fixed bins
    n_bins = 400
    edges = np.linspace(0, 800, n_bins+1) # in tu

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### per-wire ts-diff accumulator, merged across all sub-datasets
    ### structure: wire_ts_diff_all[(sl, ly)][wi] -> list of np.array chunks (concatenated at the end)
    wire_ts_diff_all = {}
    if args.plot_wire_hists:
        for sl in range(1, 4):
            for ly in range(0, 4):
                wire_ts_diff_all[(sl, ly)] = {}
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    wire_ts_diff_all[(sl, ly)][wi] = []

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"open {n_data} data files, apply timing offset and extract data for histogram...")
    print(f"CALCULATING DT HIT TIME DIFFERENCE HISTOGRAM...")
    data_to_merge = []
    for data_idx in tqdm(range(n_data)):
        sub_data_file = base_path+"/"+file_prefixes[data_idx]+"_"+dataset+".pcl"
        # pcl file import
        sub_data = data_utils.load_pickle(file=sub_data_file, silent=True)
        ## apply ts shift
        #for ts_key in ts_keys:
        #    if ts_key in sub_data.keys():
        #        sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
        ### do something with data
        ## calculate time difference between hits
        ch_list = []
        err_ch_list = []
        cut_layers = True # cut layers to calculate time difference only for hits in the same layer

        for sl in range(1,4):
            for ly in range(0,4):
                print(f"   sub_data_idx={data_idx}, sl={sl}, ly={ly}...")
                for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                    sub_data_cut = data_utils.cut_data(data=sub_data, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","==",wi)], silent=True)
                    sub_data_cut = timestamp_utils.sort_by_timestamp(hits=sub_data_cut, silent=True)
                    n_sub_data_cut = data_utils.length(sub_data_cut)
                    ts_diff_list = []
                    err_ts_diff_list = []
                    for i in range(1,n_sub_data_cut):
                        ts_diff_list.append(sub_data_cut["ts"][i] - sub_data_cut["ts"][i-1])
                        err_ts_diff_list.append( np.sqrt(sub_data_cut["err_ts"][i]**2 + sub_data_cut["err_ts"][i]**2) )
                    ts_diff_list = np.array(ts_diff_list)
                    err_ts_diff_list = np.array(err_ts_diff_list)
                    ch_list.append({"key": ts_diff_list})
                    err_ch_list.append({"key": err_ts_diff_list})

                    ## collect per-wire ts-diff data (merged across sub-datasets at the end)
                    if args.plot_wire_hists:
                        wire_ts_diff_all[(sl, ly)][wi].append(ts_diff_list)

        merged_ts_diff = data_utils.merge_dataset(split_data=ch_list, silent=True)["key"]
        merged_err_ts_diff = data_utils.merge_dataset(split_data=err_ch_list, silent=True)["key"]

        # create histogram of specified key and shifted hists to respect data error
        hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = hist_utils.calculate_histogram_and_shifted_histograms(data=merged_ts_diff, edges=edges, err_data=merged_err_ts_diff)
        # add to combined histogram
        hist += hist_
        entries += entries_
        underflow += underflow_
        overflow += overflow_
        hist_err_right += hist_err_right_
        hist_err_left += hist_err_left_

    duration = cum_ts_length
    print(f"duration = {duration*0.78*1e-9} s")

    ### error calculation for full hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    ### calculate once only stat unc
    err_hist_stat = np.sqrt(hist)

    print(f"created histogram:")
    print(f"  dataset = {dataset}")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    ### store histogram into file
    specific_data_to_store = {
        "edges": edges,
        "centers": centers,
        "hist": hist,
        "err_hist": err_hist,
        "err_hist_stat": err_hist_stat,
        "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up,
        "entries": entries,
        "underflow": underflow,
        "overflow": overflow,
    }
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_DIFF_SPECIFIC.pcl"
    print(f"storing specific data as {specific_data_file}...")
    data_utils.store_pickle(data=specific_data_to_store, file=specific_data_file)

    ### create per-wire subplot figures (max MAX_WIRES_PER_FIG wires per figure)
    if args.plot_wire_hists:
        print(f"creating per-wire histogram figures (max {MAX_WIRES_PER_FIG} wires per figure)...")
        for (sl, ly), wire_dict in wire_ts_diff_all.items():
            # concatenate ts_diff chunks (from all sub-datasets) per wire
            merged_wire_dict = {}
            for wi, chunks in wire_dict.items():
                if len(chunks) > 0:
                    merged_wire_dict[wi] = np.concatenate(chunks) if any(len(c) > 0 for c in chunks) else np.array([])
                else:
                    merged_wire_dict[wi] = np.array([])
            plot_wire_histograms(
                wire_ts_diff=merged_wire_dict,
                sl=sl,
                ly=ly,
                edges=edges,
                base_path=base_path,
                common_file_prefix=common_file_prefix,
                dataset=dataset,
            )


if __name__ == "__main__":
    main()
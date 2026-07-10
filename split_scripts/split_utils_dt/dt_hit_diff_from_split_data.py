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
        "--cuts",
        type     = str,
        help     = "cuts to apply to data in format \"key1,operator1,value1+key2,operator2,value;...\"",
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
    # cuts
    cuts_list = []
    if args.cuts:
        for cuts_str in args.cuts.split("+"):
            key, operator, value = cuts_str.split(",")
            if "params." in value:
                value = getattr(params, value.split("params.")[1])
            else:
                value = float(value)
            cuts_list.append((key, operator, value))

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
    n_bins = 2500
    edges = np.linspace(0, 5000, n_bins+1) # in tu

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

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

        ### cut data
        for i in range(len(cuts_list)):
            sub_data = data_utils.cut_data(data=sub_data, conditions=[cuts_list[i]])

        ### do something with data
        ## calculate time difference between hits
        ch_list = []
        err_ch_list = []
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


if __name__ == "__main__":
    main()
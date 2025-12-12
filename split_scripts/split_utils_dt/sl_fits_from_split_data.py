#######################
### calculate sl fit information from split data
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
    "SL_FITS", "SL_FITS_AFTERCUTS",
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

    ### prepare data
    # fitted pattern count
    pattern_count = {}
    for sl in range(1,4):
        pattern_count[sl] = {}
        for pat_type in range(6):
            pattern_count[sl][pat_type] = 0
        pattern_count[sl]["com"] = 0
    # time box hist
    n_bins = 100
    edges = np.linspace(0, 500, n_bins+1) # in tu
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"open {n_data} data files, apply timing offset and extract data for histogram...")
    print(f"CALCULATING SL FIT OCCUPANCY AND TIME BOX HISTOGRAM...")
    for data_idx in tqdm(range(n_data)):
        sub_data_file = base_path+"/"+file_prefixes[data_idx]+"_"+dataset+".pcl"
        # pcl file import
        sub_data = data_utils.load_pickle(file=sub_data_file, silent=True)
        ## apply ts shift
        #for ts_key in ts_keys:
        #    if ts_key in sub_data.keys():
        #        sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
        ### do something with data
        ### rate of patterns per superlayer
        for sl in range(1,4):
            # by pattern type
            for pat_type in range(6):
                sl_patterns_cut = data_utils.cut_data(data=sub_data, conditions=[("sl","==",sl), ("pat_type","==",pat_type)], silent=True)
                pattern_count[sl][pat_type] += data_utils.length(sl_patterns_cut)
            # all patterns combined
            sl_patterns_cut = data_utils.cut_data(data=sub_data, conditions=[("sl","==",sl)], silent=True)
            pattern_count[sl]["com"] += data_utils.length(sl_patterns_cut)
        ### time box hist for all 4 hits together
        for ly in [0,1,2,3]:
            data = sub_data[f"ts{ly}"] - sub_data[f"t0"]
            err_data = np.sqrt(sub_data[f"err_ts{ly}"]**2 + sub_data[f"err_t0"]**2)
            # create histogram of specified key and shifted hists to respect data error
            hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = hist_utils.calculate_histogram_and_shifted_histograms(data=data, edges=edges, err_data=err_data)
            # add to combined histogram
            hist += hist_
            entries += entries_
            underflow += underflow_
            overflow += overflow_
            hist_err_right += hist_err_right_
            hist_err_left += hist_err_left_
    
    duration = cum_ts_length
    print(f"duration = {duration*0.78*1e-9} s")
    duration_seconds = duration*0.78*1e-9

    ### calculate rates
    pattern_rate = {}
    err_pattern_rate = {}
    for sl in range(1,4):
        pattern_rate[sl] = {}
        err_pattern_rate[sl] = {}
        for pat_type in range(6):
            pattern_rate[sl][pat_type] = pattern_count[sl][pat_type] / duration_seconds
            err_pattern_rate[sl][pat_type] = np.sqrt(pattern_count[sl][pat_type]) / duration_seconds
        # common
        pattern_rate[sl]["com"] = pattern_count[sl]["com"] / duration_seconds
        err_pattern_rate[sl]["com"] = np.sqrt(pattern_count[sl]["com"]) / duration_seconds

    ### error calculation for full drift time hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    print(f"created histogram:")
    print(f"  dataset = {dataset}")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    ### store histogram into file
    specific_data_to_store = {
        "duration": duration,
        "pattern_count": pattern_count,
        "pattern_rate": pattern_rate,
        "err_pattern_rate": err_pattern_rate,
        "hist": hist,
        "err_hist": err_hist,
        "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up,
        "edges": edges,
        "underflow": underflow,
        "overflow": overflow,
    }
    specific_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_SPECIFIC.pcl"
    print(f"storing specific data as {specific_data_file}...")
    data_utils.store_pickle(data=specific_data_to_store, file=specific_data_file)


if __name__ == "__main__":
    main()
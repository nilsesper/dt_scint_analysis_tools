#######################
### calculate hist by summing variables of all sub datasets
# open sub datasets sequentially to save ram
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
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS", "SL_PATTERNS", "SL_FAKE_PATTERNS", "SL_FITS", "SL_FITS_AFTERCUTS", "SL_FIT_GROUPS", "DT_MUONS",
    "RAW_SCINT_HITS", "RAW_SCINT_GROUPS", "SCINT_HITS", "SCINT_AREAS",
    "DT_HIT_DIFFERENCES",
]
# possible ts keys that need to be shifted
ts_keys = [
    "ts", "t0", "tgroup",
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
        "--key",
        type     = str,
        help     = "key of dataset to create histogram from",
        required=True,
    )
    parser.add_argument(
        "--err_key",
        type     = str,
        help     = "uncertainty key of dataset to create histogram from (if available)",
    )
    parser.add_argument( # hist bin edges: linspace = "linear,start,stop,nbins+1" , arange = "range,start,stop+1"
        "--edges",
        type     = str,
        help     = "specification of histogram bin edges",
        required=True,
    )
    parser.add_argument( # if this flag is given: print hist & err_hist when finished
        "--print_hist",
        action = "store_true",
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
    # hist data key
    hist_key = args.key
    err_hist_key = None
    if args.err_key:
        err_hist_key = args.err_key

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

    ### open all data files once to find out min and max value (for automatic binning)
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    data_min_val, data_max_val = None, None
    print(f"open {n_data} data files, apply timing offset and find out min / max value...")
    data_to_merge = []
    for data_idx in tqdm(range(n_data)):
        sub_data_file = base_path+"/"+file_prefixes[data_idx]+"_"+dataset+".pcl"
        # pcl file import
        sub_data = data_utils.load_pickle(file=sub_data_file, silent=True)
        # apply ts shift
        for ts_key in ts_keys:
            if ts_key in sub_data.keys():
                sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
        ### do something with data
        # find min and max
        min_data_ = np.amin(sub_data[hist_key])
        max_data_ = np.amax(sub_data[hist_key])
        if data_min_val == None: # initial values
            data_min_val = min_data_
            data_max_val = max_data_
        else: # update minimum / maximum
            data_min_val = np.amin([data_min_val, min_data_])
            data_max_val = np.amax([data_max_val, max_data_])
    # calculate data range
    data_val_range = data_max_val - data_min_val
    if data_val_range == 0: # if all data points equal, artificially introduce bins
        data_val_range = 1
    print(f"  data_min_val = {data_min_val}")
    print(f"  data_max_val = {data_max_val}")
    print(f"  data_val_range = {data_val_range}")

    ### calculate hist bins
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg=args.edges, data_min_val=data_min_val, data_max_val=data_max_val)

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"open {n_data} data files, apply timing offset and extract data for histogram...")
    data_to_merge = []
    for data_idx in tqdm(range(n_data)):
        sub_data_file = base_path+"/"+file_prefixes[data_idx]+"_"+dataset+".pcl"
        # pcl file import
        sub_data = data_utils.load_pickle(file=sub_data_file, silent=True)
        # apply ts shift
        for ts_key in ts_keys:
            if ts_key in sub_data.keys():
                sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
        ### select data
        data = sub_data[hist_key]
        err_data = None
        if err_hist_key != None:
            err_data = sub_data[err_hist_key]
        ### do something with data
        # create histogram of specified key and shifted hists to respect data error
        hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = hist_utils.calculate_histogram_and_shifted_histograms(data=data, edges=edges, err_data=err_data)
        # add to combined histogram
        hist += hist_
        entries += entries_
        underflow += underflow_
        overflow += overflow_
        if type(err_data) != type(None):
            hist_err_right += hist_err_right_
            hist_err_left += hist_err_left_

    if type(err_data) == type(None):
        hist_err_right = None
        hist_err_left = None

    ### error calculation for full hist
    err_hist = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)

    print(f"created histogram:")
    print(f"  dataset = {dataset}  ,  key = {hist_key}  ,  err_key = {err_hist_key}")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")
    if args.print_hist:
        print(f"  edges     =  {edges}")
        print(f"  centers   =  {centers}")
        print(f"  hist      =  {hist}")
        print(f"  err_hist  =  {err_hist}")

    ### store histogram into file
    hist_to_store = {
        "edges": edges,
        "centers": centers,
        "hist": hist,
        "err_hist": err_hist,
        "entries": entries,
        "underflow": underflow,
        "overflow": overflow,
    }
    hist_key_str = hist_key.replace("/","-")
    hist_data_file = base_path+"/"+common_file_prefix+"_"+dataset+"_HIST_"+hist_key_str+".pcl"
    print(f"storing histogram as {hist_data_file}...")
    data_utils.store_pickle(data=hist_to_store, file=hist_data_file)



if __name__ == "__main__":
    main()
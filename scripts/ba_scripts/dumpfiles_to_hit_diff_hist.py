#################################################################
### import dumpfile and extract dt hits (and optionally raw scint hits)
# store dt hits (and optionally raw scint hits) as pkl file
#################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params   #, params_justus


import subprocess
import atexit
import sys
import time
from tqdm import tqdm
from scipy.optimize import curve_fit
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    ###################################################
    # IMPORTANT
    # When not using example data (dt_cosmics.txt) use params_justus
    base_path = "data_ba/"
    pcls_path = "pcls/"
    dataset_name =  "cosmic_85-15_3550-1800-1200_test1"
    dt_hits_name = dataset_name + "_hits.pcl"
    raw_scint_hits_name = dataset_name + "_raw_scint_hits.pcl"
    ts_range_name = dataset_name + "_ts_range.txt"
    input_dumpfile      = base_path + dataset_name + ".txt"
    dt_hit_diff_name = dataset_name + "_hit_diff.pcl"
    
    nodeadtime          = True  # True setzen, um dead time zu ignorieren
    deadtime_preffix = "nodeadtime" if nodeadtime else "deadtime"
    dt_hits_file        = base_path + pcls_path + dt_hits_name
    dt_hit_diff_hist_file = base_path + pcls_path +dt_hit_diff_name
    # optionale Schritte:
    use_timestamp_sync   = True   # add_timestamp + sort_by_timestamp anwenden
    extract_scint_hits    = True   # raw scint hits extrahieren und speichern
    raw_scint_hits_file   = base_path + pcls_path + raw_scint_hits_name  # nur relevant falls extract_scint_hits=True

    create_ts_file        = True   # ts_range Datei erzeugen
    
    # ---------------------------------------------------------

    #################
    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=1000)

    ### optionally sync timestamps
    if use_timestamp_sync:
        dumpfile_hits = timestamp_utils.add_timestamp(hits=dumpfile_hits)
        dumpfile_hits = timestamp_utils.sort_by_timestamp(hits=dumpfile_hits)

    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hits
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(
        hits=dumpfile_hits,
        has_timestamp=use_timestamp_sync,
        ignore_deadtime=nodeadtime,
    )
    print("dt_hits =",dt_hits)


    ### store dt hits to pcl file
    print(f"###### Storing dt hit data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)


    ####################

    ### fixed bins
    n_bins = 2500
    edges = np.linspace(0, 5000, n_bins+1) # in tu

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"CALCULATING DT HIT TIME DIFFERENCE HISTOGRAM...")

    sub_data = dt_hits
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


    ### error calculation for full hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    ### calculate once only stat unc
    err_hist_stat = np.sqrt(hist)

    print(f"created histogram:")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    ### store histogram into file
    specific_data= {
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
    specific_data_file = dt_hit_diff_hist_file 
    print(f"storing specific data as {specific_data_file}...")
    data_utils.store_pickle(data=specific_data, file=specific_data_file)

    return
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
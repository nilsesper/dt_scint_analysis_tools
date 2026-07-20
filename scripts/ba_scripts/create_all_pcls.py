#################################################################
### import dumpfile and extract dt hits without deadtime, and crate hist pcl for photopeak
### create dt hits with dead time create patterns, fits with and without vd as float, crate super patterns and super fits from super patterns
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
import argparse

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
    ### argparse for condor submissin
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type     = str,
        help     = "name of dataset or dumpfile to create pcls from for data analysis",
    )
    # ---



    ###################################################
    # IMPORTANT
    # When not using example data (dt_cosmics.txt) use params_justus
    args = parser.parse_args()
    dataset_name = args.dataset_name
    base_path = "data_ba/"
    pcls_path = "pcls/"
    #runs = ["cosmic_82-18_3625-1800-1200_run1_th20", "cosmic_82-18_3600-1800-1200_run1_th20", "cosmic_82-18_3575-1800-1200_run1_th20", "cosmic_82-18_3550-1800-1200_run1_th20", "cosmic_85-15_3550-1800-1200_run1_th20", "cosmic_85-15_3575-1800-1200_run1_th20", "cosmic_85-15_3600-1800-1200_run2_th20"]
    #runs =  ["cosmic_82-18_3625-1800-1200_run1_th20_cut100", "cosmic_82-18_3600-1800-1200_run1_th20_cut100", "cosmic_82-18_3575-1800-1200_run1_th20_cut100", "cosmic_82-18_3550-1800-1200_run1_th20_cut100", "cosmic_85-15_3550-1800-1200_run1_th20_cut100", "cosmic_85-15_3575-1800-1200_run1_th20_cut100", "cosmic_85-15_3600-1800-1200_run2_th20_cut100"] # a cut of 100 MB for quick analysis of data
    #runs =  ["cosmic_85-15_3575-1800-1200_run1_th20_cut100"] # still to do, not complete but for first batch
    max_alpha_in_deg = 15 #max value for muon angle higher values are cut away after track fit with vd = const


    

    # Ordner für dieses Dataset erstellen
    dataset_folder_pcls = base_path + pcls_path + dataset_name + "/"
    os.makedirs(dataset_folder_pcls, exist_ok=True)

    input_dumpfile = base_path + "data_tests_cuts/" + dataset_name + ".txt"

    nodeadtime = True
    use_timestamp_sync = True

    dt_hits_file = dataset_folder_pcls + dataset_name + "_hits_nodeadtime.pcl"
    dt_hit_diff_hist_file = dataset_folder_pcls + dataset_name + "_hit_diff.pcl"
    dt_hits_file_deadtime = dataset_folder_pcls + dataset_name + "_hits_wdeadtime.pcl"
    sl_patterns_file = dataset_folder_pcls + dataset_name + "_sl_patterns.pcl"

    sl_fits_file = dataset_folder_pcls + dataset_name + "_sl_fits.pcl"
    sl_refits_file = dataset_folder_pcls + dataset_name + "_sl_refits.pcl"
    super_fits_path = dataset_folder_pcls + dataset_name + "_super_fits.pcl"
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

    #print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hits
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(
        hits=dumpfile_hits,
        has_timestamp=use_timestamp_sync,
        ignore_deadtime=nodeadtime,
    )
    #print("dt_hits =",dt_hits)


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



#################################################################
### import dumpfile and extract dt hits
# store dt hits and patterns
#################################################################


    nodeadtime     = False   # True = do not apply dead time

    ### multiprocessing setup
    n_processes = 11 # no of processes running in parallel
    n_batches_clustering = 50000 # batch size for hit clustering
    do_multiprocessing = True
    # ---------------------------------------------------------

    #################

    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)
    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hit
    print(f"###### Extracting dt hits...")
    dt_hits_deadtime = dt_utils.extract_dt_hits(hits=dumpfile_hits, ignore_deadtime=nodeadtime)
    #print("dt_hits_deadtime =", dt_hits_deadtime)

    ### store to pcl file
    print(f"###### Storing data to file \"{dt_hits_file_deadtime}\"...")
    data_utils.store_pickle(data=dt_hits_deadtime, file=dt_hits_file_deadtime)

    ###############################################

    
    

    verbose = False
    simulation_only_muon_patterns = False
    fit_vd = True
    #do_timing_correction = False
    #if args.dt_tp_corrections_file:
    #    do_timing_correction = True
    #    dt_tp_corrections_file = args.dt_tp_corrections_file
    #fit_vd = False

    #################

    

    ### data import
    """
    print(f"###### Importing dt hits...")
    dt_hits = data_utils.load_pickle(file=dt_hits_file)
    """
    #if do_timing_correction:
    #    dt_tp_corrections = data_utils.load_pickle(file=dt_tp_corrections_file)

    ### optional data cut
    """
    if sl_cut:
        print(f"### Applying data cut to SL = {sl_cut}...")
        dt_hits = data_utils.cut_data(data=dt_hits, conditions=[("sl","==",int(sl_cut))])
    """
    #### optionally apply timing corrections
    #if do_timing_correction:
    #    print(f"### Applying timing correction from file \"{dt_tp_corrections_file}\"...")
    #    dt_hits = dt_utils.apply_timing_calibration(hits=dt_hits, dt_tp_corrections=dt_tp_corrections)

    #print("dt_hits =",dt_hits)

    ### dt reco
    # apply clustering algorithm
    print(f"### DT hit clustering for each superlayer...")
    if do_multiprocessing:
        sl_patterns = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_clustering, function=dt_utils.find_sl_patterns, data=dt_hits_deadtime, data_key="hits", kwargs={"verbose": verbose, "simulation_only_muon_patterns": simulation_only_muon_patterns, "fit_vd": fit_vd}, mute=True)
    else:
        sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits_deadtime, verbose=verbose, simulation_only_muon_patterns=simulation_only_muon_patterns)
    # sort by ts3 (reference timestamp)
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="ts3")
    #print("sl_patterns =",sl_patterns)

    ### store to pcl file
    print(f"###### Storing SL patterns to file \"{sl_patterns_file}\"...")
    data_utils.store_pickle(data=sl_patterns, file=sl_patterns_file)









######################################################
# Extract fits, refits with four hits, super fits from patterns and super patterns
######################################################

    verbose = False

    ### multiprocessing setup
    n_processes = 11  # no of processes running in parallel
    n_batches_sl_fitting = 1000  # batch size for sl fitting of hit clusters


    do_multiprocessing = True  # Multiprocessing aktuell deaktiviert

    ### dt reco
    # fit sl patterns
    max_alpha = np.deg2rad(max_alpha_in_deg)
    tan_alpha = np.tan(max_alpha)
    max_chi2 = 5
    print(f"### Fitting of separate SL clusters...")
    
    if do_multiprocessing:  # with multiprocessing
        sl_fits = process_utils.multiprocess_data(
            n_processes=n_processes,
            n_batches=n_batches_sl_fitting,
            function=dt_utils.fit_sl_patterns,
            data=sl_patterns,
            data_key="patterns",
            kwargs={"fit_vd": False, "suffix": ""},
            mute=True,
        )
        print("Done fitting...\nStarting cut of fits")

        

        sl_cut_fits = data_utils.cut_data(
            data=sl_fits,
            conditions=[
                ("impossible", "==", 0),
                ("chi2/ndf", "<", max_chi2),
                ("tan_alpha", ">=", (-tan_alpha)),
                ("tan_alpha", "<=", (tan_alpha)),  # FIX (#1): kein doppeltes deg2rad mehr
            ],
            silent=True,
        )





        print("Done cutting fit data...\nStarting refit...")

        sl_refits = process_utils.multiprocess_data(
            n_processes=n_processes,
            n_batches=n_batches_sl_fitting,
            function=dt_utils.fit_sl_patterns,
            data=sl_cut_fits,
            data_key="patterns",
            kwargs={"fit_vd": True, "suffix": "_refit"},
            mute=True,
        )


    else:  # without multiprocessing
        sl_fits = dt_utils.fit_sl_patterns(
            patterns=sl_patterns, verbose=verbose, fit_vd=False, suffix=""
        )
        sl_cut_fits = data_utils.cut_data(
            data=sl_fits,
            conditions=[
                ("impossible", "==", 0),
                ("chi2/ndf", "<", max_chi2),
                ("tan_alpha", ">=", (-tan_alpha)),
                ("tan_alpha", "<=", (tan_alpha)),  
            ],
            silent=True,
        )
        sl_refits = dt_utils.fit_sl_patterns(
            patterns=sl_cut_fits,
            silent=True,
            verbose=verbose,
            fit_vd=True,          
            suffix="_refit",
        )

    # sort by t0 (muon arrival time)
    sl_fits = data_utils.sort_by_key(data=sl_fits, sort_key="t0")
    sl_refits = data_utils.sort_by_key(data=sl_refits, sort_key="t0")

    ### store to pcl file
    print(f"###### Storing SL-level fits to file \"{sl_fits_file}\"...")
    data_utils.store_pickle(data=sl_fits, file=sl_fits_file)
    ### store refit to pcl file
    print(f"###### Storing SL-level refits to file \"{sl_refits_file}\"...")
    data_utils.store_pickle(data=sl_refits, file=sl_refits_file)



    print("Data saved\nBeginning with search for super patterns in both phi SLs")

    super_patterns = dt_utils.build_phi_super_patterns(sl_fits)
    suffix = "_free_vd_super_fit"
    super_fits = dt_utils.fit_super_sl_patterns(super_patterns, fit_vd=True, suffix = suffix)
    super_fits =data_utils.sort_by_key(data=super_fits, sort_key="t0" + suffix)
    print(f"Saving Superfits to {super_fits_path}...")
    data_utils.store_pickle(data = super_fits, file = super_fits_path)
    print(f"\nDone saving data under {super_fits_path}")
    



    return
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
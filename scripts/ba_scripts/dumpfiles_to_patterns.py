#################################################################
### import dumpfile and extract dt hits
# store dt hits as pkl file
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### --- manuell gesetzte Parameter (ersetzt argparse) ---
    base_path = "data_ba/"
    
    dataset_name = "cosmic_85_15_3300-1650-1100_test1"
    dumpfile_name = dataset_name + ".txt"
    dumpfile_path = base_path + dumpfile_name
    dt_hits_file   = base_path + "pcls/" + dataset_name + "_hits_wdeadtime.pcl"  # output file path: dt hits (pcl file)
    sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"   # output file path: sl_patterns
    nodeadtime     = False   # True = do not apply dead time

    ### multiprocessing setup
    n_processes = 11 # no of processes running in parallel
    n_batches_clustering = 50000 # batch size for hit clustering
    do_multiprocessing = True
    # ---------------------------------------------------------

    #################

    ### data import
    print(f"###### Importing dumpfile \"{dumpfile_path}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=dumpfile_path) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)
    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hit
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits, ignore_deadtime=nodeadtime)
    print("dt_hits =",dt_hits)

    ### store to pcl file
    print(f"###### Storing data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)

    ###############################################

    dt_hits_file = dt_hits
    

    verbose = False
    simulation_only_muon_patterns = False
    #do_timing_correction = False
    #if args.dt_tp_corrections_file:
    #    do_timing_correction = True
    #    dt_tp_corrections_file = args.dt_tp_corrections_file
    fit_vd = False

    sl_cut = 1
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

    print("dt_hits =",dt_hits)

    ### dt reco
    # apply clustering algorithm
    print(f"### DT hit clustering for each superlayer...")
    if do_multiprocessing:
        sl_patterns = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_clustering, function=dt_utils.find_sl_patterns, data=dt_hits, data_key="hits", kwargs={"verbose": verbose, "simulation_only_muon_patterns": simulation_only_muon_patterns, "fit_vd": fit_vd}, mute=True)
    else:
        sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits, verbose=verbose, simulation_only_muon_patterns=simulation_only_muon_patterns)
    # sort by ts3 (reference timestamp)
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="ts3")
    print("sl_patterns =",sl_patterns)

    ### store to pcl file
    print(f"###### Storing SL patterns to file \"{sl_patterns_file}\"...")
    data_utils.store_pickle(data=sl_patterns, file=sl_patterns_file)


if __name__ == "__main__":
    main()
    print(f"###### Done.")
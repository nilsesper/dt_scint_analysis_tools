#################################################################
### reconstruct hits from dummy cosmics
# for dt chamber: reco sl hit patterns -> sl fits -> muon tracks
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

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
# data input files:
cosmic_muon_file = pcl_path+"/dummy_cosmic_muons.pcl"
cosmic_muon_dt_hits_file = pcl_path+"/dummy_cosmic_muon_dt_hits.pcl"
# data output files:
cosmic_muon_sl_dt_patterns_file = pcl_path+"/dummy_cosmic_muon_sl_dt_patterns.pcl"
cosmic_muon_sl_dt_fits_file = pcl_path+"/dummy_cosmic_muon_dt_sl_fits.pcl"
cosmic_muon_dt_muons_file = pcl_path+"/dummy_cosmic_muon_dt_muons.pcl"

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### multiprocessing setup
    n_processes = 8 # no of processes running in parallel
    n_batches_clustering = 1000 # batch size for hit clustering
    n_batches_sl_fitting = 100 # batch size for sl fitting of hit clusters
    n_batches_muon_reco = 1000 # batch size for dt muon reco

    ### data import
    print(f"###### Importing dummy cosmic data...")
    cosmic_muons = data_utils.load_pickle(file=cosmic_muon_file)
    n_muons = len(cosmic_muons["ts"])
    cosmic_muon_dt_hits = data_utils.load_pickle(file=cosmic_muon_dt_hits_file)
    n_dt_hits = len(cosmic_muon_dt_hits["ts"])

    ### dt reco
    print(f"###### Reconstructing {n_dt_hits} DT hits and storing muons to \"{cosmic_muon_dt_muons_file}\"...")
    # apply clustering algorithm
    print(f"### DT hit clustering for each superlayer...")
    cosmic_muon_sl_dt_patterns = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_clustering, function=dt_utils.find_sl_patterns, data=cosmic_muon_dt_hits, data_key="hits", kwargs={}, mute=True)
    #cosmic_muon_sl_dt_patterns = dt_utils.find_sl_patterns(patterns=cosmic_muon_sl_dt_patterns)
    print("cosmic_muon_sl_dt_patterns =",cosmic_muon_sl_dt_patterns)
    # fit sl patterns
    print(f"### Fitting of separate SL clusters...")
    cosmic_muon_sl_dt_fits = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_sl_fitting, function=dt_utils.fit_sl_patterns, data=cosmic_muon_sl_dt_patterns, data_key="patterns", kwargs={}, mute=True)
    #cosmic_muon_sl_dt_fits = dt_utils.fit_sl_patterns(patterns=cosmic_muon_sl_dt_patterns, verbose=False)
    print("cosmic_muon_sl_dt_fits =",cosmic_muon_sl_dt_fits)
    # reco muons from fitted patterns
    print(f"### Reconstruction of muons from SL fits...")
    cosmic_muon_dt_muons = process_utils.multiprocess_data(n_processes=n_processes, n_batches=n_batches_muon_reco, function=dt_utils.reco_muons_from_sl_fits, data=cosmic_muon_sl_dt_fits, data_key="fits", kwargs={}, mute=True)
    #cosmic_muon_dt_muons = dt_utils.reco_muons_from_sl_fits(fits=cosmic_muon_sl_dt_fits, verbose=False)
    print("cosmic_muon_dt_muons =",cosmic_muon_dt_muons)
    # store to pcl file
    #data_utils.store_pickle(data=cosmic_muon_sl_dt_patterns, file=cosmic_muon_sl_dt_patterns_file)
    #data_utils.store_pickle(data=cosmic_muon_sl_dt_fits, file=cosmic_muon_sl_dt_fits_file)
    data_utils.store_pickle(data=cosmic_muon_dt_muons, file=cosmic_muon_dt_muons_file)




if __name__ == "__main__":
    main()
    print(f"###### Done.")

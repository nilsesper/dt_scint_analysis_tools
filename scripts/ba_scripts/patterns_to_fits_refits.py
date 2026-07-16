#################################################################
### use sl clusters to perform sl-level track fits
# store sl fits as pcl file
# cut fits to eliminate noise
# refit the fits with floating v_drift
# export refits as pcl for further analysis in sl_fits_analysis.py

#################################################################
import os
import argparse
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params
import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------
# main function
def main():
    base_path = "data_ba/"
    dataset_name = "cosmic_85-15_3600-1800-1200_run2_th20_cut"
    sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"  
    sl_fits_file = base_path + "pcls/" + dataset_name + "_sl_fits.pcl"
    sl_refits_file = base_path + "pcls/" + dataset_name + "_sl_refits.pcl"
    super_fits_path = base_path + "pcls/" + dataset_name + "_super_fits.pcl"

    verbose = False
   
    ### multiprocessing setup
    n_processes = 11  # no of processes running in parallel
    n_batches_sl_fitting = 1000  # batch size for sl fitting of hit clusters


    do_multiprocessing = True  # Multiprocessing aktuell deaktiviert

    ### data import
    print(f"###### Importing dt hits...")
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)

    ### dt reco
    # fit sl patterns
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

        
        max_alpha = np.deg2rad(5)
        tan_alpha = np.tan(max_alpha)
        max_chi2 = 5
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
        # cut data to restrict to chi2/ndf < 10 and |alpha| < 10 deg
        max_alpha = np.deg2rad(10)
        max_chi2 = 10
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
    print(super_patterns.keys())

    super_fits = dt_utils.fit_super_sl_patterns(super_patterns, fit_vd=True, suffix = "_free_vd_super_fit")
    super_fits =data_utils.sort_by_key(data=super_fits, sort_key="t0")
    print(f"Saving Superfits to {super_fits_path}...")
    data_utils.store_pickle(data = super_fits, file = super_fits_path)
    print(f"\nDone saving data under {super_fits_path}")
    
if __name__ == "__main__":
    main()
    print(f"###### Done.")
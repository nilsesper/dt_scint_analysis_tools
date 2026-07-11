#################################################################
### use sl clusters to perform sl-level track fits
# store sl fits as pcl file
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
    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sl_patterns_file",
        type     = str,
        help     = "input file path: sl patterns (pcl file)",
    )
    parser.add_argument(
        "--sl_fits_file",
        type     = str,
        help     = "output file path: sl fits (pcl file)",
    )
    parser.add_argument(
        "--fit_vd",
        action   = "store_true",
        help     = "fit drift velocity",
    )
    parser.add_argument(
        "--verbose",
        action   = "store_true",
        help     = "print info",
    )
    parser.add_argument(
        "--n_proc",
        type     = int,
        help     = "number of processes to run in parallel",
        default  = 16,
    )

    
    # ---
    args = parser.parse_args()
    sl_patterns_file = args.sl_patterns_file
    sl_fits_file = args.sl_fits_file


    base, ext = os.path.splitext(sl_fits_file)
    sl_refits_file = f"{base}_refits{ext}"

    verbose = args.verbose
    fit_vd = args.fit_vd

    #################
    ### multiprocessing setup
    n_processes = args.n_proc  # no of processes running in parallel
    n_batches_sl_fitting = 1000  # batch size for sl fitting of hit clusters


    do_multiprocessing = (not verbose) and False  # Multiprocessing aktuell deaktiviert

    ### data import
    print(f"###### Importing dt hits...")
    sl_patterns = data_utils.load_pickle(file=sl_patterns_file)

    ### dt reco
    # fit sl patterns
    print(f"### Fitting of separate SL clusters...")
    max_alpha = np.deg2rad(10)
    tan_alpha = np.tan(max_alpha)
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
        # FIX (#3): sl_refits war in diesem Zweig nie definiert.
        # Konsistent zum else-Zweig: cut + refit auch hier durchführen.

        max_chi2 = 20
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
        sl_refits = dt_utils.fit_sl_patterns(
            patterns=sl_cut_fits,
            silent=True,
            verbose=verbose,
            fit_vd=True,          
            suffix="_refit",
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
    # FIX (#4): sortierte Ergebnisse wieder in sl_fits/sl_refits geschrieben,
    # statt in ungenutzte Variablen sl_patterns/sl_patterns_refit,
    # damit die Sortierung beim Speichern auch tatsächlich wirkt.

    #if verbose:
        #print("sl_fits =", sl_fits)  # FIX (#7): Debug-Print jetzt an verbose gekoppelt

    ### store to pcl file
    print(f"###### Storing SL-level fits to file \"{sl_fits_file}\"...")
    data_utils.store_pickle(data=sl_fits, file=sl_fits_file)
    ### store refit to pcl file
    print(f"###### Storing SL-level refits to file \"{sl_refits_file}\"...")
    data_utils.store_pickle(data=sl_refits, file=sl_refits_file)

    refit_pcl = data_utils.load_pickle(file=sl_refits_file)
    print(refit_pcl)
    
    ### data import
    print(f"###### Importing all data...")
    arg = "vd_refit"
    # dt
    keylist = ["chi2/ndf", "chi2/ndf_refit","vd", "vd_refit", "tan_alpha", "tan_alpha_refit", "x0", "x0_refit", "t0", "t0_refit", "dt1", "dt1_refit", "dt2", "dt2_refit", "dt2", "dt2_refit"]
    refits = data_utils.load_pickle(file=sl_refits_file)
    print("### imported refits data from file: ", sl_refits_file)
    print(refits.keys())
    fig_size = (8, 6)
    for key in keylist:
        if "vd" in key:
            factor = 1 / derived_params._drift_velocity_conversion
        else:
            factor = 1

        # hist of vd distribution
        n_refits = data_utils.length(refits)
        plt.figure(figsize=fig_size)
        plt.hist(refits[key]*factor, bins=100, histtype="step", color="black")
        plt.xlabel(key + "value")
        plt.xlim(min(refits[key] * factor), max(refits[key]) * factor)
        plt.ylabel("counts")
        plt.title("distribution of " + key)
        safe_key = key.replace("/", "_")
        plt.savefig(f"example_data/plots/{safe_key}.png", bbox_inches="tight")
        print(f"### saved plot to {sl_refits_file}/plots/{safe_key}.png")
        plt.close()

        #comparison of laterality of fit and refit
        lat_fit = refits["laterality"]
        lat_refit = refits["laterality_refit"]
        change_of_lat = []
        n_lats = len(lat_refit)
        for idx in range(n_lats):
            if lat_fit[idx] != lat_refit[idx]:
                change_of_lat.append(1)
        
    n_changes = len(change_of_lat)

    print(f"{n_changes} changes of laterality in {n_lats} refits: {round(n_changes/n_lats*100, 2)}%")


if __name__ == "__main__":
    main()
    print(f"###### Done.")
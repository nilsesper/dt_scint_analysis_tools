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
import matplotlib as mpl
from scipy.optimize import curve_fit
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    base_path = "data_ba/"
    dataset_name = "cosmic_85_15_3300-1650-1100_test1"
    sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"  
    sl_fits_file = base_path + "pcls/" + dataset_name + "_sl_fits.pcl"
    sl_refits_file = base_path + "pcls/" + dataset_name + "_sl_refits.pcl"
    plot_save_path = base_path + "plots/sl_fits/" 
    plot_type = ".png"

    verbose = False
   
    ### multiprocessing setup
    n_processes = 11  # no of processes running in parallel
    n_batches_sl_fitting = 1000  # batch size for sl fitting of hit clusters


    do_multiprocessing = True  # Multiprocessing aktuell deaktiviert

    ### data import
    print(f"###### Importing fits...")
    sl_fits = data_utils.load_pickle(file = sl_fits_file)

    print(f"###### Importing refits...")
    sl_refits = data_utils.load_pickle(file = sl_refits_file)
    print("### imported refits data from file: " + sl_refits_file)
    refit_keylist = ["chi2/ndf", "chi2/ndf_refit","vd", "vd_refit", "tan_alpha", "tan_alpha_refit", "x0", "x0_refit", "t0", "dt0", "dt0_refit", "dt1", "dt1_refit", "dt2", "dt2_refit", "dt2", "dt2_refit"]
    fit_keylist = ["chi2/ndf", "vd", "tan_alpha",  "x0", "dt1",  "dt2", "dt2"]

    ###################
    # cut fits data according to nils master thesis chi2/ndf <10
    #t_ly <= 2ns
    # Drift time tly <=t_drift_max - 2ns approx 383 ns

     # cut data to restrict to chi2/ndf < 10 and |alpha| < 10 deg
    max_chi2_ndf = 10
    sl_cut_fits = data_utils.cut_data(
            data=sl_fits,
            conditions=[
                ("impossible", "==", 0),
                ("chi2/ndf", "<", max_chi2_ndf),  
            ],
            silent=True,
        )
    max_td = 450/derived_params._ts_unit # ns/0.78
    min_td = 5/derived_params._ts_unit # ns/0.78
    refits_cuts = data_utils.cut_data(
            data=sl_refits,
            conditions=[
                ("impossible_refit", "==", 0),
                ("chi2/ndf_refit", "<", 0.005)

            ],
            silent=True,
        )
    print(f"data cut succsessfull:\n max chi2/ndf in cut data = {max(sl_cut_fits["chi2/ndf"])}")
    def plot_statistics(*, keylist, fits, title):

        for key in keylist:
            if "vd" in key:
                factor = 1 / derived_params._drift_velocity_conversion
            else:
                factor = 1

            # hist of key distrubution
            n_refits = data_utils.length(fits)
            plt.figure()
            plt.hist(fits[key]*factor, bins=100, histtype="step", color="black")
            plt.xlabel(key + " value")
            plt.xlim(min(fits[key] * factor), max(fits[key] * factor))
            plt.ylabel("counts")
            plt.title("distribution of " + key + title)
            safe_key = key.replace("/", "_")
            path = f"{plot_save_path}{title}{safe_key}{dataset_name}{plot_type}"
            plt.savefig(path, bbox_inches="tight")
            print(f"### saved plot to {path}")
            plt.close()

            if key =="vd_refit":
                def gauss(x, A, mu, sigma, b, c):
                    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b*x + c

                data = fits[key] * factor
                v_max = 58
                n_bins = 80
                # Histogramm
                counts, edges = np.histogram(data, bins=n_bins, range=(50, v_max))
                centers = (edges[:-1] + edges[1:]) / 2

                # Gauß-Fit
                p0 = [counts.max(), centers[np.argmax(counts)], 1.0, 10, 10]
                params, _ = curve_fit(gauss, centers, counts, p0=p0)

                # Plot
                plt.figure()
                plt.hist(data, bins=n_bins, range=(50, v_max), histtype="step", color="black", label="Histogram")

                x = np.linspace(50, v_max, 1000)
                plt.plot(x, gauss(x, *params), "r-", lw=2,
                        label=f"$\\mu={params[1]:.3f}$\n$\\sigma={params[2]:.3f}$")

                plt.xlabel(key + " value")
                plt.ylabel("counts")
                plt.xlim(50, v_max)
                plt.title("distribution of " + key + title)
                plt.legend()

                safe_key = key.replace("/", "_")
                path = f"{plot_save_path}{title}{safe_key}_zoom_{dataset_name}{plot_type}"
                plt.savefig(path, bbox_inches="tight")
                print(f"### saved plot to {path}")
                plt.close()

        return 

    # plot_statistics(keylist= fit_keylist, fits = sl_cut_fits, title= "fit_hists")
    plot_statistics(keylist= refit_keylist, fits = refits_cuts, title="refits_cuts")
    #print(refits_cuts.keys())

    """
        # comparison of laterality of fit and refit
        lat_fit = sl_fits["laterality"]
        lat_refit = sl_refits["laterality_refit"]
        change_of_lat = []
        n_lats = len(lat_refit)
        for idx in range(n_lats):
            if lat_fit[idx] != lat_refit[idx]:
                change_of_lat.append(1)
        
    n_changes = len(change_of_lat)

    print(f"{n_changes} changes of laterality in {n_lats} refits: {round(n_changes/n_lats*100, 2)}%")
"""
    return()


if __name__ == "__main__":
    main()
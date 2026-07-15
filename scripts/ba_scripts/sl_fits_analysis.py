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
    list_of_fits = ["cosmic_85-15_3600-1800-1200_run2_th20_cut", "cosmic_85-15_3550-1800-1200_test1", "cosmic_85-15_3000-1500-1000_test3", "cosmic_85-15_3550-1800-1200_test1", "cosmic_82-18_3600-1800-1200_test1_th20"]
    base_path = "data_ba/"
    dataset_name = list_of_fits[5]

    sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"
    sl_fits_file = base_path + "pcls/" + dataset_name + "_sl_fits.pcl"
    sl_refits_file = base_path + "pcls/" + dataset_name + "_sl_refits.pcl"

    plot_save_path = base_path + "plots/sl_fits/" + dataset_name + "/"
    os.makedirs(plot_save_path, exist_ok=True)
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
    refit_keylist = [ "chi2/ndf_refit", "vd_refit", "tan_alpha_refit",  "x0_refit", "t0_refit", "dt0_refit",  "dt1_refit", "dt2_refit", "dt2_refit"]
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
            path = f"{plot_save_path}{dataset_name}{safe_key}{title}{plot_type}"
            plt.savefig(path, bbox_inches="tight")
            print(f"### saved plot to {path}")
            plt.close()

            if key =="vd_refit":
                try:
                    def gauss_lin(x, A, mu, sigma, b, c):
                        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b*x + c

                    # --- 1. build histogram from data + err_data, same as your dt_hit pipeline ---

                    x_min, x_max = 50, 62
                    n_bins = 80
                    edges = np.linspace(x_min, x_max, n_bins + 1)

                    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = \
                        hist_utils.create_empty_histogram(edges=edges)

                    hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = \
                        hist_utils.calculate_histogram_and_shifted_histograms(data=fits[key]*factor, edges=edges, err_data=fits["err_" + key]*factor)

                    hist += hist_
                    entries += entries_
                    underflow += underflow_
                    overflow += overflow_
                    hist_err_right += hist_err_right_
                    hist_err_left += hist_err_left_

                    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
                        hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True
                    )
                    err_hist_stat = np.sqrt(hist)

                    print(f"created histogram:")
                    print(f"  entries = {entries}, underflow = {underflow}, overflow = {overflow}")

                    # --- 2. fit gauss + lin background ---

                    yerr_fit = np.where(err_hist == 0, 1, err_hist)   # avoid zero-weight bins breaking curve_fit
                    p0 = [hist.max(), centers[np.argmax(hist)], 1.0, 10, 10]

                    popt, pcov = curve_fit(gauss_lin, centers, hist, p0=p0, sigma=yerr_fit, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))

                    # --- 3. plot ---

                    plt.figure()
                    plt.step(centers, hist, where="mid", color="black", label="Histogram")
                    plt.errorbar(centers, hist, yerr=err_hist, fmt="none", ecolor="black", elinewidth=1, capsize=0)

                    x = np.linspace(x_min, x_max, 1000)
                    plt.plot(x, gauss_lin(x, *popt), "r-", lw=2,
                            label=f"$\\mu={popt[1]:.3f}\\pm{perr[1]:.3f}$\n$\\sigma={popt[2]:.3f}\\pm{perr[2]:.3f}$")

                    plt.xlabel(key + " value")
                    plt.ylabel("counts")
                    plt.xlim(x_min, x_max)
                    plt.title("distribution of " + key + title)
                    plt.legend()

                    safe_key = key.replace("/", "_")
                    path = f"{plot_save_path}{dataset_name}{safe_key}{title}_zoom_{plot_type}"
                    plt.savefig(path, bbox_inches="tight")
                    print(f"### saved plot to {path}")
                    plt.close()
                except:
                    print("Fit failed...\nContinuing with other hists")

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
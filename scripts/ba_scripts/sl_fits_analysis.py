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
    list_of_fits = ["cosmic_85-15_3600-1800-1200_run2_th20_cut", "cosmic_85-15_3550-1800-1200_test1", 
                    "cosmic_85-15_3000-1500-1000_test3", "cosmic_85-15_3550-1800-1200_test1", 
                    "cosmic_82-18_3600-1800-1200_test1_th20"]
    
    base_path = "data_ba/"
    dataset_name = "cosmic_85-15_3600-1800-1200_run2_th20_cut"

    sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"
    sl_fits_file = base_path + "pcls/" + dataset_name + "_sl_fits.pcl"
    sl_refits_file = base_path + "pcls/" + dataset_name + "_sl_refits.pcl"
    super_fits_path = base_path + "pcls/" + dataset_name + "_super_fits.pcl"
    plot_save_path = base_path + "plots/sl_fits/" + dataset_name + "/"
    os.makedirs(plot_save_path, exist_ok=True)
    plot_type = ".png"


    ### data import
    print(f"###### Importing fits...")
    sl_fits = data_utils.load_pickle(file = sl_fits_file)

    print(f"###### Importing refits...")
    sl_refits = data_utils.load_pickle(file = sl_refits_file)
    print("### imported refits data from file: " + sl_refits_file)

    print(f"###### Importing super fits...")
    super_fits = data_utils.load_pickle(file = super_fits_path)
    print("### imported super fits data from file: " + super_fits_path)

    refit_keylist = [ "chi2/ndf_refit", "vd_refit", "tan_alpha_refit",  "x0_refit", "t0_refit", "dt0_refit",  "dt1_refit", "dt2_refit", "dt2_refit"]
    fit_keylist = ["chi2/ndf", "vd", "tan_alpha",  "x0", "dt1",  "dt2", "dt2", "pat_type"]

    super_fit_keylist = ['sl1', 'sl3', 'pat_type_sl1', 'pat_type_sl3', 'idx_sl1', 'idx_sl3', 'muon_id_mismatch', 'ts0', 'err_ts0', 'ts4', 'err_ts4', 'wi0_sl1', 'wi0_sl3', 'ts1', 'err_ts1', 'ts5', 'err_ts5', 'wi1_sl1', 'wi1_sl3', 'ts2', 'err_ts2', 'ts6', 'err_ts6', 'wi2_sl1', 'wi2_sl3', 'ts3', 'err_ts3', 'ts7', 'err_ts7', 'wi3_sl1', 'wi3_sl3', 'impossible_sl1', 'impossible_sl3', 'laterality_sl1', 'laterality_sl3', 't0_sl1', 't0_sl3', 'err_t0_sl1', 'err_t0_sl3', 'x0_sl1', 'x0_sl3', 'err_x0_sl1', 'err_x0_sl3', 'tan_alpha_sl1', 'tan_alpha_sl3', 'err_tan_alpha_sl1', 'err_tan_alpha_sl3', 'vd_sl1', 'vd_sl3', 'err_vd_sl1', 'err_vd_sl3', 'corr_t0_x0_sl1', 'corr_t0_x0_sl3', 'corr_t0_tan_alpha_sl1', 'corr_t0_tan_alpha_sl3', 'corr_t0_vd_sl1', 'corr_t0_vd_sl3', 'corr_x0_tan_alpha_sl1', 'corr_x0_tan_alpha_sl3', 'corr_x0_vd_sl1', 'corr_x0_vd_sl3', 'corr_tan_alpha_vd_sl1', 'corr_tan_alpha_vd_sl3', 'chi2/ndf_sl1', 'chi2/ndf_sl3', 'dt0_sl1', 'dt0_sl3', 'dt1_sl1', 'dt1_sl3', 'dt2_sl1', 'dt2_sl3', 'dt3_sl1', 'dt3_sl3', 'muon_id', 'muon_ts', 'muon_phi', 'muon_theta', 'muon_x0', 'muon_y0', 'muon_z0', 'impossible_free_vd_super_fit', 'lat_id1_free_vd_super_fit', 'lat_id2_free_vd_super_fit', 't0_free_vd_super_fit', 'x0_free_vd_super_fit', 'tan_alpha_free_vd_super_fit', 'vd_free_vd_super_fit', 'chi2/ndf_free_vd_super_fit', 'dt0_free_vd_super_fit', 'dt1_free_vd_super_fit', 'dt2_free_vd_super_fit', 'dt3_free_vd_super_fit', 'dt4_free_vd_super_fit', 'dt5_free_vd_super_fit', 'dt6_free_vd_super_fit', 'dt7_free_vd_super_fit', 'err_t0_free_vd_super_fit', 'err_x0_free_vd_super_fit', 'err_tan_alpha_free_vd_super_fit', 'err_vd_free_vd_super_fit', 'corr_t0_x0_free_vd_super_fit', 'corr_t0_tan_alpha_free_vd_super_fit', 'corr_t0_vd_free_vd_super_fit', 'corr_x0_tan_alpha_free_vd_super_fit', 'corr_x0_vd_free_vd_super_fit', 'corr_tan_alpha_vd_free_vd_super_fit', 'ref_x_free_vd_super_fit', 'ref_z_free_vd_super_fit']

    good_super_fit_keys = ["t0_sl1", 't0_sl3', "x0_free_vd_super_fit", 'tan_alpha_free_vd_super_fit', 'vd_free_vd_super_fit', 'chi2/ndf_free_vd_super_fit', 'dt1_free_vd_super_fit', 'dt2_free_vd_super_fit', 'dt3_free_vd_super_fit', 'dt4_free_vd_super_fit', 'dt5_free_vd_super_fit', 'dt6_free_vd_super_fit', 'dt7_free_vd_super_fit']



    def plot_statistics(*, keylist, fits, title):

        for key in keylist:
            if 'vd_free_vd_super_fit' in key:
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

            if key =='vd_free_vd_super_fit':
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
   

    # cut data to restrict to chi2/ndf < 10 and |alpha| < 10 deg
    max_chi2_ndf = 10
    min_x0 = 1 #mm
    super_fits_cuts = data_utils.cut_data(
            data=super_fits,
            conditions=[
                ("impossible_free_vd_super_fit", "==", 0),
                ("chi2/ndf_free_vd_super_fit", "<", 10),
                ("vd_free_vd_super_fit", "<", 70 * derived_params._drift_velocity_conversion),
                ("vd_free_vd_super_fit", ">", 30 * derived_params._drift_velocity_conversion),
                #("dt0_refit", ">", min_td),
                #("dt0_refit", "<", max_td),

            ],
            silent=True,
        )
    
    print(f"data cut succsessfull:\n max chi2/ndf in cut data = {max(sl_cut_fits["chi2/ndf"])}")

    plot_statistics(keylist= good_super_fit_keys, fits = super_fits_cuts, title="SUPER_FITS")

    return


if __name__ == "__main__":
    main()
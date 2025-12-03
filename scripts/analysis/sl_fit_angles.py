#################################################################
### sl fit angle analysis
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import matplotlib.patches as mpatches
from functools import partial
from scipy.optimize import curve_fit
from scipy import stats

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tan_alpha_hist_file",
        type     = str,
        help     = "input file path: tan_alpha histogram (pcl file)",
    )
    # plotting / store plot
    parser.add_argument(
        "--show_plots",
        action = "store_true",
        help     = "show plots flag",
    )
    parser.add_argument(
        "--store_plots",
        type     = str,
        help     = "output directory: give argument if plots should be stores, specify output path for plots here",
    )
    parser.add_argument(
        "--sim_tan_alpha_hist_file",
        type     = str,
        help     = "input file path: simulated tan_alpha histogram (pcl file)",
    )
    # -
    parser.add_argument(
        "--fig_size",
        type     = str,
        default = "12,8",
        help     = "custom fig_size of the plot in the format x_size,y_size (if desired)",
    )
    parser.add_argument(
        "--store_path",
        type     = str,
        help     = "path to store pdf plot (if desired)",
    )
    # ---
    args = parser.parse_args()
    tan_alpha_hist_file = args.tan_alpha_hist_file
    sim_tan_alpha_hist_file = args.sim_tan_alpha_hist_file
    # other 
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    tan_alpha_hist = data_utils.load_pickle(file=tan_alpha_hist_file)
    sim_tan_alpha_hist = data_utils.load_pickle(file=sim_tan_alpha_hist_file)

    entries = tan_alpha_hist["entries"]
    underflow = tan_alpha_hist["underflow"]
    overflow = tan_alpha_hist["overflow"]
    hist = tan_alpha_hist["hist"]
    err_hist = tan_alpha_hist["err_hist"]
    edges = tan_alpha_hist["edges"]
    centers = tan_alpha_hist["centers"]

    sim_entries = sim_tan_alpha_hist["entries"]
    sim_underflow = sim_tan_alpha_hist["underflow"]
    sim_overflow = sim_tan_alpha_hist["overflow"]
    sim_hist = sim_tan_alpha_hist["hist"]
    sim_err_hist = sim_tan_alpha_hist["err_hist"]
    sim_edges = sim_tan_alpha_hist["edges"]
    sim_centers = sim_tan_alpha_hist["centers"]

    ##################
    ### 

    # calculate
    centers = np.arctan(centers)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist_down=err_hist, err_hist_up=err_hist, log_scale=False)
    xlabel = "$\\alpha$ [rad]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SL_FIT_SPECIFIC_ALPHA.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    # calculate
    sim_centers = np.arctan(sim_centers)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=sim_hist, centers=sim_centers, err_hist_down=sim_err_hist, err_hist_up=sim_err_hist, log_scale=False)
    xlabel = "$\\alpha$ [rad]"
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"SL_FIT_SPECIFIC_SIM_ALPHA.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    input()
    exit()

    ##################
    ### fit

    # data import
    theta_bins = centers
    theta = hist
    err_theta = err_hist
    theta_template = sim_hist
    err_theta_template = sim_err_hist

    # restrict data to good data for fit
    idcs_fit = (theta_bins > 0) & (theta_bins < 0.8)
    theta_bins_fit = theta_bins[idcs_fit]
    theta_fit = theta[idcs_fit]
    err_theta_fit = err_theta[idcs_fit]
    theta_template_fit = theta_template[idcs_fit]
    err_theta_template_fit = err_theta_template[idcs_fit]


    ### template fit to MC
    def f_fit(x, N0):
        return N0*theta_template_fit
    def err_f_fit(x, N0, err_N0):
        return np.sqrt(
              (err_N0*theta_template_fit)**2
            + (N0*err_theta_template_fit)**2
        )
    p0 = 1
    popt, pcov, infodict, _, _ = curve_fit(f_fit, xdata=theta_bins_fit, ydata=theta_fit, p0=p0, sigma=err_theta_fit, absolute_sigma=True, full_output=True)
    N0_fit = popt[0]
    chi2 = np.sum((theta_fit - f_fit(x=theta_bins_fit, N0=N0_fit))**2/err_theta_fit**2)
    ndf = len(theta_bins_fit)-2
    chi2ndf = chi2/ndf
    err_N0_fit = np.sqrt(pcov[0][0])
    p_value = 1- stats.chi2.cdf(chi2ndf, 1)
    print(f"theta template fit:\nN0 = {N0_fit} +- {err_N0_fit}\nchi2/ndf = {chi2} / {ndf} = {chi2ndf}\np_value = {p_value}")

    # plot with fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue", label="Data")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    f_label = f"Fit to scaled template:\n$\\bullet$ $N_0={np.round(N0_fit,2):.2f}\\pm{np.round(err_N0_fit,2):.2f}$\n$\\bullet$ $\\chi^2/N_{{df}}={np.round(chi2,2)}\\;/\\;{ndf}={np.round(chi2ndf,2)}$\n$\\bullet$ $p={np.round(p_value,4):.4f}$"
    ax.step(theta_bins_fit, f_fit(theta_bins_fit, N0_fit), color="tab:red", label=f_label)
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins_fit, width=theta_bin_width, bottom=f_fit(theta_bins_fit, N0_fit)-err_f_fit(theta_bins_fit, N0_fit, err_N0_fit), height=2*err_f_fit(theta_bins_fit, N0_fit, err_N0_fit), align="center", color="tab:red", alpha=0.2)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$ (acceptance reweighted)")
    ax.legend()
    fig.tight_layout()
    fig.show()






    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

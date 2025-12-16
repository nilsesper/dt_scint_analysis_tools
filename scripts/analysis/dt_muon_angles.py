#################################################################
### dt muon angle analysis
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
        "--dt_muons_file",
        type     = str,
        help     = "input file path: dt muons (pcl file)",
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
        "--sim_dt_muons_file",
        type     = str,
        help     = "input file path: simulation dt muons (pcl file)",
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
    dt_muons_file = args.dt_muons_file
    sim_dt_muons_file = args.sim_dt_muons_file
    # other 
    arg_fig_size = args.fig_size.split(",")
    fig_size = (float(arg_fig_size[0]), float(arg_fig_size[1]))

    #################

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_muons = data_utils.load_pickle(file=dt_muons_file)
    n_dt_muons = data_utils.length(dt_muons)
    sim_dt_muons = data_utils.load_pickle(file=sim_dt_muons_file)
    n_sim_dt_muons = data_utils.length(sim_dt_muons)
            
    ### measurement duration
    duration = 0.78e-9 * (np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"])) # secs
    print(f"measurement duration = {duration} s")
    sim_duration = 0.78e-9 * (np.amax(sim_dt_muons["ts"]) - np.amin(sim_dt_muons["ts"])) # secs
    print(f"simulated measurement duration = {sim_duration} s")

    ### rate of muons
    muon_count = data_utils.length(dt_muons)
    print(f"dt muon count: {muon_count}")
    print(f"dt muon rate: {muon_count/duration} +- {np.sqrt(muon_count)/duration} Hz")
    sim_muon_count = data_utils.length(sim_dt_muons)
    print(f"simulated dt muon count: {sim_muon_count}")
    print(f"simulated dt muon rate: {sim_muon_count/sim_duration} +- {np.sqrt(sim_muon_count)/sim_duration} Hz")

    ##################
    ### solid angle weighted theta

    # calculate
    data = dt_muons["theta"]
    err_data = dt_muons["err_theta"]
    edges, n_bins, centers = hist_utils.generate_histogram_edges(arg="linear,0,1.58,100")
    hist, _, _, entries, underflow, overflow, _, _ = hist_utils.calculate_histogram_and_shifted_histograms(data=data, edges=edges, err_data=err_data)
    # solid angle factor
    hist = hist / np.sin(centers)
    err_hist = np.sqrt(hist) / np.sin(centers)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=hist, centers=centers, err_hist_down=err_hist, err_hist_up=err_hist, log_scale=False, add_info=True, entries=entries, overflow=overflow, underflow=underflow, bin_unit="rad", info_loc="top right")
    xlabel = "$\\theta$ [rad]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count / $\\sin\\theta$")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"DT_MUON_SPECIFIC_THETA_REWEIGHT.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    # calculate
    sim_data = sim_dt_muons["theta"]
    sim_err_data = sim_dt_muons["err_theta"]
    sim_edges, sim_n_bins, sim_centers = hist_utils.generate_histogram_edges(arg="linear,0,1.58,100")
    sim_hist, _, _, sim_entries, sim_underflow, sim_overflow, _, _ = hist_utils.calculate_histogram_and_shifted_histograms(data=sim_data, edges=sim_edges, err_data=sim_err_data)
    # solid angle factor
    sim_hist = sim_hist / np.sin(sim_centers)
    sim_err_hist = np.sqrt(sim_hist) / np.sin(sim_centers)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    ax = hist_utils.plot_histogram(ax=ax, hist=sim_hist, centers=sim_centers, err_hist_down=sim_err_hist, err_hist_up=sim_err_hist, log_scale=False, add_info=True, entries=sim_entries, overflow=sim_overflow, underflow=sim_underflow, bin_unit="rad", info_loc="top right")
    xlabel = "$\\theta$ [rad]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count / $\\sin\\theta$")
    fig.tight_layout()
    fig.show()
    ## store plot
    if args.store_path:
        hist_plot_file = args.store_path+"/"+f"DT_MUON_SPECIFIC_SIM_THETA_REWEIGHT.pdf"
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

    ##################
    ### cos^2(theta) fit

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


    ### cos2 fit

    # do fit
    def f_fit(x, N0):
        return N0*np.cos(x)**2
    def err_f_fit(x, N0, err_N0):
        return err_N0*np.cos(x)**2
    p0 = 1
    popt, pcov, infodict, _, _ = curve_fit(f_fit, xdata=theta_bins_fit, ydata=theta_fit, p0=p0, sigma=err_theta_fit, absolute_sigma=True, full_output=True)
    N0_fit = popt[0]
    chi2 = np.sum((theta_fit - f_fit(x=theta_bins_fit, N0=N0_fit))**2/err_theta_fit**2)
    ndf = len(theta_bins_fit)-2
    chi2ndf = chi2/ndf
    err_N0_fit = np.sqrt(pcov[0][0])
    p_value = 1- stats.chi2.cdf(chi2ndf, 1)
    print(f"theta cos^2 fit:\nN0 = {N0_fit} +- {err_N0_fit}\nchi2/ndf = {chi2} / {ndf} = {chi2ndf}\np_value = {p_value}")

    # plot with fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue", label="Data")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    f_label = f"Fit to $N(\\phi)=N_0\\cdot\\text{{cos}}^2\\theta$:\n$\\bullet$ $N_0={np.round(N0_fit,0):.0f}\\pm{np.round(err_N0_fit,0):.0f}$\n$\\bullet$ $\\chi^2/N_{{df}}={np.round(chi2,2)}\\;/\\;{ndf}={np.round(chi2ndf,2)}$\n$\\bullet$ $p={np.round(p_value,4):.4f}$"
    ax.plot(theta_bins_fit, f_fit(theta_bins_fit, N0_fit), color="tab:red", label=f_label)
    ax.fill_between(theta_bins_fit, y1=f_fit(theta_bins_fit, N0_fit)-err_f_fit(theta_bins_fit, N0_fit, err_N0_fit), y2=f_fit(theta_bins_fit, N0_fit)+err_f_fit(theta_bins_fit, N0_fit, err_N0_fit), color="tab:red", alpha=0.1)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$ (acceptance reweighted)")
    ax.legend()
    fig.tight_layout()
    fig.show()


    ### cosN fit

    # do fit
    def f_fit(x, N0, N):
        return N0*np.cos(x)**N
    def err_f_fit(x, N0, err_N0, N, err_N, cov):
        return np.sqrt(
              (err_N0*np.cos(x)**N)**2
            + (N*N0*np.cos(x)**(N-1)*err_N)**2
            + 2*(N*N0*np.cos(x)**(N-1)*err_N)*(err_N0*np.cos(x)**N)*cov)
    p0 = [1,2]
    popt, pcov, infodict, _, _ = curve_fit(f_fit, xdata=theta_bins_fit, ydata=theta_fit, p0=p0, sigma=err_theta_fit, absolute_sigma=True, full_output=True)
    N0_fit, N_fit = popt[0], popt[1]
    chi2 = np.sum((theta_fit - f_fit(x=theta_bins_fit, N0=N0_fit, N=N_fit))**2/err_theta_fit**2)
    ndf = len(theta_bins_fit)-2
    chi2ndf = chi2/ndf
    err_N0_fit, err_N_fit = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    cov_fit = pcov[0][1]
    p_value = 1- stats.chi2.cdf(chi2ndf, 1)
    print(f"theta cos^n fit:\nN0 = {N0_fit} +- {err_N0_fit}\nN = {N_fit} +- {err_N_fit}\nchi2/ndf = {chi2} / {ndf} = {chi2ndf}\np_value = {p_value}")
    
    # plot with fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue", label="Data")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    f_label = f"Fit to $N(\\phi)=N_0\\cdot\\text{{cos}}^n\\theta$:\n$\\bullet$ $N_0={np.round(N0_fit,0):.0f}\\pm{np.round(err_N0_fit,0):.0f}$\n$\\bullet$ $n={np.round(N_fit,2):.2f}\\pm{np.round(err_N_fit,2):.2f}$\n$\\bullet$ $\\chi^2/N_{{df}}={np.round(chi2,2)}\\;/\\;{ndf}={np.round(chi2ndf,2)}$\n$\\bullet$ $p={np.round(p_value,4):.4f}$"
    ax.plot(theta_bins_fit, f_fit(theta_bins_fit, N0_fit, N_fit), color="tab:red", label=f_label)
    ax.fill_between(theta_bins_fit, y1=f_fit(theta_bins_fit, N0_fit, N_fit)-err_f_fit(theta_bins_fit, N0_fit, err_N0_fit, N_fit, err_N_fit, cov_fit), y2=f_fit(theta_bins_fit, N0_fit, N_fit)+err_f_fit(theta_bins_fit, N0_fit, err_N0_fit, N_fit, err_N_fit, cov_fit), color="tab:red", alpha=0.1)
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

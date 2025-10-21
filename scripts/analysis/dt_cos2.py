#################################################################
### analyze theta distribution of dt muons
#######################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
from scipy.optimize import curve_fit
import scipy.stats as stats

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_muons_file",
        type     = str,
        help     = "input file path: reco dt muons from this simulated cosmic muon dataset (pcl file)",
    )
    parser.add_argument(
        "--geom_acceptance_file",
        type     = str,
        help     = "input file path: calculated geom acceptance factors (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_muons_file = args.dt_muons_file
    geom_acceptance_file = args.geom_acceptance_file
    
    #################

    ### data import
    print(f"###### Importing data...")
    geom_acceptance = data_utils.load_pickle(file=geom_acceptance_file)
    dt_muons = data_utils.load_pickle(file=dt_muons_file)


    """
    theta_bins = geom_acceptance["theta_bins"][1:]
    eff_theta = geom_acceptance["eff_theta"][1:]
    err_eff_theta = geom_acceptance["err_eff_theta"][1:]

    ### create theta hist
    dt_muons_theta_hists, dt_muons_theta_edges, dt_muons_theta_centers, dt_muons_theta_underflow, dt_muons_theta_overflow = hist_utils.calculate_hist(data=dt_muons, key="theta", bin_centers=theta_bins, silent=True)
    print(f"dt_muons: key \"theta\": entries={data_utils.length(dt_muons)} underflow={dt_muons_theta_underflow}, overflow={dt_muons_theta_overflow}")

    ### apply solid angle correction and geom acceptance factor
    theta = np.zeros(len(theta_bins))
    err_theta = np.zeros(len(theta_bins))
    for i in range(len(theta_bins)):
        if eff_theta[i] == 0:
            continue
        theta[i] = dt_muons_theta_hists[i] / np.sin(theta_bins[i]) / eff_theta[i]
        err_theta[i] = np.sqrt( 
            (1 / np.sin(theta_bins[i]) / eff_theta[i])**2 * np.sqrt(dt_muons_theta_hists[i])**2
            + (-dt_muons_theta_hists[i] / np.sin(theta_bins[i]) / eff_theta[i]**2)**2 * err_eff_theta[i]**2
        )
    #err_dt_muons_theta_hists = np.clip(err_dt_muons_theta_hists, 0, dt_muons_theta_hists)

    ### plot theta
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$")
    fig.tight_layout()
    fig.show()

    ### cos^2(theta) fit
    # restrict data to good data for fit
    idcs_fit = np.where(theta > 0)
    theta_bins_fit = theta_bins[idcs_fit]
    theta_fit = theta[idcs_fit]
    err_theta_fit = err_theta[idcs_fit]
    print(theta_bins_fit, theta_fit, err_theta_fit)
    print(len(theta_bins_fit), len(theta_fit), len(err_theta_fit))
    # do fit
    def f_fit(x, N0, n):
        return N0*np.cos(x)**n
    p0 = (theta[0], 2)
    popt, pcov, infodict, _, _ = curve_fit(f_fit, xdata=theta_bins_fit, ydata=theta_fit, p0=p0, sigma=err_theta_fit, absolute_sigma=True, full_output=True)
    N0_fit, n_fit = popt
    chi2 = np.sum((theta_fit - f_fit(x=theta_bins_fit, N0=N0_fit, n=n_fit))/err_theta_fit)**2
    ndf = len(theta_bins_fit)-2
    chi2ndf = chi2/ndf
    print(popt, chi2ndf)
    
    ### plot theta with fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    ax.plot(theta_bins, f_fit(theta_bins, N0_fit, n_fit), color="tab:red")
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$")
    fig.tight_layout()
    fig.show()
    #"""

    theta_edges = geom_acceptance["theta_edges"]
    theta_bins = geom_acceptance["theta_bins"]
    phi_edges = geom_acceptance["phi_edges"]
    phi_bins = geom_acceptance["phi_bins"]
    ratio_hist2d = geom_acceptance["ratio_hist2d"]
    err_ratio_hist2d = geom_acceptance["err_ratio_hist2d"]

    ## hist2d for dt muons
    dt_muons_hist2d, _, _ = np.histogram2d(x=dt_muons["theta"], y=dt_muons["phi"], bins=(theta_edges, phi_edges))
    # solid angle correction
    for i in range(len(dt_muons_hist2d)):
        dt_muons_hist2d[i] = dt_muons_hist2d[i] / np.sin(theta_bins[i])
    err_dt_muons_hist2d = np.sqrt(dt_muons_hist2d)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=dt_muons_hist2d, origin="lower", extent=[min(phi_bins), max(phi_bins), min(theta_bins), max(theta_bins)])
    ax.set_title("$N_\\text{reco muons}$")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xlabel("$\\phi$ [rad]")
    #ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()

    ### apply acceptance factor
    corr_hist2d = np.zeros(np.shape(dt_muons_hist2d))
    err_corr_hist2d = np.zeros(np.shape(dt_muons_hist2d))
    for i in range(len(dt_muons_hist2d)):
        for j in range(len(dt_muons_hist2d[i])):
            if ratio_hist2d[i][j] != 0:
                corr_hist2d[i][j] = dt_muons_hist2d[i][j] / ratio_hist2d[i][j]
                err_corr_hist2d[i][j] = np.sqrt(
                      (1 / ratio_hist2d[i][j])**2 * dt_muons_hist2d[i][j]
                    + (dt_muons_hist2d[i][j] / ratio_hist2d[i][j]**2)**2 * err_ratio_hist2d[i][j]**2
                )
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=corr_hist2d, origin="lower", extent=[min(phi_bins), max(phi_bins), min(theta_bins), max(theta_bins)])
    ax.set_title("$N_\\text{reco muons}$ (acceptance reweighted)")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xlabel("$\\phi$ [rad]")
    #ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()

    before_after_weight = np.sum(corr_hist2d)/np.sum(dt_muons_hist2d)

    ### summed 1d theta bins
    theta = np.sum(corr_hist2d, axis=1)
    err_theta = np.sum(err_corr_hist2d, axis=1)
    theta_before = np.sum(dt_muons_hist2d, axis=1) * before_after_weight
    err_theta_before = np.sum(err_dt_muons_hist2d, axis=1) * before_after_weight
    # plot theta
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    theta_bin_width = theta_bins[1]-theta_bins[0]
    # before acceptance corr
    ax.step(x=theta_bins, y=theta_before, where="mid", color="tab:orange", label="Measurement (normalized)")
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta_before-err_theta_before, height=2*err_theta_before, align="center", color="tab:orange", alpha=0.2)
    # after acceptance corr
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue", label="Acceptance reweighted")
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$")
    ax.legend()
    fig.tight_layout()
    fig.show()
    ## cos^2(theta) fit
    # restrict data to good data for fit
    idcs_fit = np.where(theta > 0)
    theta_bins_fit = theta_bins[idcs_fit]
    theta_fit = theta[idcs_fit]
    err_theta_fit = err_theta[idcs_fit]
    # do fit
    def f_fit(x, N0):
        return N0*np.cos(x)**2
    def err_f_fit(x, N0, err_N0):
        return err_N0*np.cos(x)**2
    p0 = theta[0]
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
    ax.step(x=theta_bins, y=theta, where="mid", color="tab:blue", label="Acceptance reweighted")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=theta_bins, width=theta_bin_width, bottom=theta-err_theta, height=2*err_theta, align="center", color="tab:blue", alpha=0.2)
    f_label = f"Fit to $N(\\phi)=N_0\\cdot\\text{{cos}}^2\\theta$:\n$\\bullet$ $N_0={np.round(N0_fit,0):.0f}\\pm{np.round(err_N0_fit,0):.0f}$\n$\\bullet$ $\\chi^2/N_{{df}}={np.round(chi2,2)}\\;/\\;{ndf}={np.round(chi2ndf,2)}$\n$\\bullet$ $p={np.round(p_value,4):.4f}$"
    ax.plot(theta_bins, f_fit(theta_bins, N0_fit), color="tab:red", label=f_label)
    ax.fill_between(theta_bins, y1=f_fit(theta_bins, N0_fit)-err_f_fit(theta_bins, N0_fit, err_N0_fit), y2=f_fit(theta_bins, N0_fit)+err_f_fit(theta_bins, N0_fit, err_N0_fit), color="tab:red", alpha=0.1)
    ax.set_xlabel(f"{params._key_symbols['theta']} [{params._key_units['theta']}]")
    ax.set_ylabel("$N_\\text{reco muons}$ (acceptance reweighted)")
    ax.legend()
    fig.tight_layout()
    fig.show()

    ### summed 1d phi bins
    phi = np.sum(corr_hist2d, axis=0)
    err_phi = np.sum(err_corr_hist2d, axis=0)
    phi_before = np.sum(dt_muons_hist2d, axis=0) * before_after_weight
    err_phi_before = np.sum(err_dt_muons_hist2d, axis=0) * before_after_weight
    # plot theta
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    phi_bin_width = phi_bins[1]-phi_bins[0]
    # before acceptance corr
    ax.step(x=phi_bins, y=phi_before, where="mid", color="tab:orange", label="Measurement (normalized)")
    ax.bar(x=phi_bins, width=phi_bin_width, bottom=phi_before-err_phi_before, height=2*err_phi_before, align="center", color="tab:orange", alpha=0.2)
    # after acceptance corr
    ax.step(x=phi_bins, y=phi, where="mid", color="tab:blue", label="Acceptance reweighted")
    ax.bar(x=phi_bins, width=phi_bin_width, bottom=phi-err_phi, height=2*err_phi, align="center", color="tab:blue", alpha=0.2)
    ax.set_xlabel(f"{params._key_symbols['phi']} [{params._key_units['phi']}]")
    ax.set_ylabel("$N_\\text{reco muons}$")
    ax.legend()
    fig.tight_layout()
    fig.show()
    ## flat fit
    # restrict data to good data for fit
    idcs_fit = np.where(phi > 0)
    phi_bins_fit = phi_bins[idcs_fit]
    phi_fit = phi[idcs_fit]
    err_phi_fit = err_phi[idcs_fit]
    # do fit
    def f_fit(x, N0):
        return N0*np.ones(np.shape(x))
    def err_f_fit(x, N0, err_N0):
        return err_N0*np.ones(np.shape(x))
    p0 = phi[0]
    popt, pcov, infodict, _, _ = curve_fit(f_fit, xdata=phi_bins_fit, ydata=phi_fit, p0=p0, sigma=err_phi_fit, absolute_sigma=True, full_output=True)
    N0_fit = popt[0]
    chi2 = np.sum((phi_fit - f_fit(x=phi_bins_fit, N0=N0_fit))**2/err_phi_fit**2)
    ndf = len(phi_bins_fit)-1
    chi2ndf = chi2/ndf
    err_N0_fit = np.sqrt(pcov[0][0])
    p_value = 1- stats.chi2.cdf(chi2ndf, 1)
    print(f"phi flat fit:\nN0 = {N0_fit} +- {err_N0_fit}\nchi2/ndf = {chi2} / {ndf} = {chi2ndf}\np_value = {p_value}")
    # plot with fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.step(x=phi_bins, y=phi, where="mid", color="tab:blue", label="Acceptance reweighted")
    theta_bin_width = theta_bins[1]-theta_bins[0]
    ax.bar(x=phi_bins, width=phi_bin_width, bottom=phi-err_phi, height=2*err_phi, align="center", color="tab:blue", alpha=0.2, ) #label="Data uncertainty")
    f_label = f"Fit to $N(\\phi)=N_0$:\n$\\bullet$ $N_0={np.round(N0_fit,0):.0f}\\pm{np.round(err_N0_fit,0):.0f}$\n$\\bullet$ $\\chi^2/N_{{df}}={np.round(chi2,2)}\\;/\\;{ndf}={np.round(chi2ndf,2)}$\n$\\bullet$ $p={np.round(p_value,4):.4f}$"
    ax.plot(phi_bins, f_fit(phi_bins, N0_fit), color="tab:red", label=f_label)
    ax.fill_between(phi_bins, y1=f_fit(phi_bins, N0_fit)-err_f_fit(phi_bins, N0_fit, err_N0_fit), y2=f_fit(phi_bins, N0_fit)+err_f_fit(phi_bins, N0_fit, err_N0_fit), color="tab:red", alpha=0.1, ) #label="Fit uncertainty")
    ax.set_xlabel(f"{params._key_symbols['phi']} [{params._key_units['phi']}]")
    ax.set_ylabel("$N_\\text{reco muons}$ (acceptance reweighted)")
    ax.legend()
    fig.tight_layout()
    fig.show()



    input("Press enter to exit.")
    exit()



if __name__ == "__main__":
    main()
    print(f"###### Done.")

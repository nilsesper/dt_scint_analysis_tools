#################################################################
### analyze dt hit differences
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
from scipy.optimize import curve_fit

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_hit_differences_file",
        type     = str,
        help     = "input file path: dt hit timestamp differences (pcl file)",
    )
    # ---
    args = parser.parse_args()
    dt_hit_differences_file = args.dt_hit_differences_file

    #################

    cell_half_width = 21000 # um
    err_cell_half_width = 100 # um

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_hit_differences = data_utils.load_pickle(file=dt_hit_differences_file)
    hist = dt_hit_differences["hist"]
    err_hist = dt_hit_differences["err_hist"]
    bins = dt_hit_differences["bins"]*0.78 # convert to ns

    ### plot dt hit differences
    # plot hist
    fig, ax = plt.subplots(1, 1, figsize=(12,7))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing) # relative spacing between bins
    ax.bar(bins, hist, width=barwidth, align="center")
    ax.set_xlim(0,np.amax(bins))
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.show()

    input("Press enter to exit.")
    exit()

    ### remove exponential "poisson" background
    fit_index_range = np.arange(int(1000/0.78),len(bins)-1)
    fit_bins = bins[fit_index_range]
    fit_hist = hist[fit_index_range]
    err_fit_hist = err_hist[fit_index_range]
    def f_bg_fit(x, a, b):
        return a*np.exp(-x/b)
    def err_f_bg_fit(x, a, b, err_a, err_b):
        return np.sqrt( (err_a*np.exp(-x/b))**2 + (-1/b*a*np.exp(-x/b)*err_b)**2 )
    p0 = (1000, 100)
    popt, pcov, infodict, mesg, _ = curve_fit(f=f_bg_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True, )
    a_fit, b_fit = popt
    err_a_fit = np.sqrt(pcov[0][0])
    err_b_fit = np.sqrt(pcov[1][1])
    chi2 = np.sum((fit_hist - f_bg_fit(x=fit_bins, a=a_fit, b=b_fit))**2/err_fit_hist**2)
    ndf = len(fit_hist)-2
    chi2ndf = chi2/ndf
    print(f"exp fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) ns")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    ## plot fit, with residual plot
    fig, ax = plt.subplots(2, 1, figsize=(12,7), sharex=True, height_ratios=(5,1))
    rel_spacing = 0
    # main plot
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing) # relative spacing between bins
    ax[0].bar(bins, hist, width=barwidth, align="center", label="Drift cell data")
    fit_label = f"""Exponential fit:
$f(\\Delta T) = a \\cdot e^{{-x/b}}$
$a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$
$b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
$\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
    ax[0].plot(fit_bins, f_bg_fit(fit_bins, a=a_fit, b=b_fit), color="tab:red", label=fit_label)
    ax[0].plot(bins[0:fit_index_range[0]], f_bg_fit(bins[0:fit_index_range[0]], a=a_fit, b=b_fit), color="tab:red", linestyle="--", label="Extrapolated fit")
    ax[0].fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
    ax[0].legend(loc="lower right")
    # residual plot
    residuals = fit_hist - f_bg_fit(fit_bins, a=a_fit, b=b_fit)
    err_residuals = err_fit_hist
    ax[1].axhline(y=0, color="gray", linewidth=1)
    ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1)
    # show plot
    ax[1].set_xlim(0,np.amax(bins))
    ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.show()

    ### subtract exp bg
    hist_nobg = hist - f_bg_fit(bins, a=a_fit, b=b_fit)
    err_hist_nobg = np.sqrt(
          err_hist**2 # poisson error
          + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 # bg subtraction error
    )
    bins_nobg = bins

    # plot wo bg
    fig, ax = plt.subplots(1, 1, figsize=(12,7))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing) # relative spacing between bins
    ax.bar(bins_nobg, hist_nobg, width=barwidth, align="center", label="Background subtracted")
    #ax.set_yscale("log")
    #ax.set_ylim(bottom=0.5, top=np.amax(hist_nobg)*np.exp(1.1))
    ax.set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax.set_xlim(0,600)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    ax.legend()
    fig.tight_layout()
    fig.show()

    input("Press enter to exit.")
    exit()

    ### fit parabola photopeak to determine position
    fit_index_range = np.arange(int(400/0.78), int(425/0.78))
    fit_bins = bins_nobg[fit_index_range]
    fit_hist = hist_nobg[fit_index_range]
    err_fit_hist = err_hist_nobg[fit_index_range]
    def f_peak_fit(x, a, b, c):
        return a*(x-b)**2+c
    def err_f_peak_fit(x, a, b, c, err_a, err_b, err_c):
        return np.sqrt( ( err_a*(x-b)**2 )**2 + ( -2*a*(x-b)*err_b )**2 + ( err_c )**2 )
    p0 = (-1, 500, 1000)
    popt, pcov, infodict, mesg, _ = curve_fit(f=f_peak_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True, )
    a_fit, b_fit, c_fit = popt
    err_a_fit = np.sqrt(pcov[0][0])
    err_b_fit = np.sqrt(pcov[1][1])
    err_c_fit = np.sqrt(pcov[2][2])
    chi2 = np.sum((fit_hist - f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit))**2/err_fit_hist**2)
    ndf = len(fit_hist)-2
    chi2ndf = chi2/ndf
    print(f"parabola fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) TU")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  c = {c_fit} +- {err_c_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    # plot fit
    fig, ax = plt.subplots(2, 1, figsize=(12,7), sharex=True, height_ratios=(5,1))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing) # relative spacing between bins
    ax[0].bar(bins_nobg, hist_nobg, width=barwidth, align="center", label="Background subtracted")
    fit_label = f"""Parabolic fit:
$f(\\Delta T) = a \\cdot (\\Delta T-b)^2+c$
$a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$ 1/ns${{}}^2$
$b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
$c=({np.round(c_fit,0):.0f}\\pm{np.round(err_c_fit,0):.0f})$
$\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
    ax[0].plot(fit_bins, f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit), color="tab:red", label=fit_label)
    ax[0].fill_between(fit_bins, y1=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)-err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), y2=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)+err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), color="tab:red", alpha=0.1)
    ax[0].axvline(x=b_fit, color="tab:red", linestyle="--", label="Peak position $b$")
    ax[0].axvspan(xmin=b_fit-err_b_fit, xmax=b_fit+err_b_fit, color="tab:red", alpha=0.1)
    #ax.set_yscale("log")
    ax[0].set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax[0].legend(loc="lower left")
    # residual plot
    residuals = fit_hist - f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit)
    err_residuals = err_fit_hist
    ax[1].axhline(y=0, color="gray", linewidth=1)
    ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1)
    # show plot
    ax[1].set_xlim(0,600)
    ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    fig.show()

    ### estimate drift velocity
    v_drift = cell_half_width / b_fit # um/ns
    err_v_drift = np.sqrt(
          (-cell_half_width/b_fit**2)**2 * err_b_fit**2 # peak position error
        + (1/b_fit)**2 * err_cell_half_width**2 # drift space error
    )
    print(f"v_drift = {v_drift} +- {err_v_drift} um/ns")

    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

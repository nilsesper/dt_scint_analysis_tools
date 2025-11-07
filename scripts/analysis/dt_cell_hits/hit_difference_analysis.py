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

    ### data import
    print(f"###### Importing all data...")
    # dt
    dt_hit_differences = data_utils.load_pickle(file=dt_hit_differences_file)
    hist = dt_hit_differences["hist"]
    err_hist = dt_hit_differences["err_hist"]
    bins = dt_hit_differences["bins"]

    ### plot dt hit differences
    # full bin range
    hist_utils.plot_1hist(hist=hist, centers=bins, xlabel="$\\Delta T_\\text{cell}$ [TU]", silent=True, store=False, show=True, bin_labels=False, scale="log")
    
    ### remove exponential "poisson" background
    fit_index_range = np.arange(1000,len(dt_hit_differences["bins"]-1))
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
    print(f"exp fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) TU")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    # plot exp fit
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing) # relative spacing between bins
    ax.bar(bins, hist, width=barwidth, align="center")
    plot_range = np.linspace(np.amin(bins), np.amax(bins), 10000)
    ax.plot(bins, f_bg_fit(bins, a=a_fit, b=b_fit), color="tab:red")
    ax.fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [TU]")
    fig.tight_layout()
    fig.show()

    ### subtract exp bg
    hist_nobg = hist - f_bg_fit(bins, a=a_fit, b=b_fit)
    err_hist_nobg = np.sqrt(err_hist**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2)

    # plot wo bg
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing) # relative spacing between bins
    ax.bar(bins, hist_nobg, width=barwidth, align="center")
    #ax.set_yscale("log")
    #ax.set_ylim(bottom=0.5, top=np.amax(hist_nobg)*np.exp(1.1))
    ax.set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [TU]")
    fig.tight_layout()
    fig.show()


    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

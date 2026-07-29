#################################################################
### import dumpfile and extract dt hits (and optionally raw scint hits)
# store dt hits (and optionally raw scint hits) as pkl file
#################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params   #, params_justus

import subprocess
import atexit
import sys
import time
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import re
from datetime import datetime

import matplotlib.dates as mdates


# =================================================================
# Robust secondary-peak fitting helpers
# (Parabola fit around the peak region, no valley-to-valley logic
#  -- see fit_secondary_peak_parabola for details / caveats)
# =================================================================

def parabola_vertex_form(x, A, mu, c):
    """Downward-opening parabola in vertex form: y = A - c*(x-mu)^2, with
    peak amplitude A at position mu and curvature c (c > 0 for a real
    peak). Fitting directly in this form -- with mu as its own bounded
    parameter -- is what keeps the fit from "running away": unlike the
    a2*x^2+a1*x+a0 form, mu can no longer land far outside the fit window
    just because a1/a2 happened to extrapolate there."""
    return A - c * (x - mu) ** 2


def err_parabola_vertex_form(x, A, mu, c, err_A, err_mu, err_c):
    dx = x - mu
    df_dA = np.ones_like(x)
    df_dmu = 2 * c * dx
    df_dc = -dx ** 2
    return np.sqrt(
        (df_dA * err_A) ** 2
        + (df_dmu * err_mu) ** 2
        + (df_dc * err_c) ** 2
    )


def find_secondary_peak(bins, hist, min_x, max_x, prominence_frac=0.03, smooth_window=5):
    """Locate the most prominent local maximum in [min_x, max_x] on a
    smoothed copy of the histogram, and return rough (mu, sigma, A)
    estimates to seed the fit."""
    mask = (bins >= min_x) & (bins <= max_x)
    x = bins[mask]
    y = hist[mask]

    if len(x) < smooth_window + 2:
        raise RuntimeError("Search window contains too few bins to find a peak.")

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y

    prominence = prominence_frac * np.amax(y_smooth)
    peaks, props = find_peaks(y_smooth, prominence=prominence)

    if len(peaks) == 0:
        raise RuntimeError(
            f"No peak found in x=({min_x}, {max_x}) with prominence_frac={prominence_frac}. "
            "Try lowering prominence_frac or widening the search window."
        )

    best = peaks[np.argmax(props["prominences"])]

    widths, _, _, _ = peak_widths(y_smooth, [best], rel_height=0.5)
    dx = np.mean(np.diff(x))
    fwhm_x = widths[0] * dx
    sigma_est = max(fwhm_x / 2.3548, dx)

    mu_est = x[best]
    A_est = y_smooth[best]

    return mu_est, sigma_est, A_est


def fit_secondary_peak_parabola(
    bins_nobg,
    hist_nobg,
    err_hist_nobg,
    search_min,
    search_max,
    window_sigmas_left=1.5,
    window_sigmas_right=3.0,
    min_halfwidth_left_ns=None,
    min_halfwidth_right_ns=None,
    edge_margin_frac=0.15,
    prominence_frac=0.03,
    smooth_window=5,
    max_attempts=4,
    window_growth=1.3,
    min_bins=6,
    verbose=True,
):
    """Fit a parabola (a2*x^2 + a1*x + a0) directly to the region around
    the secondary peak.

    IMPORTANT: unlike fit_secondary_peak() (the old Gaussian version),
    this function does NOT use a "valley-to-valley" (Tal-zu-Tal) search to
    define the fit boundaries. The fit window is instead a small,
    peak-centered window around the roughly located peak position. This
    is intentionally much less robust:
    - There is no check that the window actually contains a real peak
      (no valley on either side has to be crossed), so if the peak is
      barely distinguishable from the background continuum the fit can
      easily lock onto a random up/down fluctuation of the background
      instead of the true peak.
    - The parabola has no floor/background term beyond its own curvature
      (a2*x^2+a1*x+a0), so any residual slope/curvature of the true
      background inside the window biases the fitted vertex position.
    - A parabola fit is only sensible sufficiently close to the peak; if
      the window is chosen too wide the parabola will systematically
      undershoot/overshoot the true peak position, but if it is chosen
      too narrow the fit becomes very sensitive to statistical noise in
      individual bins.
    - The fit is done in VERTEX FORM, y = A - c*(x-mu)^2, with mu bounded
      to lie within the fit window. This is a deliberate safeguard: with
      the naive a2*x^2+a1*x+a0 form, mu is only computed *after* the fit
      as -a1/(2*a2), and if the window contains little real peak
      curvature (e.g. just a slight bend in an otherwise smooth
      background), a1/a2 can fit "fine" in a chi2 sense while their ratio
      extrapolates to a vertex far outside the window entirely -- giving
      a wildly wrong drift-time estimate with no error/exception raised.
      Bounding mu directly prevents that failure mode; a weak/absent peak
      now instead shows up as mu getting pushed to a window edge or as
      c coming out <= 0, both of which raise a caught, retryable error.
    Use with caution and always inspect the resulting plot/residuals.

    Controlling the fit window width (now ASYMMETRIC left/right, since
    the photopeak typically has a longer tail to the right -- towards the
    background hump -- than to the left towards the valley):
    - window_sigmas_left / window_sigmas_right: base half-widths, in
      units of sigma_est (the width automatically estimated from the raw
      peak's FWHM), applied separately to the left and right side of
      mu_est. Raise window_sigmas_right if the fit curve visibly stops
      short of the peak's right-hand shoulder.
    - min_halfwidth_left_ns / min_halfwidth_right_ns: absolute floors (in
      ns) on the respective half-width, on top of
      window_sigmas_**sigma_est. Useful when sigma_est itself is
      underestimated (common for weak/noisy peaks), since the *_sigmas
      parameters alone can't compensate for a bad sigma_est.
    - edge_margin_frac: even if the fit converges, it is treated as "too
      small" and automatically widened (via window_growth, applied to
      both sides) if the fitted vertex mu lands within this fraction of
      either edge of the window. This only catches mu sitting near an
      edge -- it does NOT catch a window that is simply too narrow
      overall while mu still sits comfortably in the middle (that's what
      window_sigmas_right / min_halfwidth_right_ns are for).
    """
    mu_est, sigma_est, A_est = find_secondary_peak(
        bins_nobg, hist_nobg,
        min_x=search_min, max_x=search_max,
        prominence_frac=prominence_frac, smooth_window=smooth_window,
    )

    win_left = window_sigmas_left
    win_right = window_sigmas_right
    last_exc = None

    for attempt in range(max_attempts):
        halfwidth_left = win_left * sigma_est
        if min_halfwidth_left_ns is not None:
            halfwidth_left = max(halfwidth_left, min_halfwidth_left_ns)

        halfwidth_right = win_right * sigma_est
        if min_halfwidth_right_ns is not None:
            halfwidth_right = max(halfwidth_right, min_halfwidth_right_ns)

        fit_min = mu_est - halfwidth_left
        fit_max = mu_est + halfwidth_right
        fit_mask = (bins_nobg >= fit_min) & (bins_nobg <= fit_max)

        fit_bins = bins_nobg[fit_mask]
        fit_hist = hist_nobg[fit_mask]
        err_fit_hist = err_hist_nobg[fit_mask]

        if len(fit_bins) < min_bins:
            if verbose:
                print(f"    (diagnostic) fit window has only {len(fit_bins)} bins "
                      f"(< {min_bins}), range = ({fit_min:.2f}, {fit_max:.2f}) ns -- widening window")
            win_left *= window_growth
            win_right *= window_growth
            last_exc = RuntimeError("fit window contains too few bins")
            continue

        # seed values for a downward-opening parabola through (mu_est, A_est),
        # in vertex form
        c_0 = A_est / max(sigma_est, 1e-6) ** 2
        p0 = (A_est, mu_est, c_0)

        # mu is bounded to the fit window itself -- this is the key change
        # vs. the old a2*x^2+a1*x+a0 form: it makes it impossible for the
        # fitted vertex to end up outside the window, no matter how weak
        # or absent the real peak signal is in this window. A weak/absent
        # peak will instead show up as mu getting pushed to (or very near)
        # one of the bounds, or as c coming out <= 0 -- both of which are
        # caught by the checks below.
        lower = (0.0, fit_bins.min(), 0.0)
        upper = (np.inf, fit_bins.max(), np.inf)

        try:
            popt, pcov = curve_fit(
                parabola_vertex_form, fit_bins, fit_hist,
                p0=p0, sigma=err_fit_hist, absolute_sigma=True,
                bounds=(lower, upper), maxfev=20000,
            )

            A_fit, mu_fit, c_fit = popt
            err_mu_fit = np.sqrt(pcov[1][1])
            # -----------------------------------------
            # Refit in ± n_bins around fitted peak
            # -----------------------------------------
            bin_width = np.mean(np.diff(bins_nobg))
            half_width = 9 * bin_width

            refit_mask = (
                (bins_nobg >= mu_fit - half_width) &
                (bins_nobg <= mu_fit + half_width)
            )

            refit_bins = bins_nobg[refit_mask]
            refit_hist = hist_nobg[refit_mask]
            refit_err = err_hist_nobg[refit_mask]

            # erster Fit als Startwert
            p0_refit = (A_fit, mu_fit, c_fit)

            # mu darf sich nur innerhalb des Refit-Fensters bewegen
            lower = (0.0, mu_fit - half_width, 0.0)
            upper = (np.inf, mu_fit + half_width, np.inf)

            popt_refit, pcov_refit = curve_fit(
                parabola_vertex_form,
                refit_bins,
                refit_hist,
                p0=p0_refit,
                sigma=refit_err,
                absolute_sigma=True,
                bounds=(lower, upper),
                maxfev=20000,
            )

            A_fit, mu_fit, c_fit = popt_refit
            err_A, err_mu_fit, err_c = np.sqrt(np.diag(pcov_refit))
            popt = popt_refit
            pcov = pcov_refit
            fit_bins = refit_bins
            fit_hist = refit_hist
            err_fit_hist = refit_err

            if c_fit <= 0:
                raise RuntimeError(
                    f"c={c_fit:.4g} <= 0 -- parabola opens upward or is flat, i.e. this "
                    "window contains a local minimum/no curvature, not a peak"
                )
            if A_fit <= 0:
                raise RuntimeError(f"fitted amplitude A={A_fit:.4g} <= 0 -- not a real peak")
            if not np.isfinite(err_mu_fit) or err_mu_fit <= 0:
                raise RuntimeError("could not determine a finite, positive error on mu")

            window_width = fit_bins.max() - fit_bins.min()
            dist_to_left_edge = mu_fit - fit_bins.min()
            dist_to_right_edge = fit_bins.max() - mu_fit
            if (dist_to_left_edge < edge_margin_frac * window_width
                    or dist_to_right_edge < edge_margin_frac * window_width):
                raise RuntimeError(
                    f"vertex mu={mu_fit:.2f} ns sits too close to the edge of the fit "
                    f"window ({fit_bins.min():.2f}, {fit_bins.max():.2f}) ns "
                    f"(margin < {edge_margin_frac*100:.0f}% of window width) -- "
                    "window is likely too small (or there's no real peak here), "
                    "the fit is being pulled/clamped towards the boundary"
                )

            fit_vals = parabola_vertex_form(fit_bins, *popt)
            chi2 = np.sum((fit_hist - fit_vals) ** 2 / err_fit_hist ** 2)
            ndf = len(fit_hist) - len(popt)
            chi2ndf = chi2 / ndf if ndf > 0 else np.inf

            if verbose:
                print(f"    (diagnostic) parabola fit window = ({fit_min:.1f}, {fit_max:.1f}) ns "
                      f"[halfwidth_left={halfwidth_left:.1f} ns, halfwidth_right={halfwidth_right:.1f} ns], "
                      f"mu = {mu_fit:.2f} +- {err_mu_fit:.2g} ns, chi2/ndf = {chi2:.1f}/{ndf} = {chi2ndf:.2f}")

            return popt, pcov, fit_bins, fit_hist, err_fit_hist, parabola_vertex_form, mu_fit, err_mu_fit

        except Exception as exc:  # noqa: BLE001 -- intentionally broad, we retry
            last_exc = exc
            win_left *= window_growth
            win_right *= window_growth

    raise RuntimeError(
        f"Parabola fit around peak (mu_est={mu_est:.1f} ns, sigma_est={sigma_est:.1f} ns) "
        f"did not converge after {max_attempts} attempts. Last error: {last_exc}\n"
        "The peak may be too weak / too close to the background here for a "
        "narrow, un-anchored parabola fit -- consider widening "
        "window_sigmas_left/window_sigmas_right (or the min_halfwidth_*_ns floors), "
        "lowering prominence_frac, or falling back to the Gaussian valley-based fit."
    )


def parse_fit_name(*, name):
        # Erwartetes Format: cosmic_<Ar>-<CO2>_<U_wire>-<U_Fieldshaper>-<U_cathode>_<rest...>
        pattern = r"^cosmic_(\d+)-(\d+)_(\d+)-(\d+)-(\d+)"
        match = re.match(pattern, name)
        if not match:
            raise ValueError(f"String hat nicht das erwartete Format: {name}")

        pct_ar, pct_co2, u_wire, u_fieldshaper, u_cathode = match.groups()

        return {
            "name": name,
            "pct_Ar": int(pct_ar),
            "pct_CO2": int(pct_co2),
            "U_wire": int(u_wire),
            "U_Fieldshaper": int(u_fieldshaper),
            "U_cathode": int(u_cathode),
        }

def parse_start_time(dataset_name: str) -> datetime:
            """
            Extract the start timestamp from a dataset name.

            Example:
                data_mic0_start_2026-07-24_18-06-10_stop_2026-07-24_18-16-11
                -> datetime(2026, 7, 24, 18, 6, 10)
            """
            match = re.search(r"start_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", dataset_name)
            if match is None:
                raise ValueError(f"Invalid dataset name: {dataset_name}")

            return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    ###################################################
    do_ramp_measurement = True

    list_of_fits = [#"cosmic_82-18_3550-1800-1200_run1_th20_cut100", no peak
                #"cosmic_82-18_3575-1800-1200_run1_th20_cut100", no peak
                #"cosmic_82-18_3600-1800-1200_run1_th20_cut100", no peak
                "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 

                #"cosmic_83-17_3650-1800-1200_run1_th20_cut100",not calculated, in calculation
                "cosmic_83-17_3625-1800-1200_run1_th20_cut100", 
                "cosmic_83-17_3600-1800-1200_run1_th20_cut100", 
                #"cosmic_83-17_3575-1800-1200_run1_th20_cut100",no peak
                #"cosmic_83-17_3550-1800-1200_run1_th20_cut100",no peak

                #"cosmic_85-15_3550-1800-1200_run1_th20_cut100", no peak
                "cosmic_85-15_3575-1800-1200_run1_th20_cut100", 
                "cosmic_85-15_3600-1800-1200_run2_th20_cut100"]



    ramp_datasets = [ "data_mic0_start_2026-07-24_18-06-10_stop_2026-07-24_18-16-11",
                    "data_mic0_start_2026-07-24_22-16-13_stop_2026-07-24_22-26-14",
                    "data_mic0_start_2026-07-25_02-26-16_stop_2026-07-25_02-36-17",
                    "data_mic0_start_2026-07-25_06-36-19_stop_2026-07-25_06-46-20",
                    "data_mic0_start_2026-07-25_10-46-22_stop_2026-07-25_10-56-23",
                    "data_mic0_start_2026-07-25_14-56-25_stop_2026-07-25_15-06-26",
                    "data_mic0_start_2026-07-25_19-06-28_stop_2026-07-25_19-16-29",
                    "data_mic0_start_2026-07-25_23-16-31_stop_2026-07-25_23-26-32",
                    "data_mic0_start_2026-07-26_03-26-34_stop_2026-07-26_03-36-35",
                    "data_mic0_start_2026-07-26_07-36-37_stop_2026-07-26_07-46-38",
                    "data_mic0_start_2026-07-26_11-46-40_stop_2026-07-26_11-56-41",
                    "data_mic0_start_2026-07-26_15-56-43_stop_2026-07-26_16-06-44",
                    "data_mic0_start_2026-07-26_20-06-46_stop_2026-07-26_20-16-47",
                    "data_mic0_start_2026-07-27_00-16-49_stop_2026-07-27_00-26-50",
                    "data_mic0_start_2026-07-27_04-26-52_stop_2026-07-27_04-36-53",
                    "data_mic0_start_2026-07-27_08-36-55_stop_2026-07-27_08-46-56",
                    #"data_mic0_start_2026-07-27_12-46-58_stop_2026-07-27_12-56-59", #not calculated
                    "data_mic0_start_2026-07-27_16-57-02_stop_2026-07-27_17-07-03",
                    "data_mic0_start_2026-07-27_21-07-05_stop_2026-07-27_21-17-06",
                    #"data_mic0_start_2026-07-28_01-17-08_stop_2026-07-28_01-27-09",
                    "data_mic0_start_2026-07-28_05-27-11_stop_2026-07-28_05-37-12",
                    "data_mic0_start_2026-07-28_09-37-15_stop_2026-07-28_09-47-16",
       
                    ]
    if do_ramp_measurement:
        list_of_fits = ramp_datasets

 
    #list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50"]
    base_path = "data_ba/"
    pcls_path = "pcls/" 
    plot_type = ".png"
    results_photopeak = {}
    

    
    #beginn for loop over all datasets here
    for i in range(len(list_of_fits)):
        dataset_name = list_of_fits[i]
        try:
            # Dataset info from name; Use parse_fit_name to extract information from dataset name
            dataset_info = parse_fit_name(name = dataset_name)
            pct_ar = dataset_info["pct_Ar"]
            pct_co2 = dataset_info["pct_CO2"]
            u_wire = dataset_info["U_wire"]
            u_fieldshaper = dataset_info["U_Fieldshaper"]
            u_cathode = dataset_info["U_cathode"]

        except:
            pct_ar = ""
            pct_co2 = ""
            u_wire = ""
            u_fieldshaper = ""
            u_cathode = ""

        # Ordner für dieses Dataset erstellen
        dataset_folder_pcls = base_path + pcls_path + dataset_name + "/"

        #input_dumpfile = base_path + "data_runs/" + dataset_name + ".txt"
        #nodeadtime = True
        #use_timestamp_sync = True
        dt_hits_file = dataset_folder_pcls + dataset_name + "_hits_nodeadtime.pcl"
        #dt_hit_diff_hist_file = dataset_folder_pcls + dataset_name + "_hit_diff.pcl"
        #dt_hits_file_deadtime = dataset_folder_pcls + dataset_name + "_hits_wdeadtime.pcl"

        dt_hit_diff_hist_file = f"data_ba/pcls/{dataset_name}/{dataset_name}_hit_diff.pcl"
        plot_save_path = base_path + f"plots/photo_peak/{dataset_name}/"
         # when set to True, the parser function extracts time information
        save_plots = True

        if save_plots:
            os.makedirs(plot_save_path, exist_ok=True)  
        




        ####################
        specific_data = data_utils.load_pickle(dt_hit_diff_hist_file)
        cell_half_width = 21000 # um
        err_cell_half_width = 100 # um

        legend_font_size = 13

        ### hist to plot
        # read data
        start_idx = 0
        hist = np.array(specific_data["hist"])[start_idx:]
        err_hist = np.array(specific_data["err_hist"])[start_idx:]
        err_hist_down = np.array(specific_data["err_hist_down"])[start_idx:]
        err_hist_up = np.array(specific_data["err_hist_up"])[start_idx:]
        edges = np.array(specific_data["edges"])[start_idx:]*0.78 # convert from tu to ns
        centers = hist_utils.centers_from_edges(edges)
        bins = centers
        overflow = specific_data["overflow"]
        underflow = specific_data["underflow"]

        ######################
        ##### poisson bg subtraction

        ### plot dt hit differences
        # plot hist, 
        wire = "wire"
        print("Plotting full t_diff hist...")
        fig_size = (8, 6)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
        ax.set_xlim(0,np.amax(bins))
        ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")

        # setting the title according to the measurement type
        if not do_ramp_measurement:
            title = f"Raw time diff hist of all cells\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"

        elif do_ramp_measurement:
            time = parse_start_time(dataset_name)
            title = f"Raw time diff hist of all cells\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"


        ax.set_title(title)
        fig.tight_layout()
        path = f"{plot_save_path}{dataset_name}_DIFF_SPECIFIC_ALL{plot_type}"
        if save_plots:
            #print(f"store histogram plot as {path}.")
            print("storing histogram...")
            fig.savefig(path)
            print(f"Done saving hist as {path}\n")

        ### remove exponential "poisson" background
        print("\nFitting exp. backgrund...")
        boarder = 2000 # ns
        fit_index_range = (bins > boarder) # > 1000 ns
        extrapol_index_range = (bins <= boarder)
        fit_bins = bins[fit_index_range]
        fit_hist = hist[fit_index_range]
        err_fit_hist = err_hist[fit_index_range]
        def f_bg_fit(x, a, b):
            return a*np.exp(-x/b)
        def err_f_bg_fit(x, a, b, err_a, err_b):
            return np.sqrt( (err_a*np.exp(-x/b))**2 + (-1/b*a*np.exp(-x/b)*err_b)**2 )
        p0 = (1000, 100)
        popt, pcov, infodict, mesg, _ = curve_fit(f=f_bg_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True)
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
        print("\nFit successfull \n beginn plotting of dt hit diff with bg fit...")
        fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5,1))
        rel_spacing = 0
        barwidth = np.mean(np.diff(bins))*(1-rel_spacing)
        ax[0] = hist_utils.plot_histogram(ax[0], hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
        fit_label = f"""Exponential fit:
        $f(\\Delta T) = a \\cdot e^{{-x/b}}$
        $a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$
        $b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
        $\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
        ax[0].plot(fit_bins, f_bg_fit(fit_bins, a=a_fit, b=b_fit), color="tab:red", label=fit_label)
        ax[0].fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
        ax[0].plot(bins[extrapol_index_range], f_bg_fit(bins[extrapol_index_range], a=a_fit, b=b_fit), color="tab:red", linestyle="--", label="Extrapolated fit")
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))

        if not do_ramp_measurement:
            title = f"Background fit time diff hist of all cells\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"

        elif do_ramp_measurement:
            time = parse_start_time(dataset_name)
            title = f"Background fit time diff hist of all cells\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"

        
        ax[0].set_title(title)
        ax[0].legend(loc="lower right", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
        residuals = fit_hist - f_bg_fit(fit_bins, a=a_fit, b=b_fit)
        err_residuals = err_fit_hist
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1, linestyle="")
        ax[1].set_xlim(0,np.amax(bins))
        ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        if save_plots:
            path = f"{plot_save_path}{dataset_name}_t_diff_bgfit{plot_type}"
            #print(f"store histogram plot as {path}.")
            print("store plot...")
            fig.savefig(path)
            print(f"plot saved to {path}")

        ### subtract exp bg
        print("\nSubtracting background from hist...")
        hist_nobg = hist - f_bg_fit(bins, a=a_fit, b=b_fit)
        err_hist_nobg = np.sqrt( err_hist**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
        err_hist_nobg_down = np.sqrt( err_hist_down**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
        err_hist_nobg_up = np.sqrt( err_hist_up**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
        bins_nobg = bins
        print("\nDone subtracting bg from hist.")

        # plot wo bg
        print("\nPlotting hist without bg...")
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        rel_spacing = 0
        barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing)
        ax = hist_utils.plot_histogram(ax, hist=hist_nobg, centers=bins_nobg, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
        info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
        ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")
        ax.set_xlim(0,600)
        ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
        if not do_ramp_measurement:
            title = f"Background subtracted fit time diff hist of all cells\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"
        
        elif do_ramp_measurement:
            time = parse_start_time(dataset_name)
            title = f"Background subtracted fit time diff hist of all cells\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"

        ax.set_title(title)
        fig.tight_layout()
        fig.show()
        ## store plot
        if save_plots:
            path = f"{plot_save_path}{dataset_name}_t_diff_nobg{plot_type}"
            #print(f"store histogram plot as {path}.")
            print(f"storing hist...")
            fig.savefig(path)
            print(f"\nSaved plot to {path}")

        ######################
        ##### fit peak position (parabola, only in the region of the peak
        ##### itself -- NOT a valley-to-valley approach, see docstring of
        ##### fit_secondary_peak_parabola for caveats)

        popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func, mu_val, err_mu = fit_secondary_peak_parabola(
            bins_nobg, hist_nobg, err_hist_nobg,
            search_min=387,
            search_max=440,
            window_sigmas_left=1.5,
            window_sigmas_right=2.5,     # größer, da der Peak rechts einen längeren Ausläufer hat
            min_halfwidth_left_ns=15,
            min_halfwidth_right_ns=40,   # absolute Untergrenze rechts, damit der Ausläufer sicher erfasst wird
            edge_margin_frac=0.15,
            window_growth=1.3,
            max_attempts=6,
            prominence_frac=0.03,
        )

        perr = np.sqrt(np.diag(pcov))
        param_names = ["A", "mu", "c"]
        fit_params = dict(zip(param_names, popt))
        errors = dict(zip(param_names, perr))

        fit_values = fit_func(fit_bins, *popt)
        chi2 = np.sum((fit_hist - fit_values)**2 / err_fit_hist**2)
        ndf = len(fit_hist) - len(popt)
        chi2ndf = chi2 / ndf

        print(f"Peak-region fit interval ΔT = ({fit_bins.min():.1f}, {fit_bins.max():.1f}) ns")
        for name in param_names:
            print(f"  {name:>5} = {fit_params[name]:.6g} ± {errors[name]:.2g}")
        print(f"  chi²/ndf = {chi2:.2f} / {ndf} = {chi2ndf:.2f}")

        # --- estimate drift velocity ---
        v_drift = cell_half_width / mu_val
        err_v_drift = np.sqrt(
            (err_cell_half_width / mu_val)**2 +
            (cell_half_width * err_mu / mu_val**2)**2
        )

        print(f"v_drift = {v_drift:.4g} ± {err_v_drift:.2g} um/ns")

        fit_label = (
            "Parabola fit\n"
            r"$f(\Delta T)=A-c\,(\Delta T-\mu)^2$"
        )
        fit_label += f"\n$\\mu=({mu_val:.3g}\\pm {err_mu:.2g})$ ns"
        fit_label += f"\n$v_{{\\mathrm{{drift}}}}=({v_drift:.3g}\\pm {err_v_drift:.2g})$"
        fit_err = err_parabola_vertex_form(fit_bins, *popt, *perr)


        
        # --- build the actual figure/axes for this plot ---
        fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5, 1))

        ax[0] = hist_utils.plot_histogram(
            ax[0], hist=hist_nobg, centers=bins_nobg,
            err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up,
            log_scale=False, power_limits=[-3, 3],
        )
        info_str = (
            f"entries = {int(np.sum(hist_nobg))}\n"
            f"bin count = {len(centers)}\n"
            f"bin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
        )
        ax[0] = hist_utils.add_infobox(ax=ax[0], info_str=info_str, info_loc="top left")

        ax[0].plot(fit_bins, fit_values, color="tab:red", label=fit_label)
        ax[0].fill_between(
            fit_bins,
            fit_values - fit_err,
            fit_values + fit_err,
            color="tab:red",
            alpha=0.1,
        )

        lims = [0, 500]
        ax[0].axvline(x=mu_val, color="tab:red", linestyle="--", label="Peak position $\\mu$")
        ax[0].axvspan(xmin=mu_val - err_mu, xmax=mu_val + err_mu, color="tab:red", alpha=0.1)
        ax[0].set_ylim(bottom=0, top=np.amax(hist_nobg) * 1.1)
        ax[0].legend(loc="lower left", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
        ax[0].set_xlim(left=lims[0], right=lims[1])
        if not do_ramp_measurement:
            title = f"Photopeak fit (Parabel)\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"
                
        elif do_ramp_measurement:
            time = parse_start_time(dataset_name)
            title = f"Photopeak fit (Parabel)\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"

        
        ax[0].set_title(title)
        # --- residuals panel ---
        residuals = fit_hist - fit_values
        err_residuals = err_fit_hist
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(
            x=fit_bins, y=residuals, yerr=err_residuals,
            color="black", marker="o", markersize=2, linewidth=1, linestyle="",
        )
        ax[1].set_xlim(left=lims[0], right=lims[1])
        ax[1].set_ylim(
            -np.amax(residuals + err_residuals) * 1.1,
            np.amax(residuals + err_residuals) * 1.1,
        )
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")

        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        if save_plots:
            path = f"{plot_save_path}{dataset_name}_t_diff_peak_fit{plot_type}"
            #print(f"store histogram plot as {path}.")
            print(f"storing histogram...")
            fig.savefig(path)
            print(f"histogram plot stored as {path}.")

        results_photopeak[dataset_name] = {
            **fit_params,
            **{f"{key}_err": value for key, value in errors.items()},
            "mu": mu_val,
            "mu_err": err_mu,
        }


        plt.close("all")
        # analyze data from all data_sets



    if not do_ramp_measurement:
        # Get all unique wire voltages
        unique_u_wires = sorted(set(
            parse_fit_name(name=dataset)["U_wire"]
            for dataset in results_photopeak.keys()
        ))

        # Create one color per wire voltage
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_u_wires)))

        # Map voltage -> color
        wire_color_map = dict(zip(unique_u_wires, colors))

        plt.figure(figsize=fig_size)

        for dataset_name, result in results_photopeak.items():

            dataset_info = parse_fit_name(name=dataset_name)

            pct_ar = dataset_info["pct_Ar"]
            pct_co2 = dataset_info["pct_CO2"]
            u_wire = dataset_info["U_wire"]

            mu = result["mu"]
            err_mu = result["mu_err"]

            vd = cell_half_width / mu
            err_vd = vd * (err_mu / mu)
            plt.errorbar(
                pct_ar,
                vd,
                yerr=err_vd,
                fmt="o",
                capsize=4,
                markersize=6,
                color=wire_color_map[u_wire],
                label=f"{pct_ar}/{pct_co2}, $U_{{wire}}={u_wire}$ V"
            )

        plt.xlabel("Ar concentration [%]")
        plt.ylabel(r"$v_d$ [$\mu$m/ns]")
        plt.title("Comparison of gas mixtures and drift velocities for photopeakmethod")
        plt.grid(True)

        # Avoid duplicate legend entries for the same voltage
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.savefig(base_path + f"plots/vd_photopeak_comparison{plot_type}")

    elif do_ramp_measurement:

        print("Analysis of ramp measurement begins...")
            
        
    

        times = []
        values = []
        errors = []
    
        for dataset, result in results_photopeak.items():
            times.append(parse_start_time(dataset))
            mu = result["mu"]
            err_mu = result["mu_err"]
            vd = cell_half_width / mu 
            err_vd = vd * (err_mu / mu) 
            print("cell_half_width =", cell_half_width)
            print("mu =", mu)
            print("vd =", vd)

            values.append(vd)
            errors.append(err_vd)
            
        plt.figure(figsize=(10, 5))
    
        plt.errorbar(
            times,
            values,
            yerr=errors,
            fmt="o",
            capsize=4,
            markersize=6,
            label=r"$U_{\mathrm{wire}} = 3600\,\mathrm{V}$"
        )
    
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        plt.gcf().autofmt_xdate()
    
        plt.xlabel("Start time")
        plt.ylabel(r"$v_d$ [$\mu$m/ns]")
        plt.title(r"Drift velocity over time ($U_{\mathrm{wire}}=3600$ V) Photopeak method")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        path = f"{base_path}plots/ramp_analysis_photo_peak{plot_type}"
        plt.savefig(path)
        print(f"vd-comparison saved to {path}")
        
    return 
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
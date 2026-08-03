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
from matplotlib.ticker import ScalarFormatter

import matplotlib.dates as mdates
import sys
from pathlib import Path

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
    min_bins_syst = 4,
    max_bis_syst = 20,

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

            # -----------------------------------------
            # Systematic uncertainty from fit window
            # -----------------------------------------
            mu_scan = []

            bin_width = np.mean(np.diff(bins_nobg))

            for n_bins in range(min_bins_syst, max_bis_syst + 1):

                half_width_syst = n_bins * bin_width

                mask = (
                    (bins_nobg >= mu_fit - half_width_syst) &
                    (bins_nobg <= mu_fit + half_width_syst)
                )

                x_syst = bins_nobg[mask]
                y_syst = hist_nobg[mask]
                err_syst = err_hist_nobg[mask]

                if len(x_syst) < 3:
                    continue

                try:
                    lower_syst = (0.0, x_syst.min(), 0.0)
                    upper_syst = (np.inf, x_syst.max(), np.inf)
                    p0_syst = (A_fit, mu_fit, c_fit)

                    popt_syst, _ = curve_fit(
                        parabola_vertex_form,
                        x_syst,
                        y_syst,
                        p0=p0_syst,
                        sigma=err_syst,
                        absolute_sigma=True,
                        bounds=(lower_syst, upper_syst),
                        maxfev=20000,
                    )

                    A_syst, mu_syst, c_syst = popt_syst

                    # Reject non-parabolic fits (c<=0 means no real curvature/peak)
                    if c_syst <= 0:
                        continue

                    # Best-fit values
                    fit_syst = parabola_vertex_form(x_syst, *popt_syst)

                    # Chi-square
                    chi2_syst = np.sum(((y_syst - fit_syst) / err_syst)**2)

                    # Degrees of freedom
                    ndf_syst = len(x_syst) - len(popt_syst)

                    # Reduced chi-square
                    chi2_ndf_syst = chi2_syst / ndf_syst if ndf_syst > 0 else np.nan

                    if verbose:
                        print(f"n_bins={n_bins:2d}, chi2/ndf={chi2_ndf_syst:.2f}, mu={mu_syst:.5f}")

                    mu_scan.append(mu_syst)

                except Exception:
                    continue

            # systematic uncertainty
            if len(mu_scan) > 1:
                mu_scan = np.array(mu_scan)

                # rms around the nominal (refit) peak position
                err_mu_syst = np.sqrt(np.mean((mu_scan - mu_fit)**2))
            else:
                err_mu_syst = 0.0

            tot_err = np.sqrt(err_mu_fit**2 + err_mu_syst**2)

            fit_results = {
                "peak_pos": mu_fit,
                "peak_err_stat": err_mu_fit,
                "popt": popt,
                "pcov": pcov,
                "peak_err_tot": tot_err,
                "peak_err_syst": err_mu_syst,
            }

            return popt, pcov, fit_bins, fit_hist, err_fit_hist, parabola_vertex_form, mu_fit, err_mu_fit, fit_results

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




# Fixed U_wire -> color mapping, light to dark, so colors stay consistent
# across every plot regardless of which subset of voltages is present in a
# given `analysis_out`. Sampled from a sequential colormap (light = low
# voltage, dark = high voltage). Extend this dict if you add more voltages.
_WIRE_VOLTAGES = [3550, 3575, 3600, 3625, 3650]
_WIRE_COLORMAP = plt.cm.Blues  # light -> dark as voltage increases
_WIRE_COLOR_MAP = {
    # skip the very lightest end (near-white) so the lowest voltage is still visible
    v: _WIRE_COLORMAP(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
 
 

def plot_vd_by_gas_mix(
    *,
    analysis_out,
    base_path,
    dataset_info_fn,
    plot_type=".png",
    fig_size=(12, 7),
    save_path=None,
    y_margin=1.0,
    verbose=True,
    method = "",
    strmethod = "",
    ):
    """
    Bar-chart comparison of fitted drift velocities, grouped by gas mixture.
 
    Replaces the errorbar-per-dataset scatter plot: instead of one point per
    dataset scattered along an Ar-concentration x-axis, every dataset
    sharing the same (pct_Ar, pct_CO2) gas mixture is grouped into one
    x-axis category, and one bar is drawn per measurement within that group
    (colored consistently by U_wire across groups, using a fixed hardcoded
    color per voltage) -- mirroring the grouped-bar comparison plot used
    elsewhere for pattern-type rates.
 
    Parameters
    ----------
    analysis_out : dict
        {dataset_name: fit_results}. Reads "peak"/"peak_err" (as produced
        by fit_parabola_peak / fit_gaussian_hist) if present, otherwise
        falls back to "v_drift"/"err_v_drift" (as produced by the photopeak
        method), so the same function works on either analysis's output.
    base_path : str
        Used to build the default output file path
        (base_path + "plots/vd_photo_peak_comparison<plot_type>").
    dataset_info_fn : callable
        Function taking `name=dataset_name` and returning a dict with keys
        "pct_Ar", "pct_CO2", "U_wire" (e.g. your existing parse_fit_name).
        Called once per dataset -- NOT once total with a stale name, which
        was the bug in the original snippet (it called
        `parse_fit_name(name=dataset_name)` using an outer-scope variable
        instead of the actual per-iteration dataset). If it raises for a
        given dataset, that dataset is skipped (with a printed warning)
        instead of silently being plotted with the wrong / blank info.
    plot_type : str, default ".png"
    fig_size : tuple, default (12, 7)
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/vd_photo_peak_comparison{plot_type}".
    y_margin : float, default 1.0
        Padding (in the same units as vd, e.g. um/ns) added below the
        lowest and above the highest (mean_vd +/- err_vd) across all
        entries, used for the y-axis limits -- instead of the default
        bar-chart behavior of always starting the y-axis at 0.
    verbose : bool, default True
        Print skipped datasets and the final save path.
 
    Returns
    -------
    fig, ax, path
    """
    # --- collect (mix, u_wire, mean_vd, err_vd) per dataset ---
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue
 
        pct_ar = int(info["pct_Ar"])
        pct_co2 = int(info["pct_CO2"])
        u_wire = int(info["U_wire"])  # force int so wide/fallback parsing paths can't mismatch
        mix_label = f"{pct_ar}/{pct_co2}"
 
        if "peak_pos" in result and "peak_err_tot" in result:
            mean_vd = result["peak_pos"]
            err_vd = result["peak_err_tot"]

        elif "v_drift" in result and "err_v_drift" in result:
            mean_vd = result["v_drift"]
            err_vd = result["err_v_drift"]

        else:
            if verbose:
                print(f"  skipping {dataset_name}: no drift velocity keys found")
            continue

        entries.append({
            "dataset": dataset_name,
            "mix": mix_label,
            "u_wire": u_wire,
            "mean_vd": mean_vd,
            "err_vd": err_vd,
        })
 
    if not entries:
        raise ValueError("No datasets could be parsed by dataset_info_fn; nothing to plot.")
 
    # --- x-axis categories: one per unique gas mix, sorted by (Ar%, CO2%) ---
    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}
 
    # --- consistent color per U_wire, from the fixed hardcoded map ---
    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    unmapped = [u for u in unique_u_wires if u not in _WIRE_COLOR_MAP]
    if unmapped:
        raise KeyError(
            f"No fixed color defined for U_wire value(s) {unmapped}. "
            f"Add them to _WIRE_VOLTAGES / _WIRE_COLOR_MAP at the top of this "
            f"module (currently defined for {_WIRE_VOLTAGES})."
        )
    wire_color_map = {u: _WIRE_COLOR_MAP[u] for u in unique_u_wires}
 
    # --- group entries by mix, sort each group by wire voltage for a stable bar order ---
    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["dataset"]))
 
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
 
    max_group_size = max(len(v) for v in grouped.values())
    group_width = 0.8
    bar_width = group_width / max_group_size
 
    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        # center this group's bars even if it has fewer entries than the widest group
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = wire_color_map[e["u_wire"]]
            ax.bar(x0 + offset, e["mean_vd"], width=bar_width * 0.9, color=color)
            ax.errorbar(
                x0 + offset, e["mean_vd"], yerr=e["err_vd"],
                fmt="none", ecolor="black", capsize=3,
            )
 
    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title(f"Comparison of gas mixtures and drift velocities from {strmethod}")
    ax.grid(True, axis="y")
 
    # y-axis scaled to the actual data range (incl. error bars) with a fixed
    # margin, rather than the default bar-chart baseline-at-0 behavior
    y_lo = min(e["mean_vd"] - e["err_vd"] for e in entries)
    y_hi = max(e["mean_vd"] + e["err_vd"] for e in entries)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
 
    # legend: one entry per U_wire value, deduplicated
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=wire_color_map[u]) for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels)
 
    fig.tight_layout()
 
    if save_path is None:
        save_path = base_path + f"plots/vd_{method}_comparison{plot_type}"
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
 
    return fig, ax, save_path


 
def analyze_specific_data(
    hits_data,
    dataset_name,
    base_path,
    plot_save_path,
    plot_type=".png",
    save_plots=True,
    verbose=True,
    ):
    """
    Run the full "SPECIFIC" occupancy/rate analysis for one dataset of DT
    chamber hits.
 
    Parameters
    ----------
    hits_data : dict
        Per-hit data as loaded from the "*_hits_nodeadtime.pcl" file. Must
        contain at least the keys "sl", "ly", "wi", "ts" (parallel arrays,
        one entry per hit).
    dataset_name : str
        Name of the dataset, used for plot titles / filenames.
    base_path : str
        Base path used together with `dataset_name` for building filenames.
    plot_save_path : str
        Directory the plots get saved to (created if it doesn't exist).
    plot_type : str, default ".png"
        File extension (including dot) used for all saved plots, e.g.
        ".png", ".pdf", ".svg".
    save_plots : bool, default True
        If True, plots are saved to `plot_save_path`.
    verbose : bool, default True
        If True, all info/status messages are also printed to stdout.
        Regardless of this flag, every message is collected in the
        returned `log` list.
 
    Returns
    -------
    results : dict
        {
            "log": list[str],                 # all print/info messages
            "cell_counts": dict,               # cell_counts[sl][ly][wi]
            "duration_seconds": float,
            "dead_cells": list[(sl, ly, wi)],
            "noisy_cells": list[(sl, ly, wi)],
            "mean_rate_all_cells": float,
            "mean_rate_all_cells_err": float,
            "avg_rate_phi1": float,
            "avg_rate_phi1_err": float,
            "avg_rate_theta": float,
            "avg_rate_theta_err": float,
            "avg_rate_phi3": float,
            "avg_rate_phi3_err": float,
            "avg_rate_phi13": float,
            "avg_rate_phi13_err": float,
            "avg_rate_chamber": float,
            "avg_rate_chamber_err": float,
        }
    """
 
    log = []
 
    def emit(msg):
        log.append(msg)
        if verbose:
            print(msg)
 
    if save_plots:
        os.makedirs(plot_save_path, exist_ok=True)
 
    layer_labels = {
        0: "SL 1, Ly 0",
        1: "SL 1, Ly 1",
        2: "SL 1, Ly 2",
        3: "SL 1, Ly 3",
        4: "SL 2, Ly 0",
        5: "SL 2, Ly 1",
        6: "SL 2, Ly 2",
        7: "SL 2, Ly 3",
        8: "SL 3, Ly 0",
        9: "SL 3, Ly 1",
        10: "SL 3, Ly 2",
        11: "SL 3, Ly 3",
    }
 
    ########################
    ####### build cell_counts from raw hits, and derive duration
 
    sl_arr = np.asarray(hits_data["sl"])
    ly_arr = np.asarray(hits_data["ly"])
    wi_arr = np.asarray(hits_data["wi"])
    ts_arr = np.asarray(hits_data["ts"])
 
    # duration: span of timestamps, converted from clock ticks to seconds
    # using the same conversion factor as the original script.
    duration_ticks = float(np.max(ts_arr) - np.min(ts_arr))
    duration_seconds = duration_ticks * 0.78 * 1e-9
    emit(f"duration = {duration_seconds} s")
 
    cell_counts = {
        sl: {
            ly: {
                wi: 0
                for wi in range(
                    params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                    params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
                )
            }
            for ly in range(0, 4)
        }
        for sl in range(1, 4)
    }
    for sl, ly, wi in zip(sl_arr, ly_arr, wi_arr):
        cell_counts[int(sl)][int(ly)][int(wi)] += 1
 
    ########################
    ####### occupancy plot (2d matrix)
 
    chamber_matrix = np.full((12, 58), np.nan)  # -1: invalid cell
    cell_hits = 0
    for sl in range(1, 4):
        for ly in range(0, 4):
            for wi in range(
                params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
            ):
                chamber_matrix[4 * (sl - 1) + ly][wi] = cell_counts[sl][ly][wi]
                cell_hits += cell_counts[sl][ly][wi]
 
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    im_obj = ax.imshow(
        X=chamber_matrix,
        origin="lower",
        extent=[0 - 0.5, 57 + 0.5, 0 - 0.5, 11 + 0.5],
        vmin=0,
    )
    ax.set_xlabel("Wire")
    ax.set_yticks(list(layer_labels.keys()))
    ax.set_yticklabels(list(layer_labels.values()))
    ax.set_aspect("auto")
    cmap = plt.get_cmap("viridis")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3])
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap, format=formatter)
    cbar.set_label("Count")
    entries = int(cell_hits)
    info_str = f"entries = {entries}"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
    fig.tight_layout()
    fig.show()
    if save_plots:
        hist_plot_file = (
            plot_save_path + dataset_name + "_SPECIFIC_" + "OCCUPANCY" + plot_type
        )
        emit(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)
    plt.close(fig)
 
    ########################
    ####### rate plot (2d matrix)
 
    chamber_matrix = np.full((12, 58), np.nan)
    cell_hits = 0
    for sl in range(1, 4):
        for ly in range(0, 4):
            for wi in range(
                params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
            ):
                chamber_matrix[4 * (sl - 1) + ly][wi] = (
                    cell_counts[sl][ly][wi] / duration_seconds
                )
                cell_hits += cell_counts[sl][ly][wi]
 
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    im_obj = ax.imshow(
        X=chamber_matrix,
        origin="lower",
        extent=[0 - 0.5, 57 + 0.5, 0 - 0.5, 11 + 0.5],
        vmin=0,
    )
    ax.set_xlabel("Wire")
    ax.set_yticks(list(layer_labels.keys()))
    ax.set_yticklabels(list(layer_labels.values()))
    ax.set_aspect("auto")
    cmap = plt.get_cmap("viridis")
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, cmap=cmap)
    cbar.set_label("Rate [Hz]")
    entries = int(cell_hits)
    info_str = f"entries = {entries}"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
    fig.tight_layout()
    fig.show()
    if save_plots:
        hist_plot_file = (
            plot_save_path + dataset_name + "_SPECIFIC_" + "RATE" + plot_type
        )
        emit(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)
    plt.close(fig)
 
    ########################
    ####### find dead & noisy cells
 
    total_count_all_cells = 0
    n_cells = 0
    for sl in range(1, 4):
        for ly in range(0, 4):
            for wi in range(
                params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
            ):
                total_count_all_cells += cell_counts[sl][ly][wi]
                n_cells += 1
 
    mean_rate_all_cells = total_count_all_cells / n_cells / duration_seconds
    mean_rate_all_cells_err = (
        np.sqrt(total_count_all_cells) / n_cells / duration_seconds
    )
    emit(
        f"total count all cells: {total_count_all_cells} "
        f"+- {np.sqrt(total_count_all_cells)}"
    )
    emit(
        f"mean count all cells: {total_count_all_cells/n_cells} "
        f"+- {np.sqrt(total_count_all_cells)/n_cells}"
    )
    emit(
        f"mean rate all cells: {mean_rate_all_cells} "
        f"+- {mean_rate_all_cells_err} Hz"
    )
 
    emit("dead and noisy cells:")
    count_thres = total_count_all_cells / n_cells
    dead_cells = []
    noisy_cells = []
    for sl in range(1, 4):
        for ly in range(0, 4):
            for wi in range(
                params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
            ):
                if cell_counts[sl][ly][wi] < 0.5 * count_thres:
                    ro_ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ro_ch"]
                    ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ch"]
                    emit(
                        f"  low occupancy in  sl={sl:1}, ly={ly:1}, wi={wi:2} "
                        f"(ro_ch={ro_ch:2}, ch={ch:3})"
                    )
                    dead_cells.append((sl, ly, wi))
                if cell_counts[sl][ly][wi] > 1.5 * count_thres:
                    ro_ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ro_ch"]
                    ch = derived_params._dt_inverted_remap_table[sl][ly][wi]["ch"]
                    emit(
                        f"  high occupancy in sl={sl:1}, ly={ly:1}, wi={wi:2} "
                        f"(ro_ch={ro_ch:2}, ch={ch:3})"
                    )
                    noisy_cells.append((sl, ly, wi))
 
    ########################
    ####### average phi and theta rates (without dead channels)
 
    phi1_total_count, phi3_total_count, theta_total_count = 0, 0, 0
    n_phi1, n_phi3, n_theta = 0, 0, 0
    for sl in range(1, 4):
        for ly in range(0, 4):
            for wi in range(
                params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
            ):
                if (sl, ly, wi) not in dead_cells:
                    if sl in [1]:
                        phi1_total_count += cell_counts[sl][ly][wi]
                        n_phi1 += 1
                    elif sl in [3]:
                        phi3_total_count += cell_counts[sl][ly][wi]
                        n_phi3 += 1
                    elif sl in [2]:
                        theta_total_count += cell_counts[sl][ly][wi]
                        n_theta += 1
 
    avg_rate_phi1 = phi1_total_count / n_phi1 / duration_seconds if n_phi1 != 0 else 0
    avg_rate_phi1_err = (
        np.sqrt(phi1_total_count) / n_phi1 / duration_seconds if n_phi1 != 0 else 0
    )
    avg_rate_theta = (
        theta_total_count / n_theta / duration_seconds if n_theta != 0 else 0
    )
    avg_rate_theta_err = (
        np.sqrt(theta_total_count) / n_theta / duration_seconds
        if n_theta != 0
        else 0
    )
    avg_rate_phi3 = phi3_total_count / n_phi3 / duration_seconds if n_phi3 != 0 else 0
    avg_rate_phi3_err = (
        np.sqrt(phi3_total_count) / n_phi3 / duration_seconds if n_phi3 != 0 else 0
    )
    avg_rate_phi13 = (
        (phi1_total_count + phi3_total_count) / (n_phi1 + n_phi3) / duration_seconds
    )
    avg_rate_phi13_err = (
        np.sqrt(phi1_total_count + phi3_total_count)
        / (n_phi1 + n_phi3)
        / duration_seconds
    )
    avg_rate_chamber = (
        (phi1_total_count + phi3_total_count + theta_total_count)
        / (n_phi1 + n_phi3 + n_theta)
        / duration_seconds
    )
    avg_rate_chamber_err = (
        np.sqrt(phi1_total_count + phi3_total_count + theta_total_count)
        / (n_phi1 + n_phi3 + n_theta)
        / duration_seconds
    )
 
    emit("* = dead or noisy cells not considered")
    emit(f"average sl 1 phi cell rate *    : {avg_rate_phi1} +- {avg_rate_phi1_err} Hz")
    emit(
        f"average sl 2 theta cell rate *  : {avg_rate_theta} +- {avg_rate_theta_err} Hz"
    )
    emit(f"average sl 3 phi cell rate *    : {avg_rate_phi3} +- {avg_rate_phi3_err} Hz")
    emit(
        f"average sl 1 & 3 phi cell rate *: {avg_rate_phi13} +- {avg_rate_phi13_err} Hz"
    )
    emit(
        f"average chamber cell rate *     : {avg_rate_chamber} +- {avg_rate_chamber_err} Hz"
    )
 
    ########################
    ####### rate plot (multiple bar plots)
 
    for sl in range(1, 4):
        fig, ax = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
        for ly in range(0, 4):
            wires = np.array(
                list(
                    range(
                        params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                        params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
                    )
                )
            )
            wire_hits = np.array([cell_counts[sl][ly][wi] for wi in wires])
            wire_rates = wire_hits / duration_seconds
            err_wire_rates = np.sqrt(wire_hits) / duration_seconds
            ax[ly].bar(wires, wire_rates, width=1, align="center")
            ax[ly].bar(
                wires,
                bottom=wire_rates - err_wire_rates,
                height=2 * err_wire_rates,
                width=1,
                align="center",
                hatch="xxx",
                fill=False,
                edgecolor="0.2",
                linestyle="",
            )
            ax[ly].set_ylim(bottom=0, top=np.amax(wire_rates + err_wire_rates) * 1.1)
            if ly == 3:
                ax[ly].set_xlabel("Wire")
            ax[ly].set_ylabel("Rate [Hz]")
            ax[ly].set_title(f"SL {sl}, Ly {ly}")
        fig.tight_layout()
        fig.show()
        if save_plots:
            hist_plot_file = (
                plot_save_path
                + dataset_name
                + "_SPECIFIC_"
                + f"SL{sl}_RATE"
                + plot_type
            )
            emit(f"store plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
        plt.close(fig)
 
    return {
        "log": log,
        "cell_counts": cell_counts,
        "duration_seconds": duration_seconds,
        "dead_cells": dead_cells,
        "noisy_cells": noisy_cells,
        "mean_rate_all_cells": mean_rate_all_cells,
        "mean_rate_all_cells_err": mean_rate_all_cells_err,
        "avg_rate_phi1": avg_rate_phi1,
        "avg_rate_phi1_err": avg_rate_phi1_err,
        "avg_rate_theta": avg_rate_theta,
        "avg_rate_theta_err": avg_rate_theta_err,
        "avg_rate_phi3": avg_rate_phi3,
        "avg_rate_phi3_err": avg_rate_phi3_err,
        "avg_rate_phi13": avg_rate_phi13,
        "avg_rate_phi13_err": avg_rate_phi13_err,
        "avg_rate_chamber": avg_rate_chamber,
        "avg_rate_chamber_err": avg_rate_chamber_err,
    }
 


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


def dataset_plots_exist(plot_save_path, dataset_name, plot_type):
    """Return True if the final plot of the per-dataset analysis pipeline
    (the peak-fit plot) already exists for this dataset. Used as an
    'analysis already done' marker so re-running the script doesn't redo
    everything for datasets that haven't changed. This is a heuristic:
    it assumes that if the *last* plot in the pipeline was written, all
    earlier steps for that dataset also completed successfully."""
    marker_file = f"{plot_save_path}{dataset_name}_t_diff_peak_fit{plot_type}"
    return os.path.exists(marker_file)

# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    ###################################################
    do_ramp_measurement = False
    save_plots = True
    only_do_analysis = True  # set True to skip the analysis and only make plots from existing results
    skip_existing_datasets = True  # set False to force re-analysis of every dataset

#define parameters
    fig_size = (8, 6)
    cell_half_width = 20500 # um 21 mm - 1mm/2 i beam thickness
    err_cell_half_width = 100 # um

    list_of_fits = [#"cosmic_82-18_3550-1800-1200_run1_th20_cut100", no peak
                #"cosmic_82-18_3575-1800-1200_run1_th20_cut100", no peak
                #"cosmic_82-18_3600-1800-1200_run1_th20_cut100", no peak
                "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 
                "cosmic_82-18_3650-1800-1200_run1_th20_cut100",

                "cosmic_83-17_3650-1800-1200_run1_th20_cut100",
                "cosmic_83-17_3625-1800-1200_run1_th20_cut100", 
                "cosmic_83-17_3600-1800-1200_run1_th20_cut100", 
                #"cosmic_83-17_3575-1800-1200_run1_th20_cut100",no peak
                #"cosmic_83-17_3550-1800-1200_run1_th20_cut100",no peak

                #"cosmic_85-15_3550-1800-1200_run1_th20_cut100", no peak
                "cosmic_85-15_3575-1800-1200_run1_th20_cut100", 
                "cosmic_85-15_3600-1800-1200_run2_th20_cut100", 

                "cosmic_87-13_3550-1800-1200_run1_th20_cut100",
                "cosmic_87-13_3575-1800-1200_run1_th20_cut100",

                ]

    #list_of_fits = ["mb1_sxa5_cosmics_10min"]

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
                    "data_mic0_start_2026-07-28_01-17-08_stop_2026-07-28_01-27-09",
                    "data_mic0_start_2026-07-28_05-27-11_stop_2026-07-28_05-37-12",
                    "data_mic0_start_2026-07-28_09-37-15_stop_2026-07-28_09-47-16",
                    "data_mic0_start_2026-07-28_13-47-18_stop_2026-07-28_13-57-19",
                    "data_mic0_start_2026-07-28_17-57-21_stop_2026-07-28_18-07-22",
                    "data_mic0_start_2026-07-28_22-07-25_stop_2026-07-28_22-17-26",
                    "data_mic0_start_2026-07-29_02-17-28_stop_2026-07-29_02-27-29",
                    "data_mic0_start_2026-07-29_06-27-31_stop_2026-07-29_06-37-32",
                    "data_mic0_start_2026-07-29_10-37-34_stop_2026-07-29_10-47-35",
                    "data_mic0_start_2026-07-29_14-47-37_stop_2026-07-29_14-57-38",
                    "data_mic0_start_2026-07-29_18-57-40_stop_2026-07-29_19-07-41",
                    "data_mic0_start_2026-07-29_23-07-43_stop_2026-07-29_23-17-44",
                    "data_mic0_start_2026-07-30_07-27-49_stop_2026-07-30_07-37-50",
                    "data_mic0_start_2026-07-30_11-37-52_stop_2026-07-30_11-47-53",
                    "data_mic0_start_2026-07-30_15-47-55_stop_2026-07-30_15-57-56",
                    "data_mic0_start_2026-07-30_19-57-58_stop_2026-07-30_20-07-59",
                    "data_mic0_start_2026-07-31_00-08-02_stop_2026-07-31_00-18-03",
                    "data_mic0_start_2026-07-31_04-18-05_stop_2026-07-31_04-28-06",
                    "data_mic0_start_2026-07-31_08-28-08_stop_2026-07-31_08-38-09",

       
                    ]
    if do_ramp_measurement:
        list_of_fits = ramp_datasets


 
    #list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50"]
    base_path = "data_ba/"
    pcls_path = "pcls/" 
    plot_type = ".png"
    analysis_out = {}
    results_raw_data = {}

    # --- load previously saved analysis results so datasets that are
    # already fully analyzed (plot + result present) can be skipped instead
    # of redone from scratch ---
    analysis_pkl_name = (
        "analysis_out_photo_peak_ramp.pcl" if do_ramp_measurement
        else "analysis_out_photo_peak_data.pcl"
    )
    analysis_pkl_path = f"{pcls_path}{analysis_pkl_name}"
    if skip_existing_datasets and os.path.exists(analysis_pkl_path):
        analysis_out_prev = data_utils.load_pickle(analysis_pkl_path)
    else:
        analysis_out_prev = {}

    datasets_to_skip = set()
    if skip_existing_datasets:
        for dataset_name in list_of_fits:
            plot_save_path = base_path + f"plots/photo_peak/{dataset_name}/"
            if (dataset_plots_exist(plot_save_path, dataset_name, plot_type)
                    and dataset_name in analysis_out_prev):
                datasets_to_skip.add(dataset_name)

    if datasets_to_skip:
        print(f"Skipping {len(datasets_to_skip)} already-analyzed dataset(s): "
              f"{sorted(datasets_to_skip)}")

    non_existing_hit_diff_hists = []
    for dataset in list_of_fits:
        if dataset in datasets_to_skip:
            continue  # no need for the raw pcl if we're not analyzing it
        file_name = f"{dataset}_hit_diff.pcl"
        dataset_path = Path(f"{base_path}pcls/{dataset}/{file_name}")

        if not dataset_path.exists():
            print(f"Error: Dataset '{file_name}' does not exist.")
            non_existing_hit_diff_hists.append(file_name)

    if len(non_existing_hit_diff_hists) >= 1:
        sys.exit(1)  # Stop the entire script

    #beginn for loop over all datasets here
    
    if not only_do_analysis:
        for i in range(len(list_of_fits)):
            dataset_name = list_of_fits[i]

            if dataset_name in datasets_to_skip:
                print(f"[{i+1}/{len(list_of_fits)}] {dataset_name}: "
                      f"already analyzed, skipping.")
                analysis_out[dataset_name] = analysis_out_prev[dataset_name]
                continue

            try:
                dataset_info = parse_fit_name(name=dataset_name)
                
                pct_ar = dataset_info["pct_Ar"]
                pct_co2 = dataset_info["pct_CO2"]
                u_wire = dataset_info["U_wire"]
                u_fieldshaper = dataset_info["U_Fieldshaper"]
                u_cathode = f"-{dataset_info["U_cathode"]}"

            except:
                if dataset_name == "mb1_sxa5_cosmics_10min":
                    pct_ar = "85"
                    pct_co2 = "15"
                    u_wire = "3600"
                    u_fieldshaper = "1800"
                    u_cathode = "-1200"

                else:
                    pct_ar = ""
                    pct_co2 = ""
                    u_wire = ""
                    u_fieldshaper = ""
                    u_cathode = ""

            # set up folder structure to find and write data
            dataset_folder_pcls = base_path + pcls_path + dataset_name + "/"

            #input_dumpfile = base_path + "data_runs/" + dataset_name + ".txt"
            #nodeadtime = True
            #use_timestamp_sync = True
            dt_hits_file = dataset_folder_pcls + dataset_name + "_hits_nodeadtime.pcl"
            #dt_hit_diff_hist_file = dataset_folder_pcls + dataset_name + "_hit_diff.pcl"
            #dt_hits_file_deadtime = dataset_folder_pcls + dataset_name + "_hits_wdeadtime.pcl"

            dt_hit_diff_hist_file = f"data_ba/pcls/{dataset_name}/{dataset_name}_hit_diff.pcl"
            plot_save_path = base_path + f"plots/photo_peak/{dataset_name}/"


            

            if save_plots:
                os.makedirs(plot_save_path, exist_ok=True)  
            

            ####################
            specific_data = data_utils.load_pickle(dt_hit_diff_hist_file)
            dt_hits_file = data_utils.load_pickle(dt_hits_file)
            print(dt_hits_file.keys())
            

            legend_font_size = 13

            analysis_out[dataset_name] = analyze_specific_data(
                hits_data=dt_hits_file,
                dataset_name=dataset_name,
                base_path=base_path,
                plot_save_path=plot_save_path,
                plot_type=plot_type,
                save_plots=save_plots,
                verbose=True,
            )


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

            popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func, mu_val, err_mu, fit_results = fit_secondary_peak_parabola(
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
            peak_err_stat = fit_results["peak_err_stat"]
            peak_err_syst = fit_results["peak_err_syst"]
            peak_err_total = np.sqrt(peak_err_stat**2 + peak_err_syst**2)
            peak_pos = fit_results["peak_pos"]

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
            v_drift = cell_half_width / peak_pos
            err_v_drift = np.sqrt(
                (err_cell_half_width / peak_pos)**2 +
                (cell_half_width * peak_err_total / peak_pos**2)**2
            )

            print(f"v_drift = {v_drift:.4g} ± {err_v_drift:.2g} um/ns")

            fit_label = (
                "Parabola fit\n"
                r"$f(\Delta T)=A-c\,(\Delta T-\mu)^2$"
            )
            fit_label += f"\n$\\mu=({peak_pos:.3g}\\pm {peak_err_total:.2g})$ ns"
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
            ax[0].axvline(x=peak_pos, color="tab:red", linestyle="--", label="Peak position $\\mu$")
            ax[0].axvspan(xmin=peak_pos - peak_err_total, xmax=peak_pos + peak_err_total, color="tab:red", alpha=0.1)
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

            analysis_out[dataset_name] = {
                **fit_params,
                **{f"{key}_err": value for key, value in errors.items()},
                "peak_pos": peak_pos,
                "peak_err_tot": peak_err_total,
                "v_drift": v_drift,
                "err_v_drift": err_v_drift,
                "peak_err_stat": peak_err_stat,
                "peak_err_syst": peak_err_syst,

            }


            plt.close("all")
            # analyze data from all data_sets


        print("\nsaving analysis results...")
        if do_ramp_measurement:
            data_utils.store_pickle(analysis_out, f"{base_path}{pcls_path}analysis_out_photo_peak_ramp.pcl")

        else:
            data_utils.store_pickle(analysis_out, f"{base_path}{pcls_path}analysis_out_photo_peak_data.pcl")

    
    if not do_ramp_measurement:
        analysis_out = data_utils.load_pickle(f"{base_path}{pcls_path}analysis_out_photo_peak_data.pcl")
        fig, ax, path = plot_vd_by_gas_mix(
                    analysis_out=analysis_out,
                    base_path=base_path,
                    dataset_info_fn=parse_fit_name,
                    plot_type=plot_type,
                    fig_size=fig_size,
                    method = "photopeak",
                    strmethod = "Photopeak Method",
                    )

    elif do_ramp_measurement:

        analysis_out = data_utils.load_pickle(f"{base_path}{pcls_path}analysis_out_photo_peak_ramp.pcl")

        print("Analysis of ramp measurement begins...")
        def exp(x, a, b, c):
            return a * np.exp(-b * x) + c

        times = []
        values = []
        errors = []
    
        for dataset, result in analysis_out.items():
            times.append(parse_start_time(dataset))
            mu = result["peak_pos"]
            err_mu = result["peak_err_tot"]
            vd = cell_half_width / mu

            err_vd = np.sqrt(
                (err_cell_half_width / mu)**2 +
                (cell_half_width * err_mu / mu**2)**2
            )
            print("cell_half_width =", cell_half_width)
            print("mu =", mu)
            print("vd =", vd)

            values.append(vd)
            errors.append(err_vd)
            print(errors)

        # numeric day values, shifted so the fit domain starts at t=0
        # (fitting on the raw absolute date-number would make b*t huge and
        # exp(-b*t) underflow to 0 for any reasonable b, matching p0's
        # "b ~ 0.6 per day" assumption requires t to actually start near 0)
        t_num = mdates.date2num(times)
        t0_num = t_num[0]              # remember the reference epoch to convert back later
        t_shifted = t_num - t0_num

        p0 = [
                values[0] - values[-1],  # amplitude ≈ 4.5
                0.6,                    # per day
                values[-1]              # equilibrium ≈ 50
                ]

        popt, pcov = curve_fit(
            exp,
            t_shifted,
            values,
            sigma=errors,
            absolute_sigma=True,
            p0=p0
        )
        a_fit, b_fit, c_fit = popt
        err_a_fit = np.sqrt(pcov[0][0])
        err_b_fit = np.sqrt(pcov[1][1])
        err_c_fit = np.sqrt(pcov[2][2])
        print(f"Exponential fit parameters: a = {a_fit:.4g} ± {err_a_fit:.2g}, b = {b_fit:.4g} ± {err_b_fit:.2g}, c = {c_fit:.4g} ± {err_c_fit:.2g}")
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

        # build the fit curve in the SAME shifted-day domain used for the
        # fit, then convert that domain back to real datetimes only for
        # plotting against the datetime-formatted x-axis
        t_fit_shifted = np.linspace(t_shifted.min(), t_shifted.max(), 100)
        exp_fit_data = exp(t_fit_shifted, *popt)
        exp_fit_times = mdates.num2date(t_fit_shifted + t0_num)

        b_expected = 10 / 800 * 24   # day^-1  (Q/V, converted h^-1 -> day^-1)
        pull_b = (b_fit - b_expected) / err_b_fit
        print(f"b_fit = {b_fit:.4g} ± {err_b_fit:.2g} day^-1, "f"b_expected = {b_expected:.4g} day^-1, pull = {pull_b:.2f} sigma")


        plt.plot(exp_fit_times, exp_fit_data, "-", label=r"$\mathrm{Exponential\ fit}$")

        exp_fit_data_expected = exp(t_fit_shifted, a_fit, b_expected, c_fit)
        plt.plot(exp_fit_times, exp_fit_data_expected, "--", color="gray", label=fr"Expected ($b={b_expected:.3g}\,\mathrm{{day}}^{{-1}}$, $\tau={1/b_expected:.2f}$ d)")

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
        print(f"vd-comparison saved to: {path}")
        
    return 
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
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
import uproot

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

def _fmt_gas_pct_display(x):
    """Format a gas percentage for on-plot labels (ticks/legend): integer
    percentages render plainly ("84"), non-integer ones keep a real
    decimal point ("84.5") instead of the filename-safe 'p' substitute
    used by _fmt_gas_pct."""
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return str(x)

def fit_secondary_peak_parabola(
    bins_nobg,
    hist_nobg,
    err_hist_nobg,
    peak_pos,
    halfwidth_left_ns=40,
    halfwidth_right_ns=40,
    edge_margin_frac=0.15,
    max_attempts=4,
    window_growth=1.3,
    min_bins=6,
    verbose=True,
    min_bins_syst=4,
    max_bis_syst=20,
):
    """
    Fit a parabola (vertex form) to a FIXED window
    [peak_pos - halfwidth_left_ns, peak_pos + halfwidth_right_ns]
    around a hardcoded peak position.

    No automatic peak search, no local width/amplitude estimate: the
    window is chosen purely from `peak_pos` and the two `halfwidth_*_ns`
    values you pass in. If `peak_pos` is wrong or the window doesn't
    actually contain a peak, the fit will fail via the existing c<=0 /
    A<=0 / edge-margin checks below -- there is no independent
    verification that a peak actually exists at that location.

    peak_pos : float
        Hardcoded peak position (ns). The fit window is centered here.
    halfwidth_left_ns, halfwidth_right_ns : float
        Fixed half-widths (in ns) of the fit window, left/right of
        `peak_pos`. Kept separate (not a single symmetric halfwidth)
        because the photopeak typically has a longer tail to the right
        (towards the background hump) than to the left (towards the
        valley) -- raise halfwidth_right_ns if the fit curve visibly
        stops short of the peak's right-hand shoulder.
    edge_margin_frac : float, default 0.15
        If the fitted vertex mu lands within this fraction of either
        edge of the window, the window is treated as too small and
        widened (via window_growth) for a retry.
    max_attempts : int, default 4
        Number of times to widen the window (by `window_growth`, applied
        to both halfwidths) and retry, if the fit fails (too few bins,
        c<=0, A<=0, or mu too close to an edge).
    window_growth : float, default 1.3
        Multiplicative factor applied to both halfwidths on each retry.
    min_bins : int, default 6
        Minimum number of bins required in the fit window.
    (min_bins_syst, max_bis_syst, verbose: as before, used for the
    systematic-uncertainty scan after a successful fit)
    """
    win_left_ns = halfwidth_left_ns
    win_right_ns = halfwidth_right_ns
    last_exc = None

    for attempt in range(max_attempts):
        fit_min = peak_pos - win_left_ns
        fit_max = peak_pos + win_right_ns
        fit_mask = (bins_nobg >= fit_min) & (bins_nobg <= fit_max)

        fit_bins = bins_nobg[fit_mask]
        fit_hist = hist_nobg[fit_mask]
        err_fit_hist = err_hist_nobg[fit_mask]

        if len(fit_bins) < min_bins:
            if verbose:
                print(f"    (diagnostic) fit window has only {len(fit_bins)} bins "
                      f"(< {min_bins}), range = ({fit_min:.2f}, {fit_max:.2f}) ns -- widening window")
            win_left_ns *= window_growth
            win_right_ns *= window_growth
            last_exc = RuntimeError("fit window contains too few bins")
            continue

        # seed values: amplitude from the bin nearest peak_pos, curvature
        # guessed from the window half-width
        idx_near = int(np.argmin(np.abs(fit_bins - peak_pos)))
        A_seed = max(fit_hist[idx_near], 1e-3)
        c_seed = A_seed / max(min(win_left_ns, win_right_ns), 1e-6) ** 2
        p0 = (A_seed, peak_pos, c_seed)

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
                      f"[halfwidth_left={win_left_ns:.1f} ns, halfwidth_right={win_right_ns:.1f} ns], "
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
                    if c_syst <= 0:
                        continue

                    fit_syst = parabola_vertex_form(x_syst, *popt_syst)
                    chi2_syst = np.sum(((y_syst - fit_syst) / err_syst) ** 2)
                    ndf_syst = len(x_syst) - len(popt_syst)
                    chi2_ndf_syst = chi2_syst / ndf_syst if ndf_syst > 0 else np.nan

                    if verbose:
                        print(f"n_bins={n_bins:2d}, chi2/ndf={chi2_ndf_syst:.2f}, mu={mu_syst:.5f}")

                    mu_scan.append(mu_syst)
                except Exception:
                    continue

            if len(mu_scan) > 1:
                mu_scan = np.array(mu_scan)
                err_mu_syst = np.sqrt(np.mean((mu_scan - mu_fit) ** 2))
            else:
                err_mu_syst = 0.0

            tot_err = np.sqrt(err_mu_fit ** 2 + err_mu_syst ** 2)

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
            win_left_ns *= window_growth
            win_right_ns *= window_growth

    raise RuntimeError(
        f"Parabola fit in fixed window around peak_pos={peak_pos:.1f} ns "
        f"did not converge after {max_attempts} attempts. Last error: {last_exc}\n"
        "There may be no real peak at this hardcoded peak_pos, or the initial "
        "halfwidth_left_ns/halfwidth_right_ns may be too small/large -- "
        "check against the *_t_diff_nobg plot."
    )

def _fmt_gas_pct(x):
    """Format a gas percentage for plot labels/filenames: integer-valued
    percentages render plainly ("84"), non-integer ones get 'p' instead
    of '.' so the string is filename-safe ("84.5" -> "84p5")."""
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return str(x).replace(".", "p")

# Fixed U_wire -> color mapping, light to dark, so colors stay consistent
# across every plot regardless of which subset of voltages is present in a
# given `analysis_out`. Sampled from a sequential colormap (light = low
# voltage, dark = high voltage). Extend this dict if you add more voltages.
_WIRE_VOLTAGES = [3550, 3575, 3600, 3625, 3650]
_WIRE_COLORMAP = plt.cm.Reds  # light -> dark as voltage increases
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
 
        pct_ar = float(info["pct_Ar"])
        pct_co2 = float(info["pct_CO2"])
        u_wire = int(info["U_wire"])  # force int so wide/fallback parsing paths can't mismatch
        mix_label = f"{_fmt_gas_pct(pct_ar)}/{_fmt_gas_pct(pct_co2)}"
 
        if "peak_pos" in result and "peak_err_tot" in result:
            mean_vd = result["v_drift"]
            err_vd = result["err_v_drift"]

        else:
            if verbose:
                print(f"  skipping {dataset_name}: no drift velocity keys found")
            continue

        entries.append({
            "dataset": dataset_name,
            "mix": mix_label,
            "mix_sort": (pct_ar, pct_co2),
            "u_wire": u_wire,
            "mean_vd": mean_vd,
            "err_vd": err_vd,
        })
 
    if not entries:
        raise ValueError("No datasets could be parsed by dataset_info_fn; nothing to plot.")
 
    # --- x-axis categories: one per unique gas mix, sorted by (Ar%, CO2%) ---
    mix_sort_key = {e["mix"]: e["mix_sort"] for e in entries}
    mixes = sorted(mix_sort_key, key=lambda m: mix_sort_key[m])
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
    plt.close("all")
    return fig, ax, save_path


def plot_metric_by_gas_mix(
    *,
    analysis_out,
    base_path,
    dataset_info_fn,
    value_key,
    err_key,
    ylabel,
    filename_prefix,
    plot_type=".png",
    fig_size=(12, 7),
    save_path=None,
    y_margin=None,
    verbose=True,
    method="",
    strmethod="",
    ar_step=0.5,
    group_width=0.4,
    ):
    """
    Generic bar-chart comparison of a scalar fit quantity (e.g. peak
    position, peak amplitude, v_drift), grouped by gas mixture and
    colored by U_wire -- shared implementation behind
    plot_vd_by_gas_mix / plot_peak_pos_by_gas_mix / plot_peak_amplitude_by_gas_mix.

    The x-axis is placed on a true numeric grid: each mix sits at its
    actual pct_Ar value (equidistant in real % Ar, e.g. 0.5 % steps),
    instead of one evenly-spaced category per unique mix regardless of
    its real gas-composition distance from its neighbors. Assumes
    pct_CO2 = 100 - pct_Ar, so pct_Ar alone fixes the x-position.

    Parameters
    ----------
    value_key, err_key : str
        Keys into each analysis_out[dataset_name] dict giving the value
        and its (total) error to plot, e.g. ("peak_pos", "peak_err_tot")
        or ("A", "A_err").
    ylabel : str
        Y-axis label, e.g. r"Photopeak amplitude [counts]".
    filename_prefix : str
        Used to build the default save_path:
        f"{base_path}plots/{filename_prefix}_{method}_comparison{plot_type}".
    y_margin : float, optional
        Padding added below/above the data range for the y-axis limits.
        Defaults to 5% of the data span if not given.
    ar_step : float, default 0.5
        Nominal spacing (in % Ar) between adjacent grid steps. Used only
        to label the axis and sanity-check bar widths; the actual x
        position of each mix comes directly from its pct_Ar value.
    group_width : float, default 0.4
        Total width (in % Ar) that one gas-mix's bars are allowed to
        span, centered on that mix's pct_Ar position. Keep this smaller
        than the smallest gap between two distinct pct_Ar values present
        in the data (0.5 by default) to avoid neighboring groups
        overlapping.
    verbose : bool, default True
        Print skipped datasets and the final save path.

    Returns
    -------
    fig, ax, path
    """
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue

        if value_key not in result or err_key not in result:
            if verbose:
                print(f"  skipping {dataset_name}: missing '{value_key}'/'{err_key}'")
            continue

        pct_ar = float(info["pct_Ar"])
        pct_co2 = float(info["pct_CO2"])
        u_wire = int(info["U_wire"])

        entries.append({
            "dataset": dataset_name,
            "mix": f"{_fmt_gas_pct_display(pct_ar)}/{_fmt_gas_pct_display(pct_co2)}",  # CHANGED
            "pct_ar": pct_ar,
            "u_wire": u_wire,
            "value": result[value_key],
            "err": result[err_key],
        })

    if not entries:
        raise ValueError(f"No datasets with '{value_key}'/'{err_key}' found; nothing to plot.")

    # --- x-axis: one position per unique pct_Ar, at its ACTUAL value ---
    unique_ar = sorted(set(e["pct_ar"] for e in entries))
    ar_gaps = np.diff(unique_ar)
    if len(ar_gaps) > 0 and np.min(ar_gaps) < group_width:
        if verbose:
            print(f"  WARNING: smallest gap between adjacent pct_Ar values "
                  f"({np.min(ar_gaps):.3g}) is smaller than group_width "
                  f"({group_width}); bars from neighboring mixes may overlap. "
                  f"Consider lowering group_width.")
    mix_to_x = {e["mix"]: e["pct_ar"] for e in entries}

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    unmapped = [u for u in unique_u_wires if u not in _WIRE_COLOR_MAP]
    if unmapped:
        raise KeyError(
            f"No fixed color defined for U_wire value(s) {unmapped}. "
            f"Add them to _WIRE_VOLTAGES / _WIRE_COLOR_MAP (currently defined for {_WIRE_VOLTAGES})."
        )
    wire_color_map = {u: _WIRE_COLOR_MAP[u] for u in unique_u_wires}

    grouped = {}
    for e in entries:
        grouped.setdefault(e["mix"], []).append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    max_group_size = max(len(v) for v in grouped.values())
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = wire_color_map[e["u_wire"]]
            ax.bar(x0 + offset, e["value"], width=bar_width * 0.9, color=color)
            ax.errorbar(
                x0 + offset, e["value"], yerr=e["err"],
                fmt="none", ecolor="black", capsize=3,
            )

    sorted_mixes = sorted(mix_to_x, key=lambda m: mix_to_x[m])
    ax.set_xticks([mix_to_x[m] for m in sorted_mixes])
    ax.set_xticklabels(sorted_mixes, rotation=45, ha="right")
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Comparison of gas mixtures\n{ylabel.split(' [')[0].lower()} from {strmethod}")
    ax.grid(True, axis="y")

    y_lo = min(e["value"] - e["err"] for e in entries)
    y_hi = max(e["value"] + e["err"] for e in entries)
    if y_margin is None:
        y_margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=wire_color_map[u]) for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/{filename_prefix}_{method}_comparison{plot_type}"
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")
    return fig, ax, save_path


def plot_peak_amplitude_by_gas_mix(
    *,
    analysis_out,
    base_path,
    dataset_info_fn,
    plot_type=".png",
    fig_size=(12, 7),
    save_path=None,
    y_margin=None,
    verbose=True,
    method="",
    strmethod="",
    ):
    """
    Bar-chart comparison of fitted photopeak amplitudes (the parabola
    fit's 'A' parameter), grouped by gas mixture, colored by U_wire.
    Thin wrapper around plot_metric_by_gas_mix -- see there for details.
    Note: amplitude is fit-window/bin-width dependent, so only compare
    datasets fit with the same binning/window settings.
    """
    return plot_metric_by_gas_mix(
        analysis_out=analysis_out,
        base_path=base_path,
        dataset_info_fn=dataset_info_fn,
        value_key="A",
        err_key="A_err",
        ylabel="Photopeak amplitude [counts]",
        filename_prefix="peak_amp",
        plot_type=plot_type,
        fig_size=fig_size,
        save_path=save_path,
        y_margin=y_margin,
        verbose=verbose,
        method=method,
        strmethod=strmethod,
    )




def plot_peak_amplitude_rate_vs_uwire_and_mix(
    *,
    analysis_out,
    base_path,
    dataset_info_fn,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    verbose=True,
    method="",
    strmethod="",
    ):
    """
    Two-panel line-plot comparison of the event-normalized photopeak
    amplitude (A_rate = fitted amplitude 'A' / number of background-
    subtracted events, int(np.sum(hist_nobg))):

      left panel  : A_rate vs U_wire, one line per gas mixture
                    -> shows how amplitude changes with wire voltage,
                       holding the gas mixture fixed.
      right panel : A_rate vs gas mixture (categorical x-axis), one
                    line per U_wire
                    -> shows how amplitude changes across gas mixtures,
                       holding the wire voltage fixed.

    Both panels are built from the same `entries` list, just grouped/
    sorted along the other axis, so a dataset missing "A_rate"/
    "A_rate_err" (or that dataset_info_fn can't parse) is skipped
    consistently in both.

    Parameters
    ----------
    analysis_out : dict
        {dataset_name: fit_results}, must contain "A_rate"/"A_rate_err"
        (as produced after switching the amplitude normalization to
        event count, see fit_secondary_peak_parabola usage in main()).
    base_path : str
        Used to build the default output file path.
    dataset_info_fn : callable
        Function taking `name=dataset_name` -> dict with "pct_Ar",
        "pct_CO2", "U_wire" (e.g. parse_fit_name). Called once per
        dataset; datasets it can't parse are skipped with a warning.
    plot_type : str, default ".png"
    fig_size : tuple, default (14, 6)
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/peak_amp_rate_vs_uwire_and_mix_{method}_comparison{plot_type}".
    verbose : bool, default True
        Print skipped datasets and the final save path.

    Returns
    -------
    fig, (ax_left, ax_right), path
    """
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue

        if "A_rate" not in result or "A_rate_err" not in result:
            if verbose:
                print(f"  skipping {dataset_name}: missing 'A_rate'/'A_rate_err'")
            continue

        pct_ar = float(info["pct_Ar"])
        pct_co2 = float(info["pct_CO2"])
        u_wire = int(info["U_wire"])

        entries.append({
            "dataset": dataset_name,
            "mix": f"{_fmt_gas_pct(pct_ar)}/{_fmt_gas_pct(pct_co2)}",
            "mix_sort": (pct_ar, pct_co2),
            "u_wire": u_wire,
            "value": result["A_rate"],
            "err": result["A_rate_err"],
        })

    if not entries:
        raise ValueError("No datasets with 'A_rate'/'A_rate_err' found; nothing to plot.")

    mix_sort_key = {e["mix"]: e["mix_sort"] for e in entries}
    mixes = sorted(mix_sort_key, key=lambda m: mix_sort_key[m])
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    unmapped = [u for u in unique_u_wires if u not in _WIRE_COLOR_MAP]
    if unmapped:
        raise KeyError(
            f"No fixed color defined for U_wire value(s) {unmapped}. "
            f"Add them to _WIRE_VOLTAGES / _WIRE_COLOR_MAP (currently defined for {_WIRE_VOLTAGES})."
        )
    wire_color_map = {u: _WIRE_COLOR_MAP[u] for u in unique_u_wires}

    # distinct color per gas mix for the left panel, independent of the
    # fixed voltage color map (different axis being compared there)
    mix_colormap = plt.cm.tab10
    mix_color_map = {mix: mix_colormap(i % 10) for i, mix in enumerate(mixes)}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=fig_size)

    # ---- left panel: A_rate vs U_wire, one line per gas mix ----
    grouped_by_mix = {mix: [] for mix in mixes}
    for e in entries:
        grouped_by_mix[e["mix"]].append(e)

    for mix in mixes:
        group = sorted(grouped_by_mix[mix], key=lambda e: e["u_wire"])
        if not group:
            continue
        x = [e["u_wire"] for e in group]
        y = [e["value"] for e in group]
        yerr = [e["err"] for e in group]
        ax_left.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=mix_color_map[mix], label=mix,
        )

    ax_left.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax_left.set_ylabel("Photopeak amplitude / events")
    ax_left.set_title("vs. wire voltage, per gas mix")
    ax_left.grid(True)
    ax_left.legend(title="Ar/CO$_2$ [%]", fancybox=False, framealpha=params._legend_alpha)

    # ---- right panel: A_rate vs gas mix, one line per U_wire ----
    grouped_by_wire = {u: [] for u in unique_u_wires}
    for e in entries:
        grouped_by_wire[e["u_wire"]].append(e)

    for u in unique_u_wires:
        group = sorted(grouped_by_wire[u], key=lambda e: mix_to_x[e["mix"]])
        if not group:
            continue
        x = [mix_to_x[e["mix"]] for e in group]
        y = [e["value"] for e in group]
        yerr = [e["err"] for e in group]
        ax_right.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=wire_color_map[u], label=f"$U_{{wire}}$ = {u} V",
        )

    ax_right.set_xticks(list(mix_to_x.values()))
    ax_right.set_xticklabels(list(mix_to_x.keys()))
    ax_right.set_xlabel("Gas mixture (Ar/CO$_2$) [%]")
    ax_right.set_ylabel("Photopeak amplitude / events")
    ax_right.set_title("vs. gas mix, per wire voltage")
    ax_right.grid(True)
    ax_right.legend(fancybox=False, framealpha=params._legend_alpha)

    fig.suptitle(f"Photopeak amplitude (normalized) comparison from {strmethod}")
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/peak_amp_norm_vs_uwire_and_mix_{method}_comparison{plot_type}"
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")
    return fig, (ax_left, ax_right), save_path


def plot_peak_pos_vs_uwire_and_mix(
    *,
    analysis_out,
    base_path,
    dataset_info_fn,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    verbose=True,
    method="",
    strmethod="",
    ):
    """
    Two-panel line-plot comparison of the fitted photopeak position
    ("peak_pos" / "peak_err_tot"), analogous to
    plot_peak_amplitude_rate_vs_uwire_and_mix but for peak position
    instead of normalized amplitude:

      left panel  : peak_pos vs U_wire, one line per gas mixture
                    -> shows how the peak position shifts with wire
                       voltage, holding the gas mixture fixed.
      right panel : peak_pos vs gas mixture (categorical x-axis), one
                    line per U_wire
                    -> shows how the peak position shifts across gas
                       mixtures, holding the wire voltage fixed.

    Both panels are built from the same `entries` list, just grouped/
    sorted along the other axis, so a dataset missing "peak_pos"/
    "peak_err_tot" (or that dataset_info_fn can't parse) is skipped
    consistently in both.

    Parameters
    ----------
    analysis_out : dict
        {dataset_name: fit_results}, must contain "peak_pos"/
        "peak_err_tot" (as produced by fit_secondary_peak_parabola /
        the main() analysis loop).
    base_path : str
        Used to build the default output file path.
    dataset_info_fn : callable
        Function taking `name=dataset_name` -> dict with "pct_Ar",
        "pct_CO2", "U_wire" (e.g. parse_fit_name). Called once per
        dataset; datasets it can't parse are skipped with a warning.
    plot_type : str, default ".png"
    fig_size : tuple, default (14, 6)
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/peak_pos_vs_uwire_and_mix_{method}_comparison{plot_type}".
    verbose : bool, default True
        Print skipped datasets and the final save path.

    Returns
    -------
    fig, (ax_left, ax_right), path
    """
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue

        if "peak_pos" not in result or "peak_err_tot" not in result:
            if verbose:
                print(f"  skipping {dataset_name}: missing 'peak_pos'/'peak_err_tot'")
            continue

        pct_ar = float(info["pct_Ar"])
        pct_co2 = float(info["pct_CO2"])
        u_wire = int(info["U_wire"])

        entries.append({
            "dataset": dataset_name,
            "mix": f"{_fmt_gas_pct(pct_ar)}/{_fmt_gas_pct(pct_co2)}",
            "mix_sort": (pct_ar, pct_co2),
            "u_wire": u_wire,
            "value": result["peak_pos"],
            "err": result["peak_err_tot"],
        })

    if not entries:
        raise ValueError("No datasets with 'peak_pos'/'peak_err_tot' found; nothing to plot.")

    mix_sort_key = {e["mix"]: e["mix_sort"] for e in entries}
    mixes = sorted(mix_sort_key, key=lambda m: mix_sort_key[m])
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    unmapped = [u for u in unique_u_wires if u not in _WIRE_COLOR_MAP]
    if unmapped:
        raise KeyError(
            f"No fixed color defined for U_wire value(s) {unmapped}. "
            f"Add them to _WIRE_VOLTAGES / _WIRE_COLOR_MAP (currently defined for {_WIRE_VOLTAGES})."
        )
    wire_color_map = {u: _WIRE_COLOR_MAP[u] for u in unique_u_wires}

    # distinct color per gas mix for the left panel, independent of the
    # fixed voltage color map (different axis being compared there)
    mix_colormap = plt.cm.tab10
    mix_color_map = {mix: mix_colormap(i % 10) for i, mix in enumerate(mixes)}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=fig_size)

    # ---- left panel: peak_pos vs U_wire, one line per gas mix ----
    grouped_by_mix = {mix: [] for mix in mixes}
    for e in entries:
        grouped_by_mix[e["mix"]].append(e)

    for mix in mixes:
        group = sorted(grouped_by_mix[mix], key=lambda e: e["u_wire"])
        if not group:
            continue
        x = [e["u_wire"] for e in group]
        y = [e["value"] for e in group]
        yerr = [e["err"] for e in group]
        ax_left.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=mix_color_map[mix], label=mix,
        )

    ax_left.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax_left.set_ylabel(r"Peak position $\mu$ [ns]")
    ax_left.set_title("vs. wire voltage, per gas mix")
    ax_left.grid(True)
    ax_left.legend(title="Ar/CO$_2$ [%]", fancybox=False, framealpha=params._legend_alpha)

    # ---- right panel: peak_pos vs gas mix, one line per U_wire ----
    grouped_by_wire = {u: [] for u in unique_u_wires}
    for e in entries:
        grouped_by_wire[e["u_wire"]].append(e)

    for u in unique_u_wires:
        group = sorted(grouped_by_wire[u], key=lambda e: mix_to_x[e["mix"]])
        if not group:
            continue
        x = [mix_to_x[e["mix"]] for e in group]
        y = [e["value"] for e in group]
        yerr = [e["err"] for e in group]
        ax_right.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3,
            color=wire_color_map[u], label=f"$U_{{wire}}$ = {u} V",
        )

    ax_right.set_xticks(list(mix_to_x.values()))
    ax_right.set_xticklabels(list(mix_to_x.keys()))
    ax_right.set_xlabel("Gas mixture (Ar/CO$_2$) [%]")
    ax_right.set_ylabel(r"Peak position $\mu$ [ns]")
    ax_right.set_title("vs. gas mix, per wire voltage")
    ax_right.grid(True)
    ax_right.legend(fancybox=False, framealpha=params._legend_alpha)

    fig.suptitle(f"Photopeak position comparison from {strmethod}")
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/peak_pos_vs_uwire_and_mix_{method}_comparison{plot_type}"
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")
    return fig, (ax_left, ax_right), save_path

def analyze_specific_data(
    cell_counts,
    duration_seconds,
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
    cell_counts : dict
        cell_counts[sl][ly][wi] -> int, per-cell hit counts as produced
        by the streaming pipeline's chunked accumulation (or, for older
        datasets, derived from the raw hit arrays as a fallback in
        main()).
    duration_seconds : float
        Run duration in seconds, as derived from ts_max - ts_min.
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

    emit(f"duration = {duration_seconds} s")

    ########################
    ####### occupancy plot (2d matrix)
    # cell_counts is taken directly from the parameter -- no rebuild here

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
    # Ar/CO2 percentages may be integers ("84") or decimals written with
    # "p" instead of "." ("84p5" -> 84.5)
    pattern = r"^cosmic_(\d+(?:p\d+)?)-(\d+(?:p\d+)?)_(\d+)-(\d+)-(\d+)"
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"String hat nicht das erwartete Format: {name}")

    pct_ar, pct_co2, u_wire, u_fieldshaper, u_cathode = match.groups()

    def _to_number(s):
        """'84' -> 84 (int), '84p5' -> 84.5 (float)"""
        if "p" in s:
            return float(s.replace("p", "."))
        return int(s)

    return {
        "name": name,
        "pct_Ar": _to_number(pct_ar),
        "pct_CO2": _to_number(pct_co2),
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
    only_do_analysis = False  # set True to skip the analysis and only make plots from existing results
    skip_existing_datasets = False  # set False to force re-analysis of every dataset

#define parameters
    fig_size = (8, 6)
    cell_half_width = 20500 # um 21 mm - 1mm/2 i beam thickness
    err_cell_half_width = 100 # um

    legend_font_size = mpl.rcParams['font.size'] + 1

    list_of_fits = [



                ["cosmic_84p5-15p5_3625-1800-1200_run1_th20_cut100", 409],
                ["cosmic_84p5-15p5_3550-1800-1200_run1_th20_cut100", 409], 
                ["cosmic_84p5-15p5_3575-1800-1200_run1_th20_cut100", 409], 
                ["cosmic_84p5-15p5_3600-1800-1200_run1_th20_cut100", 409],
                ["cosmic_84p5-15p5_3625-1800-1200_run1_th20_cut100", 409],
                ["cosmic_84p5-15p5_3650-1800-1200_run1_th20_cut100", 409],


                ["cosmic_85p5-14p5_3625-1800-1200_run1_th20_cut100", 420],
                ["cosmic_85p5-14p5_3550-1800-1200_run1_th20_cut100", 420], 
                ["cosmic_85p5-14p5_3575-1800-1200_run1_th20_cut100", 420], 
                ["cosmic_85p5-14p5_3600-1800-1200_run1_th20_cut100", 420],
                ["cosmic_85p5-14p5_3625-1800-1200_run1_th20_cut100", 420],
                ["cosmic_85p5-14p5_3650-1800-1200_run1_th20_cut100", 420],

                ["cosmic_82-18_3650-1800-1200_run1_th20_cut100", 391],        
                ["cosmic_82-18_3625-1800-1200_run1_th20_cut100", 387], 
                #["cosmic_82-18_3550-1800-1200_run1_th20_cut100", 387], no peak
                #["cosmic_82-18_3575-1800-1200_run1_th20_cut100", 387], no peak
                #["cosmic_82-18_3600-1800-1200_run1_th20_cut100", 387], no peak

                ["cosmic_83-17_3650-1800-1200_run1_th20_cut100", 400],
                ["cosmic_83-17_3625-1800-1200_run1_th20_cut100", 400], 
                ["cosmic_83-17_3600-1800-1200_run1_th20_cut100", 392],
                #["cosmic_83-17_3575-1800-1200_run1_th20_cut100", 392], # no peak
                #["cosmic_83-17_3550-1800-1200_run1_th20_cut100", 392], # no peak

                ["cosmic_84-16_3650-1800-1200_run1_th20_cut100", 405], 
                ["cosmic_84-16_3625-1800-1200_run1_th20_cut100", 405], 
                ["cosmic_84-16_3600-1800-1200_run1_th20_cut100", 405],
                ["cosmic_84-16_3575-1800-1200_run1_th20_cut100", 405],
                ["cosmic_84-16_3550-1800-1200_run1_th20_cut100", 405],


                ["cosmic_85-15_3600-1800-1200_run2_th20_cut100", 413],
                ["cosmic_85-15_3575-1800-1200_run1_th20_cut100", 411],                
                ["cosmic_85-15_3550-1800-1200_run1_th20_cut100", 411], #no peak

                

                ["cosmic_86-14_3650-1800-1200_run1_th20_cut100", 430],#issues with data proccessing
                ["cosmic_86-14_3625-1800-1200_run1_th20_cut100", 430],
                ["cosmic_86-14_3600-1800-1200_run1_th20_cut100", 430],
                ["cosmic_86-14_3575-1800-1200_run1_th20_cut100", 430],
                ["cosmic_86-14_3550-1800-1200_run1_th20_cut100",430],#full

                ["cosmic_87-13_3600-1800-1200_run1_th20_cut100", 440], # stopped because of tripping
                ["cosmic_87-13_3575-1800-1200_run1_th20_cut100", 440],
                ["cosmic_87-13_3550-1800-1200_run1_th20_cut100", 440],



                ]
    

    #list_of_fits = [["cosmic_85-15_3600-1800-1200_test4_th20", 408]]
    #list_of_fits = ["mb1_sxa5_cosmics_10min"]

    ramp_datasets = [
        ["data_mic0_start_2026-07-24_18-06-10_stop_2026-07-24_18-16-11", 395],
        ["data_mic0_start_2026-07-24_22-16-13_stop_2026-07-24_22-26-14", 401],
        ["data_mic0_start_2026-07-25_02-26-16_stop_2026-07-25_02-36-17", 404],
        ["data_mic0_start_2026-07-25_06-36-19_stop_2026-07-25_06-46-20", 407],
        ["data_mic0_start_2026-07-25_10-46-22_stop_2026-07-25_10-56-23", 409],
        ["data_mic0_start_2026-07-25_14-56-25_stop_2026-07-25_15-06-26", 412],
        ["data_mic0_start_2026-07-25_19-06-28_stop_2026-07-25_19-16-29", 415],
        ["data_mic0_start_2026-07-25_23-16-31_stop_2026-07-25_23-26-32", 416],
        ["data_mic0_start_2026-07-26_03-26-34_stop_2026-07-26_03-36-35", 418],
        ["data_mic0_start_2026-07-26_07-36-37_stop_2026-07-26_07-46-38", 420],
        ["data_mic0_start_2026-07-26_11-46-40_stop_2026-07-26_11-56-41", 422],
        ["data_mic0_start_2026-07-26_15-56-43_stop_2026-07-26_16-06-44", 424],
        ["data_mic0_start_2026-07-26_20-06-46_stop_2026-07-26_20-16-47", 425],
        ["data_mic0_start_2026-07-27_00-16-49_stop_2026-07-27_00-26-50", 426],
        ["data_mic0_start_2026-07-27_04-26-52_stop_2026-07-27_04-36-53", 428],
        ["data_mic0_start_2026-07-27_08-36-55_stop_2026-07-27_08-46-56", 429],
        ["data_mic0_start_2026-07-27_12-46-58_stop_2026-07-27_12-56-59", 405],  #not calculated
        ["data_mic0_start_2026-07-27_16-57-02_stop_2026-07-27_17-07-03", 430],
        ["data_mic0_start_2026-07-27_21-07-05_stop_2026-07-27_21-17-06", 431],
        ["data_mic0_start_2026-07-28_01-17-08_stop_2026-07-28_01-27-09", 432],
        ["data_mic0_start_2026-07-28_05-27-11_stop_2026-07-28_05-37-12", 433],
        ["data_mic0_start_2026-07-28_09-37-15_stop_2026-07-28_09-47-16", 434],
        ["data_mic0_start_2026-07-28_13-47-18_stop_2026-07-28_13-57-19", 435],
        ["data_mic0_start_2026-07-28_17-57-21_stop_2026-07-28_18-07-22", 435],
        ["data_mic0_start_2026-07-28_22-07-25_stop_2026-07-28_22-17-26", 437],
        ["data_mic0_start_2026-07-29_02-17-28_stop_2026-07-29_02-27-29", 436],
        ["data_mic0_start_2026-07-29_06-27-31_stop_2026-07-29_06-37-32", 436],
        ["data_mic0_start_2026-07-29_10-37-34_stop_2026-07-29_10-47-35", 436],
        ["data_mic0_start_2026-07-29_14-47-37_stop_2026-07-29_14-57-38", 437],
        ["data_mic0_start_2026-07-29_18-57-40_stop_2026-07-29_19-07-41", 437],
        ["data_mic0_start_2026-07-29_23-07-43_stop_2026-07-29_23-17-44", 436],
        ["data_mic0_start_2026-07-30_07-27-49_stop_2026-07-30_07-37-50", 437],
        ["data_mic0_start_2026-07-30_11-37-52_stop_2026-07-30_11-47-53", 438],
        ["data_mic0_start_2026-07-30_15-47-55_stop_2026-07-30_15-57-56", 438],
        ["data_mic0_start_2026-07-30_19-57-58_stop_2026-07-30_20-07-59", 439],
        ["data_mic0_start_2026-07-31_00-08-02_stop_2026-07-31_00-18-03", 439],
        ["data_mic0_start_2026-07-31_04-18-05_stop_2026-07-31_04-28-06", 440],
        ["data_mic0_start_2026-07-31_08-28-08_stop_2026-07-31_08-38-09", 440],
    ]
    if do_ramp_measurement:
        list_of_fits = ramp_datasets


 
    #list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50"]
    base_path = "data_ba/"
    pcls_path = "pcls/"
    # absolute path where the actual dt_hits/root/pcl input files live;
    # everything derived from base_path (plots, analysis_out pickle) stays
    # relative and untouched
    data_path = "/net/data_cms3a-1/tacke/pcls/"
    plot_type = ".pdf"
    analysis_out = {}


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
        for dataset_name, _ in list_of_fits:
            plot_save_path = base_path + f"plots/photo_peak/{dataset_name}/"
            if (dataset_plots_exist(plot_save_path, dataset_name, plot_type)
                    and dataset_name in analysis_out_prev):
                datasets_to_skip.add(dataset_name)
    if datasets_to_skip:
        print(f"Skipping {len(datasets_to_skip)} already-analyzed dataset(s): "
              f"{sorted(datasets_to_skip)}")

    non_existing_hit_diff_hists = []
    for dataset_name, _ in list_of_fits:
        if dataset_name in datasets_to_skip:
            continue
        file_name = f"{dataset_name}_hit_diff.pcl"
        dataset_path = Path(f"{data_path}{dataset_name}/{file_name}")
        if not dataset_path.exists():
            print(f"Error: Dataset '{file_name}' does not exist.")
            non_existing_hit_diff_hists.append(file_name)

        if not dataset_path.exists():
            print(f"Error: Dataset '{file_name}' does not exist.")
            non_existing_hit_diff_hists.append(file_name)

    if len(non_existing_hit_diff_hists) >= 1:
        sys.exit(1)  # Stop the entire script

    #beginn for loop over all datasets here
    
    if not only_do_analysis:
        for i in range(len(list_of_fits)):
            dataset_name = list_of_fits[i][0]
            dataset_peak_position = list_of_fits[i][1]

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
            dataset_folder_pcls = data_path + dataset_name + "/"

            #input_dumpfile = base_path + "data_runs/" + dataset_name + ".txt"
            #nodeadtime = True
            #use_timestamp_sync = True
            dt_hits_file = dataset_folder_pcls + dataset_name + "_hits_nodeadtime.pcl"
            dt_hits_root_file = dataset_folder_pcls + dataset_name + "_dt_hits.root"
            #dt_hit_diff_hist_file = dataset_folder_pcls + dataset_name + "_hit_diff.pcl"
            #dt_hits_file_deadtime = dataset_folder_pcls + dataset_name + "_hits_wdeadtime.pcl"

            dt_hit_diff_hist_file = f"{data_path}{dataset_name}/{dataset_name}_hit_diff.pcl"
            plot_save_path = base_path + f"plots/photo_peak/{dataset_name}/"


            

            if save_plots:
                os.makedirs(plot_save_path, exist_ok=True)  
            

            ####################
            cell_counts_file = dataset_folder_pcls + dataset_name + "_cell_counts.pcl"
            if not os.path.exists(cell_counts_file):
                raise FileNotFoundError(
                    f"Missing '{cell_counts_file}'. Run root_streaming_pipeline_v3.py "
                    f"for dataset '{dataset_name}' first."
                )
            cc = data_utils.load_pickle(cell_counts_file)
            cell_counts = cc["cell_counts"]
            duration_seconds = cc["duration_seconds"]

            """

            # fallback for datasets not yet reprocessed through the new pipeline
            dt_hits_file_loaded = data_utils.load_pickle(dt_hits_file)
            sl_arr = np.asarray(dt_hits_file_loaded["sl"])
            ly_arr = np.asarray(dt_hits_file_loaded["ly"])
            wi_arr = np.asarray(dt_hits_file_loaded["wi"])
            ts_arr = np.asarray(dt_hits_file_loaded["ts"])
            duration_seconds = float(np.max(ts_arr) - np.min(ts_arr)) * 0.78 * 1e-9
            cell_counts = {
                sl: {ly: {wi: 0 for wi in range(
                        params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                        params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1)}
                    for ly in range(0, 4)}
                for sl in range(1, 4)
            }
            for sl, ly, wi in zip(sl_arr, ly_arr, wi_arr):
                cell_counts[int(sl)][int(ly)][int(wi)] += 1
            """
            specific_data = data_utils.load_pickle(dt_hit_diff_hist_file)

            specific_results = analyze_specific_data(
                cell_counts=cell_counts,
                duration_seconds=duration_seconds,
                dataset_name=dataset_name,
                base_path=base_path,
                plot_save_path=plot_save_path,
                plot_type=plot_type,
                save_plots=save_plots,
                verbose=True,
            )
            duration_seconds = specific_results["duration_seconds"]
            analysis_out[dataset_name] = specific_results


 
            ### hist to plot
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

            ### plot dt hit differences (raw, log scale) -- unchanged
            wire = "wire"
            print("Plotting full t_diff hist...")

            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
            ax.set_xlim(0,np.amax(bins))
            ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")

            if not do_ramp_measurement:
                title = f"Raw time diff hist of all cells\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"
            elif do_ramp_measurement:
                time = parse_start_time(dataset_name)
                title = f"Raw time diff hist of all cells\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"

            ax.set_title(title)
            fig.tight_layout()
            path = f"{plot_save_path}{dataset_name}_DIFF_SPECIFIC_ALL{plot_type}"
            if save_plots:
                print("storing histogram...")
                fig.savefig(path)
                print(f"Done saving hist as {path}\n")

            ######################
            ##### fit peak position DIRECTLY on the RAW histogram -- no bg
            ##### fit / subtraction step at all anymore. Parabola fit only
            ##### in the region of the peak itself (NOT valley-to-valley),
            ##### see fit_secondary_peak_parabola's docstring for caveats.
            print("\nFitting photopeak directly to raw histogram (no bg subtraction)...")

            popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func, mu_val, err_mu, fit_results = fit_secondary_peak_parabola(
                bins, hist, err_hist,
                peak_pos=dataset_peak_position,
                halfwidth_left_ns=20,
                halfwidth_right_ns=20,
                edge_margin_frac=0.15,
                window_growth=1.3,
                max_attempts=6,
            )
            peak_err_stat = fit_results["peak_err_stat"]
            peak_err_syst = fit_results["peak_err_syst"]
            peak_err_total = np.sqrt(peak_err_stat**2 + peak_err_syst**2)
            peak_pos = fit_results["peak_pos"]

            perr = np.sqrt(np.diag(pcov))
            param_names = ["A", "mu", "c"]
            fit_params = dict(zip(param_names, popt))
            errors = dict(zip(param_names, perr))

            # --- normalization: A_rate = fitted amplitude / total (raw)
            # events in the histogram. Since we no longer subtract
            # background, A now includes whatever background sits under
            # the peak at this Delta_T -- A_rate therefore compares
            # "peak-region counts per event" across datasets, correcting
            # for run-to-run statistics/duration, but NOT for differences
            # in the background level itself. ---
            excluded_cells = set()
            for sl in range(1, 4):
                for ly in range(0, 4):
                    for wi in params._dt_dead_wires.get(sl, {}).get(ly, []):
                        excluded_cells.add((sl, ly, wi))
                    for wi in params._dt_wire_mask.get(sl, {}).get(ly, []):
                        excluded_cells.add((sl, ly, wi))

            n_events = 0
            for sl in range(1, 4):
                for ly in range(0, 4):
                    for wi in range(
                        params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"],
                        params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1,
                    ):
                        if (sl, ly, wi) in excluded_cells:
                            continue
                        n_events += cell_counts[sl][ly][wi]

            if n_events == 0:
                raise ValueError(
                    f"{dataset_name}: all cells excluded as dead/masked -- "
                    "n_events is 0, cannot normalize A_rate."
                )

            A_rate = fit_params["A"] / n_events
            A_rate_err = errors["A"] / n_events

            fit_values = fit_func(fit_bins, *popt)
            chi2 = np.sum((fit_hist - fit_values)**2 / err_fit_hist**2)
            ndf = len(fit_hist) - len(popt)
            chi2ndf = chi2 / ndf

            print(f"Peak-region fit interval ΔT = ({fit_bins.min():.1f}, {fit_bins.max():.1f}) ns")
            for name in param_names:
                print(f"  {name:>5} = {fit_params[name]:.6g} ± {errors[name]:.2g}")
            print(f"  chi²/ndf = {chi2:.2f} / {ndf} = {chi2ndf:.2f}")
            print(f"  A_rate = {A_rate:.6g} ± {A_rate_err:.2g} (counts / event)")

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

            # --- build the actual figure/axes for this plot (raw hist, log scale) ---
            fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5, 1))

            ax[0] = hist_utils.plot_histogram(
                ax[0], hist=hist, centers=bins,
                err_hist_down=err_hist_down, err_hist_up=err_hist_up,
                log_scale=False, power_limits=[-4, 4],
            )
            info_str = (
                f"entries = {int(np.sum(hist))}\n"
                f"bin count = {len(centers)}\n"
                f"bin width = {np.mean(np.diff(bins)):.3g} ns"
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

            max_dt = 700
            lims = [0, max_dt]
            ax[0].axvline(x=peak_pos, color="tab:red", linestyle="--", label="Peak position $\\mu$")
            ax[0].axvspan(xmin=peak_pos - peak_err_total, xmax=peak_pos + peak_err_total, color="tab:red", alpha=0.1)
            ax[0].legend(loc="lower left", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
            ax[0].set_xlim(left=lims[0], right=lims[1])
            i_max_dt = int(np.argmin(np.abs(bins - max_dt)))
            y_bottom = hist[i_max_dt]
            y_top = 1.1 * np.amax(hist)
            ax[0].set_ylim(y_bottom, y_top)

            if not do_ramp_measurement:
                title = f"Photopeak fit (Parabel, raw hist)\n{pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{wire}}$ = {u_wire}V"
            elif do_ramp_measurement:
                time = parse_start_time(dataset_name)
                title = f"Photopeak fit (Parabel, raw hist)\nRamp measurement $t_{{\\mathrm{{start}}}}$ = {time}, $U_{{\\mathrm{{wire}}}}$ = 3600 V"
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
                print(f"storing histogram...")
                fig.savefig(path)
                print(f"histogram plot stored as {path}.")

            analysis_out[dataset_name] = {
                **specific_results,
                **fit_params,
                **{f"{key}_err": value for key, value in errors.items()},
                "peak_pos": peak_pos,
                "peak_err_tot": peak_err_total,
                "v_drift": v_drift,
                "err_v_drift": err_v_drift,
                "peak_err_stat": peak_err_stat,
                "peak_err_syst": peak_err_syst,
                "A_rate": A_rate,
                "A_rate_err": A_rate_err,
                "n_events": n_events,
            }

            plt.close("all")
            # analyze data from all data_sets
            # analyze data from all data_sets


        print("\nsaving analysis results...")
        if do_ramp_measurement:
            data_utils.store_pickle(analysis_out, f"{base_path}{pcls_path}analysis_out_photo_peak_ramp.pcl")

        else:
            data_utils.store_pickle(analysis_out, f"{base_path}{pcls_path}analysis_out_photo_peak_data.pcl")

    
    if not do_ramp_measurement:
        analysis_out = data_utils.load_pickle(f"{base_path}{pcls_path}analysis_out_photo_peak_data.pcl")

        fig, ax, path = plot_metric_by_gas_mix(
                    analysis_out=analysis_out,
                    base_path=base_path,
                    dataset_info_fn=parse_fit_name,
                    value_key="v_drift",
                    err_key="err_v_drift",
                    ylabel=r"$v_d$ [$\mu$m/ns]",
                    filename_prefix="vd",
                    plot_type=plot_type,
                    fig_size=fig_size,
                    method="photopeak",
                    strmethod="Photopeak Method",
                    )

        fig, ax, path = plot_metric_by_gas_mix(
                    analysis_out=analysis_out,
                    base_path=base_path,
                    dataset_info_fn=parse_fit_name,
                    value_key="peak_pos",
                    err_key="peak_err_tot",
                    ylabel=r"Peak position $\mu$ [ns]",
                    filename_prefix="peak_pos",
                    plot_type=plot_type,
                    fig_size=fig_size,
                    method="photopeak",
                    strmethod="Photopeak Method",
                    )

        fig, ax, path = plot_metric_by_gas_mix(
                    analysis_out=analysis_out,
                    base_path=base_path,
                    dataset_info_fn=parse_fit_name,
                    value_key="A_rate",
                    err_key="A_rate_err",
                    ylabel="Photopeak normalized amplitude A_photopeak/counts",
                    filename_prefix="peak_amp",
                    plot_type=plot_type,
                    fig_size=fig_size,
                    method="photopeak",
                    strmethod="Photopeak Method",
                    )
        fig, (ax_left, ax_right), path = plot_peak_amplitude_rate_vs_uwire_and_mix(
            analysis_out=analysis_out,
            base_path=base_path,
            dataset_info_fn=parse_fit_name,
            plot_type=plot_type,
            method="photopeak",
            strmethod="Photopeak Method",
            )



        fig, ax, path = plot_metric_by_gas_mix(
            analysis_out=analysis_out,
            base_path=base_path,
            dataset_info_fn=parse_fit_name,
            value_key="avg_rate_chamber",
            err_key="avg_rate_chamber_err",
            ylabel="Chamber rate [Hz]",
            filename_prefix="chamber_rate",
            plot_type=plot_type,
            fig_size=fig_size,
            method="photopeak",
            strmethod="Photopeak Method",
            )


        fig, (ax_left, ax_right), path = plot_peak_pos_vs_uwire_and_mix(
            analysis_out=analysis_out,
            base_path=base_path,
            dataset_info_fn=parse_fit_name,
            plot_type=plot_type,
            method="photopeak",
            strmethod="Photopeak Method",
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
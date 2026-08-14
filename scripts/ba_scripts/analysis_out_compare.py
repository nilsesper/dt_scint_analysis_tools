#################################################################
### import analysis_out files from both photopeak and track fit method
### and compare both methods
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


_WIRE_VOLTAGES = [3550, 3575, 3600, 3625, 3650]

_CMAP_PHOTOPEAK = plt.cm.Reds
_CMAP_TRACKFIT = plt.cm.Blues

_WIRE_COLOR_MAP_PHOTOPEAK = {
    v: _CMAP_PHOTOPEAK(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_WIRE_COLOR_MAP_TRACKFIT = {
    v: _CMAP_TRACKFIT(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_MIX_CMAP = plt.cm.tab10


def _wire_color(u_wire, color_map, fallback_cmap):
    """Look up a fixed color for u_wire; fall back to a generated one
    (with a printed warning) instead of raising, since the comparison
    script may see voltage points the two individual pipelines didn't."""
    if u_wire in color_map:
        return color_map[u_wire]
    print(f"  warning: no fixed color defined for U_wire={u_wire}, generating one on the fly")
    idx = sorted(set(list(color_map.keys()) + [u_wire])).index(u_wire)
    n = len(color_map) + 1
    return fallback_cmap(0.3 + 0.7 * idx / max(n - 1, 1))


# extract vd from photopeak analysis out
def get_vd_photopeak(*, dataset_name, result):
    """Return (v_drift, err_v_drift) [um/ns] from one photopeak analysis_out entry."""
    if "v_drift" in result and "err_v_drift" in result:
        return float(result["v_drift"]), float(result["err_v_drift"])
    raise KeyError(f"{dataset_name}: no 'v_drift'/'err_v_drift' in photopeak result")

# extract vd from Track fit analysis out
def get_vd_trackfit(*, dataset_name, result):
    """Return (v_drift, err_v_drift) [um/ns] from one track-fit analysis_out entry."""
    if "peak" in result:
        err = result.get("tot_err", result.get("peak_err"))
        if err is None:
            raise KeyError(f"{dataset_name}: 'peak' present but no 'tot_err'/'peak_err'")
        return float(result["peak"]), float(err)
    raise KeyError(f"{dataset_name}: no 'peak' key in track-fit result")



# extract hit rate from photopeak analysis out
def get_rate_photopeak(*, dataset_name, result, rate_key="avg_rate_chamber", err_key="avg_rate_chamber_err"):
    """Return (rate, err_rate) [Hz] from one photopeak analysis_out entry.
    Defaults to the overall chamber rate; pass rate_key/err_key
    (e.g. "avg_rate_phi1"/"avg_rate_phi1_err") for a specific SL's rate."""
    if rate_key in result and err_key in result:
        return float(result[rate_key]), float(result[err_key])
    raise KeyError(f"{dataset_name}: no '{rate_key}'/'{err_key}' in photopeak result")

# extract track fit rate from track fit analysis out
def get_rate_trackfit(*, dataset_name, result):
    """Return (rate, err_rate) [Hz] from one track-fit analysis_out entry.
    Requires "track_rate"/"track_rate_err" -- see the patch documented
    above; existing track-fit pickles produced before that patch will not
    have these keys and will raise KeyError here."""
    if "track_rate" in result and "track_rate_err" in result:
        return float(result["track_rate"]), float(result["track_rate_err"])
    raise KeyError(
        f"{dataset_name}: no 'track_rate'/'track_rate_err' in track-fit result "
        "-- rerun sl_fits_analysis.py with the track_rate patch (see comment "
        "above get_rate_photopeak) to populate this."
    )



def build_comparison_entries(
    *,
    analysis_out_photopeak,
    analysis_out_track_fit,
    dataset_info_fn=parse_fit_name,
    verbose=True,
    ):
    """
    Find analysis out data that was extracted for both datasets and create comparisson plots
    for example: Photopeak analysis does not contain all measurements, thats why when a file was only analyzed with track fit, it wont be displayed

    Returns
    -------
    entries : list[dict]
        Each entry:
        {
            "dataset": str,
            "mix": "Ar/CO2" string,
            "u_wire": int,
            "vd_photopeak": float, "err_vd_photopeak": float,
            "vd_trackfit": float,  "err_vd_trackfit": float,
            "diff": float,       # vd_photopeak - vd_trackfit
            "err_diff": float,   # sqrt(err_pp^2 + err_tf^2)
            "pull": float,       # diff / err_diff
        }
    """
    # find common datasets, datasets that are only in Photopeak, and only in track fit
    common = sorted(set(analysis_out_photopeak) & set(analysis_out_track_fit))
    only_tf = sorted(set(analysis_out_track_fit) - set(analysis_out_photopeak))
    only_pp = sorted(set(analysis_out_photopeak) - set(analysis_out_track_fit))

    if verbose:
        print(f"{len(common)} dataset(s) present in both analysis_out dicts.")
        if only_pp:
            print(f"  {len(only_pp)} dataset(s) only in photopeak results, skipped: {only_pp}")
        if only_tf:
            print(f"  {len(only_tf)} dataset(s) only in track-fit results, skipped: {only_tf}")

    entries = []
    for name in common:
        try:
            info = dataset_info_fn(name=name)
        except Exception as e:
            if verbose:
                print(f"  skipping {name}: could not parse dataset info ({e})")
            continue

        # try extracting vd and err_vd
        try:
            vd_pp, err_pp = get_vd_photopeak(dataset_name=name, result=analysis_out_photopeak[name])
            vd_tf, err_tf = get_vd_trackfit(dataset_name=name, result=analysis_out_track_fit[name])
        except KeyError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
            continue

        # calculate diff and error propagation
        diff = vd_pp - vd_tf
        err_diff = np.sqrt(err_pp ** 2 + err_tf ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan

        entries.append({
            "dataset": name,
            "mix": f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}",
            "u_wire": int(info["U_wire"]),
            "vd_photopeak": vd_pp, "err_vd_photopeak": err_pp,
            "vd_trackfit": vd_tf, "err_vd_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
        })

    return entries



def build_ramp_comparison_entries(
    *,
    analysis_out_photopeak_ramp,
    analysis_out_track_fit_ramp,
    verbose=True,
    ):
    """
    extracts data to form a single plot to show change in vd over time for the ramp measurement from 83/17 to 87/13

    Returns
    -------
    entries : list[dict]
        Each entry:
        {
            "dataset": str, "time": datetime,
            "vd_photopeak": float, "err_vd_photopeak": float,
            "vd_trackfit": float,  "err_vd_trackfit": float,
            "diff": float, "err_diff": float, "pull": float,
        }
        sorted by time.
    """
    # find common and non common datasets in both analysis
    common = sorted(set(analysis_out_photopeak_ramp) & set(analysis_out_track_fit_ramp))
    only_tf = sorted(set(analysis_out_track_fit_ramp) - set(analysis_out_photopeak_ramp))
    only_pp = sorted(set(analysis_out_photopeak_ramp) - set(analysis_out_track_fit_ramp))

    if verbose:
        print(f"{len(common)} ramp dataset(s) present in both analysis_out dicts.")
        if only_pp:
            print(f"  {len(only_pp)} dataset(s) only in photopeak ramp results, skipped: {only_pp}")
        if only_tf:
            print(f"  {len(only_tf)} dataset(s) only in track-fit ramp results, skipped: {only_tf}")

    entries = []
    #extract start time
    for name in common:
        try:
            t = parse_start_time(name)
        except Exception as e:
            if verbose:
                print(f"  skipping {name}: could not parse start time ({e})")
            continue
        #extract vd
        try:
            vd_pp, err_pp = get_vd_photopeak(dataset_name=name, result=analysis_out_photopeak_ramp[name])
            vd_tf, err_tf = get_vd_trackfit(dataset_name=name, result=analysis_out_track_fit_ramp[name])
        except KeyError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
            continue
        # calculate diff, and do error propagation
        diff = vd_pp - vd_tf
        err_diff = np.sqrt(err_pp ** 2 + err_tf ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan
        # form return
        entries.append({
            "dataset": name, "time": t,
            "vd_photopeak": vd_pp, "err_vd_photopeak": err_pp,
            "vd_trackfit": vd_tf, "err_vd_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
        })

    entries.sort(key=lambda e: e["time"])
    return entries


def build_rate_comparison_entries(
    *,
    analysis_out_photopeak,
    analysis_out_track_fit,
    dataset_info_fn=parse_fit_name,
    photopeak_rate_key="avg_rate_chamber",
    photopeak_err_key="avg_rate_chamber_err",
    verbose=True,
    ):
    """
    extract rates from photopeak analysis out to build rate comparrison plots

    Returns
    -------
    entries : list[dict]
        Each entry:
        {
            "dataset": str, "mix": str, "u_wire": int,
            "rate_photopeak": float, "err_rate_photopeak": float,
            "rate_trackfit": float,  "err_rate_trackfit": float,
            "diff": float, "err_diff": float, "pull": float,
        }
    """
    common = sorted(set(analysis_out_photopeak) & set(analysis_out_track_fit))
    only_tf = sorted(set(analysis_out_track_fit) - set(analysis_out_photopeak))
    only_pp = sorted(set(analysis_out_photopeak) - set(analysis_out_track_fit))

    if verbose:
        print(f"{len(common)} dataset(s) present in both analysis_out dicts (rate comparison).")
        if only_pp:
            print(f"  {len(only_pp)} dataset(s) only in photopeak results, skipped: {only_pp}")
        if only_tf:
            print(f"  {len(only_tf)} dataset(s) only in track-fit results, skipped: {only_tf}")

    entries = []
    n_missing_track_rate = 0
    for name in common:
        try:
            info = dataset_info_fn(name=name)
        except Exception as e:
            if verbose:
                print(f"  skipping {name}: could not parse dataset info ({e})")
            continue

        try:
            rate_pp, err_pp = get_rate_photopeak(
                dataset_name=name, result=analysis_out_photopeak[name],
                rate_key=photopeak_rate_key, err_key=photopeak_err_key,
            )
        except KeyError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
            continue

        try:
            rate_tf, err_tf = get_rate_trackfit(dataset_name=name, result=analysis_out_track_fit[name])
        except KeyError:
            n_missing_track_rate += 1
            continue

        diff = rate_pp - rate_tf
        err_diff = np.sqrt(err_pp ** 2 + err_tf ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan

        entries.append({
            "dataset": name,
            "mix": f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}",
            "u_wire": int(info["U_wire"]),
            "rate_photopeak": rate_pp, "err_rate_photopeak": err_pp,
            "rate_trackfit": rate_tf, "err_rate_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
        })

    if n_missing_track_rate and verbose:
        print(f"  {n_missing_track_rate} dataset(s) skipped: track-fit result has no "
              "'track_rate' (rerun sl_fits_analysis.py with the patch to populate it).")

    return entries


def build_ramp_rate_comparison_entries(
    *,
    analysis_out_photopeak_ramp,
    analysis_out_track_fit_ramp,
    photopeak_rate_key="avg_rate_chamber",
    photopeak_err_key="avg_rate_chamber_err",
    verbose=True,
    ):
    """Rate analogue of build_ramp_comparison_entries(); see that function
    and build_rate_comparison_entries() for details. Entries carry "time"
    (datetime) instead of (mix, u_wire), sorted by time."""
    common = sorted(set(analysis_out_photopeak_ramp) & set(analysis_out_track_fit_ramp))
    if verbose:
        print(f"{len(common)} ramp dataset(s) present in both analysis_out dicts (rate comparison).")

    entries = []
    n_missing_track_rate = 0
    for name in common:
        try:
            t = parse_start_time(name)
        except Exception as e:
            if verbose:
                print(f"  skipping {name}: could not parse start time ({e})")
            continue

        try:
            rate_pp, err_pp = get_rate_photopeak(
                dataset_name=name, result=analysis_out_photopeak_ramp[name],
                rate_key=photopeak_rate_key, err_key=photopeak_err_key,
            )
        except KeyError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
            continue

        try:
            rate_tf, err_tf = get_rate_trackfit(dataset_name=name, result=analysis_out_track_fit_ramp[name])
        except KeyError:
            n_missing_track_rate += 1
            continue

        diff = rate_pp - rate_tf
        err_diff = np.sqrt(err_pp ** 2 + err_tf ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan

        entries.append({
            "dataset": name, "time": t,
            "rate_photopeak": rate_pp, "err_rate_photopeak": err_pp,
            "rate_trackfit": rate_tf, "err_rate_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
        })

    if n_missing_track_rate and verbose:
        print(f"  {n_missing_track_rate} dataset(s) skipped: track-fit result has no 'track_rate'.")

    entries.sort(key=lambda e: e["time"])
    return entries


def plot_vd_comparison_bars_by_gas_mix(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(14, 7),
    save_path=None,
    y_margin=1.0,
    verbose=True,
    ):
    """
    Bar-chart comparison of fitted drift velocities from BOTH methods,
    grouped by gas mixture. Within each gas-mixture group, every dataset
    (i.e. every U_wire present for that mix) gets TWO adjacent bars: the
    photopeak result (Reds colormap, matching photo_peak_analysis.py's own
    plot) and the track-fit result (Blues colormap, matching
    sl_fits_analysis.py's own plot) -- so the color already tells you both
    "which U_wire" and "which method" at a glance.

    Parameters
    ----------
    entries : list[dict]
        Output of build_comparison_entries().
    base_path : str
        Used to build the default output path.
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/compare/vd_comparison_bars{plot_type}".
    y_margin : float, default 1.0
        Padding (um/ns) below/above the data range for the y-axis limits.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # each dataset contributes 2 bars (photopeak, trackfit)
    max_group_size = max(len(v) for v in grouped.values()) * 2
    group_width = 0.85
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        # centers of each dataset's (pp, tf) bar-pair, evenly spaced
        pair_centers = (np.arange(n) - (n - 1) / 2) * (2 * bar_width)
        for e, pc in zip(group_entries, pair_centers):
            color_pp = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)
            color_tf = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_TRACKFIT, _CMAP_TRACKFIT)

            x_pp = x0 + pc - bar_width / 2
            x_tf = x0 + pc + bar_width / 2

            ax.bar(x_pp, e["vd_photopeak"], width=bar_width * 0.95, color=color_pp)
            ax.errorbar(x_pp, e["vd_photopeak"], yerr=e["err_vd_photopeak"],
                        fmt="none", ecolor="black", capsize=2)

            ax.bar(x_tf, e["vd_trackfit"], width=bar_width * 0.95, color=color_tf)
            ax.errorbar(x_tf, e["vd_trackfit"], yerr=e["err_vd_trackfit"],
                        fmt="none", ecolor="black", capsize=2)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title("Drift velocity comparison: Photopeak vs. Track-fit method")
    ax.grid(True, axis="y")

    y_lo = min(min(e["vd_photopeak"] - e["err_vd_photopeak"],
                    e["vd_trackfit"] - e["err_vd_trackfit"]) for e in entries)
    y_hi = max(max(e["vd_photopeak"] + e["err_vd_photopeak"],
                    e["vd_trackfit"] + e["err_vd_trackfit"]) for e in entries)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles = []
    legend_labels = []
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)))
        legend_labels.append(f"Photopeak, $U_{{wire}}$={u} V")
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_TRACKFIT, _CMAP_TRACKFIT)))
        legend_labels.append(f"Track-fit, $U_{{wire}}$={u} V")
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_comparison_bars{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path




def plot_vd_vs_uwire_both_methods(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(10, 7),
    save_path=None,
    verbose=True,
    ):
    """
    Trend plot: v_drift vs. U_wire, one line per gas mixture, drawn TWICE
    (solid = photopeak, dashed = track-fit) in the SAME color per mix --
    so the two methods' trends can be compared directly, mix by mix.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_color_map = {mix: _MIX_CMAP(i % 10) for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for mix in mixes:
        group = sorted(grouped[mix], key=lambda e: e["u_wire"])
        if not group:
            continue
        x = [e["u_wire"] for e in group]
        y_pp = [e["vd_photopeak"] for e in group]
        yerr_pp = [e["err_vd_photopeak"] for e in group]
        y_tf = [e["vd_trackfit"] for e in group]
        yerr_tf = [e["err_vd_trackfit"] for e in group]

        color = mix_color_map[mix]
        ax.errorbar(x, y_pp, yerr=yerr_pp, marker="o", linestyle="-",
                    capsize=3, color=color, label=f"{mix} (photopeak)")
        ax.errorbar(x, y_tf, yerr=yerr_tf, marker="^", linestyle="--",
                    capsize=3, color=color, label=f"{mix} (track-fit)")


    y_margin_up = 2
    y_margin_down = 0.5
    y_lo = min(min(e["vd_photopeak"] - e["err_vd_photopeak"],
                e["vd_trackfit"] - e["err_vd_trackfit"]) for e in entries)
    y_hi = max(max(e["vd_photopeak"] + e["err_vd_photopeak"],
                e["vd_trackfit"] + e["err_vd_trackfit"]) for e in entries)
    ax.set_ylim(y_lo - y_margin_down, y_hi + y_margin_up)
    ax.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title("Drift velocity vs. wire voltage -- both methods")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Ar/CO$_2$ [%] (method)", fontsize=9, ncol=2,
              fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_vs_uwire_both_methods{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")

    return fig, ax, save_path



def plot_method_difference_by_gas_mix(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    verbose=True,
    ):
    """
    For each dataset: diff = vd_photopeak - vd_trackfit, with
    err_diff = sqrt(err_pp^2 + err_tf^2), plotted as a bar per dataset,
    grouped by gas mixture and colored by U_wire (using the photopeak
    color map, since this isn't "a photopeak value" or "a track-fit
    value" specifically). A horizontal line at 0 marks perfect agreement;
    shaded bands at +-1 sigma (of err_diff, dataset by dataset, so drawn
    as the error bars themselves) show whether the discrepancy is
    statistically significant.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

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
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)
            ax.bar(x0 + offset, e["diff"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["diff"], yerr=e["err_diff"],
                        fmt="none", ecolor="black", capsize=3)

    ax.axhline(y=0, color="gray", linewidth=1.2, linestyle="--",
               label="perfect agreement")

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_{d,\mathrm{photopeak}} - v_{d,\mathrm{track\!-\!fit}}$ [$\mu$m/ns]")
    ax.set_title("Method difference (photopeak $-$ track-fit)")
    ax.grid(True, axis="y")

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                       color=_wire_color(u, _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_method_difference{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path

# pull dist
def plot_pull_distribution(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(8, 6),
    save_path=None,
    n_bins=15,
    verbose=True,
    ):
    """
    Histogram of pull = (vd_photopeak - vd_trackfit) / sqrt(err_pp^2 + err_tf^2)
    across all datasets, with a standard-normal N(0,1) curve overlaid for
    reference. If the two methods' uncertainties are correctly estimated
    and there's no systematic offset between them, this should scatter
    around a unit Gaussian centered at 0.

    Returns
    -------
    fig, ax, path
    """
    pulls = np.array([e["pull"] for e in entries if np.isfinite(e["pull"])])
    if pulls.size == 0:
        raise ValueError("No finite pull values to plot.")

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.hist(pulls, bins=n_bins, density=True, color="tab:purple", alpha=0.6,
            edgecolor="black", label=f"pulls (N={pulls.size})")

    x = np.linspace(min(-4, pulls.min() - 0.5), max(4, pulls.max() + 0.5), 300)
    ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color="black",
            linestyle="--", label=r"$\mathcal{N}(0,1)$")

    mean_pull = np.mean(pulls)
    std_pull = np.std(pulls, ddof=1) if pulls.size > 1 else np.nan
    info_str = f"mean = {mean_pull:.2f}\nstd = {std_pull:.2f}"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")

    ax.set_xlabel(r"pull $= \dfrac{v_{d,\mathrm{photopeak}} - v_{d,\mathrm{track\!-\!fit}}}"
                  r"{\sqrt{\sigma_{\mathrm{pp}}^2+\sigma_{\mathrm{tf}}^2}}$")
    ax.set_ylabel("density")
    ax.set_title("Pull distribution: photopeak vs. track-fit method")
    ax.legend(fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_pull_distribution{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path


#both methods ramp measurement

def plot_ramp_comparison(
    *,
    ramp_entries,
    base_path,
    plot_type=".png",
    fig_size=(11, 6),
    save_path=None,
    verbose=True,
    ):
    """
    Time-series comparison of the ramp measurement, both methods on the
    same axes -- the direct analogue of the individual
    "ramp_analysis_photo_peak"/"ramp_analysis_track_fit" plots produced by
    the two source scripts, but overlaid so any time-dependent divergence
    between the methods (e.g. one tracking the HV ramp-up differently than
    the other) is visible directly.

    Returns
    -------
    fig, ax, path
    """
    if not ramp_entries:
        raise ValueError("No ramp entries to plot.")

    times = [e["time"] for e in ramp_entries]
    vd_pp = [e["vd_photopeak"] for e in ramp_entries]
    err_pp = [e["err_vd_photopeak"] for e in ramp_entries]
    vd_tf = [e["vd_trackfit"] for e in ramp_entries]
    err_tf = [e["err_vd_trackfit"] for e in ramp_entries]

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.errorbar(times, vd_pp, yerr=err_pp, fmt="o-", capsize=3, markersize=5,
                color="tab:red", label="Photopeak method")
    ax.errorbar(times, vd_tf, yerr=err_tf, fmt="^--", capsize=3, markersize=5,
                color="tab:blue", label="Track-fit method")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Start time")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title(r"Ramp measurement ($U_{\mathrm{wire}}=3600$ V): "
                 "photopeak vs. track-fit method")
    ax.grid(True, alpha=0.3)
    ax.legend(fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/ramp_comparison{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


# rate comparison
def plot_rate_comparison_bars_by_gas_mix(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(14, 7),
    save_path=None,
    y_margin=None,
    verbose=True,
    ):
    """
    Bar-chart comparison of the chamber rate [Hz] from both methods,
    grouped by gas mixture -- same layout/coloring convention as
    plot_vd_comparison_bars_by_gas_mix() (Reds = photopeak, Blues =
    track-fit, colored by U_wire). `entries` is the output of
    build_rate_comparison_entries().

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    max_group_size = max(len(v) for v in grouped.values()) * 2
    group_width = 0.85
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        pair_centers = (np.arange(n) - (n - 1) / 2) * (2 * bar_width)
        for e, pc in zip(group_entries, pair_centers):
            color_pp = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)
            color_tf = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_TRACKFIT, _CMAP_TRACKFIT)

            x_pp = x0 + pc - bar_width / 2
            x_tf = x0 + pc + bar_width / 2

            ax.bar(x_pp, e["rate_photopeak"], width=bar_width * 0.95, color=color_pp)
            ax.errorbar(x_pp, e["rate_photopeak"], yerr=e["err_rate_photopeak"],
                        fmt="none", ecolor="black", capsize=2)

            ax.bar(x_tf, e["rate_trackfit"], width=bar_width * 0.95, color=color_tf)
            ax.errorbar(x_tf, e["rate_trackfit"], yerr=e["err_rate_trackfit"],
                        fmt="none", ecolor="black", capsize=2)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel("Chamber rate [Hz]")
    ax.set_title("Chamber rate comparison: Photopeak (cell rate) vs. Track-fit (muon rate)")
    ax.grid(True, axis="y")

    y_lo = min(min(e["rate_photopeak"] - e["err_rate_photopeak"],
                    e["rate_trackfit"] - e["err_rate_trackfit"]) for e in entries)
    y_hi = max(max(e["rate_photopeak"] + e["err_rate_photopeak"],
                    e["rate_trackfit"] + e["err_rate_trackfit"]) for e in entries)
    if y_margin is None:
        y_margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(max(0, y_lo - y_margin), y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles, legend_labels = [], []
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)))
        legend_labels.append(f"Photopeak, $U_{{wire}}$={u} V")
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_TRACKFIT, _CMAP_TRACKFIT)))
        legend_labels.append(f"Track-fit, $U_{{wire}}$={u} V")
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/rate_comparison_bars{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


# rate cs u_wire

def plot_rate_vs_uwire_both_methods(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(10, 7),
    save_path=None,
    verbose=True,
    ):
    """Rate analogue of plot_vd_vs_uwire_both_methods(): rate [Hz] vs.
    U_wire, one line per gas mixture, solid = photopeak, dashed =
    track-fit, same color per mix."""
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_color_map = {mix: _MIX_CMAP(i % 10) for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for mix in mixes:
        group = sorted(grouped[mix], key=lambda e: e["u_wire"])
        if not group:
            continue
        x = [e["u_wire"] for e in group]
        y_pp = [e["rate_photopeak"] for e in group]
        yerr_pp = [e["err_rate_photopeak"] for e in group]
        y_tf = [e["rate_trackfit"] for e in group]
        yerr_tf = [e["err_rate_trackfit"] for e in group]

        color = mix_color_map[mix]
        ax.errorbar(x, y_pp, yerr=yerr_pp, marker="o", linestyle="-",
                    capsize=3, color=color, label=f"{mix} (photopeak)")
        ax.errorbar(x, y_tf, yerr=yerr_tf, marker="^", linestyle="--",
                    capsize=3, color=color, label=f"{mix} (track-fit)")

    ax.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax.set_ylabel("Chamber rate [Hz]")
    ax.set_title("Chamber rate vs. wire voltage -- both methods")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Ar/CO$_2$ [%] (method)", fontsize=9, ncol=2,
              fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/rate_vs_uwire_both_methods{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


# ramp measurement rate

def plot_ramp_rate_comparison(
    *,
    ramp_entries,
    base_path,
    plot_type=".png",
    fig_size=(11, 6),
    save_path=None,
    verbose=True,
    ):
    """Rate analogue of plot_ramp_comparison(): chamber rate [Hz] vs. time
    for both methods, from build_ramp_rate_comparison_entries()."""
    if not ramp_entries:
        raise ValueError("No ramp entries to plot.")

    times = [e["time"] for e in ramp_entries]
    rate_pp = [e["rate_photopeak"] for e in ramp_entries]
    err_pp = [e["err_rate_photopeak"] for e in ramp_entries]
    rate_tf = [e["rate_trackfit"] for e in ramp_entries]
    err_tf = [e["err_rate_trackfit"] for e in ramp_entries]

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.errorbar(times, rate_pp, yerr=err_pp, fmt="o-", capsize=3, markersize=5,
                color="tab:red", label="Photopeak method (cell rate)")
    ax.errorbar(times, rate_tf, yerr=err_tf, fmt="^--", capsize=3, markersize=5,
                color="tab:blue", label="Track-fit method (muon rate)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Start time")
    ax.set_ylabel("Chamber rate [Hz]")
    ax.set_title(r"Ramp measurement ($U_{\mathrm{wire}}=3600$ V): rate, both methods")
    ax.grid(True, alpha=0.3)
    ax.legend(fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/ramp_rate_comparison{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path



def build_rate_entries_single_method(
    *,
    analysis_out,
    dataset_info_fn=parse_fit_name,
    rate_key="avg_rate_chamber",
    err_key="avg_rate_chamber_err",
    verbose=True,
    ):
    """
    One entry per dataset, read directly out of a single analysis_out dict
    (no cross-method matching). Use rate_key/err_key to pick which rate,
    e.g. "avg_rate_phi1"/"avg_rate_phi1_err", "avg_rate_theta"/
    "avg_rate_theta_err", "avg_rate_phi3"/"avg_rate_phi3_err" -- default
    is the overall chamber rate.

    Returns
    -------
    entries : list[dict]
        Each entry: {"dataset": str, "mix": str, "u_wire": int,
                      "rate": float, "err_rate": float}
    """
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue
        if rate_key not in result or err_key not in result:
            if verbose:
                print(f"  skipping {dataset_name}: missing '{rate_key}'/'{err_key}'")
            continue
        entries.append({
            "dataset": dataset_name,
            "mix": f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}",
            "u_wire": int(info["U_wire"]),
            "rate": float(result[rate_key]),
            "err_rate": float(result[err_key]),
        })
    return entries


def plot_rate_bars_by_gas_mix(
    *,
    entries,
    base_path,
    rate_label="Chamber rate [Hz]",
    method_label="",
    plot_type=".png",
    fig_size=(12, 7),
    save_path=None,
    y_margin=None,
    verbose=True,
    ):
    """
    Bar-chart of `entries` (from build_rate_entries_single_method()),
    grouped by gas mixture, one bar per U_wire, colored by U_wire (same
    fixed color map used everywhere else in this file). This is the most
    direct "how does rate depend on gas mix and wire voltage" view: read
    across a group to see the U_wire dependence at fixed mix, read across
    groups (same bar color) to see the mix dependence at fixed U_wire.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

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
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK)
            ax.bar(x0 + offset, e["rate"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["rate"], yerr=e["err_rate"],
                        fmt="none", ecolor="black", capsize=3)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(rate_label)
    title = "Rate vs. gas mixture, colored by wire voltage"
    if method_label:
        title += f" ({method_label})"
    ax.set_title(title)
    ax.grid(True, axis="y")

    y_lo = min(e["rate"] - e["err_rate"] for e in entries)
    y_hi = max(e["rate"] + e["err_rate"] for e in entries)
    if y_margin is None:
        y_margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(max(0, y_lo - y_margin), y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                       color=_wire_color(u, _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels, fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        tag = f"_{method_label}" if method_label else ""
        save_path = base_path + f"plots/compare/rate_bars_by_gas_mix{tag}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


def plot_rate_vs_uwire_trend(
    *,
    entries,
    base_path,
    rate_label="Chamber rate [Hz]",
    method_label="",
    plot_type=".png",
    fig_size=(10, 7),
    save_path=None,
    verbose=True,
    ):
    """
    Trend plot: rate vs. U_wire, one line per gas mixture -- isolates the
    U_wire dependence cleanly, with the mix-to-mix comparison available
    via line color.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_color_map = {mix: _MIX_CMAP(i % 10) for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for mix in mixes:
        group = sorted(grouped[mix], key=lambda e: e["u_wire"])
        if not group:
            continue
        x = [e["u_wire"] for e in group]
        y = [e["rate"] for e in group]
        yerr = [e["err_rate"] for e in group]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3,
                    color=mix_color_map[mix], label=mix)

    ax.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax.set_ylabel(rate_label)
    title = "Rate vs. wire voltage, per gas mixture"
    if method_label:
        title += f" ({method_label})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Ar/CO$_2$ [%]", fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        tag = f"_{method_label}" if method_label else ""
        save_path = base_path + f"plots/compare/rate_vs_uwire_trend{tag}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


def plot_rate_heatmap(
    *,
    entries,
    base_path,
    rate_label="Chamber rate [Hz]",
    method_label="",
    plot_type=".png",
    fig_size=(10, 6),
    save_path=None,
    cmap="viridis",
    verbose=True,
    ):
    """
    2D heatmap of rate over the (gas mix, U_wire) grid -- the most compact
    view of how BOTH parameters jointly influence the rate at once. Cells
    with no measured dataset are left blank (NaN, shown as background
    color); each measured cell is annotated with its rate value.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    u_wires = sorted(set(e["u_wire"] for e in entries))
    mix_to_row = {m: i for i, m in enumerate(mixes)}
    wire_to_col = {u: i for i, u in enumerate(u_wires)}

    grid = np.full((len(mixes), len(u_wires)), np.nan)
    for e in entries:
        grid[mix_to_row[e["mix"]], wire_to_col[e["u_wire"]]] = e["rate"]

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(u_wires)))
    ax.set_xticklabels(u_wires)
    ax.set_yticks(range(len(mixes)))
    ax.set_yticklabels(mixes)
    ax.set_xlabel(r"$U_{\mathrm{wire}}$ [V]")
    ax.set_ylabel("Gas mixture (Ar/CO2) [%]")
    title = "Rate vs. gas mixture & wire voltage"
    if method_label:
        title += f" ({method_label})"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05)
    cbar.set_label(rate_label)

    for e in entries:
        i = mix_to_row[e["mix"]]
        j = wire_to_col[e["u_wire"]]
        # pick a readable text color depending on the cell's brightness
        norm_val = (grid[i, j] - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-12)
        text_color = "white" if norm_val < 0.6 else "black"
        ax.text(j, i, f"{e['rate']:.1f}\n$\\pm${e['err_rate']:.1f}",
                ha="center", va="center", color=text_color, fontsize=8)

    fig.tight_layout()

    if save_path is None:
        tag = f"_{method_label}" if method_label else ""
        save_path = base_path + f"plots/compare/rate_heatmap{tag}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, ax, save_path


def build_rate_evolution_entries_photopeak(
    *,
    analysis_out_photopeak,
    dataset_info_fn=parse_fit_name,
    rate_key="avg_rate_chamber",
    err_key="avg_rate_chamber_err",
    verbose=True,
    ):
    """
    One entry per photopeak dataset, ordered chronologically -- traces how
    the rate evolves across the measurement campaign.

    The cosmic-scan dataset names ("cosmic_82-18_...") carry no real
    timestamp anywhere (not in the name, not in the stored fit result
    dict -- only the ramp run's "data_mic0_start_..." names do). What IS
    meaningful is processing order: analysis_out is built by
    photo_peak_analysis.py iterating its hand-ordered `list_of_fits`
    (82/18 -> 87/13), which follows the actual campaign order, so dict
    insertion order is used as the x-axis here instead.

    Returns
    -------
    entries : list[dict]
        Each entry: {"dataset": str, "seq": int, "mix": str,
                      "u_wire": int or None, "rate": float, "err_rate": float}
        in analysis_out_photopeak's insertion (= campaign) order.
    """
    entries = []
    for seq_idx, (dataset_name, result) in enumerate(analysis_out_photopeak.items()):
        if rate_key not in result or err_key not in result:
            if verbose:
                print(f"  skipping {dataset_name}: missing '{rate_key}'/'{err_key}'")
            continue

        try:
            info = dataset_info_fn(name=dataset_name)
            mix = f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}"
            u_wire = int(info["U_wire"])
        except Exception:
            mix, u_wire = "unknown", None

        entries.append({
            "dataset": dataset_name, "seq": seq_idx, "mix": mix, "u_wire": u_wire,
            "rate": float(result[rate_key]), "err_rate": float(result[err_key]),
        })

    return entries

def plot_rate_evolution_photopeak(
    *,
    entries,
    base_path,
    rate_label="Chamber rate [Hz]",
    method_label="",
    plot_type=".png",
    fig_size=(13, 6),
    save_path=None,
    verbose=True,
    ):
    """
    Rate-evolution plot for the photopeak method across the campaign,
    x-axis = processing/campaign order (see
    build_rate_evolution_entries_photopeak for why this isn't wall-clock
    time), points colored by gas mixture so any drift lines up visibly
    with a mix change. Dataset names are shown as tick labels (rotated)
    so a specific run can still be identified.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")

    mixes = sorted(
        set(e["mix"] for e in entries if e["mix"] != "unknown"),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    if any(e["mix"] == "unknown" for e in entries):
        mixes.append("unknown")
    mix_color_map = {mix: (_MIX_CMAP(i % 10) if mix != "unknown" else "gray")
                      for i, mix in enumerate(mixes)}

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    seqs = [e["seq"] for e in entries]
    rates = [e["rate"] for e in entries]
    errs = [e["err_rate"] for e in entries]

    ax.plot(seqs, rates, color="lightgray", linewidth=1, zorder=1)
    ax.errorbar(seqs, rates, yerr=errs, fmt="none", ecolor="black",
                capsize=2, zorder=2)
    for mix in mixes:
        mix_seqs = [e["seq"] for e in entries if e["mix"] == mix]
        mix_rates = [e["rate"] for e in entries if e["mix"] == mix]
        ax.scatter(mix_seqs, mix_rates, color=mix_color_map[mix], label=mix,
                   zorder=3, s=40, edgecolor="black", linewidth=0.5)

    ax.set_xticks(seqs)
    ax.set_xticklabels([e["dataset"] for e in entries], rotation=90, fontsize=7)
    ax.set_xlabel("Dataset (campaign order)")
    ax.set_ylabel(rate_label)
    title = "Rate evolution over the measurement campaign (photopeak method)"
    if method_label:
        title += f" [{method_label}]"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Ar/CO$_2$ [%]", fontsize=9, ncol=2,
              fancybox=False, framealpha=params._legend_alpha)
    fig.tight_layout()

    if save_path is None:
        tag = f"_{method_label}" if method_label else ""
        save_path = base_path + f"plots/compare/rate_evolution_photopeak{tag}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path
# idndividual vd plots

def build_vd_entries_single_method(
    *,
    analysis_out,
    method,
    dataset_info_fn=parse_fit_name,
    verbose=True,
    ):
    """
    One entry per dataset, read directly out of a single analysis_out dict
    (no cross-method matching) -- the vd analogue of
    build_rate_entries_single_method(). `method` selects the extractor:
    "photopeak" (reads "v_drift"/"err_v_drift" via get_vd_photopeak) or
    "trackfit" (reads "peak"/"tot_err" via get_vd_trackfit) -- see those
    two functions for why the pickles store this differently. Since this
    doesn't require the dataset to exist in the OTHER method's dict, it
    also picks up datasets that were only processed by one pipeline.

    Returns
    -------
    entries : list[dict]
        Each entry: {"dataset": str, "mix": str, "u_wire": int,
                      "vd": float, "err_vd": float}
    """
    if method not in ("photopeak", "trackfit"):
        raise ValueError(f"method must be 'photopeak' or 'trackfit', got {method!r}")
    getter = get_vd_photopeak if method == "photopeak" else get_vd_trackfit

    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name=dataset_name)
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue
        try:
            vd, err_vd = getter(dataset_name=dataset_name, result=result)
        except KeyError as e:
            if verbose:
                print(f"  skipping {dataset_name}: {e}")
            continue
        entries.append({
            "dataset": dataset_name,
            "mix": f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}",
            "u_wire": int(info["U_wire"]),
            "vd": vd, "err_vd": err_vd,
        })
    return entries


def plot_vd_bars_by_gas_mix_single_method(
    *,
    entries,
    base_path,
    method_label,
    plot_type=".png",
    fig_size=(12, 7),
    save_path=None,
    y_margin=1.0,
    verbose=True,
    ):
    """
    Bar-chart of fitted drift velocities from ONE method, grouped by gas
    mixture, one bar per U_wire, colored by U_wire -- same layout as
    plot_rate_bars_by_gas_mix(), just for v_drift instead of rate.
    `method_label` must be "photopeak" or "trackfit"; picks the matching
    Reds/Blues color convention used everywhere else in this file.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")
    if method_label not in ("photopeak", "trackfit"):
        raise ValueError(f"method_label must be 'photopeak' or 'trackfit', got {method_label!r}")

    color_map = _WIRE_COLOR_MAP_PHOTOPEAK if method_label == "photopeak" else _WIRE_COLOR_MAP_TRACKFIT
    cmap = _CMAP_PHOTOPEAK if method_label == "photopeak" else _CMAP_TRACKFIT
    method_title = "Photopeak method" if method_label == "photopeak" else "Track-fit method"

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

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
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = _wire_color(e["u_wire"], color_map, cmap)
            ax.bar(x0 + offset, e["vd"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["vd"], yerr=e["err_vd"],
                        fmt="none", ecolor="black", capsize=3)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title(f"Drift velocity vs. gas mixture, colored by wire voltage ({method_title})")
    ax.grid(True, axis="y")

    y_lo = min(e["vd"] - e["err_vd"] for e in entries)
    y_hi = max(e["vd"] + e["err_vd"] for e in entries)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_wire_color(u, color_map, cmap))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels, fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_bars_{method_label}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")
    return fig, ax, save_path


#tex table

def make_comparison_tex_table(*, entries, float_precision=3):
    """
    LaTeX table listing, per dataset, both methods' drift velocities, the
    difference, and the pull -- mirrors the tex-table style already used
    by analyze_pattern_type_data() in sl_fits_analysis.py.

    Returns
    -------
    tex_table : str
    """
    fp = float_precision
    lines = [
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"    \hline",
        r"    Dataset & Mix & $U_{\mathrm{wire}}$ [V] & $v_{d,\mathrm{pp}}$ [$\mu$m/ns] "
        r"& $v_{d,\mathrm{tf}}$ [$\mu$m/ns] & pull \\ \hline",
    ]
    for e in sorted(entries, key=lambda e: (e["mix"], e["u_wire"])):
        lines.append(
            f"    {e['dataset'].replace('_', r'\\_')} & {e['mix']} & {e['u_wire']} & "
            f"${np.round(e['vd_photopeak'], fp):.{fp}f} \\pm {np.round(e['err_vd_photopeak'], fp):.{fp}f}$ & "
            f"${np.round(e['vd_trackfit'], fp):.{fp}f} \\pm {np.round(e['err_vd_trackfit'], fp):.{fp}f}$ & "
            f"${np.round(e['pull'], 2):.2f}$ \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# =================================================================
# main function
# =================================================================
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12})
def main(save_plots=True, do_cut_data=True):
    plot_type = ".png"
    fig_size = (8, 6)

    base_path = "data_ba/"
    pcls_path = "pcls/"
    pcls_file_path = f"{base_path}{pcls_path}"
    plot_save_path = base_path + "plots/compare/"

    if save_plots:
        os.makedirs(plot_save_path, exist_ok=True)


    analysis_out_photopeak = data_utils.load_pickle(
        f"{pcls_file_path}analysis_out_photo_peak_data.pcl"
    )
    analysis_out_track_fit = data_utils.load_pickle(
        f"{pcls_file_path}analysis_out_track_fit.pcl"
    )

    analysis_out_photopeak_ramp = data_utils.load_pickle(
        f"{pcls_file_path}analysis_out_photo_peak_ramp.pcl"
    )
    analysis_out_track_fit_ramp = data_utils.load_pickle(
        f"{pcls_file_path}analysis_out_track_fit_ramp.pcl"
    )

    # ---- cosmic gas-mix scan comparison ----
    entries = build_comparison_entries(
        analysis_out_photopeak=analysis_out_photopeak,
        analysis_out_track_fit=analysis_out_track_fit,
        dataset_info_fn=parse_fit_name,
    )

    if entries:
        plot_vd_comparison_bars_by_gas_mix(
            entries=entries, base_path=base_path, plot_type=plot_type,
        )
        plot_vd_vs_uwire_both_methods(
            entries=entries, base_path=base_path, plot_type=plot_type,
        )
        plot_method_difference_by_gas_mix(
            entries=entries, base_path=base_path, plot_type=plot_type,
        )
        plot_pull_distribution(
            entries=entries, base_path=base_path, plot_type=plot_type,
        )

        tex_table = make_comparison_tex_table(entries=entries)
        print(tex_table)
        tex_path = plot_save_path + "vd_comparison_table.tex"
        with open(tex_path, "w") as f:
            f.write(tex_table)
        print(f"store tex table as {tex_path}.")
    else:
        print("No overlapping cosmic-scan datasets found between the two methods; "
              "skipping gas-mix comparison plots.")

    # ---- vd bar chart, plotted individually per method (doesn't require
    # the dataset to be present in the other method's dict) ----
    for method in ("photopeak", "trackfit"):
        analysis_out_this = analysis_out_photopeak if method == "photopeak" else analysis_out_track_fit
        vd_entries_single = build_vd_entries_single_method(
            analysis_out=analysis_out_this, method=method, dataset_info_fn=parse_fit_name,
        )
        if not vd_entries_single:
            print(f"No datasets with a valid vd for method={method!r}; skipping.")
            continue
        plot_vd_bars_by_gas_mix_single_method(
            entries=vd_entries_single, base_path=base_path,
            method_label=method, plot_type=plot_type,
        )

    # ---- ramp measurement comparison ----
    ramp_entries = build_ramp_comparison_entries(
        analysis_out_photopeak_ramp=analysis_out_photopeak_ramp,
        analysis_out_track_fit_ramp=analysis_out_track_fit_ramp,
    )

    if ramp_entries:
        plot_ramp_comparison(
            ramp_entries=ramp_entries, base_path=base_path, plot_type=plot_type,
        )
        plot_pull_distribution(
            entries=ramp_entries, base_path=base_path, plot_type=plot_type,
            save_path=plot_save_path + f"ramp_pull_distribution{plot_type}",
        )
    else:
        print("No overlapping ramp datasets found between the two methods; "
              "skipping ramp comparison plots.")

    # ---- rate comparison (requires the track_rate patch in sl_fits_analysis.py,
    # see the comment above get_rate_photopeak/get_rate_trackfit) ----
    rate_entries = build_rate_comparison_entries(
        analysis_out_photopeak=analysis_out_photopeak,
        analysis_out_track_fit=analysis_out_track_fit,
        dataset_info_fn=parse_fit_name,
    )

    if rate_entries:
        plot_rate_comparison_bars_by_gas_mix(
            entries=rate_entries, base_path=base_path, plot_type=plot_type,
        )
        plot_rate_vs_uwire_both_methods(
            entries=rate_entries, base_path=base_path, plot_type=plot_type,
        )
    else:
        print("No overlapping datasets with both a photopeak and a track-fit "
              "rate found; skipping rate comparison plots (see the track_rate "
              "patch comment above get_rate_photopeak/get_rate_trackfit).")

    ramp_rate_entries = build_ramp_rate_comparison_entries(
        analysis_out_photopeak_ramp=analysis_out_photopeak_ramp,
        analysis_out_track_fit_ramp=analysis_out_track_fit_ramp,
    )

    if ramp_rate_entries:
        plot_ramp_rate_comparison(
            ramp_entries=ramp_rate_entries, base_path=base_path, plot_type=plot_type,
        )
    else:
        print("No overlapping ramp datasets with both rates found; "
              "skipping ramp rate comparison plot.")

    # ---- rate vs. gas mix / wire voltage (single method: photopeak,
    # since that's the analysis_out that already has rates populated) ----
    for rate_key, err_key, label in [
        ("avg_rate_chamber", "avg_rate_chamber_err", "Chamber rate [Hz]"),
        ("avg_rate_phi1",    "avg_rate_phi1_err",    "SL1 (phi) rate [Hz]"),
        ("avg_rate_theta",   "avg_rate_theta_err",   "SL2 (theta) rate [Hz]"),
        ("avg_rate_phi3",    "avg_rate_phi3_err",    "SL3 (phi) rate [Hz]"),
    ]:
        rate_entries_single = build_rate_entries_single_method(
            analysis_out=analysis_out_photopeak,
            dataset_info_fn=parse_fit_name,
            rate_key=rate_key, err_key=err_key,
        )
        if not rate_entries_single:
            print(f"No datasets with '{rate_key}' found; skipping.")
            continue

        plot_rate_bars_by_gas_mix(
            entries=rate_entries_single, base_path=base_path, rate_label=label,
            method_label=rate_key, plot_type=plot_type,
        )
        plot_rate_vs_uwire_trend(
            entries=rate_entries_single, base_path=base_path, rate_label=label,
            method_label=rate_key, plot_type=plot_type,
        )
        plot_rate_heatmap(
            entries=rate_entries_single, base_path=base_path, rate_label=label,
            method_label=rate_key, plot_type=plot_type,
        )

        # rate evolution over time, photopeak method
        rate_evolution_entries = build_rate_evolution_entries_photopeak(
            analysis_out_photopeak=analysis_out_photopeak,
            dataset_info_fn=parse_fit_name,
            rate_key=rate_key, err_key=err_key,
        )
        if rate_evolution_entries:
            plot_rate_evolution_photopeak(
                entries=rate_evolution_entries, base_path=base_path,
                rate_label=label, method_label=rate_key, plot_type=plot_type,
            )
        else:
            print(f"No datasets with a parseable start time for '{rate_key}'; "
                  "skipping rate evolution plot.")

    return


if __name__ == "__main__":
    main()
    print("####done####")
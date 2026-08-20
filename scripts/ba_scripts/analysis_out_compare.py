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


def parse_sim_name(*, name):
    # Expected format: ar-<Ar%>_co2-<CO2%>_anode<U_wire>V
    # (matches dataset_key = f"ar-{ar_pct:.1f}_co2-{co2_pct:.1f}_anode{anode_voltage_V:.0f}V"
    # from the simulation script's main())
    pattern = r"^ar-(\d+(?:\.\d+)?)_co2-(\d+(?:\.\d+)?)_anode(\d+(?:\.\d+)?)V$"
    match = re.match(pattern, name)
    if not match:
        raise ValueError(f"String does not match the expected simulation dataset format: {name}")

    pct_ar, pct_co2, u_wire = match.groups()

    return {
        "name": name,
        "pct_Ar": int(round(float(pct_ar))),
        "pct_CO2": int(round(float(pct_co2))),
        "U_wire": int(round(float(u_wire))),
    }



_WIRE_VOLTAGES = [3550, 3575, 3600, 3625, 3650]

_CMAP_PHOTOPEAK = plt.cm.Reds
_CMAP_TRACKFIT = plt.cm.Blues
_CMAP_SIM = plt.cm.Purples
# for the sim-internal pp-vs-tf comparison: keep both bars in the purple
# family (still reads as "simulation"), but shade each toward the color of
# the measurement method it's the analogue of, so the two bars are easy to
# tell apart without a legend lookup -- RdPu (red-purple) for the
# photopeak-style estimate, BuPu (blue-purple) for the track-fit-style one.
_CMAP_SIM_PP = plt.cm.RdPu
_CMAP_SIM_TF = plt.cm.BuPu

_WIRE_COLOR_MAP_PHOTOPEAK = {
    v: _CMAP_PHOTOPEAK(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_WIRE_COLOR_MAP_TRACKFIT = {
    v: _CMAP_TRACKFIT(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_WIRE_COLOR_MAP_SIM = {
    v: _CMAP_SIM(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_WIRE_COLOR_MAP_SIM_PP = {
    v: _CMAP_SIM_PP(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
    for i, v in enumerate(_WIRE_VOLTAGES)
}
_WIRE_COLOR_MAP_SIM_TF = {
    v: _CMAP_SIM_TF(0.3 + 0.7 * i / (len(_WIRE_VOLTAGES) - 1))
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


def _ratio_and_err(*, num, err_num, den, err_den):
    """Shared helper: ratio = num/den with standard error propagation
    err_ratio = |ratio| * sqrt((err_num/num)^2 + (err_den/den)^2).
    Returns (nan, nan) if num or den is zero, rather than raising, since a
    zero denominator/numerator can legitimately occur in edge-case fits
    and callers just want to skip it downstream (matching how "pull" is
    already handled with np.nan for err_diff == 0)."""
    if den == 0 or num == 0:
        return np.nan, np.nan
    ratio = num / den
    err_ratio = abs(ratio) * np.sqrt((err_num / num) ** 2 + (err_den / den) ** 2)
    return ratio, err_ratio


def fit_constant(*, values, errors):
    """
    Fit a single constant c to a set of (value, error) pairs by
    inverse-variance-weighted least squares -- i.e. the minimum-chi2
    value of a flat line y = c through the data:
        c        = sum(v_i / err_i^2) / sum(1 / err_i^2)
        err_c    = sqrt(1 / sum(1 / err_i^2))
        chi2     = sum(((v_i - c) / err_i)^2)
        ndof     = n - 1
    This is the "factor you're looking for" behind a set of per-dataset
    ratios (or any other per-dataset bar values): the single number that
    best represents all the bars at once, weighted by how precisely each
    one is known. Non-finite values/errors and errors <= 0 are dropped
    before fitting.

    Parameters
    ----------
    values, errors : array-like
        Per-dataset values and their 1-sigma uncertainties.

    Returns
    -------
    const, err_const, chi2, ndof : float, float, float, int
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    values = values[mask]
    errors = errors[mask]
    if values.size == 0:
        raise ValueError("No finite (value, error) pairs to fit a constant to.")

    weights = 1.0 / errors ** 2
    const = np.sum(values * weights) / np.sum(weights)
    err_const = np.sqrt(1.0 / np.sum(weights))
    chi2 = np.sum(((values - const) / errors) ** 2)
    ndof = values.size - 1
    return const, err_const, chi2, ndof


def _draw_constant_fit(*, ax, const, err_const, chi2, ndof, color="black"):
    """Shared helper: overlay a fit_constant() result on a bar-plot axis
    as a solid horizontal line at `const` plus a shaded +-1 sigma band,
    and return the (handle, label) pair to add to the legend so the fit
    value is readable directly instead of just eyeballed off the line."""
    line = ax.axhline(y=const, color=color, linewidth=1.6, linestyle="-", zorder=5)
    ax.axhspan(const - err_const, const + err_const, color=color, alpha=0.15, zorder=0)
    ndof_str = f"{ndof}" if ndof > 0 else "0"
    label = (rf"const. fit: ${const:.4f} \pm {err_const:.4f}$"
             rf" ($\chi^2$/ndof = {chi2:.1f}/{ndof_str})")
    return line, label


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


# extract vd from simulation analysis out
def get_vd_sim(*, dataset_name, result, key="v_drift_tf"):
    """Return (v_drift, err_v_drift) [um/ns] from one simulation analysis_out
    entry. The simulation produces two independent drift-velocity estimates
    per dataset, mirroring the two measurement methods:
      - "v_drift_tf" -- from the primary track's space-time linear fit
        (the track-fit-method analogue)
      - "v_drift_pp" -- from the secondary/photoelectron peak position
        (the photopeak-method analogue)
    `key` selects which one to read; pass the matching "<key>_err" pair."""
    err_key = f"{key}_err"
    if key in result and err_key in result:
        return float(result[key]), float(result[err_key])
    raise KeyError(f"{dataset_name}: no '{key}'/'{err_key}' in simulation result")



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
            "ratio": float,      # vd_photopeak / vd_trackfit
            "err_ratio": float,  # error-propagated uncertainty on ratio
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
        ratio, err_ratio = _ratio_and_err(num=vd_pp, err_num=err_pp, den=vd_tf, err_den=err_tf)

        entries.append({
            "dataset": name,
            "mix": f"{int(info['pct_Ar'])}/{int(info['pct_CO2'])}",
            "u_wire": int(info["U_wire"]),
            "vd_photopeak": vd_pp, "err_vd_photopeak": err_pp,
            "vd_trackfit": vd_tf, "err_vd_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
            "ratio": ratio, "err_ratio": err_ratio,
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
            "ratio": float, "err_ratio": float,
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
        ratio, err_ratio = _ratio_and_err(num=vd_pp, err_num=err_pp, den=vd_tf, err_den=err_tf)
        # form return
        entries.append({
            "dataset": name, "time": t,
            "vd_photopeak": vd_pp, "err_vd_photopeak": err_pp,
            "vd_trackfit": vd_tf, "err_vd_trackfit": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
            "ratio": ratio, "err_ratio": err_ratio,
        })

    entries.sort(key=lambda e: e["time"])
    return entries


def build_sim_vs_measurement_vd_entries(
    *,
    analysis_out_sim,
    analysis_out_measurement,
    sim_vd_key,
    measurement_getter,
    sim_info_fn=parse_sim_name,
    measurement_info_fn=parse_fit_name,
    verbose=True,
    ):
    """
    Match simulation datasets to measurement datasets by (pct_Ar, pct_CO2,
    U_wire) -- NOT by dataset name. The simulation script's dataset keys
    ("ar-85.0_co2-15.0_anode3600V") and the measurement scripts' dataset
    names ("cosmic_85-15_3600-1800-1200_run1_..." /
    "data_mic0_start_..._stop_...") are unrelated strings even when they
    describe the same gas mixture and wire voltage, so matching is done on
    the parsed (pct_Ar, pct_CO2, U_wire) tuple instead of set intersection
    over raw names (contrast with build_comparison_entries()).

    Parameters
    ----------
    analysis_out_sim : dict
        {dataset_name: fit_results} from the simulation script's
        analysis_out_simulation.pcl.
    analysis_out_measurement : dict
        {dataset_name: fit_results} from EITHER measurement method
        (analysis_out_photopeak or analysis_out_track_fit) -- pick the
        matching `measurement_getter` to go with it.
    sim_vd_key : str
        "v_drift_tf" or "v_drift_pp" -- which simulated drift velocity to
        compare (see get_vd_sim() docstring for which measurement method
        each one is the analogue of).
    measurement_getter : callable
        get_vd_photopeak or get_vd_trackfit.
    sim_info_fn, measurement_info_fn : callable
        Parsers for the sim / measurement dataset names, each returning a
        dict with "pct_Ar", "pct_CO2", "U_wire".

    Returns
    -------
    entries : list[dict]
        Each entry:
        {
            "sim_dataset": str, "measurement_dataset": str,
            "mix": "Ar/CO2" string, "u_wire": int,
            "vd_sim": float, "err_vd_sim": float,
            "vd_measurement": float, "err_vd_measurement": float,
            "diff": float,       # vd_sim - vd_measurement
            "err_diff": float,   # sqrt(err_sim^2 + err_measurement^2)
            "pull": float,       # diff / err_diff
            "ratio": float,      # vd_sim / vd_measurement
            "err_ratio": float,  # error-propagated uncertainty on ratio
        }
    """
    # index simulation datasets by (pct_Ar, pct_CO2, U_wire)
    sim_by_key = {}
    for name, result in analysis_out_sim.items():
        try:
            info = sim_info_fn(name=name)
        except Exception as e:
            if verbose:
                print(f"  skipping sim dataset {name}: could not parse dataset info ({e})")
            continue
        sim_by_key[(info["pct_Ar"], info["pct_CO2"], info["U_wire"])] = (name, result)

    entries = []
    n_missing_sim_vd = 0
    n_missing_meas_vd = 0
    n_unmatched = 0
    for meas_name, meas_result in analysis_out_measurement.items():
        try:
            info = measurement_info_fn(name=meas_name)
        except Exception as e:
            if verbose:
                print(f"  skipping measurement dataset {meas_name}: could not parse dataset info ({e})")
            continue

        key = (info["pct_Ar"], info["pct_CO2"], info["U_wire"])
        if key not in sim_by_key:
            n_unmatched += 1
            continue
        sim_name, sim_result = sim_by_key[key]

        try:
            vd_sim, err_sim = get_vd_sim(dataset_name=sim_name, result=sim_result, key=sim_vd_key)
        except KeyError as e:
            n_missing_sim_vd += 1
            if verbose:
                print(f"  skipping {key}: {e}")
            continue

        try:
            vd_meas, err_meas = measurement_getter(dataset_name=meas_name, result=meas_result)
        except KeyError as e:
            n_missing_meas_vd += 1
            if verbose:
                print(f"  skipping {key}: {e}")
            continue

        diff = vd_sim - vd_meas
        err_diff = np.sqrt(err_sim ** 2 + err_meas ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan
        ratio, err_ratio = _ratio_and_err(num=vd_sim, err_num=err_sim, den=vd_meas, err_den=err_meas)

        entries.append({
            "sim_dataset": sim_name, "measurement_dataset": meas_name,
            "mix": f"{info['pct_Ar']}/{info['pct_CO2']}",
            "u_wire": int(info["U_wire"]),
            "vd_sim": vd_sim, "err_vd_sim": err_sim,
            "vd_measurement": vd_meas, "err_vd_measurement": err_meas,
            "diff": diff, "err_diff": err_diff, "pull": pull,
            "ratio": ratio, "err_ratio": err_ratio,
        })

    if verbose:
        print(f"{len(entries)} dataset(s) matched between simulation ('{sim_vd_key}') "
              f"and measurement (by gas mix + U_wire).")
        if n_unmatched:
            print(f"  {n_unmatched} measurement dataset(s) had no matching (mix, U_wire) in the simulation.")
        if n_missing_sim_vd:
            print(f"  {n_missing_sim_vd} matched dataset(s) skipped: missing '{sim_vd_key}' in sim result.")
        if n_missing_meas_vd:
            print(f"  {n_missing_meas_vd} matched dataset(s) skipped: missing vd in measurement result.")

    return entries


def build_sim_pp_vs_tf_entries(
    *,
    analysis_out_sim,
    sim_info_fn=parse_sim_name,
    verbose=True,
    ):
    """
    Compare the simulation's own two drift-velocity estimates against each
    other, per dataset: v_drift_pp (photopeak-style, from the secondary/
    photoelectron peak) vs. v_drift_tf (track-fit-style, from the primary
    track's space-time linear fit). Unlike build_sim_vs_measurement_vd_entries,
    this stays entirely within analysis_out_sim -- no name/key matching
    across dicts is needed since both values live in the same per-dataset
    result.

    Returns
    -------
    entries : list[dict]
        Each entry:
        {
            "dataset": str, "mix": "Ar/CO2" string, "u_wire": int,
            "vd_pp": float, "err_vd_pp": float,
            "vd_tf": float, "err_vd_tf": float,
            "diff": float,       # vd_pp - vd_tf
            "err_diff": float,   # sqrt(err_pp^2 + err_tf^2)
            "pull": float,       # diff / err_diff
        }
    """
    entries = []
    n_missing = 0
    for name, result in analysis_out_sim.items():
        try:
            info = sim_info_fn(name=name)
        except Exception as e:
            if verbose:
                print(f"  skipping sim dataset {name}: could not parse dataset info ({e})")
            continue

        try:
            vd_pp, err_pp = get_vd_sim(dataset_name=name, result=result, key="v_drift_pp")
            vd_tf, err_tf = get_vd_sim(dataset_name=name, result=result, key="v_drift_tf")
        except KeyError as e:
            n_missing += 1
            if verbose:
                print(f"  skipping {name}: {e}")
            continue

        diff = vd_pp - vd_tf
        err_diff = np.sqrt(err_pp ** 2 + err_tf ** 2)
        pull = diff / err_diff if err_diff > 0 else np.nan

        entries.append({
            "dataset": name,
            "mix": f"{info['pct_Ar']}/{info['pct_CO2']}",
            "u_wire": int(info["U_wire"]),
            "vd_pp": vd_pp, "err_vd_pp": err_pp,
            "vd_tf": vd_tf, "err_vd_tf": err_tf,
            "diff": diff, "err_diff": err_diff, "pull": pull,
        })

    if verbose:
        print(f"{len(entries)} simulation dataset(s) have both 'v_drift_pp' and 'v_drift_tf'.")
        if n_missing:
            print(f"  {n_missing} dataset(s) skipped: missing one or both keys.")

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



def plot_vd_comparison_bars_sim_vs_measurement(
    *,
    entries,
    base_path,
    measurement_label,
    plot_type=".png",
    fig_size=(14, 7),
    save_path=None,
    y_margin=1.0,
    verbose=True,
    ):
    """
    Bar-chart comparison of drift velocity: simulation vs. ONE measurement
    method, grouped by gas mixture -- the simulation analogue of
    plot_vd_comparison_bars_by_gas_mix(). Every matched dataset gets two
    adjacent bars: simulation (Purples colormap, colored by U_wire) and
    the chosen measurement method (Reds for photopeak / Blues for
    track-fit, matching that method's own color convention elsewhere in
    this file).

    Parameters
    ----------
    entries : list[dict]
        Output of build_sim_vs_measurement_vd_entries().
    base_path : str
        Used to build the default output path.
    measurement_label : str
        "photopeak" or "trackfit" -- selects the measurement bars' color
        map and the plot title/filename.
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/compare/vd_comparison_bars_sim_vs_{measurement_label}{plot_type}".
    y_margin : float, default 1.0
        Padding (um/ns) below/above the data range for the y-axis limits.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")
    if measurement_label not in ("photopeak", "trackfit"):
        raise ValueError(f"measurement_label must be 'photopeak' or 'trackfit', got {measurement_label!r}")

    meas_color_map = _WIRE_COLOR_MAP_PHOTOPEAK if measurement_label == "photopeak" else _WIRE_COLOR_MAP_TRACKFIT
    meas_cmap = _CMAP_PHOTOPEAK if measurement_label == "photopeak" else _CMAP_TRACKFIT
    meas_title = "Photopeak method" if measurement_label == "photopeak" else "Track-fit method"

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["measurement_dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # each dataset contributes 2 bars (simulation, measurement)
    max_group_size = max(len(v) for v in grouped.values()) * 2
    group_width = 0.85
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        pair_centers = (np.arange(n) - (n - 1) / 2) * (2 * bar_width)
        for e, pc in zip(group_entries, pair_centers):
            color_sim = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_SIM, _CMAP_SIM)
            color_meas = _wire_color(e["u_wire"], meas_color_map, meas_cmap)

            x_sim = x0 + pc - bar_width / 2
            x_meas = x0 + pc + bar_width / 2

            ax.bar(x_sim, e["vd_sim"], width=bar_width * 0.95, color=color_sim)
            ax.errorbar(x_sim, e["vd_sim"], yerr=e["err_vd_sim"],
                        fmt="none", ecolor="black", capsize=2)

            ax.bar(x_meas, e["vd_measurement"], width=bar_width * 0.95, color=color_meas)
            ax.errorbar(x_meas, e["vd_measurement"], yerr=e["err_vd_measurement"],
                        fmt="none", ecolor="black", capsize=2)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title(f"Drift velocity comparison: Simulation vs. {meas_title}")
    ax.grid(True, axis="y")

    y_lo = min(min(e["vd_sim"] - e["err_vd_sim"],
                    e["vd_measurement"] - e["err_vd_measurement"]) for e in entries)
    y_hi = max(max(e["vd_sim"] + e["err_vd_sim"],
                    e["vd_measurement"] + e["err_vd_measurement"]) for e in entries)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles, legend_labels = [], []
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_SIM, _CMAP_SIM)))
        legend_labels.append(f"Simulation, $U_{{wire}}$={u} V")
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, meas_color_map, meas_cmap)))
        legend_labels.append(f"{meas_title}, $U_{{wire}}$={u} V")
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_comparison_bars_sim_vs_{measurement_label}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")

    return fig, ax, save_path



def plot_vd_comparison_bars_sim_pp_vs_tf(
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
    Bar-chart comparison of the simulation's own two drift-velocity
    estimates -- v_drift_pp (photopeak-style) vs. v_drift_tf (track-fit-
    style) -- grouped by gas mixture, purely within the simulation
    (no measurement data involved). Both bars stay in the purple family to
    read as "simulation", shaded toward RdPu (pp) / BuPu (tf) so the two
    are distinguishable without checking the legend.

    Parameters
    ----------
    entries : list[dict]
        Output of build_sim_pp_vs_tf_entries().
    base_path : str
        Used to build the default output path.
    save_path : str, optional
        Full output path. Defaults to
        f"{base_path}plots/compare/vd_comparison_bars_sim_pp_vs_tf{plot_type}".
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

    # each dataset contributes 2 bars (pp, tf)
    max_group_size = max(len(v) for v in grouped.values()) * 2
    group_width = 0.85
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        pair_centers = (np.arange(n) - (n - 1) / 2) * (2 * bar_width)
        for e, pc in zip(group_entries, pair_centers):
            color_pp = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_SIM_PP, _CMAP_SIM_PP)
            color_tf = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_SIM_TF, _CMAP_SIM_TF)

            x_pp = x0 + pc - bar_width / 2
            x_tf = x0 + pc + bar_width / 2

            ax.bar(x_pp, e["vd_pp"], width=bar_width * 0.95, color=color_pp)
            ax.errorbar(x_pp, e["vd_pp"], yerr=e["err_vd_pp"],
                        fmt="none", ecolor="black", capsize=2)

            ax.bar(x_tf, e["vd_tf"], width=bar_width * 0.95, color=color_tf)
            ax.errorbar(x_tf, e["vd_tf"], yerr=e["err_vd_tf"],
                        fmt="none", ecolor="black", capsize=2)

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_d$ [$\mu$m/ns]")
    ax.set_title("Simulation drift velocity: photopeak-style vs. track-fit-style estimate")
    ax.grid(True, axis="y")

    y_lo = min(min(e["vd_pp"] - e["err_vd_pp"],
                    e["vd_tf"] - e["err_vd_tf"]) for e in entries)
    y_hi = max(max(e["vd_pp"] + e["err_vd_pp"],
                    e["vd_tf"] + e["err_vd_tf"]) for e in entries)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles, legend_labels = [], []
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_SIM_PP, _CMAP_SIM_PP)))
        legend_labels.append(f"Sim (photopeak-style), $U_{{wire}}$={u} V")
    for u in unique_u_wires:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                               color=_wire_color(u, _WIRE_COLOR_MAP_SIM_TF, _CMAP_SIM_TF)))
        legend_labels.append(f"Sim (track-fit-style), $U_{{wire}}$={u} V")
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_comparison_bars_sim_pp_vs_tf{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")
    plt.close("all")

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


def plot_method_ratio_by_gas_mix(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    y_half_range=0.15,
    verbose=True,
    ):
    """
    Ratio analogue of plot_method_difference_by_gas_mix(): for each
    dataset, ratio = vd_photopeak / vd_trackfit, with err_ratio
    error-propagated (see _ratio_and_err()), plotted as a bar per
    dataset, grouped by gas mixture and colored by U_wire (photopeak
    color map, same convention as the difference plot). A horizontal
    line at 1 marks perfect agreement between the two methods. A second
    horizontal line (+shaded band) shows a single constant fitted to all
    the ratio bars via inverse-variance-weighted least squares (see
    fit_constant()) -- this is "the factor" between the two methods,
    with its own uncertainty and a chi2/ndof to judge how well a single
    constant actually describes the scatter of bars.

    Parameters
    ----------
    y_half_range : float, default 0.15
        The y-axis is fixed to [1 - y_half_range, 1 + y_half_range] (NOT
        auto-scaled to the data), so the "distance from 1" is always
        visually comparable across figures/reruns. Lower it to zoom in
        further on datasets that cluster tightly around 1; raise it if
        any bar or its error bar would otherwise fall outside the range
        (a warning is printed in that case, since matplotlib would
        silently clip it).

    Returns
    -------
    fig, ax, path, const, err_const, chi2, ndof
    """
    plot_entries = [e for e in entries if np.isfinite(e["ratio"])]
    if not plot_entries:
        raise ValueError("No entries with a finite ratio to plot.")

    mixes = sorted(
        set(e["mix"] for e in plot_entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in plot_entries:
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
            ax.bar(x0 + offset, e["ratio"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["ratio"], yerr=e["err_ratio"],
                        fmt="none", ecolor="black", capsize=3)

    ax.axhline(y=1, color="gray", linewidth=1.2, linestyle="--",
               label="perfect agreement")

    const, err_const, chi2, ndof = fit_constant(
        values=[e["ratio"] for e in plot_entries],
        errors=[e["err_ratio"] for e in plot_entries],
    )
    fit_line, fit_label = _draw_constant_fit(ax=ax, const=const, err_const=err_const,
                                              chi2=chi2, ndof=ndof, color="black")
    if verbose:
        print(f"  constant fit to photopeak/track-fit ratio: "
              f"{const:.4f} +/- {err_const:.4f} (chi2/ndof = {chi2:.2f}/{ndof})")

    y_lo, y_hi = 1 - y_half_range, 1 + y_half_range
    ax.set_ylim(y_lo, y_hi)
    if verbose:
        clipped = [e["dataset"] for e in plot_entries
                   if e["ratio"] - e["err_ratio"] < y_lo or e["ratio"] + e["err_ratio"] > y_hi]
        if clipped:
            print(f"  warning: y_half_range={y_half_range} clips {len(clipped)} bar(s)/error "
                  f"bar(s) out of view: {clipped}")

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_{d,\mathrm{photopeak}} \, / \, v_{d,\mathrm{track\!-\!fit}}$")
    ax.set_title("Method ratio (photopeak $/$ track-fit)")
    ax.grid(True, axis="y")

    unique_u_wires = sorted(set(e["u_wire"] for e in plot_entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                       color=_wire_color(u, _WIRE_COLOR_MAP_PHOTOPEAK, _CMAP_PHOTOPEAK))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    legend_handles += [plt.Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--"), fit_line]
    legend_labels += ["perfect agreement", fit_label]
    ax.legend(legend_handles, legend_labels, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_method_ratio{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path, const, err_const, chi2, ndof

# pull dist
def plot_pull_distribution(
    *,
    entries,
    base_path,
    plot_type=".png",
    fig_size=(8, 6),
    save_path=None,
    n_bins=15,
    xlabel=None,
    title=None,
    verbose=True,
    ):
    """
    Histogram of pull = diff / err_diff across all datasets, with a
    standard-normal N(0,1) curve overlaid for reference. If the compared
    quantities' uncertainties are correctly estimated and there's no
    systematic offset between them, this should scatter around a unit
    Gaussian centered at 0. Works on the output of any of this file's
    build_*_entries() functions, as long as each entry has a "pull" key
    (all of them do).

    Parameters
    ----------
    xlabel, title : str, optional
        Override the default photopeak-vs-track-fit axis label/title --
        pass these when plotting pulls from a different comparison (e.g.
        simulation vs. measurement) so the text matches what's actually
        being compared.

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

    if xlabel is None:
        xlabel = (r"pull $= \dfrac{v_{d,\mathrm{photopeak}} - v_{d,\mathrm{track\!-\!fit}}}"
                  r"{\sqrt{\sigma_{\mathrm{pp}}^2+\sigma_{\mathrm{tf}}^2}}$")
    if title is None:
        title = "Pull distribution: photopeak vs. track-fit method"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
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


def plot_vd_difference_sim_vs_measurement(
    *,
    entries,
    base_path,
    measurement_label,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    verbose=True,
    ):
    """
    Simulation-vs-measurement analogue of plot_method_difference_by_gas_mix():
    for each matched dataset, diff = vd_sim - vd_measurement, with
    err_diff = sqrt(err_sim^2 + err_measurement^2), plotted as a bar per
    dataset, grouped by gas mixture and colored by U_wire using the
    simulation's own purple color map (this difference is attributed to
    "the simulation" -- it's what the simulation over/under-predicts
    relative to the given measurement method). A horizontal line at 0
    marks perfect agreement.

    Parameters
    ----------
    entries : list[dict]
        Output of build_sim_vs_measurement_vd_entries().
    measurement_label : str
        "photopeak" or "trackfit" -- only used for the title/filename.

    Returns
    -------
    fig, ax, path
    """
    if not entries:
        raise ValueError("No entries to plot.")
    if measurement_label not in ("photopeak", "trackfit"):
        raise ValueError(f"measurement_label must be 'photopeak' or 'trackfit', got {measurement_label!r}")

    meas_title = "Photopeak method" if measurement_label == "photopeak" else "Track-fit method"

    mixes = sorted(
        set(e["mix"] for e in entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["measurement_dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    max_group_size = max(len(v) for v in grouped.values())
    group_width = 0.8
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_SIM, _CMAP_SIM)
            ax.bar(x0 + offset, e["diff"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["diff"], yerr=e["err_diff"],
                        fmt="none", ecolor="black", capsize=3)

    ax.axhline(y=0, color="gray", linewidth=1.2, linestyle="--",
               label="perfect agreement")

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_{d,\mathrm{sim}} - v_{d,\mathrm{meas}}$ [$\mu$m/ns]")
    ax.set_title(f"Simulation $-$ Experiment difference ({meas_title})")
    ax.grid(True, axis="y")

    unique_u_wires = sorted(set(e["u_wire"] for e in entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                       color=_wire_color(u, _WIRE_COLOR_MAP_SIM, _CMAP_SIM))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    ax.legend(legend_handles, legend_labels, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_difference_sim_vs_{measurement_label}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path


def plot_vd_ratio_sim_vs_measurement(
    *,
    entries,
    base_path,
    measurement_label,
    plot_type=".png",
    fig_size=(14, 6),
    save_path=None,
    y_half_range=0.15,
    verbose=True,
    ):
    """
    Ratio analogue of plot_vd_difference_sim_vs_measurement(): for each
    matched dataset, ratio = vd_sim / vd_measurement, with err_ratio
    error-propagated (see _ratio_and_err()), plotted as a bar per
    dataset, grouped by gas mixture and colored by U_wire using the
    simulation's own purple color map. A horizontal line at 1 marks
    perfect agreement -- values above 1 mean the simulation over-predicts
    the drift velocity relative to the given measurement method, values
    below 1 mean it under-predicts. A second horizontal line (+shaded
    band) shows a single constant fitted to all the ratio bars via
    inverse-variance-weighted least squares (see fit_constant()) -- the
    overall sim/measurement scale factor, with its own uncertainty and a
    chi2/ndof to judge how well one constant describes the bars.

    Parameters
    ----------
    entries : list[dict]
        Output of build_sim_vs_measurement_vd_entries().
    measurement_label : str
        "photopeak" or "trackfit" -- only used for the title/filename.
    y_half_range : float, default 0.15
        The y-axis is fixed to [1 - y_half_range, 1 + y_half_range] (NOT
        auto-scaled to the data). The simulation-vs-measurement ratios
        tend to sit further from 1 than the two measurement methods do
        against each other (see the console log's diff/pull columns), so
        you may want a larger value here than for
        plot_method_ratio_by_gas_mix() -- a warning is printed if any
        bar or its error bar falls outside the chosen range.

    Returns
    -------
    fig, ax, path, const, err_const, chi2, ndof
    """
    plot_entries = [e for e in entries if np.isfinite(e["ratio"])]
    if not plot_entries:
        raise ValueError("No entries with a finite ratio to plot.")
    if measurement_label not in ("photopeak", "trackfit"):
        raise ValueError(f"measurement_label must be 'photopeak' or 'trackfit', got {measurement_label!r}")

    meas_title = "Photopeak method" if measurement_label == "photopeak" else "Track-fit method"

    mixes = sorted(
        set(e["mix"] for e in plot_entries),
        key=lambda m: tuple(int(v) for v in m.split("/")),
    )
    mix_to_x = {mix: i for i, mix in enumerate(mixes)}

    grouped = {mix: [] for mix in mixes}
    for e in plot_entries:
        grouped[e["mix"]].append(e)
    for mix in grouped:
        grouped[mix].sort(key=lambda e: (e["u_wire"], e["measurement_dataset"]))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    max_group_size = max(len(v) for v in grouped.values())
    group_width = 0.8
    bar_width = group_width / max_group_size

    for mix, group_entries in grouped.items():
        x0 = mix_to_x[mix]
        n = len(group_entries)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
        for e, offset in zip(group_entries, offsets):
            color = _wire_color(e["u_wire"], _WIRE_COLOR_MAP_SIM, _CMAP_SIM)
            ax.bar(x0 + offset, e["ratio"], width=bar_width * 0.9, color=color)
            ax.errorbar(x0 + offset, e["ratio"], yerr=e["err_ratio"],
                        fmt="none", ecolor="black", capsize=3)

    ax.axhline(y=1, color="gray", linewidth=1.2, linestyle="--",
               label="perfect agreement")

    const, err_const, chi2, ndof = fit_constant(
        values=[e["ratio"] for e in plot_entries],
        errors=[e["err_ratio"] for e in plot_entries],
    )
    fit_line, fit_label = _draw_constant_fit(ax=ax, const=const, err_const=err_const,
                                              chi2=chi2, ndof=ndof, color="black")
    if verbose:
        print(f"  constant fit to sim/{measurement_label} ratio: "
              f"{const:.4f} +/- {err_const:.4f} (chi2/ndof = {chi2:.2f}/{ndof})")

    y_lo, y_hi = 1 - y_half_range, 1 + y_half_range
    ax.set_ylim(y_lo, y_hi)
    if verbose:
        clipped = [e["measurement_dataset"] for e in plot_entries
                   if e["ratio"] - e["err_ratio"] < y_lo or e["ratio"] + e["err_ratio"] > y_hi]
        if clipped:
            print(f"  warning: y_half_range={y_half_range} clips {len(clipped)} bar(s)/error "
                  f"bar(s) out of view: {clipped}")

    ax.set_xticks(list(mix_to_x.values()))
    ax.set_xticklabels(list(mix_to_x.keys()))
    ax.set_xlabel("Gas mixture (Ar/CO2) [%]")
    ax.set_ylabel(r"$v_{d,\mathrm{sim}} \, / \, v_{d,\mathrm{meas}}$")
    ax.set_title(f"Simulation $/$ Experiment ratio ({meas_title})")
    ax.grid(True, axis="y")

    unique_u_wires = sorted(set(e["u_wire"] for e in plot_entries))
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                       color=_wire_color(u, _WIRE_COLOR_MAP_SIM, _CMAP_SIM))
                       for u in unique_u_wires]
    legend_labels = [f"$U_{{wire}}$ = {u} V" for u in unique_u_wires]
    legend_handles += [plt.Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--"), fit_line]
    legend_labels += ["perfect agreement", fit_label]
    ax.legend(legend_handles, legend_labels, fontsize=9,
              fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/compare/vd_ratio_sim_vs_{measurement_label}{plot_type}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    plt.close("all")
    return fig, ax, save_path, const, err_const, chi2, ndof


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


def make_sim_summary_tex_table(*, analysis_out_sim, sim_info_fn=parse_sim_name, float_precision=3):
    """
    LaTeX table summarizing the simulation's own results, per dataset:
    gas mixture, wire voltage, and both drift-velocity estimates
    (v_drift_pp = photopeak-style, v_drift_tf = track-fit-style).

    Returns
    -------
    tex_table : str
    """
    fp = float_precision
    rows = []
    for name, result in analysis_out_sim.items():
        try:
            info = sim_info_fn(name=name)
        except Exception:
            continue
        if "v_drift_pp" not in result or "v_drift_tf" not in result:
            continue
        rows.append({
            "dataset": name,
            "mix": f"{info['pct_Ar']}/{info['pct_CO2']}",
            "u_wire": int(info["U_wire"]),
            "vd_pp": result["v_drift_pp"],
            "err_vd_pp": result.get("v_drift_pp_err", float("nan")),
            "vd_tf": result["v_drift_tf"],
            "err_vd_tf": result.get("v_drift_tf_err", float("nan")),
        })

    if not rows:
        raise ValueError("No simulation datasets with 'v_drift_pp'/'v_drift_tf' to tabulate.")

    lines = [
        r"\begin{tabular}{|l|c|c|c|c|}",
        r"    \hline",
        r"    Dataset & Mix & $U_{\mathrm{wire}}$ [V] & $v_{d,\mathrm{pp}}$ [$\mu$m/ns] "
        r"& $v_{d,\mathrm{tf}}$ [$\mu$m/ns] \\ \hline",
    ]
    for r in sorted(rows, key=lambda r: (r["mix"], r["u_wire"])):
        lines.append(
            f"    {r['dataset'].replace('_', r'\\_')} & {r['mix']} & {r['u_wire']} & "
            f"${np.round(r['vd_pp'], fp):.{fp}f} \\pm {np.round(r['err_vd_pp'], fp):.{fp}f}$ & "
            f"${np.round(r['vd_tf'], fp):.{fp}f} \\pm {np.round(r['err_vd_tf'], fp):.{fp}f}$ \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_sim_pp_vs_tf_tex_table(*, entries, float_precision=3):
    """
    LaTeX table listing, per simulation dataset, the photopeak-style vs.
    track-fit-style drift velocities, the difference, and the pull.
    `entries` is the output of build_sim_pp_vs_tf_entries().

    Returns
    -------
    tex_table : str
    """
    if not entries:
        raise ValueError("No entries to tabulate.")

    fp = float_precision
    lines = [
        r"\begin{tabular}{|l|c|c|c|c|c|c|}",
        r"    \hline",
        r"    Dataset & Mix & $U_{\mathrm{wire}}$ [V] & $v_{d,\mathrm{pp}}$ [$\mu$m/ns] "
        r"& $v_{d,\mathrm{tf}}$ [$\mu$m/ns] & diff [$\mu$m/ns] & pull \\ \hline",
    ]
    for e in sorted(entries, key=lambda e: (e["mix"], e["u_wire"])):
        lines.append(
            f"    {e['dataset'].replace('_', r'\\_')} & {e['mix']} & {e['u_wire']} & "
            f"${np.round(e['vd_pp'], fp):.{fp}f} \\pm {np.round(e['err_vd_pp'], fp):.{fp}f}$ & "
            f"${np.round(e['vd_tf'], fp):.{fp}f} \\pm {np.round(e['err_vd_tf'], fp):.{fp}f}$ & "
            f"${np.round(e['diff'], fp):.{fp}f}$ & "
            f"${np.round(e['pull'], 2):.2f}$ \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_sim_vs_measurement_tex_table(*, entries, measurement_label, float_precision=3):
    """
    LaTeX table listing, per matched (simulation, measurement) dataset
    pair, both drift velocities, the difference, and the pull. `entries`
    is the output of build_sim_vs_measurement_vd_entries();
    `measurement_label` ("photopeak" or "trackfit") only affects the
    column header wording.

    Returns
    -------
    tex_table : str
    """
    if not entries:
        raise ValueError("No entries to tabulate.")
    if measurement_label not in ("photopeak", "trackfit"):
        raise ValueError(f"measurement_label must be 'photopeak' or 'trackfit', got {measurement_label!r}")

    fp = float_precision
    meas_word = "Photopeak" if measurement_label == "photopeak" else "Track-fit"
    lines = [
        r"\begin{tabular}{|l|l|c|c|c|c|c|c|}",
        r"    \hline",
        rf"    Sim dataset & {meas_word} dataset & Mix & $U_{{\mathrm{{wire}}}}$ [V] & "
        r"$v_{d,\mathrm{sim}}$ [$\mu$m/ns] & $v_{d,\mathrm{meas}}$ [$\mu$m/ns] & "
        r"diff [$\mu$m/ns] & pull \\ \hline",
    ]
    for e in sorted(entries, key=lambda e: (e["mix"], e["u_wire"])):
        lines.append(
            f"    {e['sim_dataset'].replace('_', r'\\_')} & "
            f"{e['measurement_dataset'].replace('_', r'\\_')} & "
            f"{e['mix']} & {e['u_wire']} & "
            f"${np.round(e['vd_sim'], fp):.{fp}f} \\pm {np.round(e['err_vd_sim'], fp):.{fp}f}$ & "
            f"${np.round(e['vd_measurement'], fp):.{fp}f} \\pm {np.round(e['err_vd_measurement'], fp):.{fp}f}$ & "
            f"${np.round(e['diff'], fp):.{fp}f}$ & "
            f"${np.round(e['pull'], 2):.2f}$ \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_tex_table(*, tex_table, path, verbose=True):
    """Write `tex_table` to its own file at `path` -- a small shared
    helper so every LaTeX table in this file (comparison, sim summary,
    sim-vs-measurement, ...) is saved the same way, each to a distinct
    file rather than being appended together."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tex_table)
    if verbose:
        print(f"store tex table as {path}.")
    return path



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

    analysis_out_sim = data_utils.load_pickle(
        f"{pcls_file_path}analysis_out_simulation.pcl"
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
        plot_method_ratio_by_gas_mix(
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

    # ---- simulation vs. measurement drift-velocity comparison (matched by
    # gas mix + U_wire, not dataset name -- see build_sim_vs_measurement_vd_entries) ----
    sim_vs_measurement_specs = [
        ("photopeak", "v_drift_pp", analysis_out_photopeak, get_vd_photopeak),
        ("trackfit",  "v_drift_tf", analysis_out_track_fit, get_vd_trackfit),
    ]
    for measurement_label, sim_vd_key, analysis_out_meas, measurement_getter in sim_vs_measurement_specs:
        sim_vs_meas_entries = build_sim_vs_measurement_vd_entries(
            analysis_out_sim=analysis_out_sim,
            analysis_out_measurement=analysis_out_meas,
            sim_vd_key=sim_vd_key,
            measurement_getter=measurement_getter,
            sim_info_fn=parse_sim_name,
            measurement_info_fn=parse_fit_name,
        )
        if not sim_vs_meas_entries:
            print(f"No datasets matched between simulation and {measurement_label} "
                  "measurement (by gas mix + U_wire); skipping.")
            continue
        plot_vd_comparison_bars_sim_vs_measurement(
            entries=sim_vs_meas_entries, base_path=base_path,
            measurement_label=measurement_label, plot_type=plot_type,
        )
        plot_vd_difference_sim_vs_measurement(
            entries=sim_vs_meas_entries, base_path=base_path,
            measurement_label=measurement_label, plot_type=plot_type,
        )
        plot_vd_ratio_sim_vs_measurement(
            entries=sim_vs_meas_entries, base_path=base_path,
            measurement_label=measurement_label, plot_type=plot_type,
        )
        meas_title = "Photopeak method" if measurement_label == "photopeak" else "Track-fit method"
        plot_pull_distribution(
            entries=sim_vs_meas_entries, base_path=base_path, plot_type=plot_type,
            save_path=plot_save_path + f"vd_pull_distribution_sim_vs_{measurement_label}{plot_type}",
            xlabel=(r"pull $= \dfrac{v_{d,\mathrm{sim}} - v_{d,\mathrm{meas}}}"
                    r"{\sqrt{\sigma_{\mathrm{sim}}^2+\sigma_{\mathrm{meas}}^2}}$"),
            title=f"Pull distribution: Simulation vs. {meas_title}",
        )

        sim_vs_meas_tex_table = make_sim_vs_measurement_tex_table(
            entries=sim_vs_meas_entries, measurement_label=measurement_label,
        )
        print(sim_vs_meas_tex_table)
        save_tex_table(
            tex_table=sim_vs_meas_tex_table,
            path=plot_save_path + f"vd_comparison_table_sim_vs_{measurement_label}.tex",
        )

    # ---- simulation-internal comparison: photopeak-style vs. track-fit-
    # style drift velocity, both from analysis_out_sim (no measurement data) ----
    sim_pp_vs_tf_entries = build_sim_pp_vs_tf_entries(
        analysis_out_sim=analysis_out_sim, sim_info_fn=parse_sim_name,
    )
    if sim_pp_vs_tf_entries:
        plot_vd_comparison_bars_sim_pp_vs_tf(
            entries=sim_pp_vs_tf_entries, base_path=base_path, plot_type=plot_type,
        )

        sim_pp_vs_tf_tex_table = make_sim_pp_vs_tf_tex_table(entries=sim_pp_vs_tf_entries)
        print(sim_pp_vs_tf_tex_table)
        save_tex_table(
            tex_table=sim_pp_vs_tf_tex_table,
            path=plot_save_path + "vd_comparison_table_sim_pp_vs_tf.tex",
        )
    else:
        print("No simulation datasets with both 'v_drift_pp' and 'v_drift_tf'; "
              "skipping sim pp-vs-tf comparison plot and table.")

    # ---- simulation summary table (all sim datasets, independent of any
    # measurement matching) ----
    try:
        sim_summary_tex_table = make_sim_summary_tex_table(
            analysis_out_sim=analysis_out_sim, sim_info_fn=parse_sim_name,
        )
        print(sim_summary_tex_table)
        save_tex_table(
            tex_table=sim_summary_tex_table,
            path=plot_save_path + "vd_sim_summary_table.tex",
        )
    except ValueError as e:
        print(f"Skipping simulation summary table: {e}")

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
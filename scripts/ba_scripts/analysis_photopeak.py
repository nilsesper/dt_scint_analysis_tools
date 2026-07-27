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
# (Gaussian + linear background, automatic peak detection,
#  adaptive fit window with retries)
# =================================================================
def gauss_plus_quad(x, A, mu, sigma, a2, a1, a0):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + a2 * x ** 2 + a1 * x + a0


def err_gauss_plus_quad(x, A, mu, sigma, a2, a1, a0, err_A, err_mu, err_sigma, err_a2, err_a1, err_a0):
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    df_dA = gauss
    df_dmu = A * gauss * (x - mu) / sigma ** 2
    df_dsigma = A * gauss * (x - mu) ** 2 / sigma ** 3
    df_da2 = x ** 2
    df_da1 = x
    df_da0 = np.ones_like(x)
    return np.sqrt(
        (df_dA * err_A) ** 2
        + (df_dmu * err_mu) ** 2
        + (df_dsigma * err_sigma) ** 2
        + (df_da2 * err_a2) ** 2
        + (df_da1 * err_a1) ** 2
        + (df_da0 * err_a0) ** 2
    )

def gauss_plus_linear(x, A, mu, sigma, m, b):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + b

def err_gauss_plus_linear(
    x, A, mu, sigma, m, b,
    err_A, err_mu, err_sigma, err_m, err_b,
):
    gauss = np.exp(-0.5*((x-mu)/sigma)**2)

    df_dA = gauss
    df_dmu = A*gauss*(x-mu)/sigma**2
    df_dsigma = A*gauss*(x-mu)**2/sigma**3
    df_dm = x
    df_db = np.ones_like(x)

    return np.sqrt(
        (df_dA*err_A)**2 +
        (df_dmu*err_mu)**2 +
        (df_dsigma*err_sigma)**2 +
        (df_dm*err_m)**2 +
        (df_db*err_b)**2
    )

def find_peak_window_by_valley_height(
    bins, hist, mu_est, valley_x,
    smooth_window=9, tol_bins=0, search_max_x=None,
):
    """
    Bestimmt ein Fit-Fenster [min_x, max_x] um einen Peak, indem ausgehend
    vom Peak nach rechts gelaufen wird, bis der (geglättete) Histogrammwert
    wieder auf das Niveau des zuvor gefundenen Valleys zurückgefallen ist.
    Die linke Grenze ist einfach das Valley selbst. Beide Grenzen werden
    um `tol_bins` nach außen erweitert, um etwas Untergrund-Baseline auf
    beiden Seiten für den quadratischen Term mitzunehmen.

    Motivation: bei sehr hohen, schmalen Peaks auf einer breiten Schulter
    ist ein festes Vielfaches von sigma als Fensterbreite ungeeignet, weil
    peak/valley-Kontrast stark variiert. "Gleiche Höhe wie das Valley" ist
    dagegen ein adaptives, von der tatsächlichen Peakform abgeleitetes
    Kriterium und nutzt aus, dass Peak und Valley in guter Näherung
    symmetrisch um mu liegen.

    Gibt (min_x, max_x, right_idx_found) zurück. right_idx_found=False
    bedeutet, dass innerhalb von search_max_x kein Rückgang auf
    Valley-Niveau gefunden wurde und stattdessen symmetrisch gespiegelt
    wurde (Fallback).
    """
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        y_smooth = np.convolve(hist, kernel, mode="same")
    else:
        y_smooth = hist

    valley_idx = int(np.argmin(np.abs(bins - valley_x)))
    peak_idx = int(np.argmin(np.abs(bins - mu_est)))
    valley_level = y_smooth[valley_idx]

    if peak_idx <= valley_idx:
        raise ValueError("mu_est muss rechts vom Valley liegen.")

    max_idx = len(bins) - 1
    if search_max_x is not None:
        candidates = np.where(bins <= search_max_x)[0]
        if len(candidates):
            max_idx = candidates.max()

    right_idx = None
    for i in range(peak_idx, max_idx + 1):
        if abs(y_smooth[i] - valley_level) / valley_level < 0.05:
            right_idx = i
            break

    found = right_idx is not None
    if not found:
        # Kein Rückgang auf Valley-Niveau gefunden -> symmetrisch zum
        # Valley um den Peak spiegeln als Fallback.
        half_width_idx = peak_idx - valley_idx
        right_idx = min(peak_idx + half_width_idx, max_idx)

    left_idx = max(valley_idx - tol_bins, 0)
    right_idx = min(right_idx + tol_bins, len(bins) - 1)

    return bins[left_idx], bins[right_idx], found

def find_all_valleys_after_main_peak(bins, hist, smooth_window=9, valley_prominence_frac=0.01):
    """Find the main (largest) peak, then ALL local minima ("valleys") to
    its right, sorted by position (closest first). Returns (main_peak_x,
    [valley_x_1, valley_x_2, ...]).

    A single "take the first valley" approach is fragile: real histograms
    can have a spurious dip on the main peak's shoulder (residual structure
    from an imperfect background subtraction, detector artifacts, etc.)
    that isn't the genuine gap before the secondary peak. Returning all
    candidates lets the caller try them in order and validate each fit,
    falling through to the next valley if a candidate produces an
    unphysical result.
    """
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        y_smooth = np.convolve(hist, kernel, mode="same")
    else:
        y_smooth = hist

    main_idx = int(np.argmax(y_smooth))

    tail = y_smooth[main_idx:]
    if len(tail) < 3:
        raise RuntimeError("Not enough bins to the right of the main peak to find a valley.")

    # require valleys to be reasonably separated so we don't pick up
    # adjacent noise fluctuations as distinct candidates
    min_distance = max(3, len(tail) // 100)
    valley_candidates, props = find_peaks(
        -tail, prominence=valley_prominence_frac * np.amax(y_smooth), distance=min_distance,
    )

    if len(valley_candidates) == 0:
        # fallback: no clear valley found, just start searching a bit past the main peak
        valley_idxs = [main_idx + max(1, len(tail) // 10)]
    else:
        valley_idxs = [main_idx + i for i in valley_candidates]

    return bins[main_idx], [bins[i] for i in valley_idxs]

def find_and_fit_secondary_peak(
    bins, hist, err_hist,
    bins_nobg, hist_nobg, err_hist_nobg,
    smooth_window_valley=9, valley_prominence_frac=0.01,
    smooth_window_peak=5, prominence_frac=0.03,
    tol_bins=0,
    window_sigmas=5, max_attempts=6, window_growth=1.3, max_sigma=None,
    min_snr=5.0, max_chi2ndf=None,
):
    """Full pipeline: find main peak -> candidate valleys -> for each
    valley, locate the secondary peak, derive an adaptive (valley-height
    symmetric) fit window, refine the peak estimate inside that window,
    and attempt the Gaussian+quad fit. Falls through to the next valley
    candidate if a fit fails, since some valleys may be spurious shoulder
    dips rather than the real gap before the secondary peak."""

    main_peak_x, valley_candidates = find_all_valleys_after_main_peak(
        bins, hist, smooth_window=smooth_window_valley,
        valley_prominence_frac=valley_prominence_frac,
    )

    last_exc = None
    for valley_x in valley_candidates:
        try:
            # 1) grobe Peak-Suche rechts vom Valley bis zum Ende, um ein
            #    erstes mu_est/sigma_est/A_est zu bekommen
            mu_est, sigma_est, A_est = find_secondary_peak(
                bins, hist, valley_x, bins.max(),
                prominence_frac=prominence_frac,
                smooth_window=smooth_window_peak,
            )

            # 2) adaptives Fenster: Valley links, rechts der Punkt, an dem
            #    das Histogramm wieder auf Valley-Niveau zurückfällt
            min_x, max_x, found_right = find_peak_window_by_valley_height(
                bins, hist, mu_est, valley_x,
                smooth_window=smooth_window_valley, tol_bins=tol_bins,
            )

            # 3) Peak-Suche im engeren Fenster wiederholen für sauberere
            #    Startwerte (weniger Kontamination durch die Schulter)
            mu_est, sigma_est, A_est = find_secondary_peak(
                bins, hist, min_x, max_x,
                prominence_frac=prominence_frac,
                smooth_window=smooth_window_peak,
            )

            # 4) eigentlicher Fit, startend mit dem expliziten Fenster
            popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func = _fit_gaussian_window(
                bins_nobg, hist_nobg, err_hist_nobg,
                mu_est, sigma_est, A_est,
                window_sigmas=window_sigmas, max_attempts=max_attempts,
                window_growth=window_growth, max_sigma=max_sigma,
                min_snr=min_snr, max_chi2ndf=max_chi2ndf,
                explicit_window=(min_x, max_x),
            )

            return popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func, valley_x

        except Exception as exc:  # noqa: BLE001 -- try next valley candidate
            last_exc = exc
            print(f"    valley candidate x={valley_x:.2f} failed: {exc}")
            continue

    raise RuntimeError(
        f"No valley candidate produced a valid fit (tried {len(valley_candidates)} candidates). "
        f"Last error: {last_exc}"
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

def _fit_gaussian_window(
    bins_nobg, hist_nobg, err_hist_nobg,
    mu_est, sigma_est, A_est,
    window_sigmas, max_attempts, window_growth, max_sigma,
    min_snr=5.0, max_chi2ndf=None,
    explicit_window=None,
):
    """Try to fit a Gaussian+quadratic-background window around a single
    (mu_est, sigma_est, A_est) seed, widening the window on failure.
    Rejects fits whose sigma exceeds max_sigma (a physical upper bound on
    how wide the *real* peak is allowed to be -- catches cases where the
    fit locks onto a broad shoulder feature instead of the genuine,
    narrower peak), or whose amplitude SNR is too low.

    If `explicit_window=(min_x, max_x)` is given, the FIRST attempt uses
    that window as-is (e.g. a valley-symmetric window computed upstream)
    instead of mu_est +/- window_sigmas * sigma_est. If that attempt fails
    for any reason, subsequent attempts fall back to the normal
    sigma-based window and widen it with window_growth as before, so no
    robustness is lost relative to the original behavior.

    Note: chi2/ndf is NOT a reliable accept/reject criterion here. With
    high-statistics histograms (small per-bin errors), even a correct fit
    of the real peak can have chi2/ndf >> 1 simply because a quadratic
    background can't perfectly capture the true continuum shape -- while
    an incorrect fit to a flatter, featureless region can look artificially
    "good" because there's nothing there to fit badly. chi2/ndf is reported
    for diagnostics and only used as a hard cutoff if you explicitly set
    max_chi2ndf to a number."""
    win = window_sigmas
    last_exc = None

    for attempt in range(max_attempts):
        if explicit_window is not None:
            lo, hi = explicit_window
        else:
            lo, hi = mu_est - win*sigma_est, mu_est + win*sigma_est

        fit_mask = (bins_nobg >= lo) & (bins_nobg <= hi)
        fit_bins = bins_nobg[fit_mask]
        fit_hist = hist_nobg[fit_mask]
        err_fit_hist = err_hist_nobg[fit_mask]

        if len(fit_bins) < 10:
            win *= window_growth
            last_exc = RuntimeError("fit window contains too few bins")
            continue

        n_edge = max(3, len(fit_bins) // 8)
        edge_x = np.concatenate([fit_bins[:n_edge], fit_bins[-n_edge:]])
        edge_y = np.concatenate([fit_hist[:n_edge], fit_hist[-n_edge:]])
        try:
            m_est, b_est = np.polyfit(edge_x, edge_y, 1)
        except np.linalg.LinAlgError:
            a2_est, a1_est, a0_est = 0.0, 0.0, np.median(fit_hist)

        p0 = ( A_est,mu_est,sigma_est,m_est,b_est,)
        lower = [
            0,
            fit_bins.min(),
            1e-6,
            -np.inf,
            -np.inf,
        ]

        upper = [
            np.inf,
            fit_bins.max(),
            fit_bins.max()-fit_bins.min(),
            np.inf,
            np.inf,
        ]
        try:
            popt, pcov = curve_fit(
                gauss_plus_linear, fit_bins, fit_hist,
                p0=p0, sigma=err_fit_hist, absolute_sigma=True,
                bounds=(lower, upper), maxfev=20000,
            )

            A_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
            err_A_fit = np.sqrt(pcov[0][0])
            err_sigma_fit = np.sqrt(pcov[2][2])

            fit_vals = gauss_plus_linear(fit_bins, *popt)
            chi2 = np.sum((fit_hist - fit_vals) ** 2 / err_fit_hist ** 2)
            ndf = len(fit_hist) - len(popt)
            chi2ndf = chi2 / ndf if ndf > 0 else np.inf

            if not (fit_bins.min() < mu_fit < fit_bins.max()):
                raise RuntimeError(f"fitted mu={mu_fit:.2f} left the fit window")
            if not (0 < sigma_fit < (fit_bins.max() - fit_bins.min())):
                raise RuntimeError(f"unreasonable sigma={sigma_fit:.2f}")
            if A_fit <= 0:
                raise RuntimeError("fitted amplitude <= 0")
            if max_sigma is not None and sigma_fit > max_sigma:
                raise RuntimeError(f"sigma={sigma_fit:.2f} exceeds max_sigma={max_sigma} "
                                    "(likely locked onto a broad shoulder feature, not the real peak)")
            if err_sigma_fit > sigma_fit:
                raise RuntimeError(f"sigma error ({err_sigma_fit:.2f}) exceeds sigma itself "
                                    f"({sigma_fit:.2f}) -- fit is not well constrained")
            if err_A_fit <= 0 or A_fit / err_A_fit < min_snr:
                raise RuntimeError(f"amplitude SNR too low (A={A_fit:.1f} ± {err_A_fit:.1f}) "
                                    "-- likely a statistical fluctuation, not a real peak")
            if max_chi2ndf is not None and chi2ndf > max_chi2ndf:
                raise RuntimeError(f"chi2/ndf={chi2ndf:.2f} exceeds max_chi2ndf={max_chi2ndf} "
                                    "-- poor fit quality, likely wrong feature")

            print(f"    (diagnostic) chi2/ndf = {chi2:.1f}/{ndf} = {chi2ndf:.2f}, "
                  f"SNR = {A_fit/err_A_fit:.1f}")

            return popt, pcov, fit_bins, fit_hist, err_fit_hist, gauss_plus_linear

        except Exception as exc:  # noqa: BLE001 -- intentionally broad, we retry
            last_exc = exc
            win *= window_growth

    raise RuntimeError(f"did not converge after {max_attempts} attempts. Last error: {last_exc}")



def fit_secondary_peak(
    bins_nobg,
    hist_nobg,
    err_hist_nobg,
    search_min=None,
    search_max=None,
    window_sigmas=3.0,
    prominence_frac=0.03,
    smooth_window=5,
    max_attempts=4,
    window_growth=1.4,
    max_sigma=60.0,
    min_snr=3.0,
    max_chi2ndf=None,
    valley_prominence_frac=0.01,
    smooth_window_valley=9,
    tol_bins=0,
    verbose=True,
):
    """Automatically find and robustly fit the secondary peak with a
    Gaussian + quadratic background model.

    For each valley candidate, an adaptive fit window is derived via
    find_peak_window_by_valley_height (valley on the left, the point
    where the histogram falls back to valley height on the right), which
    is far more robust for very high/narrow peaks on a broad shoulder
    than a fixed multiple of sigma. This window seeds an explicit-window
    attempt in _fit_gaussian_window, which falls back to the normal
    sigma-based widening if that first attempt fails.

    If search_min is None, ALL valleys right of the main peak are found
    and tried in order (closest first); if search_min is given, it is
    used directly as the "valley" boundary for a single-candidate fit
    (useful when you already know roughly where the peak's left edge is).

    max_sigma should be set to a physically reasonable upper bound on the
    real peak's width for your detector (default 60 ns).
    max_chi2ndf is off by default (see _fit_gaussian_window docstring for
    why chi2/ndf is unreliable here) -- set it to a number to re-enable
    it as a hard cutoff.
    """
    if search_max is None:
        search_max = bins_nobg.max()

    def _fit_one_valley(valley_x):
        # 1) grobe Peak-Suche vom Valley bis search_max, für erstes
        #    mu_est/sigma_est/A_est
        mu_est, sigma_est, A_est = find_secondary_peak(
            bins_nobg, hist_nobg,
            min_x=valley_x, max_x=search_max,
            prominence_frac=prominence_frac, smooth_window=smooth_window,
        )

        # 2) adaptives Fenster: Valley links, rechts der Punkt, an dem
        #    das Histogramm wieder auf Valley-Niveau zurückfällt
        min_x, max_x, found_right = find_peak_window_by_valley_height(
            bins_nobg, hist_nobg, mu_est, valley_x,
            smooth_window=smooth_window_valley, tol_bins=tol_bins,
            search_max_x=search_max,
        )

        # 3) Peak-Suche im engeren Fenster wiederholen für sauberere
        #    Startwerte (weniger Kontamination durch die Schulter)
        mu_est, sigma_est, A_est = find_secondary_peak(
            bins_nobg, hist_nobg, min_x=min_x, max_x=max_x,
            prominence_frac=prominence_frac, smooth_window=smooth_window,
        )

        # 4) eigentlicher Fit, startend mit dem expliziten Fenster
        return _fit_gaussian_window(
            bins_nobg, hist_nobg, err_hist_nobg, mu_est, sigma_est, A_est,
            window_sigmas, max_attempts, window_growth, max_sigma,
            min_snr=min_snr, max_chi2ndf=max_chi2ndf,
            explicit_window=(min_x, max_x),
        )

    if search_min is not None:
        # explizite Region gegeben -- als einzelner Valley-Kandidat behandeln
        return _fit_one_valley(search_min)

    # automatic mode: try every valley candidate in order
    main_peak_x, valley_candidates = find_all_valleys_after_main_peak(
        bins_nobg, hist_nobg, smooth_window=smooth_window_valley,
        valley_prominence_frac=valley_prominence_frac,
    )

    last_exc = None
    for i, valley_x in enumerate(valley_candidates):
        if verbose:
            print(f"  [candidate {i+1}/{len(valley_candidates)}] main peak at {main_peak_x:.1f} ns, "
                  f"trying valley at {valley_x:.1f} ns -> searching in ({valley_x:.1f}, {search_max:.1f}) ns")
        try:
            result = _fit_one_valley(valley_x)
            if verbose:
                print(f"  [candidate {i+1}] accepted: mu={result[0][1]:.1f} ns, sigma={result[0][2]:.1f} ns")
            return result
        except Exception as exc:  # noqa: BLE001 -- try next valley candidate
            if verbose:
                print(f"  [candidate {i+1}] rejected: {exc}")
            last_exc = exc
            continue

    raise RuntimeError(
        f"No valley candidate produced a valid secondary-peak fit "
        f"(tried {len(valley_candidates)} candidates). Last error: {last_exc}\n"
        f"Try lowering max_sigma / valley_prominence_frac, or pass an explicit search_min."
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

# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    ###################################################

    list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut100", 
                "cosmic_82-18_3575-1800-1200_run1_th20_cut100", "cosmic_82-18_3600-1800-1200_run1_th20_cut100", 
                "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 
                "cosmic_85-15_3550-1800-1200_run1_th20_cut100", "cosmic_85-15_3575-1800-1200_run1_th20_cut100", 
                "cosmic_85-15_3600-1800-1200_run2_th20_cut100"]
    list_of_fits = ["cosmic_82-18_3600-1800-1200_run1_th20_cut100", 
                    "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 
                    "cosmic_85-15_3600-1800-1200_run2_th20_cut100"]


    ramp_datasets = [
        "data_mic0_start_2026-07-24_18-06-10_stop_2026-07-24_18-16-11",
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
        ]
    do_ramp_measurement = True
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
        print("Plotting full t_diff hist...")
        fig_size = (8, 6)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
        ax.set_xlim(0,np.amax(bins))
        ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
        fig.tight_layout()
        path = f"{plot_save_path}{dataset_name}_DIFF_SPECIFIC_ALL{plot_type}"
        if save_plots:
            print(f"store histogram plot as {path}.")
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
            print(f"store histogram plot as {path}.")
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
        fig.tight_layout()
        fig.show()
        ## store plot
        if save_plots:
            path = f"{plot_save_path}{dataset_name}_t_diff_nobg{plot_type}"
            print(f"store histogram plot as {path}.")
            fig.savefig(path)
            print(f"\nSaved plot to {path}")

        ######################
        ##### fit peak position (robust, automatic)

        popt, pcov, fit_bins, fit_hist, err_fit_hist, fit_func = fit_secondary_peak(
            bins_nobg, hist_nobg, err_hist_nobg,
            search_min=387,  # auto: first valley right of the main peak
            search_max=410,  # auto: end of histogram
            window_sigmas=3.0,
            prominence_frac=0.03,
        )

        perr = np.sqrt(np.diag(pcov))
        param_names = ["A", "mu","sigma","m","b",]
        fit_params = dict(zip(param_names, popt))
        errors = dict(zip(param_names, perr))

        fit_values = fit_func(fit_bins, *popt)
        chi2 = np.sum((fit_hist - fit_values)**2 / err_fit_hist**2)
        ndf = len(fit_hist) - len(popt)
        chi2ndf = chi2 / ndf

        print(f"Auto-detected fit interval ΔT = ({fit_bins.min():.1f}, {fit_bins.max():.1f}) ns")
        for name in param_names:
            print(f"  {name:>5} = {fit_params[name]:.6g} ± {errors[name]:.2g}")
        print(f"  chi²/ndf = {chi2:.2f} / {ndf} = {chi2ndf:.2f}")

        # --- estimate drift velocity ---
        mu_val = fit_params["mu"]
        err_mu = errors["mu"]
        v_drift = cell_half_width / mu_val
        err_v_drift = np.sqrt(
            (err_cell_half_width / mu_val)**2 +
            (cell_half_width * err_mu / mu_val**2)**2
        )

        print(f"v_drift = {v_drift:.4g} ± {err_v_drift:.2g} um/ns")

        fit_label = (
            "Gaussian + linear bg fit\n"
            r"$f(\Delta T)=A\,e^{-\frac{1}{2}((\Delta T-\mu)/\sigma)^2}+a_2\Delta T^2+a_1\Delta T+a_0$"
        )
        for name in ["A", "mu", "sigma"]:
            fit_label += f"\n${name}=({fit_params[name]:.3g}\\pm {errors[name]:.2g})$"
        fit_label += f"\n$v_{{\\mathrm{{drift}}}}=({v_drift:.3g}\\pm {err_v_drift:.2g})$"
        fit_err = err_gauss_plus_linear(fit_bins, *popt, *perr)

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
            print(f"store histogram plot as {path}.")
            fig.savefig(path)

        perr = np.sqrt(np.diag(pcov))

        param_names = ["A", "mu", "sigma", "a2", "a1", "a0"]

        fit_params = dict(zip(param_names, popt))
        errors = dict(zip(param_names, perr))

        results_photopeak[dataset_name] = {
            **fit_params,
            **{f"{key}_err": value for key, value in errors.items()}
        }
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
        plt.title(r"Drift velocity over time ($U_{\mathrm{wire}}=3600$ V)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_path  + f"plots/ramp_analysis_photo_peak{plot_type}")
        
    return 
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
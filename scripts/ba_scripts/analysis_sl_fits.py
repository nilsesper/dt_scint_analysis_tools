#################################################################
### use sl clusters to perform sl-level track fits
# store sl fits as pcl file
# cut fits to eliminate noise
# refit the fits with floating v_drift
# export refits as pcl for further analysis in sl_fits_analysis.py

#################################################################
import os
import argparse
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import re
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D 
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import uproot
# ---------------------------------------------------------------

vd_factor = 1 / derived_params._drift_velocity_conversion
def build_hist_general(
    *,
    data_list,
    err_data_list=None,
    edges=None,
    n_bins=500,
    edge_min=None,
    edge_max=None,
    margin_frac=0.02,
    do_stat_err=True,
    verbose=True,
    
):
    """
    Build a histogram dict from one or more arrays of *any* quantity.

    Fully generic: there is nothing here specific to time differences, energy,
    charge, or any other physical quantity. You pass in the raw data (and,
    optionally, per-entry errors), and get back a dict that
    `plot_hist_general` can plot directly. This is the piece that used to be
    hardcoded/pasted inside the plotting function -- now it's separate and
    reusable for any quantity you want to histogram.

    Binning is automatic by default: if you don't pass `edges` or explicit
    `edge_min`/`edge_max`, the range is taken from the actual min/max of the
    data (across all arrays in `data_list`, if you pass several), with a
    small margin so points don't sit exactly on the outer edge. This means
    you can call this with any quantity -- whatever its natural range is --
    without hardcoding numbers per-key.

    Parameters
    ----------
    data_list : array-like or list of array-like
        The quantity to histogram. Pass a single array, or a list of arrays
        if you want to merge several datasets into one combined histogram
        (e.g. multiple runs/channels of the same quantity).
    err_data_list : array-like or list of array-like, optional
        Per-entry errors for `data_list`, used to build shifted histograms
        for asymmetric uncertainty estimation. Must match `data_list` in
        shape (single array <-> single array, list <-> list of same length).
        If None, no data-driven error is added (only statistical error, if
        `do_stat_err=True`).
    edges : array-like, optional
        Explicit bin edges to use. If given, everything else about binning
        (`n_bins`/`edge_min`/`edge_max`/`margin_frac`) is ignored.
    n_bins : int
        Number of bins, used only if `edges` is not given (default 2500).
    edge_min, edge_max : float, optional
        Range of the histogram, used only if `edges` is not given. Either
        or both can be left as None to auto-detect from the data's actual
        min/max (this is the default -- pass both explicitly to fully
        override auto-ranging).
    margin_frac : float
        Fractional padding added to each side of an auto-detected range
        (default 0.02 = 2% of the data span on each side), so the min/max
        data points don't land exactly on the histogram edge. Ignored for
        any side of the range you set explicitly via edge_min/edge_max.
    do_stat_err : bool
        Whether to include statistical error (sqrt(N)) in the combined
        uncertainty via hist_utils.calculate_hist_uncertainty.
    verbose : bool
        Whether to print entry/underflow/overflow counts and the auto-picked
        range.

    Returns
    -------
    specific_data : dict
        Dict with keys: edges, centers, hist, err_hist, err_hist_stat,
        err_hist_down, err_hist_up, entries, underflow, overflow.
        Suitable to pass straight into `plot_hist_general`.
    """
    # normalize inputs to lists so a single array or multiple arrays both work
    if not isinstance(data_list, (list, tuple)):
        data_list = [data_list]
    if err_data_list is not None and not isinstance(err_data_list, (list, tuple)):
        err_data_list = [err_data_list]
    if err_data_list is not None and len(err_data_list) != len(data_list):
        raise ValueError("err_data_list must be the same length as data_list")

    # coerce every array to clean float64 up front -- this also gives us
    # the cleaned arrays to (a) auto-detect the range from and (b) reuse
    # below without converting twice
    clean_data_list = []
    for i, data in enumerate(data_list):
        try:
            data = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"data_list[{i}] could not be converted to a float64 array "
                f"(contains None/NaN/strings/ragged entries?): {e}"
            ) from e
        n_bad = int(np.sum(~np.isfinite(data)))
        if n_bad > 0 and verbose:
            print(f"  warning: data_list[{i}] has {n_bad} non-finite entries (NaN/inf)")
        clean_data_list.append(data)

    clean_err_data_list = None
    if err_data_list is not None:
        clean_err_data_list = []
        for i, err_data in enumerate(err_data_list):
            try:
                err_data = np.asarray(err_data, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"err_data_list[{i}] could not be converted to a float64 array: {e}"
                ) from e
            clean_err_data_list.append(err_data)

    ### determine bin edges
    if edges is not None:
        edges = np.asarray(edges)
    else:
        auto_min = edge_min
        auto_max = edge_max
        if auto_min is None or auto_max is None:
            finite_vals = np.concatenate([d[np.isfinite(d)] for d in clean_data_list])
            if finite_vals.size == 0:
                raise ValueError("no finite data points found to auto-determine binning range")
            data_min, data_max = float(np.min(finite_vals)), float(np.max(finite_vals))
            span = data_max - data_min
            pad = span * margin_frac if span > 0 else (abs(data_max) * margin_frac or 1.0)
            if auto_min is None:
                auto_min = data_min - pad
            if auto_max is None:
                auto_max = data_max + pad
        if verbose and (edge_min is None or edge_max is None):
            print(f"  auto-detected binning range: [{auto_min:.6g}, {auto_max:.6g}]")
        edges = np.linspace(auto_min, auto_max, n_bins + 1)

    ### prepare empty combined hist
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = (
        hist_utils.create_empty_histogram(edges=edges)
    )

    ### fill and combine, one dataset at a time
    for i, data in enumerate(clean_data_list):
        err_data = clean_err_data_list[i] if clean_err_data_list is not None else None

        (
            hist_,
            _,
            _,
            entries_,
            underflow_,
            overflow_,
            hist_err_right_,
            hist_err_left_,
        ) = hist_utils.calculate_histogram_and_shifted_histograms(
            data=data, edges=edges, err_data=err_data
        )

        # defensive cast: some hist_utils code paths can hand back object
        # dtype (e.g. when err_data is None internally); force float64 so
        # the accumulation below can never hit a casting error
        hist_ = np.asarray(hist_, dtype=np.float64)
        hist_err_right_ = np.asarray(hist_err_right_, dtype=np.float64)
        hist_err_left_ = np.asarray(hist_err_left_, dtype=np.float64)

        hist += hist_
        entries += entries_
        underflow += underflow_
        overflow += overflow_
        hist_err_right += hist_err_right_
        hist_err_left += hist_err_left_

    ### error calculation for full hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
        hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=do_stat_err
    )
    ### stat-only uncertainty, kept separately in case you want it
    err_hist_stat = np.sqrt(hist)

    if verbose:
        print("created histogram:")
        print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    specific_data = {
        "edges": edges,
        "centers": centers,
        "hist": hist,
        "err_hist": err_hist,
        "err_hist_stat": err_hist_stat,
        "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up,
        "entries": entries,
        "underflow": underflow,
        "overflow": overflow,
    }
    return specific_data



 
def plot_hist_general(
    *,
    specific_data,
    dataset_name,
    plot_save_path,
    plot_type=".png",
    xlabel,
    ylabel="counts",
    filename_suffix="ALL",
    start_idx=0,
    scale_factor=1,          # e.g. tu -> ns conversion; set to 1.0 to disable
    log_scale=False,
    power_limits=[-4, 4],
    bin_unit="",
    add_info=True,
    legend_font_size=13,
    fig_size=(8, 6),
    xlim=None,                # None -> (0, max(bins)); pass tuple to override; False to skip xlim
    hist_key="hist",
    err_hist_key="err_hist",
    err_hist_down_key="err_hist_down",
    err_hist_up_key="err_hist_up",
    edges_key="edges",
    overflow_key="overflow",
    underflow_key="underflow",
    save=True,
    verbose=True,
    title="",
    speckey=None,
    # --- A*cos^2(alpha) fit, only attempted when speckey is not None ---
    fit_cos2=True,
    fit_range=None,           # (lo, hi) central x-range the fit is restricted to;
                               # None -> auto: symmetric central band, width =
                               # fit_range_frac * max(|xmin|, |xmax|)
                               # around 0
    fit_range_frac=0.5,
    fit_cos2_p0=None,
    fit_cos2_bounds=(0, np.inf),
    fit_cos2_x_kind="alpha_rad",   # "alpha_rad", "alpha_deg", or "tan_alpha"
    fit_cos2_color="red",
):
    """
    General histogram plotting function.
 
    Reads a histogram (values, edges, asymmetric errors, over/underflow) out of
    `specific_data`, and either:
      - plots it as a bar histogram via hist_utils.plot_histogram (default,
        used for every histogram in this pipeline), or
      - if speckey is given and fit_cos2=True: plots it as a SCATTER with a
        central-range A*cos^2(alpha) fit overlaid (solid in-range, dashed
        extrapolated) -- see module docstring for why.
 
    This function does NOT build or mutate `specific_data` in any way -- it only
    reads from it. Use `build_hist_general` (or your own logic) to construct the
    dict beforehand, e.g.:
 
        specific_data = build_hist_general(
            data_list=my_quantity,
            err_data_list=my_quantity_err,
        )
        fig, ax, path = plot_hist_general(
            specific_data=specific_data,
            dataset_name="run1",
            plot_save_path="./plots/",
            xlabel="my quantity (units)",
        )
 
    Parameters
    ----------
    specific_data : dict
        Dict containing the histogram arrays under the *_key entries below.
    dataset_name : str
        Used to build the output filename.
    plot_save_path : str
        Directory (or path prefix) to save the plot into.
    plot_type : str
        File extension / suffix appended to filename (e.g. ".png").
    xlabel : str
        X-axis label. Overridden by `speckey` if given.
    filename_suffix : str
        Inserted into the filename to distinguish plots, e.g. "ALL", "SUBSET_1".
    start_idx : int
        Index to slice hist/edges/errors from (drops leading bins).
    scale_factor : float
        Multiplier applied to edges (e.g. unit conversion). Use 1.0 for none.
    log_scale, power_limits, bin_unit, add_info : see hist_utils.plot_histogram;
        only used in the default (non-alpha) bar-histogram path.
    legend_font_size : int
        Font size used for the cos^2-fit legend entry.
    fig_size : tuple
        Figure size.
    xlim : tuple, None, or False
        If None, defaults to (min(bins), max(bins)). If False, xlim is not
        set. If a tuple, used directly as ax.set_xlim(*xlim).
    hist_key, err_hist_key, err_hist_down_key, err_hist_up_key, edges_key,
    overflow_key, underflow_key : str
        Keys used to pull data out of `specific_data`.
    save : bool
        Whether to save the figure to disk.
    verbose : bool
        Whether to print progress messages.
    title : str
        Plot title. Any "tan alpha" / "tan_alpha" wording (case-insensitive,
        with or without underscore/space) is automatically normalized to
        "alpha" -- see module docstring point 5.
    speckey : str or None
        If given, overrides xlabel with this value AND (when fit_cos2=True)
        switches to the scatter + central-range cos^2(alpha) fit described
        above.
    fit_cos2 : bool, default True
        Whether to do the cos^2 fit (and switch to scatter display) when
        speckey is given. Set False to keep the bar-histogram display even
        with speckey set (xlabel override only, no fit).
    fit_range : (float, float) or None
        Central x-range (post scale_factor) the fit is restricted to.
        Everything outside is still plotted (scatter points + extrapolated
        dashed fit) but doesn't influence the fit. If None, auto-computed
        as a symmetric band around 0 covering `fit_range_frac` of the
        larger x-extent.
    fit_range_frac : float
        Used only when fit_range is None; fraction of max(|xmin|, |xmax|)
        used as the auto central half-width... actually used as the full
        fraction of the extent (see code) -- default 0.5.
    fit_cos2_p0 : list[float] or None
        Initial guess for [A]. Defaults to [max(hist) within fit_range].
    fit_cos2_bounds : tuple
        Bounds passed to curve_fit for A. Default (0, inf).
    fit_cos2_x_kind : "alpha_rad", "alpha_deg", or "tan_alpha"
        What the histogram's x-axis actually contains. Default "alpha_rad"
        (the histogrammed quantity is alpha itself). "tan_alpha" uses the
        closed form A/(1+x^2) == A*cos^2(arctan(x)).
    fit_cos2_color : str
        Line color for the fit curve.
 
    Returns
    -------
    fig, ax, path
        The created figure, axis, and the save path (path is None if save=False).
        ax.fit_results holds the cos^2-fit results dict (or None if the
        bar-histogram path was used, or the fit failed) -- keys: "A",
        "A_err", "chi2", "ndf", "chi2_ndf", "chi2_ndf_full_range",
        "popt", "pcov", "fit_range".
    """
    # --- title normalization: "tan alpha"/"tan_alpha" (any spacing/case) -> "alpha" ---
    title = re.sub(r"tan[\s_]*alpha", "alpha", title, flags=re.IGNORECASE)
 
    # read data
    hist = np.array(specific_data[hist_key])[start_idx:]
    err_hist_down = np.array(specific_data[err_hist_down_key])[start_idx:]
    err_hist_up = np.array(specific_data[err_hist_up_key])[start_idx:]
    edges = np.array(specific_data[edges_key])[start_idx:] * scale_factor
    centers = hist_utils.centers_from_edges(edges)
    bins = centers
    overflow = specific_data[overflow_key]
    underflow = specific_data[underflow_key]
 
    if verbose:
        print(f"Plotting histogram for {dataset_name} ({filename_suffix})...")
 
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
 
    do_alpha_fit = speckey is not None and fit_cos2
 
    # -----------------------------------------------------------------
    # histogram display: always the normal bar histogram, unchanged
    # -----------------------------------------------------------------
    ax = hist_utils.plot_histogram(
        ax,
        hist=hist,
        centers=bins,
        err_hist_down=err_hist_down,
        err_hist_up=err_hist_up,
        log_scale=log_scale,
        power_limits=power_limits,
        add_info=add_info,
        entries=int(np.sum(hist)),
        overflow=overflow,
        underflow=underflow,
        bin_unit=bin_unit,
    )
    fit_results = None
 
    if do_alpha_fit:
        # -----------------------------------------------------------------
        # alpha path: overlay a central-range A*cos^2(alpha) fit on top of
        # the bar histogram, extrapolated (dashed) outside fit_range
        # -----------------------------------------------------------------
        err_hist_sym = (err_hist_up + err_hist_down) / 2.0
        err_hist_safe = np.where(err_hist_sym <= 0, 1.0, err_hist_sym)
 
        def cos2_model(x, A):
            if fit_cos2_x_kind == "tan_alpha":
                # A*cos^2(arctan(x)) == A/(1+x^2), closed form
                return A / (1.0 + x ** 2)
            elif fit_cos2_x_kind == "alpha_deg":
                return A * np.cos(np.deg2rad(x)) ** 2
            else:  # "alpha_rad"
                return A * np.cos(x) ** 2
 
        if fit_range is None:
            x_extent = max(abs(np.amin(bins)), abs(np.amax(bins)))
            half_width = x_extent * fit_range_frac
            fit_range = (-half_width, half_width)
 
        fit_mask = (bins >= fit_range[0]) & (bins <= fit_range[1])
        n_fit_pts = int(np.sum(fit_mask))
 
        if n_fit_pts < 2:
            if verbose:
                print(f"  cos^2 fit skipped for {dataset_name} ({filename_suffix}): "
                      f"fit_range={fit_range} selects only {n_fit_pts} bin(s).")
        else:
            p0 = fit_cos2_p0 if fit_cos2_p0 is not None else (
                [np.max(hist[fit_mask]) if hist[fit_mask].size else 1.0]
            )
            try:
                popt, pcov = curve_fit(
                    cos2_model, bins[fit_mask], hist[fit_mask],
                    p0=p0, sigma=err_hist_safe[fit_mask], absolute_sigma=True,
                    bounds=fit_cos2_bounds,
                )
                perr = np.sqrt(np.diag(pcov))
 
                model_vals_fit = cos2_model(bins[fit_mask], *popt)
                resid_fit = (hist[fit_mask] - model_vals_fit) / err_hist_safe[fit_mask]
                chi2 = float(np.sum(resid_fit ** 2))
                ndf = max(n_fit_pts - len(popt), 1)
                chi2_ndf = chi2 / ndf
 
                model_vals_full = cos2_model(bins, *popt)
                resid_full = (hist - model_vals_full) / err_hist_safe
                chi2_full = float(np.sum(resid_full ** 2))
                ndf_full = max(len(bins) - len(popt), 1)
                chi2_ndf_full = chi2_full / ndf_full
 
                fit_results = {
                    "A": popt[0], "A_err": perr[0],
                    "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2_ndf,
                    "chi2_ndf_full_range": chi2_ndf_full,
                    "popt": popt, "pcov": pcov,
                    "fit_range": fit_range,
                }
                if verbose:
                    print(f"  cos^2 fit ({fit_cos2_x_kind}), central range {fit_range} "
                          f"({n_fit_pts}/{len(bins)} bins): A = {popt[0]:.4g} +/- {perr[0]:.4g}")
                    print(f"    chi2/ndf (fit range only)  = {chi2:.4g} / {ndf} = {chi2_ndf:.4g}")
                    print(f"    chi2/ndf (full range, ref) = {chi2_full:.4g} / {ndf_full} = {chi2_ndf_full:.4g}")
            except (RuntimeError, ValueError) as e:
                if verbose:
                    print(f"  cos^2 fit failed for {dataset_name} ({filename_suffix}): {e}")
 
        # --- overlay fit range shading + fit curve on top of the bars ---
        ax.axvspan(fit_range[0], fit_range[1], color="gray", alpha=0.12, zorder=2, label="fit range")
 
        if fit_results is not None:
            popt = fit_results["popt"]
            x_full = np.linspace(np.amin(bins), np.amax(bins), 1000)
            y_full = cos2_model(x_full, *popt)
            mid_mask = (x_full >= fit_range[0]) & (x_full <= fit_range[1])
            left_mask = x_full < fit_range[0]
            right_mask = x_full > fit_range[1]
 
            model_str = {
                "tan_alpha": r"$A\cos^2(\alpha),\ \alpha=\arctan(\tan\alpha)$",
                "alpha_deg": r"$A\cos^2(\alpha)$",
                "alpha_rad": r"$A\cos^2(\alpha)$",
            }[fit_cos2_x_kind]
            fit_label = (
                f"{model_str} fit (central range)\n"
                f"$A = {fit_results['A']:.3g} \\pm {fit_results['A_err']:.3g}$\n"
                f"$\\chi^2/N_{{df}} = {fit_results['chi2_ndf']:.2f}$ (fit range only)"
            )
            ax.plot(x_full[mid_mask], y_full[mid_mask], color=fit_cos2_color, linewidth=2,
                     label=fit_label, zorder=4)
 
            first_extrap_label = "extrapolated"
            for m in (left_mask, right_mask):
                if np.any(m):
                    ax.plot(x_full[m], y_full[m], color=fit_cos2_color, linewidth=2,
                             linestyle="--", label=first_extrap_label, zorder=4)
                    first_extrap_label = None  # only label the first dashed segment
 
        ax.legend(prop={"size": legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
 
    if xlim is None:
        ax.set_xlim(np.amin(bins), np.amax(bins))
    elif xlim is not False:
        ax.set_xlim(*xlim)
 
    if speckey != None:
        xlabel = speckey
 
    ax.fit_results = fit_results
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
 
    path = None
    if save:
        path = f"{plot_save_path}{dataset_name}_{filename_suffix}{plot_type}"
        if verbose:
            print(f"store histogram plot as {path}.")
        fig.savefig(path)
        if verbose:
            print(f"Done saving hist as {path}\n")
 
    return fig, ax, path

def measured_alpha_hist_path(pcls_path, pct_ar, pct_co2, u_wire, suffix="w_cut"):
    """
    Canonical path for a measured-alpha-histogram export, keyed by gas mix
    and wire voltage. Used by both sl_fits_analysis.py (exporting) and the
    Garfield analysis script (importing) -- keep in sync between the two.
    """
    return os.path.join(
        pcls_path, "measured_alpha_hists",
        f"measured_alpha_hist_ar-{int(pct_ar)}_co2-{int(pct_co2)}_anode{int(u_wire)}V_{suffix}.npz"
    )

def detector_track(
    *,
    super_fits_cuts,
    dataset_info,
    plot_idcs,
    suffix,
    plot_save_path,
    dataset_name,
    fit_suffix="_free_vd_super_fit",
    plot_type=".png",
    zoom=True,
    zoom_margin=20.0,
    orient="phi",
    wire_marker_size=4,
    wire_marker_color="black",
    sl1_fit_color="tab:orange",
    sl2_fit_color="tab:purple",
    save = True
):
    """
    Reproduce every plot from plot_super_sl_pattern_fits.py's per-fit loop,
    for one or more individual super-pattern fits (sl1 + sl2, 8 layers).

    For each idx in `plot_idcs`, saves:
    - "{dataset_name}_ts_vs_fit_{suffix}_{idx}{plot_type}"
            measured vs. fitted timestamps (8 layers) + residuals
    - "{dataset_name}_local_track_{suffix}_{idx}{plot_type}"
            projected hits + fitted track in the local fit frame
    - "{dataset_name}_detector_track_{suffix}_{idx}{plot_type}"
            fitted track drawn inside the full DT chamber geometry
    - "{dataset_name}_detector_track_zoom_{suffix}_{idx}{plot_type}"
            same, zoomed in x (only if zoom=True)

    Additional plots (super fit + standalone sl1/sl3 fits):
    - "{dataset_name}_detector_track_individual_fits_{suffix}_{idx}{plot_type}"
            chamber view comparing the super fit track against the standalone
            sl1-only and sl3-only fit tracks
    - "{dataset_name}_detector_track_individual_fits_zoom_{suffix}_{idx}{plot_type}"
            same, zoomed in x (only if zoom=True)
    - "{dataset_name}_ts_vs_fit_individual_fits_{suffix}_{idx}{plot_type}"
            same style/structure as ts_vs_fit above (measured vs. fitted
            timestamps + residuals), but using the standalone sl1-only /
            sl3-only fit predictions instead of the super fit

    Parameters
    ----------
    super_fits_cuts : dict of arrays
        The (already cut) super-fit dataset, e.g. the output of
        data_utils.cut_data() on the super_fits pcl. Must contain per-fit
        arrays for sl1, sl3, pat_type_sl1/sl3, wi{0-3}_sl1/sl3, ts{0-7},
        err_ts{0-7}, the fitted parameters selected by `fit_suffix`
        (lat_id1/2, t0, x0, tan_alpha, vd, their errors, all corr_* terms,
        ref_x, ref_z, chi2/ndf), AND the standalone single-SL fit results
        (t0_sl1/sl3, x0_sl1/sl3, tan_alpha_sl1/sl3, vd_sl1/sl3, their errors,
        all corr_*_sl1/sl3 terms, chi2/ndf_sl1/sl3). There is no stored
        reference point for the standalone fits, so each one's local origin
        is derived here as that SL's own top wire (see in-code comments).
    dataset_info : dict
        Dict as returned by parse_fit_name(), with keys "pct_Ar", "pct_CO2",
        "U_wire", "U_Fieldshaper", "U_cathode". Used in the chamber-view title.
    plot_idcs : list of int
        Indices into `super_fits_cuts` selecting which fits to plot.
    suffix : str
        Cut-stage label, e.g. "no_cut" or "w_cut". Used only in titles/filenames.
    plot_save_path : str
        Directory the figures are saved into (created if missing).
    dataset_name : str
        Used as a prefix for output filenames.
    fit_suffix : str
        Which stored fit variant to draw, e.g. "_free_vd_super_fit" (default)
        or "_refit" etc. Selects the keys read off `super_fits_cuts`.
    plot_type : str
        File extension for the saved figures, e.g. ".png".
    zoom : bool
        If True, also produce the x-zoomed chamber view.
    zoom_margin : float
        Margin in mm added around hits/track for the zoomed view.
    orient : str
        Chamber orientation to draw ("phi" for sl1+sl3 super patterns -> x-axis;
        would need orient="theta" / a y-axis view for the other pairing).
    wire_marker_size : float
        Marker size (in points^2, passed to scatter) used to re-draw the
        highlighted-cell wire dots on top of the aqua cell fill, so they
        stay visible instead of being hidden by the highlight color.
    wire_marker_color : str
        Color used for the re-drawn wire dots.
    sl1_fit_color : str
        Line color for the standalone sl1 fit track in the individual-fits
        comparison plot.
    sl2_fit_color : str
        Line color for the standalone sl2 (sl3) fit track in the
        individual-fits comparison plot.

    Returns
    -------
    saved_paths : dict
        {idx: {"ts_vs_fit": path, "local_track": path,
            "detector_track": path, "detector_track_zoom": path_or_None,
            "detector_track_individual_fits": path,
            "detector_track_individual_fits_zoom": path_or_None,
            "ts_vs_fit_individual_fits": path}}
        for every index in `plot_idcs`.
    """

    saved_paths = {}

    pct_ar = dataset_info["pct_Ar"]
    pct_co2 = dataset_info["pct_CO2"]
    u_wire = dataset_info["U_wire"]
    u_fieldshaper = dataset_info["U_Fieldshaper"]
    u_cathode = dataset_info["U_cathode"]

    ref_axis = 0 if orient == "phi" else 1  # phi view -> x-axis, theta view -> y-axis

    for idx in plot_idcs:

        fit = {k: super_fits_cuts[k][idx] for k in super_fits_cuts.keys()}
        paths = {}

        # -------------------------------------------------------------
        # pattern / laterality bookkeeping (mirrors fit_super_sl_patterns)
        # -------------------------------------------------------------
        sl1 = int(fit["sl1"])
        sl2 = int(fit["sl3"])
        pat_type_sl1 = int(fit["pat_type_sl1"])
        pat_type_sl2 = int(fit["pat_type_sl3"])
        pat_name_sl1 = list(params._dt_sl_patterns.keys())[pat_type_sl1]
        pat_name_sl2 = list(params._dt_sl_patterns.keys())[pat_type_sl2]
        lats1 = params._dt_sl_patterns[pat_name_sl1]["laterality"]
        lats2 = params._dt_sl_patterns[pat_name_sl2]["laterality"]

        lat_id1 = int(fit["lat_id1" + fit_suffix])
        lat_id2 = int(fit["lat_id2" + fit_suffix])
        laterality = np.array(list(lats1[lat_id1]) + list(lats2[lat_id2]))  # length 8

        wi_sl1 = [int(fit[f"wi{ly}_sl1"]) for ly in range(4)]
        wi_sl2 = [int(fit[f"wi{ly}_sl3"]) for ly in range(4)]

        lys = np.arange(0, 8)
        ts = np.array([fit[f"ts{ly}"] for ly in range(8)])
        err_ts = np.array([fit[f"err_ts{ly}"] for ly in range(8)])

        # -------------------------------------------------------------
        # geometry: rebuilt exactly as in fit_super_sl_patterns
        # (global frame -> shifted into local frame via stored ref_x/ref_z)
        # -------------------------------------------------------------
        z_arr_glob, x_cell_glob = np.full(8, 0, dtype=np.float64), np.full(8, 0, dtype=np.float64)
        for ly in range(4):
            x_cell_glob[ly],     z_arr_glob[ly]     = derived_params.super_pattern_geometry(sl1, ly, wi_sl1[ly])
            x_cell_glob[ly + 4], z_arr_glob[ly + 4] = derived_params.super_pattern_geometry(sl2, ly, wi_sl2[ly])

        ref_x = fit["ref_x" + fit_suffix]
        ref_z = fit["ref_z" + fit_suffix]
        x_cell = x_cell_glob - ref_x
        z_arr = z_arr_glob - ref_z

        # -------------------------------------------------------------
        # SUPER FIT results
        # -------------------------------------------------------------
        t0 = fit["t0" + fit_suffix]
        x0 = fit["x0" + fit_suffix]
        tan_alpha = fit["tan_alpha" + fit_suffix]
        vd = fit["vd" + fit_suffix]
        err_t0 = fit["err_t0" + fit_suffix]
        err_x0 = fit["err_x0" + fit_suffix]
        err_tan_alpha = fit["err_tan_alpha" + fit_suffix]
        err_vd = fit["err_vd" + fit_suffix]
        corr_t0_x0 = fit["corr_t0_x0" + fit_suffix]
        corr_t0_tan_alpha = fit["corr_t0_tan_alpha" + fit_suffix]
        corr_t0_vd = fit["corr_t0_vd" + fit_suffix]
        corr_x0_tan_alpha = fit["corr_x0_tan_alpha" + fit_suffix]
        corr_x0_vd = fit["corr_x0_vd" + fit_suffix]
        corr_tan_alpha_vd = fit["corr_tan_alpha_vd" + fit_suffix]
        chi2ndf = fit["chi2/ndf" + fit_suffix]

        # -------------------------------------------------------------
        # standalone single-SL FIT results
        # -------------------------------------------------------------
        t0_sl1 = fit["t0_sl1"]
        t0_sl3 = fit["t0_sl3"]

        x0_sl1 = fit["x0_sl1"]
        x0_sl3 = fit["x0_sl3"]

        tan_alpha_sl1 = fit["tan_alpha_sl1"]
        tan_alpha_sl3 = fit["tan_alpha_sl3"]

        vd_sl1 = fit["vd_sl1"]
        vd_sl3 = fit["vd_sl3"]

        err_t0_sl1 = fit["err_t0_sl1"]
        err_t0_sl3 = fit["err_t0_sl3"]

        err_x0_sl1 = fit["err_x0_sl1"]
        err_x0_sl3 = fit["err_x0_sl3"]

        err_tan_alpha_sl1 = fit["err_tan_alpha_sl1"]
        err_tan_alpha_sl3 = fit["err_tan_alpha_sl3"]

        err_vd_sl1 = fit["err_vd_sl1"]
        err_vd_sl3 = fit["err_vd_sl3"]

        corr_t0_x0_sl1 = fit["corr_t0_x0_sl1"]
        corr_t0_x0_sl3 = fit["corr_t0_x0_sl3"]

        corr_t0_tan_alpha_sl1 = fit["corr_t0_tan_alpha_sl1"]
        corr_t0_tan_alpha_sl3 = fit["corr_t0_tan_alpha_sl3"]

        corr_t0_vd_sl1 = fit["corr_t0_vd_sl1"]
        corr_t0_vd_sl3 = fit["corr_t0_vd_sl3"]

        corr_x0_tan_alpha_sl1 = fit["corr_x0_tan_alpha_sl1"]
        corr_x0_tan_alpha_sl3 = fit["corr_x0_tan_alpha_sl3"]

        corr_x0_vd_sl1 = fit["corr_x0_vd_sl1"]
        corr_x0_vd_sl3 = fit["corr_x0_vd_sl3"]

        corr_tan_alpha_vd_sl1 = fit["corr_tan_alpha_vd_sl1"]
        corr_tan_alpha_vd_sl3 = fit["corr_tan_alpha_vd_sl3"]

        chi2ndf_sl1 = fit["chi2/ndf_sl1"]
        chi2ndf_sl3 = fit["chi2/ndf_sl3"]

        # There is no stored per-SL reference point in the data, so each
        # standalone fit's local origin is taken to be that SL's OWN top
        # wire (among its own 4 layers), evaluated in the same
        # super_pattern_geometry() frame used for x_cell_glob/z_arr_glob.
        # This mirrors the convention already used (and visually confirmed
        # to work) for the chamber-frame track drawing below, just applied
        # in the fit's own geometry frame instead of the plotting frame.
        # NOTE: reusing the super fit's shared ref_x/ref_z here is WRONG
        # whenever the global top wire doesn't happen to lie within that
        # SL's own 4 layers -- that was the bug causing the huge residuals.
        top_wire_idx_sl1_geom = int(np.argmax(z_arr_glob[0:4]))
        ref_x_sl1 = x_cell_glob[0:4][top_wire_idx_sl1_geom]
        ref_z_sl1 = z_arr_glob[0:4][top_wire_idx_sl1_geom]

        top_wire_idx_sl3_geom = int(np.argmax(z_arr_glob[4:8]))
        ref_x_sl3 = x_cell_glob[4:8][top_wire_idx_sl3_geom]
        ref_z_sl3 = z_arr_glob[4:8][top_wire_idx_sl3_geom]
        # -------------------------------------------------------------

        # fit function evaluated at all 8 layers (super fit)
        fit_ts, err_fit_ts = np.zeros(8), np.zeros(8)
        for ly in range(8):
            fit_ts[ly] = derived_params.f_ts_fit(
                x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly],
                laterality=laterality[ly], vd=vd,
            )
            err_fit_ts[ly] = derived_params.err_f_ts_fit(
                x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly],
                laterality=laterality[ly], vd=vd,
                err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd,
                corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=corr_t0_vd,
                corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=corr_x0_vd, corr_tan_alpha_vd=corr_tan_alpha_vd,
            )

        # fit function evaluated at each SL's own 4 layers, using ONLY that
        # SL's standalone fit parameters and its own local frame
        x_cell_sl1_local = x_cell_glob[0:4] - ref_x_sl1
        z_arr_sl1_local = z_arr_glob[0:4] - ref_z_sl1
        x_cell_sl3_local = x_cell_glob[4:8] - ref_x_sl3
        z_arr_sl3_local = z_arr_glob[4:8] - ref_z_sl3

        fit_ts_sl1_own, err_fit_ts_sl1_own = np.zeros(4), np.zeros(4)
        for ly in range(4):
            fit_ts_sl1_own[ly] = derived_params.f_ts_fit(
                x_cell=x_cell_sl1_local[ly], t0=t0_sl1, x0=x0_sl1, tan_alpha=tan_alpha_sl1,
                z=z_arr_sl1_local[ly], laterality=laterality[ly], vd=vd_sl1,
            )
            err_fit_ts_sl1_own[ly] = derived_params.err_f_ts_fit(
                x_cell=x_cell_sl1_local[ly], t0=t0_sl1, x0=x0_sl1, tan_alpha=tan_alpha_sl1,
                z=z_arr_sl1_local[ly], laterality=laterality[ly], vd=vd_sl1,
                err_t0=err_t0_sl1, err_x0=err_x0_sl1, err_tan_alpha=err_tan_alpha_sl1, err_vd=err_vd_sl1,
                corr_t0_x0=corr_t0_x0_sl1, corr_t0_tan_alpha=corr_t0_tan_alpha_sl1, corr_t0_vd=corr_t0_vd_sl1,
                corr_x0_tan_alpha=corr_x0_tan_alpha_sl1, corr_x0_vd=corr_x0_vd_sl1, corr_tan_alpha_vd=corr_tan_alpha_vd_sl1,
            )

        fit_ts_sl3_own, err_fit_ts_sl3_own = np.zeros(4), np.zeros(4)
        for ly in range(4):
            fit_ts_sl3_own[ly] = derived_params.f_ts_fit(
                x_cell=x_cell_sl3_local[ly], t0=t0_sl3, x0=x0_sl3, tan_alpha=tan_alpha_sl3,
                z=z_arr_sl3_local[ly], laterality=laterality[ly + 4], vd=vd_sl3,
            )
            err_fit_ts_sl3_own[ly] = derived_params.err_f_ts_fit(
                x_cell=x_cell_sl3_local[ly], t0=t0_sl3, x0=x0_sl3, tan_alpha=tan_alpha_sl3,
                z=z_arr_sl3_local[ly], laterality=laterality[ly + 4], vd=vd_sl3,
                err_t0=err_t0_sl3, err_x0=err_x0_sl3, err_tan_alpha=err_tan_alpha_sl3, err_vd=err_vd_sl3,
                corr_t0_x0=corr_t0_x0_sl3, corr_t0_tan_alpha=corr_t0_tan_alpha_sl3, corr_t0_vd=corr_t0_vd_sl3,
                corr_x0_tan_alpha=corr_x0_tan_alpha_sl3, corr_x0_vd=corr_x0_vd_sl3, corr_tan_alpha_vd=corr_tan_alpha_vd_sl3,
            )

        ts_label = "Hit timestamps"
        fit_label = f"""Track fit:
$T_0=({np.round(t0,0):.0f}\\pm{np.round(err_t0,0):.0f})$ {params._key_units['t0']}
$x_0=({np.round(x0,1):.1f}\\pm{np.round(err_x0,1):.1f})$ {params._key_units['x0']}
$\\tan\\alpha=({np.round(tan_alpha,2):.2f}\\pm{np.round(err_tan_alpha,2):.2f})$ {params._key_units['tan_alpha']}
$V_d=({np.round(vd * vd_factor,4):.2f}\\pm{np.round(err_vd * vd_factor,4):.2f} )\\mu m/ns$ 
$\\chi^2/N_{{df}}={np.round(chi2ndf,2):.2f}$
Track ID = {idx}"""

        ################################
        ###### plot 1: timestamps vs fit, 8 layers, with residuals

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(5, 1))
        ax[0].errorbar(x=lys - 0.04, y=ts, yerr=err_ts, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        ax[0].errorbar(x=lys + 0.04, y=fit_ts, yerr=err_fit_ts, color="tab:red", marker="v", markersize=7, linestyle="", label=fit_label)
        ax[0].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[0].set_ylabel("Timestamp $T_{ly}$ [TU]")
        ax[0].legend(prop={"size": 16}, fancybox=False, framealpha=params._legend_alpha)
        ax[0].set_title(f"SL {sl1} + SL {sl2}, Patterns {pat_name_sl1}/{pat_name_sl2}, Laterality {[int(l) for l in laterality]}")
        ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        residuals = ts - fit_ts
        err_residuals = err_ts
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[1].errorbar(x=lys, y=residuals, yerr=err_residuals, color="black", marker="o", markersize=7, linestyle="")
        y_span = np.amax(np.abs(residuals) + err_residuals) * 1.1
        ax[1].set_ylim(-y_span, y_span)
        ax[1].set_ylabel("Residuals [TU]")
        ax[1].set_xlabel(f"Layer $ly$ (0-3: SL{sl1}, 4-7: SL{sl2})")
        ax[1].set_xticks([i for i in range(8)])
        ax[1].set_xticklabels([f"{i}" for i in range(8)])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)

        path = f"{plot_save_path}{dataset_name}_ts_vs_fit_{suffix}_{idx}{plot_type}"
        fig.savefig(fname=path)
        plt.close(fig)
        paths["ts_vs_fit"] = path

        
        ################################
        ###### plot 2: projected local track (8 hits, spanning both SLs)

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        for ly in range(8):
            color = "tab:gray" if ly < 4 else "tab:olive"
            ax.scatter(x_cell[ly], z_arr[ly], marker="s", s=60, facecolors="none", edgecolors=color, zorder=1)

        x_hits = x_cell + laterality * (ts - t0) * vd
        err_x_hits = np.sqrt(
            (laterality * vd) ** 2 * err_ts ** 2
            + (-laterality * vd) ** 2 * err_t0 ** 2
            + (-laterality * (ts - t0) * vd ** 2) ** 2 * err_vd ** 2
            + 2 * (laterality * vd) * (-laterality * (ts - t0) * vd ** 2) * corr_t0_vd
        )
        ax.errorbar(x=x_hits, y=z_arr, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label, zorder=2)

        z_range = np.linspace(np.amin(z_arr) - params._cell_height, np.amax(z_arr) + params._cell_height, 1000)
        track = derived_params.f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha)
        err_track = derived_params.err_f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        ax.plot(track, z_range, linewidth=2, color="tab:red", label=fit_label, zorder=3)
        ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:red", alpha=0.2, zorder=0)

        ax.legend(prop={"size": 16}, loc="center left", fancybox=False, framealpha=params._legend_alpha)
        ax.set_xlabel("$x-x_\\text{wire,top}$ [mm]")
        ax.set_ylabel("$z-z_\\text{wire,top}$ [mm]")
        ax.set_ylim(np.amin(z_range), np.amax(z_range))
        ax.set_title(f"SL {sl1} + SL {sl2}, Patterns {pat_name_sl1}/{pat_name_sl2}, Laterality {[int(l) for l in laterality]}")
        fig.tight_layout()

        path = f"{plot_save_path}{dataset_name}_local_track_{suffix}_{idx}{plot_type}"
        #fig.savefig(fname=path)
        plt.close(fig)
        paths["local_track"] = path
        
        ################################
        ###### plot 3: track inside the full detector geometry (global chamber view)
        # super patterns span sl1 + sl3, i.e. the "phi" superlayers

        dt_cell_data = dt_utils._chamber_data()
        for ly in range(4):
            dt_cell_data[sl1][ly][wi_sl1[ly]]["color"] = "aqua"
            dt_cell_data[sl2][ly][wi_sl2[ly]]["color"] = "aqua"

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=True)

        # -----------------------------------------------------------------
        # IMPORTANT: super_pattern_geometry() (used inside fit_super_sl_patterns,
        # and thus the fit's ref_x/ref_z) lives in a DIFFERENT coordinate frame
        # than _dt_cell_coordinates (the source chamber_ax/dt_cell_data draw from).
        # Mixing the two would offset the track/hits from the cells. So here we
        # independently rebuild the 8-layer geometry from _dt_cell_coordinates --
        # the same source the chamber background uses -- and re-derive our own
        # "top wire" reference in THAT frame.
        # -----------------------------------------------------------------
        z_arr_ch, x_cell_ch = np.full(8, 0, dtype=np.float64), np.full(8, 0, dtype=np.float64)
        for ly in range(4):
            x_cell_ch[ly]     = derived_params._dt_cell_coordinates[sl1][ly][wi_sl1[ly]][ref_axis + 3]
            z_arr_ch[ly]      = derived_params._dt_cell_coordinates[sl1][ly][wi_sl1[ly]][5]
            x_cell_ch[ly + 4] = derived_params._dt_cell_coordinates[sl2][ly][wi_sl2[ly]][ref_axis + 3]
            z_arr_ch[ly + 4]  = derived_params._dt_cell_coordinates[sl2][ly][wi_sl2[ly]][5]

        top_wire_idx = int(np.argmax(z_arr_ch))
        x_ref_ch = x_cell_ch[top_wire_idx]
        z_ref_ch = z_arr_ch[top_wire_idx]

        # hit positions directly in chamber coordinates -- the drift-distance term
        # (laterality * (ts - t0) * vd) is a pure physical offset in mm and is
        # translation-invariant, so it's valid to add it to the chamber-frame cell centers
        x_hits_ch = x_cell_ch + laterality * (ts - t0) * vd

        # extrapolate the track across the FULL chamber z-range, not just the
        # local 8-layer hit height, so it renders as an actual track through
        # the detector rather than a short segment
        z_range_glob = np.linspace(
            derived_params._dt_cell_coordinates[1][0][1][5] - params._cell_height * 2,
            derived_params._dt_cell_coordinates[3][3][1][5] + params._cell_height * 2,
            1000,
        )
        z_range_local = z_range_glob - z_ref_ch
        track_local = derived_params.f_x_muon(z=z_range_local, x0=x0, tan_alpha=tan_alpha)
        err_track_local = derived_params.err_f_x_muon(z=z_range_local, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        track_glob = track_local + x_ref_ch

        # re-draw the wire markers for the highlighted cells ON TOP of the
        # aqua cell fill, so they stay visible instead of being hidden by it
        ax.scatter(
            x_cell_ch, z_arr_ch, marker="o", s=wire_marker_size,
            color=wire_marker_color, zorder=4,
        )

        ax.errorbar(x=x_hits_ch, y=z_arr_ch, xerr=err_x_hits, color="tab:blue", marker="o", markersize=4, linestyle="", label=ts_label, zorder=5)
        ax.plot(track_glob, z_range_glob, linewidth=2, color="tab:red", label=fit_label, zorder=6)
        ax.fill_betweenx(x1=track_glob - err_track_local, x2=track_glob + err_track_local, y=z_range_glob, color="tab:red", alpha=0.2, zorder=0)

        ax.legend(prop={"size": 14}, fancybox=False, framealpha=params._legend_alpha, loc="center right")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))
        ax.set_title(
            f"DT chamber ($\\phi$ view) SL{sl1}+SL{sl2} track fit -- "
            f"{pct_ar}/{pct_co2} Ar/CO2, U_wire={u_wire}, {suffix}"
        )
        fig.tight_layout()

        path = f"{plot_save_path}{dataset_name}_detector_track_{suffix}_{idx}{plot_type}"
        fig.savefig(fname=path)
        plt.close(fig)
        paths["detector_track"] = path

        ################################
        ###### standalone SL1/SL3 fit tracks in chamber coordinates
        ###### (computed here so both the zoomed plot and plot 5 can use them)

        # each standalone SL fit is drawn using ITS OWN top wire (among its
        # own 4 layers) as the local-frame reference in chamber coordinates --
        # same convention as plot 3 above, just restricted to one SL's cells
        top_wire_idx_sl1 = int(np.argmax(z_arr_ch[0:4]))
        x_ref_ch_sl1 = x_cell_ch[0:4][top_wire_idx_sl1]
        z_ref_ch_sl1 = z_arr_ch[0:4][top_wire_idx_sl1]

        top_wire_idx_sl3 = int(np.argmax(z_arr_ch[4:8]))
        x_ref_ch_sl3 = x_cell_ch[4:8][top_wire_idx_sl3]
        z_ref_ch_sl3 = z_arr_ch[4:8][top_wire_idx_sl3]

        z_range_sl1 = np.linspace(
            np.amin(z_arr_ch[0:4]) - params._cell_height * 2,
            np.amax(z_arr_ch[0:4]) + params._cell_height * 2,
            500,
        )
        z_range_sl3 = np.linspace(
            np.amin(z_arr_ch[4:8]) - params._cell_height * 2,
            np.amax(z_arr_ch[4:8]) + params._cell_height * 2,
            500,
        )

        track_sl1_local = derived_params.f_x_muon(z=z_range_sl1 - z_ref_ch_sl1, x0=x0_sl1, tan_alpha=tan_alpha_sl1)
        err_track_sl1_local = derived_params.err_f_x_muon(
            z=z_range_sl1 - z_ref_ch_sl1, x0=x0_sl1, tan_alpha=tan_alpha_sl1,
            err_x0=err_x0_sl1, err_tan_alpha=err_tan_alpha_sl1, corr_x0_tan_alpha=corr_x0_tan_alpha_sl1,
        )
        track_sl1_glob = track_sl1_local + x_ref_ch_sl1

        track_sl3_local = derived_params.f_x_muon(z=z_range_sl3 - z_ref_ch_sl3, x0=x0_sl3, tan_alpha=tan_alpha_sl3)
        err_track_sl3_local = derived_params.err_f_x_muon(
            z=z_range_sl3 - z_ref_ch_sl3, x0=x0_sl3, tan_alpha=tan_alpha_sl3,
            err_x0=err_x0_sl3, err_tan_alpha=err_tan_alpha_sl3, corr_x0_tan_alpha=corr_x0_tan_alpha_sl3,
        )
        track_sl3_glob = track_sl3_local + x_ref_ch_sl3

        ################################
        ###### plot 4: zoomed chamber view (x-axis only)

        paths["detector_track_zoom"] = None
        if zoom:
            fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(8, 10))
            ax_zoom = geoplot_utils.chamber_ax(
                ax=ax_zoom,
                orient=orient,
                cell_data=dt_cell_data,
                wire=True,
            )

            # same fix as the non-zoomed view: keep the wire dots visible
            # on top of the aqua-highlighted cells
            ax_zoom.scatter(
                x_cell_ch, z_arr_ch, marker="o", s=wire_marker_size,
                color=wire_marker_color, zorder=4,
            )

            ax_zoom.errorbar(
                x=x_hits_ch, y=z_arr_ch, xerr=err_x_hits,
                color="tab:blue", marker="o", markersize=7, linestyle="",
                label=ts_label, zorder=5,
            )
            ax_zoom.plot(track_glob, z_range_glob, linewidth=2, color="tab:red", label=fit_label, zorder=6)
            ax_zoom.fill_betweenx(
                y=z_range_glob, x1=track_glob - err_track_local, x2=track_glob + err_track_local,
                color="tab:red", alpha=0.2, zorder=0,
            )

            xmin = min(
                np.min(track_glob - err_track_local),
                np.min(x_hits_ch - err_x_hits),
            )
            xmax = max(
                np.max(track_glob + err_track_local),
                np.max(x_hits_ch + err_x_hits),
            )
            ax_zoom.set_xlim(xmin - zoom_margin, xmax + zoom_margin)
            ax_zoom.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))

            ax_zoom.set_xlabel("$x$ [mm]")
            ax_zoom.set_ylabel("$z$ [mm]")
            ax_zoom.set_title(f"DT chamber ($\\phi$ view) -- Zoomed track fit, {suffix}")
            ax_zoom.legend(
                prop={"size": 14}, fancybox=False, framealpha=params._legend_alpha, loc="center right",
            )
            fig_zoom.tight_layout()

            path_zoom = f"{plot_save_path}{dataset_name}_detector_track_zoom_{suffix}_{idx}{plot_type}"
            fig_zoom.savefig(fname=path_zoom)
            plt.close(fig_zoom)
            paths["detector_track_zoom"] = path_zoom

        ################################
        ###### plot 5 (NEW): chamber view comparing the super fit track
        ###### against the standalone sl1-only and sl3-only fit tracks
        ###### (track data computed earlier, above the zoom block)

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=True)

        ax.scatter(x_cell_ch, z_arr_ch, marker="o", s=wire_marker_size, color=wire_marker_color, zorder=4)
        ax.errorbar(x=x_hits_ch, y=z_arr_ch, xerr=err_x_hits, color="tab:blue", marker="o", markersize=4, linestyle="", label=ts_label, zorder=5)

        ax.plot(track_glob, z_range_glob, linewidth=2, color="tab:red",
                label=f"Super fit ($\\chi^2/N_{{df}}={chi2ndf:.2f}$)", zorder=6)
        ax.fill_betweenx(x1=track_glob - err_track_local, x2=track_glob + err_track_local, y=z_range_glob, color="tab:red", alpha=0.15, zorder=0)

        ax.plot(track_sl1_glob, z_range_sl1, linewidth=2, linestyle="--", color=sl1_fit_color,
                label=f"SL{sl1}-only fit ($\\chi^2/N_{{df}}={chi2ndf_sl1:.2f}$)", zorder=6)
        ax.fill_betweenx(x1=track_sl1_glob - err_track_sl1_local, x2=track_sl1_glob + err_track_sl1_local, y=z_range_sl1, color=sl1_fit_color, alpha=0.15, zorder=0)

        ax.plot(track_sl3_glob, z_range_sl3, linewidth=2, linestyle="--", color=sl2_fit_color,
                label=f"SL{sl2}-only fit ($\\chi^2/N_{{df}}={chi2ndf_sl3:.2f}$)", zorder=6)
        ax.fill_betweenx(x1=track_sl3_glob - err_track_sl3_local, x2=track_sl3_glob + err_track_sl3_local, y=z_range_sl3, color=sl2_fit_color, alpha=0.15, zorder=0)

        ax.legend(prop={"size": 13}, fancybox=False, framealpha=params._legend_alpha, loc="center right")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))
        ax.set_title(
            f"DT chamber ($\\phi$ view) -- super fit vs. standalone SL{sl1}/SL{sl2} fits, "
            f"{pct_ar}/{pct_co2} Ar/CO2, U_wire={u_wire}, {suffix}"
        )
        fig.tight_layout()

        path = f"{plot_save_path}{dataset_name}_detector_track_individual_fits_{suffix}_{idx}{plot_type}"
        fig.savefig(fname=path)
        plt.close(fig)
        paths["detector_track_individual_fits"] = path

        ################################
        ###### plot 5b (NEW): zoomed version of plot 5

        paths["detector_track_individual_fits_zoom"] = None
        if zoom:
            fig_zoom2, ax_zoom2 = plt.subplots(1, 1, figsize=(8, 10))
            ax_zoom2 = geoplot_utils.chamber_ax(ax=ax_zoom2, orient=orient, cell_data=dt_cell_data, wire=True)

            ax_zoom2.scatter(x_cell_ch, z_arr_ch, marker="o", s=wire_marker_size, color=wire_marker_color, zorder=4)
            ax_zoom2.errorbar(x=x_hits_ch, y=z_arr_ch, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label, zorder=5)

            ax_zoom2.plot(track_glob, z_range_glob, linewidth=2, color="tab:red",
                          label=f"Super fit ($\\chi^2/N_{{df}}={chi2ndf:.2f}$)", zorder=6)
            ax_zoom2.fill_betweenx(x1=track_glob - err_track_local, x2=track_glob + err_track_local, y=z_range_glob, color="tab:red", alpha=0.15, zorder=0)

            ax_zoom2.plot(track_sl1_glob, z_range_sl1, linewidth=2, linestyle="--", color=sl1_fit_color,
                          label=f"SL{sl1}-only fit ($\\chi^2/N_{{df}}={chi2ndf_sl1:.2f}$)", zorder=6)
            ax_zoom2.fill_betweenx(x1=track_sl1_glob - err_track_sl1_local, x2=track_sl1_glob + err_track_sl1_local, y=z_range_sl1, color=sl1_fit_color, alpha=0.15, zorder=0)

            ax_zoom2.plot(track_sl3_glob, z_range_sl3, linewidth=2, linestyle="--", color=sl2_fit_color,
                          label=f"SL{sl2}-only fit ($\\chi^2/N_{{df}}={chi2ndf_sl3:.2f}$)", zorder=6)
            ax_zoom2.fill_betweenx(x1=track_sl3_glob - err_track_sl3_local, x2=track_sl3_glob + err_track_sl3_local, y=z_range_sl3, color=sl2_fit_color, alpha=0.15, zorder=0)

            xmin = min(
                np.min(track_glob - err_track_local),
                np.min(track_sl1_glob - err_track_sl1_local),
                np.min(track_sl3_glob - err_track_sl3_local),
                np.min(x_hits_ch - err_x_hits),
            )
            xmax = max(
                np.max(track_glob + err_track_local),
                np.max(track_sl1_glob + err_track_sl1_local),
                np.max(track_sl3_glob + err_track_sl3_local),
                np.max(x_hits_ch + err_x_hits),
            )
            ax_zoom2.set_xlim(xmin - zoom_margin, xmax + zoom_margin)
            ax_zoom2.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))

            ax_zoom2.set_xlabel("$x$ [mm]")
            ax_zoom2.set_ylabel("$z$ [mm]")
            ax_zoom2.set_title(f"DT chamber ($\\phi$ view) -- Zoomed, super vs. standalone SL{sl1}/SL{sl2} fits, {suffix}")
            ax_zoom2.legend(prop={"size": 12}, fancybox=False, framealpha=params._legend_alpha, loc="center right")
            fig_zoom2.tight_layout()

            path_zoom2 = f"{plot_save_path}{dataset_name}_detector_track_individual_fits_zoom_{suffix}_{idx}{plot_type}"
            fig_zoom2.savefig(fname=path_zoom2)
            plt.close(fig_zoom2)
            paths["detector_track_individual_fits_zoom"] = path_zoom2

        ################################
        ###### plot 6 (NEW): same style/structure as plot 1 (ts_vs_fit),
        ###### but using the standalone sl1-only / sl3-only fit predictions
        ###### instead of the super fit

        fit_ts_indiv = np.concatenate([fit_ts_sl1_own, fit_ts_sl3_own])
        err_fit_ts_indiv = np.concatenate([err_fit_ts_sl1_own, err_fit_ts_sl3_own])

        fit_label_indiv = f"""Standalone SL fits:
SL{sl1}: $T_0=({np.round(t0_sl1,0):.0f}\\pm{np.round(err_t0_sl1,0):.0f})$ {params._key_units['t0']}, $\\chi^2/N_{{df}}={np.round(chi2ndf_sl1,2):.2f}$
SL{sl2}: $T_0=({np.round(t0_sl3,0):.0f}\\pm{np.round(err_t0_sl3,0):.0f})$ {params._key_units['t0']}, $\\chi^2/N_{{df}}={np.round(chi2ndf_sl3,2):.2f}$
Track ID = {idx}"""

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(5, 1))
        ax[0].errorbar(x=lys - 0.04, y=ts, yerr=err_ts, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        ax[0].errorbar(x=lys + 0.04, y=fit_ts_indiv, yerr=err_fit_ts_indiv, color="tab:red", marker="v", markersize=7, linestyle="", label=fit_label_indiv)
        ax[0].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[0].set_ylabel("Timestamp $T_{ly}$ [TU]")
        ax[0].legend(prop={"size": 16}, fancybox=False, framealpha=params._legend_alpha)
        ax[0].set_title(f"SL {sl1} + SL {sl2}, standalone fits, Patterns {pat_name_sl1}/{pat_name_sl2}, Laterality {[int(l) for l in laterality]}")
        ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        residuals_indiv = ts - fit_ts_indiv
        err_residuals_indiv = err_ts
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[1].errorbar(x=lys, y=residuals_indiv, yerr=err_residuals_indiv, color="black", marker="o", markersize=7, linestyle="")
        y_span = np.amax(np.abs(residuals_indiv) + err_residuals_indiv) * 1.1
        ax[1].set_ylim(-y_span, y_span)
        ax[1].set_ylabel("Residuals [TU]")
        ax[1].set_xlabel(f"Layer $ly$ (0-3: SL{sl1}, 4-7: SL{sl2})")
        ax[1].set_xticks([i for i in range(8)])
        ax[1].set_xticklabels([f"{i}" for i in range(8)])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)

        path = f"{plot_save_path}{dataset_name}_ts_vs_fit_individual_fits_{suffix}_{idx}{plot_type}"
        fig.savefig(fname=path)
        plt.close(fig)
        paths["ts_vs_fit_individual_fits"] = path

        saved_paths[idx] = paths
        

        ################################
        ###### fine-binned histogram of x0, converted into actual chamber-frame
        ###### x-coordinates [mm] -- with dead-cell regions (zero hits) shaded red

        x0_arr  = super_fits_cuts["x0" + fit_suffix]
        sl1_arr = super_fits_cuts["sl1"]
        sl3_arr = super_fits_cuts["sl3"]
        wi3_sl1_arr = super_fits_cuts["wi3_sl1"]
        wi3_sl3_arr = super_fits_cuts["wi3_sl3"]

        x0_glob_arr = np.empty(len(x0_arr), dtype=np.float64)
        for i in range(len(x0_arr)):
            sl1_val = int(sl1_arr[i])
            sl3_val = int(sl3_arr[i])
            wi3_sl1 = int(wi3_sl1_arr[i])
            wi3_sl3 = int(wi3_sl3_arr[i])

            x_sl1_top = derived_params._dt_cell_coordinates[sl1_val][3][wi3_sl1][ref_axis + 3]
            z_sl1_top = derived_params._dt_cell_coordinates[sl1_val][3][wi3_sl1][5]
            x_sl3_top = derived_params._dt_cell_coordinates[sl3_val][3][wi3_sl3][ref_axis + 3]
            z_sl3_top = derived_params._dt_cell_coordinates[sl3_val][3][wi3_sl3][5]
            x_ref_ch = x_sl1_top if z_sl1_top > z_sl3_top else x_sl3_top

            x0_glob_arr[i] = x0_arr[i] + x_ref_ch

        # --- SL3 wire positions (layer 0, as a representative x-per-wire mapping) ---
        sl3_wire_dict = derived_params._dt_cell_coordinates[3][0]
        sl3_wires_sorted = sorted(sl3_wire_dict.keys())
        sl3_x_vals = np.array([sl3_wire_dict[wi][ref_axis + 3] for wi in sl3_wires_sorted])
        x_lo, x_hi = np.amin(sl3_x_vals), np.amax(sl3_x_vals)

        # --- per-cell counts: nearest-wire assignment of every x0_glob value,
        # used ONLY to find dead cells, not for the plotted histogram itself ---
        cell_edges = np.empty(len(sl3_x_vals) + 1)
        cell_edges[1:-1] = (sl3_x_vals[:-1] + sl3_x_vals[1:]) / 2.0
        cell_edges[0] = sl3_x_vals[0] - (cell_edges[1] - sl3_x_vals[0])
        cell_edges[-1] = sl3_x_vals[-1] + (sl3_x_vals[-1] - cell_edges[-2])

        cell_hit_counts, _ = np.histogram(x0_glob_arr, bins=cell_edges)
        dead_cell_mask = cell_hit_counts == 0
        n_dead = int(np.sum(dead_cell_mask))

        # --- fine-binned histogram (unchanged from before) ---
        x0_glob_hist_data = build_hist_general(
            data_list=x0_glob_arr,
            n_bins=500,
            edge_min=x_lo - 20,
            edge_max=x_hi + 20,
            verbose=False,
        )

        filename_suffix_x0g = f"x0_globframe_{suffix}"

        fig_x0g, ax_x0g, _ = plot_hist_general(
            specific_data=x0_glob_hist_data,
            dataset_name=dataset_name,
            plot_save_path=plot_save_path,
            xlabel="$x_0$ [mm] (chamber frame)",
            ylabel="counts",
            filename_suffix=filename_suffix_x0g,
            plot_type=plot_type,
            title=f"Fitted track $x_0$ position in chamber frame, {suffix}",
            xlim=(x_lo - 20, x_hi + 20),
            save=False,          # save after adding the dead-cell shading
        )

        path_x0g = f"{plot_save_path}{dataset_name}_{filename_suffix_x0g}{plot_type}"

        # --- shade dead-cell x-ranges in red ---
        dead_indices = np.where(dead_cell_mask)[0]
        first_label = True
        for idx in dead_indices:
            lo, hi = cell_edges[idx], cell_edges[idx + 1]
            ax_x0g.axvspan(
                lo, hi, color="red", alpha=0.25, zorder=0,
                label="dead cell (0 fits)" if first_label else None,
            )
            first_label = False

        if n_dead > 0:
            ax_x0g.legend(loc="upper right", fontsize=10)

        fig_x0g.tight_layout()
        fig_x0g.savefig(path_x0g)   # save unconditionally now, path is always valid
        plt.close(fig_x0g)

        saved_paths["rate_plots"] = {
            #"top_ref_wire_counts": path_top_ref,
            "x0_globframe_hist": path_x0g,
            "n_dead_cells": n_dead,
        }

    return saved_paths

def fit_gaussian_hist(
    *,
    specific_data,
    dataset_name,
    plot_save_path,
    plot_type=".png",
    xlabel,
    ylabel="counts",
    filename_suffix="ALL",
    start_idx=0,
    scale_factor=vd_factor,
    log_scale=False,
    power_limits=[-4, 4],
    bin_unit=f"$\\mu$m/ns",
    add_info=True,
    legend_font_size=13,
    fig_size=(9, 8),          # slightly taller to fit residual panel
    xlim=None,
    hist_key="hist",
    err_hist_key="err_hist",
    err_hist_down_key="err_hist_down",
    err_hist_up_key="err_hist_up",
    edges_key="edges",
    overflow_key="overflow",
    underflow_key="underflow",
    save=True,
    verbose=True,
    title="",
    p0=None,
    fit_range=None,
    fit_color="red",
    component_colors=("tab:orange", "tab:green"),
    show_components=True,
    show_residuals=True,      # NEW: toggle the residual sub-panel
    residual_height_ratio=1,  # NEW: relative height of residual panel vs main panel (main=3)
    max_retries=5,            # NEW: number of alternate-p0 attempts if errors/chi2 are bad
    chi2_ndf_max= 2 ,         # NEW: refit if chi2/ndf exceeds this
    ):
    #from scipy.optimize import curve_fit

    def gaussian(x, amplitude, mean, sigma):
        return np.abs(amplitude) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
        return gaussian(x, amp1, mean1, sigma1) + gaussian(x, amp2, mean2, sigma2)

    # --- read data ---
    hist = np.array(specific_data[hist_key])[start_idx:]
    err_hist_down = np.array(specific_data[err_hist_down_key])[start_idx:]
    err_hist_up = np.array(specific_data[err_hist_up_key])[start_idx:]
    edges = np.array(specific_data[edges_key])[start_idx:] * scale_factor
    centers = hist_utils.centers_from_edges(edges)
    overflow = specific_data[overflow_key]
    underflow = specific_data[underflow_key]
    lo, hi = np.amin(centers), np.amax(centers)
    if verbose:
        print(f"Fitting double Gaussian to {dataset_name} ({filename_suffix})...")

    err_hist_sym = (err_hist_up + err_hist_down) / 2.0
    err_hist_safe = np.where(err_hist_sym <= 0, 1.0, err_hist_sym)

    if fit_range is not None:
        mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    else:
        mask = np.ones_like(centers, dtype=bool)

    n_params = 6
    n_fit_pts = int(np.sum(mask))
    ndf = max(n_fit_pts - n_params, 1)  # avoid div by zero

    if p0 is None:
        h_masked = hist[mask]
        c_masked = centers[mask]
        if h_masked.size and np.sum(h_masked) > 0:
            amp0 = np.max(h_masked)
            mean0 = np.average(c_masked, weights=h_masked)
            spread0 = np.sqrt(np.average((c_masked - mean0) ** 2, weights=h_masked))
        else:
            amp0 = np.max(hist) if hist.size else 1.0
            mean0 = np.mean(c_masked) if c_masked.size else 0.0
            spread0 = (np.amax(centers) - np.amin(centers)) / 6
        spread0 = spread0 if spread0 > 0 else (np.amax(centers) - np.amin(centers)) / 6
        span = np.amax(centers) - np.amin(centers) if centers.size else 1.0
        p0 = [
            amp0, mean0 - 0.1 * span, spread0 * 0.7,
            amp0 * 0.5, mean0 + 0.1 * span, spread0 * 1.5,
        ]
    else:
        span = np.amax(centers) - np.amin(centers) if centers.size else 1.0

    def compute_chi2(popt_):
        """chi2 over the masked/fitted points for a given parameter set."""
        model_vals = double_gaussian(centers[mask], *popt_)
        resid = (hist[mask] - model_vals) / err_hist_safe[mask]
        return float(np.sum(resid ** 2))
    
    def canonicalize_components(popt_, pcov_):
            """Ensure component 1 is always the narrower (smaller sigma) component."""
            amp1_, mean1_, sigma1_, amp2_, mean2_, sigma2_ = popt_
            if np.abs(sigma1_) > np.abs(sigma2_):
                # swap the two (amp, mean, sigma) triplets
                order = [3, 4, 5, 0, 1, 2]
                popt_ = popt_[order]
                if pcov_ is not None:
                    pcov_ = pcov_[np.ix_(order, order)]
            return popt_, pcov_
    
    # --- first attempt ---
    def run_fit(p0_guess):
            try:
                popt_, pcov_ = curve_fit(
                    double_gaussian,
                    centers[mask],
                    hist[mask],
                    p0=p0_guess,
                    sigma=err_hist_safe[mask],
                    absolute_sigma=True,
                    maxfev=10000,
                )
                popt_, pcov_ = canonicalize_components(popt_, pcov_)
                perr_ = np.sqrt(np.diag(pcov_))
                chi2_ = compute_chi2(popt_)
                chi2_ndf_ = chi2_ / ndf
                return popt_, pcov_, perr_, chi2_ndf_
            except (RuntimeError, ValueError):
                return None, None, None, None

    def bad_result(perr_, chi2_ndf_):
        if perr_ is None or not np.all(np.isfinite(perr_)):
            return True
        if chi2_ndf_ is None or not np.isfinite(chi2_ndf_) or chi2_ndf_ > chi2_ndf_max:
            return True
        return False


    popt, pcov, perr, chi2_ndf = run_fit(p0)

    # --- retry with alternate starting parameters if errors are inf/nan or chi2/ndf too high ---
    if bad_result(perr, chi2_ndf):
        if verbose:
            if popt is None:
                reason = "fit did not converge"
            elif not np.all(np.isfinite(perr)):
                reason = "fit gave non-finite errors"
            else:
                reason = f"chi2/ndf too high ({chi2_ndf:.3f} > {chi2_ndf_max})"
            print(f"  Initial fit: {reason} for {dataset_name} ({filename_suffix}); trying alternate starting parameters...")

        # keep the best-so-far attempt around, in case no retry clears the threshold
        best_popt, best_pcov, best_perr, best_chi2_ndf = popt, pcov, perr, chi2_ndf

        rng = np.random.default_rng(0)
        attempt = 0
        while bad_result(perr, chi2_ndf) and attempt < max_retries:
            jitter = 1 + rng.uniform(-0.3, 0.3, size=n_params)
            alt_p0 = np.array(p0, dtype=float) * jitter
            popt_try, pcov_try, perr_try, chi2_ndf_try = run_fit(alt_p0)

            # track the best valid attempt by chi2/ndf, even if it doesn't clear the threshold
            if perr_try is not None and np.all(np.isfinite(perr_try)) and chi2_ndf_try is not None and np.isfinite(chi2_ndf_try):
                if best_chi2_ndf is None or not np.isfinite(best_chi2_ndf) or chi2_ndf_try < best_chi2_ndf:
                    best_popt, best_pcov, best_perr, best_chi2_ndf = popt_try, pcov_try, perr_try, chi2_ndf_try

            if not bad_result(perr_try, chi2_ndf_try):
                popt, pcov, perr, chi2_ndf = popt_try, pcov_try, perr_try, chi2_ndf_try
                if verbose:
                    print(f"  Refit succeeded on attempt {attempt + 1} for {dataset_name} ({filename_suffix}) "
                          f"(chi2/ndf = {chi2_ndf:.3f}).")
                break
            attempt += 1

        if bad_result(perr, chi2_ndf):
            # no attempt cleared the threshold -- fall back to the best one we saw
            if best_popt is not None:
                popt, pcov, perr, chi2_ndf = best_popt, best_pcov, best_perr, best_chi2_ndf
                if verbose:
                    print(f"  No refit attempt met chi2/ndf <= {chi2_ndf_max} for {dataset_name} ({filename_suffix}); "
                          f"using best available (chi2/ndf = {chi2_ndf:.3f}).")
            else:
                if verbose:
                    print(f"  All {max_retries} refit attempts failed outright for {dataset_name} ({filename_suffix}).")
                popt = np.full(n_params, np.nan)
                perr = np.full(n_params, np.nan)
                pcov = None
                chi2_ndf = np.nan

    amp1, mean1, sigma1, amp2, mean2, sigma2 = popt
    sigma1, sigma2 = np.abs(sigma1), np.abs(sigma2)
    amp1_err, mean1_err, sigma1_err, amp2_err, mean2_err, sigma2_err = perr

    # --- chi2 / ndf from the fitted (masked) points ---
    chi2 = chi2_ndf * ndf if np.isfinite(chi2_ndf) else np.nan
    chi2_ndf = chi2_ndf
    residuals = None       # raw residuals over ALL bins (for plotting)
    pull = None             # residuals / error, i.e. "sigma away from fit", over ALL bins

    if not np.isnan(mean1):
        fit_vals_masked = double_gaussian(centers[mask], *popt)
        resid_masked = hist[mask] - fit_vals_masked
        pull_masked = resid_masked / err_hist_safe[mask]

        n_points = np.sum(mask)
        ndf = n_points - n_params
        chi2 = np.sum(pull_masked ** 2)
        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

        if verbose:
            print(f"  chi2 / ndf = {chi2:.4g} / {ndf} = {chi2_ndf:.4g}")

        # residuals/pull over the full (unmasked) range, for the residual panel
        fit_vals_all = double_gaussian(centers, *popt)
        residuals = hist - fit_vals_all
        pull = residuals / err_hist_safe

    if verbose and not np.isnan(mean1):
        print(f"  component 1: amp = {amp1:.4g} +/- {amp1_err:.4g}, "
            f"mean = {mean1:.4g} +/- {mean1_err:.4g}, sigma = {sigma1:.4g} +/- {sigma1_err:.4g}")
        print(f"  component 2: amp = {amp2:.4g} +/- {amp2_err:.4g}, "
            f"mean = {mean2:.4g} +/- {mean2_err:.4g}, sigma = {sigma2:.4g} +/- {sigma2_err:.4g}")

    if show_residuals:
            fig, (ax, ax_res) = plt.subplots(
                2, 1, figsize=fig_size, sharex=True,
                gridspec_kw={"height_ratios": [3, residual_height_ratio], "hspace": 0.05},
            )
    else:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax_res = None

    # --- NEW: build the hist-info string ourselves (plot_histogram won't draw it) ---
    barwidth = np.mean(np.diff(centers))
    barwidth_str = f"{barwidth:.3g}"
    if bin_unit:
        barwidth_str += f" {bin_unit}"
    entries_total = int(np.sum(hist))
    info_str = (
        f"entries = {entries_total}\n"
        f"underflow = {underflow}\n"
        f"overflow = {overflow}\n"
        f"total = {entries_total + overflow + underflow}\n"
        f"bin count = {len(centers)}\n"
        f"bin width = {barwidth_str}"
    )

    ax = hist_utils.plot_histogram(
        ax,
        hist=hist,
        centers=centers,
        err_hist_down=err_hist_down,
        err_hist_up=err_hist_up,
        log_scale=log_scale,
        power_limits=power_limits,
        add_info=False,          # <-- CHANGED: was `add_info` (or True); suppress the separate box
        entries=entries_total,
        overflow=overflow,
        underflow=underflow,
        bin_unit=bin_unit,
    )

    # --- NEW: collect everything into one legend ---
    handles, labels = [], []

    if add_info:  # only add the info entry if the caller still wants it shown
        info_handle = Line2D([], [], linestyle="none")
        handles.append(info_handle)
        labels.append(info_str)

    if not np.isnan(mean1):
        x_fit = np.linspace(np.amin(centers), np.amax(centers), 500)
        y_fit = double_gaussian(x_fit, *popt)
        fit_label = (
            f"Double Gaussian fit\n"
            f"($\\chi^2$/ndf = {chi2_ndf:.3f})\n"
            f"$\\mu_1$ = ({mean1:.2f} $\\pm$ {mean1_err:.2f}) $\\mu$m/ns\n"
            f"$\\sigma_1$ = ({sigma1:.2f} $\\pm$ {sigma1_err:.2f}) $\\mu$m/ns\n"
            f"$\\mu_2$ = ({mean2:.2f} $\\pm$ {mean2_err:.2f}) $\\mu$m/ns\n"
            f"$\\sigma_2$ = ({sigma2:.2f} $\\pm$ {sigma2_err:.2f}) $\\mu$m/ns"
        )
        (fit_line,) = ax.plot(x_fit, y_fit, color=fit_color, linewidth=2, label=fit_label)
        handles.append(fit_line)
        labels.append(fit_label)

        if show_components:
            y1 = gaussian(x_fit, amp1, mean1, sigma1)
            y2 = gaussian(x_fit, amp2, mean2, sigma2)
            (comp1_line,) = ax.plot(x_fit, y1, color=component_colors[0], linewidth=1.5,
                                    linestyle="--", label="component 1")
            (comp2_line,) = ax.plot(x_fit, y2, color=component_colors[1], linewidth=1.5,
                                    linestyle="--", label="component 2")
            handles += [comp1_line, comp2_line]
            labels += ["component 1", "component 2"]

    if handles:
        ax.legend(handles, labels, fontsize=legend_font_size, loc="center right")

    if xlim is None:
        ax.set_xlim(np.amin(centers), np.amax(centers))
    elif xlim is not False:
        ax.set_xlim(*xlim)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_residuals:
        ax.tick_params(labelbottom=False)  
    else:
        ax.set_xlabel(xlabel)

    # --- NEW: residual panel ---
    if show_residuals and ax_res is not None:
        if pull is not None:
            ax_res.errorbar(
                centers[::10], residuals[::10], yerr=err_hist_safe[::10], fmt="o", markersize=3,
                color="black", ecolor="gray", elinewidth=1, capsize=2,
            )
        ax_res.axhline(0, color=fit_color, linewidth=1, linestyle="--")
        ax_res.set_xlabel(xlabel)
        ax_res.set_ylabel(f"data-fit")
        if xlim is None:
            ax_res.set_xlim(np.amin(centers), np.amax(centers))
        elif xlim is not False:
            ax_res.set_xlim(*xlim)

    fig.tight_layout()

    path = None
    if save:
        path = f"{plot_save_path}{dataset_name}_{filename_suffix}_doublegaussfit{plot_type}"
        if verbose:
            print(f"store double-gaussian-fit plot as {path}.")
        fig.savefig(path)
        if verbose:
            print(f"Done saving fit plot as {path}\n")

    if not np.isnan(amp1) and amp1 >= amp2:
        dom_mean, dom_mean_err = mean1, mean1_err
        dom_sigma, dom_sigma_err = sigma1, sigma1_err
    else:
        dom_mean, dom_mean_err = mean2, mean2_err
        dom_sigma, dom_sigma_err = sigma2, sigma2_err

    fit_results = {
        "amplitude_1": amp1, "amplitude_1_err": amp1_err,
        "mean_1": mean1, "mean_1_err": mean1_err,
        "sigma_1": sigma1, "sigma_1_err": sigma1_err,
        "amplitude_2": amp2, "amplitude_2_err": amp2_err,
        "mean_2": mean2, "mean_2_err": mean2_err,
        "sigma_2": sigma2, "sigma_2_err": sigma2_err,
        "mean": dom_mean, "mean_err": dom_mean_err,
        "sigma": dom_sigma, "sigma_err": dom_sigma_err,
        "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2_ndf,   
        "residuals": residuals, "pull": pull,             
        "popt": popt,
        "pcov": pcov,
    }

    return fig, ax, path, fit_results

def fit_parabola_peak(
    *,
    specific_data,
    dataset_name,
    plot_save_path,
    xlabel,
    ylabel="counts",
    title="",
    filename_suffix="ALL",
    fit_half_width=5,
    fit_half_range = 1,
    scale_factor=1,
    min_bins_syst = 7,
    max_bins_syst = 18,

    ):
    # Draw histogram using your existing function
    fig, ax, path = plot_hist_general(
        specific_data=specific_data,
        dataset_name=dataset_name,
        plot_save_path=plot_save_path,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        filename_suffix=filename_suffix,
        scale_factor=scale_factor,
        save=False,          # save after adding fit


    )

    # Read histogram
    hist = np.asarray(specific_data["hist"])
    err = 0.5 * (
        np.asarray(specific_data["err_hist_up"])
        + np.asarray(specific_data["err_hist_down"])
    )
    err = np.where(err <= 0, 1.0, err)

    edges = np.asarray(specific_data["edges"]) * scale_factor
    centers = hist_utils.centers_from_edges(edges)

    # Maximum bin
    peak_fraction = 0.80
    n_consecutive = 4

    imax = np.argmax(hist)
    threshold = peak_fraction * hist[imax]

    # Search left
    lo = imax
    count = 0
    while lo > 0:
        lo -= 1
        if hist[lo] < threshold:
            count += 1
            if count >= n_consecutive:
                lo += n_consecutive - 1  # last bin above threshold
                break
        else:
            count = 0

    # Search right
    hi = imax
    count = 0
    while hi < len(hist) - 1:
        hi += 1
        if hist[hi] < threshold:
            count += 1
            if count >= n_consecutive:
                hi -= n_consecutive - 1  # last bin above threshold
                break
        else:
            count = 0

    x = centers[lo:hi+1]
    y = hist[lo:hi+1]
    sigma = err[lo:hi+1]

    # Parabola
    def parabola(x, a, b, c):
        return a*x**2 + b*x + c

    popt, pcov = curve_fit(
        parabola,
        x,
        y,
        sigma=sigma,
        absolute_sigma=True,
    )

    a, b, c = popt

    peak = -b/(2*a)

    # error propagation
    da = b/(2*a**2)
    db = -1/(2*a)

    peak_err = np.sqrt(
        da**2 * pcov[0,0]
        + db**2 * pcov[1,1]
        + 2*da*db*pcov[0,1]
    )



    # -----------------------------------------
    # Systematic uncertainty from fit window
    # -----------------------------------------

    mu_scan = []

    bin_width = np.mean(np.diff(centers))

    for n_bins in range(min_bins_syst, max_bins_syst + 1):

        half_width = n_bins * bin_width

        mask = (
            (centers >= peak - half_width) &
            (centers <= peak + half_width)
        )
        #print(mask)
        x_syst = centers[mask]
        y_syst = hist[mask]
        err_syst = err[mask]

        if len(x_syst) < 3:
            continue

        try:
            popt_syst, _ = curve_fit(
                parabola,
                x_syst,
                y_syst,
                sigma=err_syst,
                absolute_sigma=True,
            )

            a_syst, b_syst, c_syst = popt_syst

            # Reject non-parabolic fits
            if a_syst >= 0:
                continue

            mu_syst = -b_syst / (2*a_syst)


            # Best-fit values
            fit = parabola(x_syst, *popt_syst)

            # Chi-square
            chi2 = np.sum(((y_syst - fit) / err_syst)**2)

            # Degrees of freedom
            ndf = len(x_syst) - len(popt_syst)

            # Reduced chi-square
            chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

            print(f"n_bins={n_bins:2d}, chi2/ndf={chi2_ndf:.2f}, mu={mu_syst:.5f}")
            print(chi2_ndf)
            mu_scan.append(mu_syst)

        except RuntimeError:
            continue



    # systematic uncertainty
    if len(mu_scan) > 1:
        syst_err = np.std(mu_scan, ddof=1)
    else:
        syst_err = 0.0


    # -----------------------------------------
    # Calculate systematic uncertainty
    # -----------------------------------------
    if len(mu_scan) > 1:
        mu_scan = np.array(mu_scan)

        # Option 1: rms
        err_mu_syst = np.sqrt(np.mean((np.asarray(mu_scan)-peak)**2))

        # Option 2: half the full range (more conservative)
        # err_mu_syst = 0.5 * (np.max(mu_scan) - np.min(mu_scan))

    else:
        err_mu_syst = 0.0

    # Plot fit
    xx = np.linspace(x[0], x[-1], 200)
    ax.plot(xx, parabola(xx, *popt), "r-", lw=2,
            label=f"Parabola\nPeak = {peak:.3f} ± {peak_err:.3f}")

    ax.axvline(peak, color="red", ls="--", alpha=0.7)

    ax.legend()
    plt.close(fig)
    # Save
    path = f"{plot_save_path}{dataset_name}_{filename_suffix}_parabolafit.png"
    fig.savefig(path)


    tot_err = np.sqrt(peak_err**2 + err_mu_syst**2)
    print(peak_err)
    print(tot_err)

    fit_results = {
        "peak": peak,
        "peak_err": peak_err,
        "popt": popt,
        "pcov": pcov,
        "tot_err": tot_err,
        "syst_err": err_mu_syst,
    }

    return fig, ax, path, fit_results


def muon_heatmap_from_fits(
    *,
    fits_cuts,
    plot_save_path,
    dataset_name,
    fit_suffix="_free_vd_super_fit",
    plot_type=".pdf",
    orient="phi",
    z_bin_width=5.0,
    x_bin_width=5.0,
    z_margin=100.0,
    x_margin=100.0,
    n_z_eval=1000,
    save=True,
    cut_suff = "",
    dataset_info = "",
    masked_cell_color="orange",
    dead_cell_color="tab:red",
):
    """
    Build an x-z occupancy heatmap of incoming muons directly from fitted
    tracks, with no dt_muons reconstruction file required.
 
    Every muon's straight-line trajectory x(z) is reconstructed purely
    from its per-fit parameters (x0, tan_alpha, selected by `fit_suffix`)
    combined with that event's own geometric reference point -- the same
    top-wire convention used in detector_track()'s chamber-view plots,
    rebuilt from derived_params._dt_cell_coordinates (the same source the
    chamber background is drawn from, so track and geometry line up).
    The track is then extrapolated across the full chamber z-height and
    every event's extrapolated line is accumulated into a single 2D
    histogram -- effectively "unrolling" every fit into the positions it
    would have crossed at every z, rather than only its own 4-8 hit
    layers.
 
    Because these super-pattern fits only constrain the x-z (phi)
    projection (SL1 + SL3), this only produces the x-z view. There is no
    y/theta information without an equivalent SL2 (theta-view) fit
    dataset.
 
    Masked (known dead/low-occupancy) wires are read directly from
    params._dt_wire_mask -- {sl: {ly: [wire_ids]}} -- and shaded red on
    top of the chamber geometry.
 
    Parameters
    ----------
    fits_cuts : dict of arrays
        The (already cut) fit dataset, same format as detector_track's
        `super_fits_cuts`: per-fit arrays for sl1, sl3, wi3_sl1, wi3_sl3,
        and the fitted x0/tan_alpha selected by `fit_suffix`.
    plot_save_path : str
        Directory the figure is saved into (created if missing).
    dataset_name : str
        Prefix used for the output filename.
    fit_suffix : str
        Which stored fit variant to use, e.g. "_free_vd_super_fit".
    plot_type : str
        File extension for the saved figure, e.g. ".pdf" or ".png".
    orient : str
        Passed to geoplot_utils.chamber_ax; "phi" is what SL1+SL3 fits
        support.
    z_bin_width, x_bin_width : float
        Bin widths [mm] for the heatmap.
    z_margin, x_margin : float
        Margins [mm] added around the chamber extent.
    n_z_eval : int
        Number of z points each track is evaluated at when extrapolated
        across the chamber (higher = smoother heatmap, slower/more
        memory: n_fits * n_z_eval floats).
    save : bool
        If False, the figure is built but not written to disk.
 
    Returns
    -------
    saved_paths : dict
        {"xz_heatmap": path}
    """

    saved_paths = {}
 
    n_fits = len(fits_cuts["sl1"])
    sl1_vals = fits_cuts["sl1"].astype(int)
    sl3_vals = fits_cuts["sl3"].astype(int)
    used_sls = sorted(set(np.unique(sl1_vals)).union(np.unique(sl3_vals)))
 
    x0_arr = np.asarray(fits_cuts["x0" + fit_suffix], dtype=np.float64)
    tan_alpha_arr = np.asarray(fits_cuts["tan_alpha" + fit_suffix], dtype=np.float64)
 
    wi3_sl1_arr = fits_cuts["wi3_sl1"].astype(int)
    wi3_sl3_arr = fits_cuts["wi3_sl3"].astype(int)
 
    # -----------------------------------------------------------------
    # per-event reference point (same top-wire convention as
    # detector_track's chamber-view plots), computed once and reused
    # -----------------------------------------------------------------
    x_sl1_top = np.empty(n_fits); z_sl1_top = np.empty(n_fits)
    x_sl3_top = np.empty(n_fits); z_sl3_top = np.empty(n_fits)
    for i in range(n_fits):
        x_sl1_top[i] = derived_params._dt_cell_coordinates[sl1_vals[i]][3][wi3_sl1_arr[i]][3]
        z_sl1_top[i] = derived_params._dt_cell_coordinates[sl1_vals[i]][3][wi3_sl1_arr[i]][5]
        x_sl3_top[i] = derived_params._dt_cell_coordinates[sl3_vals[i]][3][wi3_sl3_arr[i]][3]
        z_sl3_top[i] = derived_params._dt_cell_coordinates[sl3_vals[i]][3][wi3_sl3_arr[i]][5]
 
    use_sl1 = z_sl1_top > z_sl3_top
    x_ref_ch = np.where(use_sl1, x_sl1_top, x_sl3_top)
    z_ref_ch = np.where(use_sl1, z_sl1_top, z_sl3_top)
 
    # -----------------------------------------------------------------
    # extrapolate every fitted track across the full chamber z-range
    # -----------------------------------------------------------------
    z_chamber_min = min(derived_params.sl_z_min[sl] for sl in used_sls) - z_margin
    z_chamber_max = max(derived_params.sl_z_max[sl] for sl in used_sls) + z_margin
    z_eval = np.linspace(z_chamber_min, z_chamber_max, n_z_eval)
 
    z_local = z_eval[None, :] - z_ref_ch[:, None]                 # (n_fits, n_z_eval)
    track_local = derived_params.f_x_muon(z=z_local, x0=x0_arr[:, None], tan_alpha=tan_alpha_arr[:, None])
    track_glob = track_local + x_ref_ch[:, None]
 
    all_x = track_glob.ravel()
    all_z = np.broadcast_to(z_eval, (n_fits, n_z_eval)).ravel()
 
    # -----------------------------------------------------------------
    # x-z occupancy heatmap over all extrapolated fitted tracks
    # -----------------------------------------------------------------
    sl_x_coord = (min(derived_params.sl_x_min[sl] for sl in used_sls),
                  max(derived_params.sl_x_max[sl] for sl in used_sls))
    x_edges = np.arange(sl_x_coord[0] - x_margin, sl_x_coord[1] + x_margin, x_bin_width)
    z_edges = np.arange(z_chamber_min, z_chamber_max, z_bin_width)
    x_bins = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_bins = (z_edges[:-1] + z_edges[1:]) / 2.0
 
    hist2d, _, _ = np.histogram2d(x=all_z, y=all_x, bins=(z_edges, x_edges))
 
# pass 0: base chamber cell grid (all cells, neutral outline)
    dt_cell_data_base = dt_utils._chamber_data()
    for sl_key in dt_cell_data_base:
        for ly_key in dt_cell_data_base[sl_key]:
            for wi_key in dt_cell_data_base[sl_key][ly_key]:
                dt_cell_data_base[sl_key][ly_key][wi_key]["color"] = "white"

    # pass 1: full chamber geometry with dead cells (params._dt_dead_wires)
    dt_cell_data_dead = dt_utils._chamber_data()
    for sl_key in dt_cell_data_dead:
        for ly_key in dt_cell_data_dead[sl_key]:
            for wi_key in dt_cell_data_dead[sl_key][ly_key]:
                dt_cell_data_dead[sl_key][ly_key][wi_key]["color"] = "none"

    n_dead = 0
    for sl in used_sls:
        for ly, wire_ids in params._dt_dead_wires.get(sl, {}).items():
            for wi in wire_ids:
                if wi in dt_cell_data_dead[sl][ly]:
                    dt_cell_data_dead[sl][ly][wi]["color"] = dead_cell_color
                    n_dead += 1

    # pass 2: masked/noisy cells (params._dt_wire_mask)
    dt_cell_data_masked = dt_utils._chamber_data()
    for sl_key in dt_cell_data_masked:
        for ly_key in dt_cell_data_masked[sl_key]:
            for wi_key in dt_cell_data_masked[sl_key][ly_key]:
                dt_cell_data_masked[sl_key][ly_key][wi_key]["color"] = "none"

    n_masked = 0
    for sl in used_sls:
        for ly, wire_ids in params._dt_wire_mask.get(sl, {}).items():
            for wi in wire_ids:
                if wi in dt_cell_data_masked[sl][ly]:
                    dt_cell_data_masked[sl][ly][wi]["color"] = masked_cell_color
                    n_masked += 1

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im_obj = ax.imshow(X=hist2d, origin="lower",
                        extent=[x_bins.min(), x_bins.max(), z_bins.min(), z_bins.max()],
                        aspect="auto")

    # draw base cell grid, then dead cells, then overlay masked cells on top
    ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data_base, wire=False, transparent=True)
    ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data_dead, wire=False, transparent=True)
    ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data_masked, wire=False, transparent=True)
    ax.set_title(f"DT tracks (reconstructed from fits), {cut_suff}", fontsize=20)
    ax.set_ylabel("$z$ [mm]")
    ax.set_xlabel("$x$ [mm]")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3])
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, format=formatter)
    cbar.set_label("Track-crossing density\n(extrapolated fits)")

    legend_entries = {"Chamber geometry": mpatches.Patch(edgecolor="white", facecolor="none")}
    if n_dead > 0:
        legend_entries["Dead cells"] = mpatches.Patch(edgecolor=dead_cell_color, facecolor="none")
    if n_masked > 0:
        legend_entries["Masked/noisy cells"] = mpatches.Patch(edgecolor=masked_cell_color, facecolor="none")

    if n_dead > 0 or n_masked > 0:
        ax.legend(legend_entries.values(), legend_entries.keys(), prop={"size": 12}, loc="lower center",
                  fancybox=False, framealpha=params._legend_alpha)

    fig.tight_layout()

    path = f"{plot_save_path}{dataset_name}_muon_heatmap_fits_xz_CHAMBER_{cut_suff}{plot_type}"
    if save:
        fig.savefig(path)
        print(f"Detector heatmap saved to: {path}")
    plt.close(fig)
    saved_paths["xz_heatmap"] = path

    return saved_paths


def _sl_display(sl):
    """Human-readable label for an SL identifier, for titles/filenames."""
    return sl.upper() if isinstance(sl, str) else f"SL{sl}"

def _get_pattern_and_mask(
    *,
    data,
    sl,
    pattern_key,
    impossible_key,
    sl_selector_key,
    exclude_impossible,
    ):
    """
    Pull the pattern-type array for one SL out of `data`, handling two
    possible data layouts, auto-detected per call:
 
    - "wide" layout (e.g. the super_fit dataset): a separate key per SL,
      built as f"{pattern_key}_{sl}" (and f"{impossible_key}_{sl}" for the
      validity flag), e.g. "pat_type_sl1" / "impossible_sl1".
 
    - "long" layout (e.g. the single-SL sl_fit dataset): one shared
      `pattern_key` column (e.g. "pat_type") and one shared `impossible_key`
      column (e.g. "impossible"), both selected via a per-row
      `sl_selector_key` column (e.g. "sl") matching the given `sl` value.
 
    Returns
    -------
    pat_type_valid : np.ndarray
        pattern-type values for this SL, with impossible fits already
        removed (if exclude_impossible and the flag is available).
    n_excluded : int
    n_total_for_sl : int
        entry count for this SL before excluding impossible fits.
    """
    wide_pat_key = f"{pattern_key}_{sl}"
    wide_imp_key = f"{impossible_key}_{sl}"
 
    if wide_pat_key in data:
        # --- wide layout ---
        pat_type_arr = np.asarray(data[wide_pat_key])
        if exclude_impossible and wide_imp_key in data:
            impossible_arr = np.asarray(data[wide_imp_key]).astype(bool)
        else:
            impossible_arr = np.zeros(len(pat_type_arr), dtype=bool)
 
    elif pattern_key in data and sl_selector_key in data:
        # --- long layout ---
        sl_arr = np.asarray(data[sl_selector_key])
        sl_mask = sl_arr == sl
        pat_type_arr = np.asarray(data[pattern_key])[sl_mask]
        if exclude_impossible and impossible_key in data:
            impossible_arr = np.asarray(data[impossible_key])[sl_mask].astype(bool)
        else:
            impossible_arr = np.zeros(len(pat_type_arr), dtype=bool)
 
    else:
        raise KeyError(
            f"Could not find pattern data for sl={sl!r}: neither '{wide_pat_key}' "
            f"(wide layout: separate key per SL) nor the pair "
            f"('{pattern_key}', '{sl_selector_key}') (long layout: shared "
            "column + SL selector column) are present in `data`. Pass "
            "pattern_key / sl_selector_key explicitly if your keys are "
            "named differently."
        )
 
    valid_mask = ~impossible_arr
    return pat_type_arr[valid_mask], int(np.sum(impossible_arr)), int(len(pat_type_arr))
 
def _auto_detect_sl_list(*, data, pattern_key, sl_selector_key):
    """
    Figure out which SLs to analyze when `sl_list` isn't given explicitly:
    - wide layout -> collect every suffix from keys named f"{pattern_key}_*"
    - long layout -> collect every unique value in data[sl_selector_key]
    """
    wide_prefix = pattern_key + "_"
    wide_suffixes = [k[len(wide_prefix):] for k in data.keys() if k.startswith(wide_prefix)]
    if wide_suffixes:
        return tuple(sorted(wide_suffixes))
 
    if sl_selector_key in data:
        unique_sls = np.unique(np.asarray(data[sl_selector_key]))
        return tuple(int(s) if np.issubdtype(unique_sls.dtype, np.integer) else s for s in unique_sls)
 
    raise ValueError(
        "Could not auto-detect sl_list from `data` (no "
        f"'{pattern_key}_<sl>' keys and no '{sl_selector_key}' selector "
        "column found). Pass sl_list explicitly."
    )
 
def analyze_pattern_type_data(
    data,
    dataset_name,
    plot_save_path,
    plot_type=".png",
    save_plots=True,
    verbose=True,
    sl_list=None,
    pattern_key="pat_type",
    impossible_key="impossible",
    sl_selector_key="sl",
    pattern_labels=None,
    exclude_impossible=True,
    duration_seconds=None,
    duration_key="ts0",
    dataset_info = "",
    suffix = "",
    fit_type = "",
    add_title_info = "",



    ):
    """
    Run the pattern-type occupancy/rate analysis for one dataset, for each
    SL in `sl_list`. Histograms the pattern values, converts to rates, and
    produces comparison plots + a tex table -- mirroring the structure of
    `analyze_specific_data`.
 
    Works with either of two data layouts, auto-detected per SL (mixing is
    even fine within the same call, though that would be unusual):
 
    - "wide" layout, e.g. the super_fit dataset: a separate key per SL,
      "pat_type_sl1" / "pat_type_sl3", "impossible_sl1" / "impossible_sl3".
      Here `sl_list` entries are the key suffixes, e.g. ("sl1", "sl3").
 
    - "long" layout, e.g. the single-SL sl_fit dataset: one shared
      "pat_type" column and one shared "impossible" column, both selected
      per-SL via a shared "sl" column (data["sl"] == sl). Here `sl_list`
      entries are the values found in the "sl" column, e.g. (1, 2, 3).
 
    All of the relevant key names (pattern_key, impossible_key,
    sl_selector_key) are overridable if your dataset names them
    differently.
 
    Parameters
    ----------
    data : dict
        Loaded fit pickle (or a cuts-filtered slice of it), in either the
        "wide" or "long" layout described above. Must also contain a
        per-row timestamp array under `duration_key` (default "ts0") to
        derive the run duration -- NOT "muon_ts", which has been observed
        to be an unpopulated/placeholder field (all zeros) in at least one
        of these pipelines' outputs; "ts0" etc. are the actual populated
        hit timestamps used throughout the rest of the analysis.
    dataset_name : str
        Name of the dataset, used for plot titles / filenames.
    plot_save_path : str
        Directory the plots get saved to (created if it doesn't exist).
    plot_type : str, default ".png"
        File extension (including dot) used for all saved plots.
    save_plots : bool, default True
        If True, plots are saved to `plot_save_path`.
    verbose : bool, default True
        If True, all info/status messages are also printed to stdout.
        Regardless of this flag, every message is collected in the
        returned `log` list.
    sl_list : tuple, optional
        Which SLs to analyze. If None (default), auto-detected from
        `data`: every f"{pattern_key}_<suffix>" key found (wide layout),
        or every unique value in data[sl_selector_key] (long layout).
    pattern_key : str, default "pat_type"
        Base name of the pattern-type field. Wide layout looks for
        f"{pattern_key}_{sl}"; long layout looks for data[pattern_key]
        filtered by the sl selector column.
    impossible_key : str, default "impossible"
        Base name of the fit-validity flag field, same wide/long lookup
        pattern as `pattern_key`.
    sl_selector_key : str, default "sl"
        Column used to select rows per SL in the long layout.
    pattern_labels : list, optional
        Labels for the pattern ids, in id order (label at index p == the
        pattern with value p). Defaults to ["0", "1", "2", "3", "4", "5"].
    exclude_impossible : bool, default True
        If True and the impossible-flag field is available for a given SL
        (wide or long layout), fits flagged as impossible are excluded
        from the counts/rates.
    duration_seconds : float, optional
        Run duration to use for the rate calculation. If None (default),
        it is derived from `data[duration_key]` (max - min, *0.78e-9). Pass
        this explicitly if you track the run duration separately (e.g.
        from a run-info file) rather than deriving it per-call.
        A ValueError is raised if the derived duration is 0 or the array
        has fewer than 2 entries -- that indicates `duration_key` is
        missing, unfiltered, or otherwise out of sync with the rest of
        `data`, which is worth fixing at the source rather than working
        around here.
    duration_key : str, default "ts0"
        Key in `data` holding a per-row timestamp array (in the same TU
        units as the rest of the pipeline), used to derive the run
        duration when `duration_seconds` is not given.
 
    Returns
    -------
    results : dict
        {
            "log": list[str],
            "duration_seconds": float,
            "pat_counts": dict,      # pat_counts[sl][pattern_id | "com"]
            "pat_rate": dict,        # pat_rate[sl][pattern_id | "com"]  (Hz)
            "err_pat_rate": dict,    # err_pat_rate[sl][pattern_id | "com"]
            "n_entries": dict,       # n_entries[sl] -> valid entry count
            "tex_table": str,
        }
    """
 
    log = []
 
    def emit(msg):
        log.append(msg)
        if verbose:
            print(msg)
 
 
    if pattern_labels is None:
        pattern_labels = [str(i) for i in range(6)]
 
    if sl_list is None:
        sl_list = _auto_detect_sl_list(
            data=data, pattern_key=pattern_key, sl_selector_key=sl_selector_key,
        )
        emit(f"auto-detected sl_list = {sl_list}")


    pct_ar = dataset_info["pct_Ar"]
    pct_co2 = dataset_info["pct_CO2"]
    u_wire = dataset_info["U_wire"]
    """
    #u_fieldshaper = dataset_info["U_Fieldshaper"]
    #u_cathode = f"-{dataset_info["U_cathode"]}"
    """
    ########################
    ####### duration from a per-row timestamp array (or an explicit override)
 
    if duration_seconds is None:
        ts_arr = np.asarray(data[duration_key])
        emit(f"{duration_key}: {len(ts_arr)} entries, {len(np.unique(ts_arr))} unique values")
        if len(ts_arr) < 2:
            raise ValueError(
                f"Cannot derive a duration from {duration_key}: only {len(ts_arr)} "
                "entries present. Pass duration_seconds explicitly."
            )
        duration_ticks = float(np.max(ts_arr) - np.min(ts_arr))
        duration_seconds = duration_ticks * 0.78 * 1e-9
        if duration_seconds == 0.0:
            raise ValueError(
                f"Derived duration is 0 s (all {duration_key} entries are identical). "
                f"Try a different duration_key, or pass duration_seconds explicitly "
                "to bypass derivation entirely."
            )
    emit(f"duration = {duration_seconds} s")
 
    ########################
    ####### per-SL pattern type counts / rates
 
    pat_counts = {}
    pat_rate = {}
    err_pat_rate = {}
    n_entries = {}
 
    for sl in sl_list:
        pat_type_valid, n_excluded, n_total = _get_pattern_and_mask(
            data=data,
            sl=sl,
            pattern_key=pattern_key,
            impossible_key=impossible_key,
            sl_selector_key=sl_selector_key,
            exclude_impossible=exclude_impossible,
        )
        n_entries[sl] = int(len(pat_type_valid))
 
        counts, rates, errs = {}, {}, {}
        for p, label in enumerate(pattern_labels):
            c = int(np.sum(pat_type_valid == p))
            counts[p] = c
            rates[p] = c / duration_seconds
            errs[p] = np.sqrt(c) / duration_seconds
        counts["com"] = int(len(pat_type_valid))
        rates["com"] = counts["com"] / duration_seconds
        errs["com"] = np.sqrt(counts["com"]) / duration_seconds
 
        pat_counts[sl] = counts
        pat_rate[sl] = rates
        err_pat_rate[sl] = errs
 
        emit(f"[{_sl_display(sl)}] total valid entries: {counts['com']} "
             f"(excluded {n_excluded} impossible fits out of {n_total})")
        for p, label in enumerate(pattern_labels):
            emit(f"  pattern {label}: {counts[p]} counts, rate = ({rates[p]:.3f} +- {errs[p]:.3f}) Hz")
 
    ########################
    ####### bar plot: pattern type rate distribution, one plot per SL
 
    for sl in sl_list:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x = np.arange(len(pattern_labels))
        heights = [pat_rate[sl][p] for p in range(len(pattern_labels))]
        errors = [err_pat_rate[sl][p] for p in range(len(pattern_labels))]
        ax.bar(x, heights, width=0.6, align="center")
        ax.errorbar(x, heights, yerr=errors, fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(pattern_labels)
        ax.set_xlabel("Pattern type")
        ax.set_ylabel("Rate [Hz]")
        ax.set_title(f"{add_title_info}: {_sl_display(sl)} pattern type rates\nfor {pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{\\mathrm{{wire}}}} = {u_wire}$, {suffix}")
        info_str = f"entries = {n_entries[sl]}"
        ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="upper right")
        fig.tight_layout()
        fig.show()
        if save_plots:
            hist_plot_file = plot_save_path + dataset_name + f"_PAT_TYPE_{_sl_display(sl)}_{suffix}" + plot_type
            emit(f"store plot as {hist_plot_file}.")
            fig.savefig(hist_plot_file)
        plt.close(fig)
 
    ########################
    ####### comparison plot: grouped bars across all requested SLs
 
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_sl = len(sl_list)
    width = 0.8 / n_sl
    x = np.arange(len(pattern_labels))
    for i, sl in enumerate(sl_list):
        heights = [pat_rate[sl][p] for p in range(len(pattern_labels))]
        errors = [err_pat_rate[sl][p] for p in range(len(pattern_labels))]
        offset = (i - (n_sl - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=_sl_display(sl))
        ax.errorbar(x + offset, heights, yerr=errors, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_labels)
    ax.set_xlabel("Pattern type")
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(
    f"pattern type rate comparison\nfor {pct_ar}/{pct_co2} Ar/CO$_2$, "
    f"$U_{{\\mathrm{{wire}}}} = {u_wire}$, {suffix}"
)
    ax.legend()
    fig.tight_layout()
    fig.show()
    if save_plots:
        hist_plot_file = plot_save_path + dataset_name + f"_{fit_type}_PAT_TYPE_COMPARISON_{suffix}" + plot_type
        emit(f"store plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)
    plt.close(fig)
 
    ########################
    ####### tex table (mirrors the layout from the original plot_specific_hist.py)
 
    float_precision = 2
    header_cols = " & ".join(f"\\ac{{{_sl_display(sl)}}}" for sl in sl_list)
    tex_lines = [
        f"\\begin{{tabular}}{{|c|{'c|' * n_sl}}}",
        "    \\hline",
        f"    Pattern & {header_cols} \\\\ \\hline",
    ]
    for p, label in enumerate(pattern_labels):
        row_vals = " & ".join(
            f"$({np.round(pat_rate[sl][p], float_precision):.{float_precision}f} \\pm "
            f"{np.round(err_pat_rate[sl][p], float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$"
            for sl in sl_list
        )
        tex_lines.append(f"    ${label}$ & {row_vals} \\\\")
    tex_lines.append("    \\hline")
    com_vals = " & ".join(
        f"$({np.round(pat_rate[sl]['com'], float_precision):.{float_precision}f} \\pm "
        f"{np.round(err_pat_rate[sl]['com'], float_precision):.{float_precision}f})\\;\\si{{\\hertz}}$"
        for sl in sl_list
    )
    tex_lines.append(f"    Cumulative & {com_vals} \\\\ \\hline")
    tex_lines.append("\\end{tabular}")
    tex_table = "\n".join(tex_lines)
    emit(tex_table)
 
    return {
        "log": log,
        "duration_seconds": duration_seconds,
        "pat_counts": pat_counts,
        "pat_rate": pat_rate,
        "err_pat_rate": err_pat_rate,
        "n_entries": n_entries,
        "tex_table": tex_table,
    }


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
    method="",
    strmethod="",
    ):
    entries = []
    for dataset_name, result in analysis_out.items():
        try:
            info = dataset_info_fn(name = dataset_name)   # FIXED: was calling stray `dataset_info` kwarg
        except Exception as e:
            if verbose:
                print(f"  skipping {dataset_name}: could not parse dataset info ({e})")
            continue
 
        pct_ar = int(info["pct_Ar"])
        pct_co2 = int(info["pct_CO2"])
        u_wire = int(info["U_wire"])  # force int so wide/fallback parsing paths can't mismatch
        mix_label = f"{pct_ar}/{pct_co2}"
 
        try:
            entries.append({
                "dataset": dataset_name,
                "mix": mix_label,
                "u_wire": u_wire,
                "mean_vd": result["peak"],
                "err_vd": result["tot_err"],
            })
        except KeyError:
            entries.append({
                "dataset": dataset_name,
                "mix": mix_label,
                "u_wire": u_wire,
                "mean_vd": result["v_drift"],
                "err_vd": result["err_v_drift"],
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



# quantity key -> (column suffix appended to "_free_vd_super_fit", y-axis label, log-y)
_QUANTITIES = [
    #("tan_alpha",     r"$\tan\alpha$",              False),
    ("err_vd",        r"$\sigma(v_d)$",             True),
    ("err_t0",        r"$\sigma(t_0)$",              True),
    ("err_tan_alpha", r"$\sigma(\tan\alpha)$",       True),
    ("err_x0",        r"$\sigma(x_0)$",              True),
]

def plot_super_fit_errors_vs_tan_alpha(
    cut_super_fits,
    *,
    base_path,
    plot_type=".png",
    fig_size=(12, 10),
    save_path=None,
    point_size=8,
    alpha=0.35,
    verbose=True,
    method="",
    strmethod="",
    dataset_name="",
    dataset_info="",
    suffix=""
    ):
    """
    ... (docstring unchanged) ...
    """

    pct_ar = dataset_info["pct_Ar"]
    pct_co2 = dataset_info["pct_CO2"]
    u_wire = dataset_info["U_wire"]
    tan_alpha = np.asarray(cut_super_fits["tan_alpha_free_vd_super_fit"], dtype=float)

    mask = np.isfinite(tan_alpha)
    impossible_col = "impossible_free_vd_super_fit"
    if impossible_col in cut_super_fits:
        mask &= ~np.asarray(cut_super_fits[impossible_col], dtype=bool)

    if not np.any(mask):
        raise ValueError("No valid entries left after masking; nothing to plot.")

    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.ravel()

    err_threshold = 20

    for ax, (qty, ylabel, logy) in zip(axes, _QUANTITIES):
        col = f"{qty}_free_vd_super_fit"
        # these y-columns ARE the fit-error quantities themselves,
        # there's no separate "error on the error" column
        err_col = col

        if col not in cut_super_fits:
            if verbose:
                print(f"  skipping '{col}': column not found")
            ax.set_visible(False)
            continue

        y = np.asarray(cut_super_fits[col], dtype=float)
        yerr = np.asarray(cut_super_fits[err_col], dtype=float)
        m = mask & np.isfinite(y) & np.isfinite(yerr)

        ax.scatter(tan_alpha[m], y[m], s=point_size, alpha=alpha, color="tab:blue")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\tan\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + r" vs $\tan\alpha$")
        ax.grid(True, alpha=0.3)

        # "bad" fits: error above threshold
        m_bad = m & (yerr > err_threshold)

        m_bad_pos = m_bad & (tan_alpha > 0)
        m_bad_neg = m_bad & (tan_alpha < 0)

        if m_bad_pos.any():
            # furthest-out bad fit on the positive side = edge of the bad region
            ax.axvline(np.max(tan_alpha[m_bad_pos]), color="black", linestyle="--", alpha=0.5, label=f"bad fit threshold max: {np.rad2deg(np.max(tan_alpha[m_bad_pos])):.4f} deg")
            ax.legend(loc="upper left", fontsize=8)
        if m_bad_neg.any():
            # furthest-out bad fit on the negative side (most negative = furthest from 0)
            ax.axvline(np.min(tan_alpha[m_bad_neg]), color="black", linestyle="--", alpha=0.5, label=f"bad fit threshold min: {np.rad2deg(np.min(tan_alpha[m_bad_neg])):.4f} deg")
            ax.legend(loc="upper left", fontsize=8)

    title = (
        f"SUPER fit diagnostics vs tan_alpha\n"
        f"for {pct_ar}/{pct_co2} Ar/CO$_2$, $U_{{\\mathrm{{wire}}}} = {u_wire}$"
    )
    if strmethod:
        title += f" ({strmethod})"
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is None:
        save_path = base_path + f"plots/sl_fits/{dataset_name}/{dataset_name}_super_fit_errors_vs_tan_alpha{plot_type}"
    fig.savefig(save_path)
    if verbose:
        print(f"store plot as {save_path}.")

    return fig, axes, save_path


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

def data_to_hist_2d (*, data_x, data_y, x_label, y_label, title, colorbar_label = "counts", save_path = None, n_bins = 80):
    plt.figure()
    plt.hist2d(data_x, data_y, bins=n_bins, cmap='viridis')
    plt.colorbar(label='Anzahl Einträge')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return


def get_dataset_marker_path(*, plot_save_path, dataset_name, plot_type,
                             do_ramp_measurement, do_only_vd_peak_fit,
                             do_refit_full_analysis):
    """Path of the LAST plot main() writes for one dataset, for the given
    combination of mode flags. Used as an 'analysis already done' marker
    for skip_existing_datasets.

    This has to mirror main()'s control flow exactly -- which plot is
    written last depends on do_ramp_measurement / do_only_vd_peak_fit /
    do_refit_full_analysis, AND on the order of the internal
    `for i in range(2)` cut loops (super-fit loop: i=0 -> w_cut,
    i=1 -> no_cut; refit loop: i=0 -> no_cut, i=1 -> w_cut -- these are
    NOT the same order). If you change what's plotted or reorder those
    loops, update this function too.
    """
    if do_ramp_measurement or do_only_vd_peak_fit:
        # only the w_cut parabola-peak fit of vd is produced per dataset
        return f"{plot_save_path}{dataset_name}_w_cut_parabolafit{plot_type}"

    if do_refit_full_analysis:
        # refit loop's last iteration (i=1) uses suffix=w_cut, fit_type="fit"
        return f"{plot_save_path}{dataset_name}_fit_PAT_TYPE_COMPARISON_w_cut{plot_type}"

    # full super-fit analysis only (no refit): super-fit loop's last
    # iteration (i=1) uses suffix=no_cut, fit_type="super_fit"
    return f"{plot_save_path}{dataset_name}_super_fit_PAT_TYPE_COMPARISON_no_cut{plot_type}"


# -------------------------------------------------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    



    do_only_vd_peak_fit = False # when set to true, only the gaussian fit of the photopeak is performed, no super-fit analysis
    do_ramp_measurement = False
    do_refit_full_analysis = False
    do_super_fit_analysis = True
    skip_existing_datasets = False  # set False to force re-analysis of every dataset



    #set parameters for the super fit analysis cuts 
    max_err_to_free_vd_superfit = 20
    max_err_x0_free_vd_superfit = 1
    max_err_vd_free_vd_superfit = 2
    max_err_tan_alpha_free_vd_superfit = 0.1
    max_chi2ndf_frree_vd_superfit = 20
    max_angle_rad = max(alpha_max for _, (_, alpha_max) in params._dt_pattern_alpha_range.items())
    max_angle_deg = np.rad2deg(max_angle_rad)
    max_tan_alpha = np.tan(max_angle_rad)

    print(max_angle_rad)  # 1.0164888305933455
    print(max_angle_deg)  # 58.240519915187214
    
    
    list_of_fits = [



                "cosmic_82-18_3550-1800-1200_run1_th20_cut100",
                "cosmic_82-18_3575-1800-1200_run1_th20_cut100", 
                "cosmic_82-18_3600-1800-1200_run1_th20_cut100",
                "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 
                "cosmic_82-18_3650-1800-1200_run1_th20_cut100",#full

                "cosmic_83-17_3650-1800-1200_run1_th20_cut100",
                "cosmic_83-17_3625-1800-1200_run1_th20_cut100", 
                "cosmic_83-17_3600-1800-1200_run1_th20_cut100", 
                "cosmic_83-17_3575-1800-1200_run1_th20_cut100", 
                "cosmic_83-17_3550-1800-1200_run1_th20_cut100",#full

                "cosmic_84-16_3650-1800-1200_run1_th20_cut100",
                "cosmic_84-16_3625-1800-1200_run1_th20_cut100", 
                "cosmic_84-16_3600-1800-1200_run1_th20_cut100", 
                "cosmic_84-16_3575-1800-1200_run1_th20_cut100", 
                "cosmic_84-16_3550-1800-1200_run1_th20_cut100",#full

                "cosmic_85-15_3550-1800-1200_run1_th20_cut100",
                "cosmic_85-15_3575-1800-1200_run1_th20_cut100", 
                "cosmic_85-15_3600-1800-1200_run2_th20_cut100",#missing 3625, 3650 (not measured)

                "cosmic_86-14_3650-1800-1200_run1_th20_cut100", #issues with data proccessing
                "cosmic_86-14_3625-1800-1200_run1_th20_cut100", 
                "cosmic_86-14_3600-1800-1200_run1_th20_cut100", 
                "cosmic_86-14_3575-1800-1200_run1_th20_cut100", 
                "cosmic_86-14_3550-1800-1200_run1_th20_cut100",#full

                "cosmic_87-13_3550-1800-1200_run1_th20_cut100",
                "cosmic_87-13_3575-1800-1200_run1_th20_cut100",
                "cosmic_87-13_3600-1800-1200_run1_th20_cut100", # stopped because of tripping
                ]
    #list_of_fits = ["mb1_sxa5_cosmics_10min"]
    #list_of_fits = ["cosmic_85-15_3600-1800-1200_test4_th20"]

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
        #"data_mic0_start_2026-07-27_12-46-58_stop_2026-07-27_12-56-59",
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
    #list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50"]
    
        
    if do_ramp_measurement:
        do_only_vd_peak_fit = True
        do_super_fit_analysis = True
        list_of_fits = ramp_datasets

    base_path = "data_ba/"
    plot_type = ".png"

    fig_size = (8,6)
    #dataset_name = "cosmic_82-18_3550-1800-1200_run1_th20_cut_50"

    #check that datasets exist
    pcls_path = f"{base_path}pcls/"

    # --- load previously saved analysis results so already-analyzed
    # datasets can be skipped instead of redone from scratch, AND so
    # results accumulate across runs ---
    analysis_pkl_name = (
        "analysis_out_track_fit_ramp.pcl" if do_ramp_measurement
        else "analysis_out_track_fit.pcl"
    )
    analysis_pkl_path = f"{pcls_path}{analysis_pkl_name}"
    if os.path.exists(analysis_pkl_path):
        analysis_out_prev = data_utils.load_pickle(analysis_pkl_path)
    else:
        analysis_out_prev = {}

    # seed this run's results with everything already stored, so datasets
    # not touched this run are carried forward instead of dropped
    analysis_out = dict(analysis_out_prev)

    datasets_to_skip = set()
    if skip_existing_datasets:
        for dataset_name in list_of_fits:
            plot_save_path = base_path + f"plots/sl_fits/{dataset_name}/"
            marker_path = get_dataset_marker_path(
                plot_save_path=plot_save_path,
                dataset_name=dataset_name,
                plot_type=plot_type,
                do_ramp_measurement=do_ramp_measurement,
                do_only_vd_peak_fit=do_only_vd_peak_fit,
                do_refit_full_analysis=do_refit_full_analysis,
            )
            if os.path.exists(marker_path) and dataset_name in analysis_out_prev:
                datasets_to_skip.add(dataset_name)

    if datasets_to_skip:
        print(f"Skipping {len(datasets_to_skip)} already-analyzed dataset(s): "
              f"{sorted(datasets_to_skip)}")
    # check that source datasets exist (skip the check for datasets we're
    # not going to touch anyway)
    non_existing_super_fits = []

    for dataset in list_of_fits:
        if dataset in datasets_to_skip:
            continue

        root_file_name = f"{dataset}_super_fits.root"
        pcl_file_name = f"{dataset}_super_fits.pcl"
        root_path = Path(f"{base_path}pcls/{dataset}/{root_file_name}")
        pcl_path = Path(f"{base_path}pcls/{dataset}/{pcl_file_name}")

        if root_path.exists() or pcl_path.exists():
            continue

        print(f"Error: Dataset '{root_file_name}' (or '{pcl_file_name}') does not exist.")
        non_existing_super_fits.append(root_file_name)

    if len(non_existing_super_fits) >= 1:
        sys.exit(1)  # Stop the entire script

    print("All datasets found. Continuing...")
    for dataset_idx in range(len(list_of_fits)):

        dataset_name = list_of_fits[dataset_idx]

        if dataset_name in datasets_to_skip:
            print(f"[{dataset_idx+1}/{len(list_of_fits)}] {dataset_name}: "
                  f"already analyzed, skipping.")
            analysis_out[dataset_name] = analysis_out_prev[dataset_name]
            continue

        

        #when using pcls use this
        #sl_patterns_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_patterns.pcl"
        #sl_fits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_fits.pcl"
        #sl_refits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_refits.pcl"
        #super_fits_path = base_path + f"pcls/{dataset_name}/" + dataset_name + "_super_fits.pcl"
        #plot_save_path = base_path + f"plots/sl_fits/{dataset_name}/" 
        #print(f"###### Importing super fits...")
        #super_fits = data_utils.load_pickle(file = super_fits_path)
        #print("### imported super fits data from file: " + super_fits_path)



        #when using root files, use this
        sl_patterns_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_patterns.root"
        sl_fits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_fits.root"
        sl_refits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_refits.root"
        super_fits_path = base_path + f"pcls/{dataset_name}/" + dataset_name + "_super_fits.root"
        plot_save_path = base_path + f"plots/sl_fits/{dataset_name}/" 

        super_fits = uproot.open(f"{super_fits_path}:tree").arrays(library="np")
        print("### imported super fits data from file: " + super_fits_path)


        os.makedirs(plot_save_path, exist_ok=True)
    

        ### data import
        #print(f"###### Importing fits...")
        



        #refit_keylist = [ "chi2/ndf_refit", "vd_refit", "tan_alpha_refit",  "x0_refit", "t0_refit", "dt0_refit",  "dt1_refit", "dt2_refit", "dt2_refit"]
        #fit_keylist = ["chi2/ndf", "vd", "tan_alpha",  "x0", "dt1",  "dt2", "dt2", "pat_type"]






        
        
        
        no_cut = "no_cut"
        w_cut = "w_cut"


        ###########################
    


        #super_fit_keylist = ['sl1', 'sl3', 'pat_type_sl1', 'pat_type_sl3', 'idx_sl1', 'idx_sl3', 'muon_id_mismatch', 'ts0', 'err_ts0', 'ts4', 'err_ts4', 'wi0_sl1', 'wi0_sl3', 'ts1', 'err_ts1', 'ts5', 'err_ts5', 'wi1_sl1', 'wi1_sl3', 'ts2', 'err_ts2', 'ts6', 'err_ts6', 'wi2_sl1', 'wi2_sl3', 'ts3', 'err_ts3', 'ts7', 'err_ts7', 'wi3_sl1', 'wi3_sl3', 'impossible_sl1', 'impossible_sl3', 'laterality_sl1', 'laterality_sl3', 't0_sl1', 't0_sl3', 'err_t0_sl1', 'err_t0_sl3', 'x0_sl1', 'x0_sl3', 'err_x0_sl1', 'err_x0_sl3', 'tan_alpha_sl1', 'tan_alpha_sl3', 'err_tan_alpha_sl1', 'err_tan_alpha_sl3', 'vd_sl1', 'vd_sl3', 'err_vd_sl1', 'err_vd_sl3', 'corr_t0_x0_sl1', 'corr_t0_x0_sl3', 'corr_t0_tan_alpha_sl1', 'corr_t0_tan_alpha_sl3', 'corr_t0_vd_sl1', 'corr_t0_vd_sl3', 'corr_x0_tan_alpha_sl1', 'corr_x0_tan_alpha_sl3', 'corr_x0_vd_sl1', 'corr_x0_vd_sl3', 'corr_tan_alpha_vd_sl1', 'corr_tan_alpha_vd_sl3', 'chi2/ndf_sl1', 'chi2/ndf_sl3', 'dt0_sl1', 'dt0_sl3', 'dt1_sl1', 'dt1_sl3', 'dt2_sl1', 'dt2_sl3', 'dt3_sl1', 'dt3_sl3', 'muon_id', 'muon_ts', 'muon_phi', 'muon_theta', 'muon_x0', 'muon_y0', 'muon_z0', 'impossible_free_vd_super_fit', 'lat_id1_free_vd_super_fit', 'lat_id2_free_vd_super_fit', 't0_free_vd_super_fit', 'x0_free_vd_super_fit', 'tan_alpha_free_vd_super_fit', 'vd_free_vd_super_fit', 'chi2/ndf_free_vd_super_fit', 'dt0_free_vd_super_fit', 'dt1_free_vd_super_fit', 'dt2_free_vd_super_fit', 'dt3_free_vd_super_fit', 'dt4_free_vd_super_fit', 'dt5_free_vd_super_fit', 'dt6_free_vd_super_fit', 'dt7_free_vd_super_fit', 'err_t0_free_vd_super_fit', 'err_x0_free_vd_super_fit', 'err_tan_alpha_free_vd_super_fit', 'err_vd_free_vd_super_fit', 'corr_t0_x0_free_vd_super_fit', 'corr_t0_tan_alpha_free_vd_super_fit', 'corr_t0_vd_free_vd_super_fit', 'corr_x0_tan_alpha_free_vd_super_fit', 'corr_x0_vd_free_vd_super_fit', 'corr_tan_alpha_vd_free_vd_super_fit', 'ref_x_free_vd_super_fit', 'ref_z_free_vd_super_fit']

        # Format:[              key,      Plot title,                             factor,     Unit of measurement,  xlabel, ylabel, gas_mix, U_wire]
        
        
        good_super_fit_keys = [["t0_sl1", "T0 distribution of cosmic muons in SL1", 1, "TS", "T0 [TS]", "counts"], 
                            ['t0_sl3', "T0 distribution of cosmic muons in SL3", 1, "TS", "T0 [TS]", "counts"], 
                            ["x0_free_vd_super_fit", "x0 distribution of muon super fits", 1, "mm", "x0 [mm]", "counts"], 
                            ['tan_alpha_free_vd_super_fit', "tan alpha distribution of cosmic muon super fits", 1, "", "tan alpha", "counts"], 
                            ['vd_free_vd_super_fit', "electron drift velocity distribution from fits", vd_factor, "um/ns", "v_drift [um/ns]", "counts"], 
                            ['chi2/ndf_free_vd_super_fit', "chi2/ndf distribution from fits", 1, "", "chi2/ndf", "counts"], 
                            ['dt0_free_vd_super_fit', "drift time distribution of wire 0", 1, "TS", "dt1 [TU]", "counts"],
                            ['dt1_free_vd_super_fit', "drift time distribution of wire 1", 1, "TS", "dt1 [TU]", "counts"], 
                            ['dt2_free_vd_super_fit', "drift time distribution of wire 2", 1, "TS", "dt1 [TU]", "counts"], 
                            ['dt3_free_vd_super_fit', "drift time distribution of wire 3", 1, "TS", "dt1 [TU]", "counts"], 
                            ['dt4_free_vd_super_fit', "drift time distribution of wire 4", 1, "TS", "dt1 [TU]", "counts"],  
                            ['dt5_free_vd_super_fit', "drift time distribution of wire 5", 1, "TS", "dt1 [TU]", "counts"], 
                            ['dt6_free_vd_super_fit', "drift time distribution of wire 6", 1, "TS", "dt1 [TU]", "counts"],  
                            ['dt7_free_vd_super_fit', "drift time distribution of wire 7", 1, "TS", "dt1 [TU]", "counts"], 
                            ]
        goood_fit_keys = ["tan_alpha"]


        #print(super_fits.keys())
        try:
            # Dataset info from name; Use parse_fit_name to extract information from dataset name
            dataset_info = parse_fit_name(name = dataset_name)
            pct_ar = dataset_info['pct_Ar']
            pct_co2 = dataset_info['pct_CO2']
            u_wire = dataset_info['U_wire']
            u_fieldshaper = dataset_info['U_Fieldshaper']
            u_cathode = f"-{dataset_info['U_cathode']}"

        except:
            if dataset_name == "mb1_sxa5_cosmics_10min":
                dataset_info = {}
                pct_ar = "85"
                pct_co2 = "15"
                u_wire = "3600"
                u_fieldshaper = "1800"
                u_cathode = "-1200"
                dataset_info["pct_Ar"] = pct_ar
                dataset_info["pct_CO2"] = pct_co2
                dataset_info["U_wire"] = u_wire
                dataset_info["U_Fieldshaper"] = u_fieldshaper
                dataset_info["U_cathode"] = u_cathode 

            else:
                pct_ar = ""
                pct_co2 = ""
                u_wire = ""
                u_fieldshaper = ""
                u_cathode = ""
        uncut = data_utils.cut_data(
            data=super_fits,
            conditions=[("impossible_free_vd_super_fit", "==", 0)],
            silent=True,
        )
        for key in ["err_t0_free_vd_super_fit", "err_x0_free_vd_super_fit",
                    "err_vd_free_vd_super_fit", "err_tan_alpha_free_vd_super_fit"]:
            arr = np.asarray(uncut[key], dtype=np.float64)
            n_inf = np.sum(np.isinf(arr))
            n_nan = np.sum(np.isnan(arr))
            n_finite = np.sum(np.isfinite(arr))
            finite = arr[np.isfinite(arr)]
            print(f"{key}: n={len(arr)}, n_inf={n_inf}, n_nan={n_nan}, n_finite={n_finite}, "
                f"median_finite={np.median(finite) if n_finite else 'n/a'}")

        arr = np.asarray(uncut["chi2/ndf_free_vd_super_fit"], dtype=np.float64)
        print(f"chi2/ndf: n={len(arr)}, n_inf={np.sum(np.isinf(arr))}, n_nan={np.sum(np.isnan(arr))}, "
            f"n_finite={np.sum(np.isfinite(arr))}")
        finite = arr[np.isfinite(arr)]
        print(f"median={np.median(finite):.4g}, p10={np.percentile(finite,10):.4g}, "
            f"p50={np.percentile(finite,50):.4g}, p90={np.percentile(finite,90):.4g}, max={np.max(finite):.4g}")
        print(f"fraction < 10: {np.mean(finite < 10):.3f}")



        if do_super_fit_analysis:

            for i in range(2):

                if i == 1:

                    # beginning with analysis of all fits that are flagged as "possible" (impossible == 0)
                    super_fits_cuts = data_utils.cut_data(
                        data=super_fits,
                        conditions=[
                            ("impossible_free_vd_super_fit", "==", 0),
                        ],
                        silent=True,
                    )
                    suffix = no_cut


                elif i == 0:

                    print(np.histogram(np.rad2deg(np.arctan(super_fits["tan_alpha_free_vd_super_fit"])), bins=20))
                    # The analyisis of more restrictive cuts beginns here
                    super_fits_cuts = data_utils.cut_data(
                        data=super_fits,
                        conditions=[
                            ("impossible_free_vd_super_fit", "==", 0),
                            ("chi2/ndf_free_vd_super_fit", "<", max_chi2ndf_frree_vd_superfit),
                            ("err_t0_free_vd_super_fit", "<", max_err_to_free_vd_superfit),
                            ("err_x0_free_vd_super_fit", "<", max_err_x0_free_vd_superfit),
                            ("err_vd_free_vd_super_fit", "<", max_err_vd_free_vd_superfit),
                            ("err_tan_alpha_free_vd_super_fit", "<", max_err_tan_alpha_free_vd_superfit),
                            #dt_i must be greater than zero, otherwise the fit is not valid
                            ("dt0_free_vd_super_fit", ">", 0),
                            ("dt1_free_vd_super_fit", ">", 0),
                            ("dt2_free_vd_super_fit", ">", 0),
                            ("dt3_free_vd_super_fit", ">", 0),
                            ("dt4_free_vd_super_fit", ">", 0),
                            ("dt5_free_vd_super_fit", ">", 0),
                            ("dt6_free_vd_super_fit", ">", 0),
                            ("dt7_free_vd_super_fit", ">", 0),


                            #("tan_alpha_free_vd_super_fit", "<", max_tan_alpha), # max alpha is defined over pattern max angles
                            #("tan_alpha_free_vd_super_fit", ">", -max_tan_alpha),

                            #("chi2/ndf_free_vd_super_fit", ">", 0.5),
                            #("vd_free_vd_super_fit", "<", 59 * derived_params._drift_velocity_conversion),
                            #("vd_free_vd_super_fit", ">", 51 * derived_params._drift_velocity_conversion),
                            #("vd_free_vd_super_fit", ">", 40 * derived_params._drift_velocity_conversion),
                            #("dt0_refit", ">", min_td),
                            #("dt0_refit", "<", max_td),
                
                        ],
                        silent=True,
                    )
                    suffix = w_cut
                    # gauss fit of drift velocity

                    print("fitting gaussian to cut drift velocity")
                    key = 'vd_free_vd_super_fit'

                    if not do_ramp_measurement:
                        title = f"Gaussian fit to drift velocity histogram\n{pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {suffix}"
                    if do_ramp_measurement:
                        title = f"Gaussian fit to drift velocity histogram\nRamp measurement U_wire = 3600V"
                    factor = vd_factor
                    unit = "um/ns"
                    x_label = f"drift velocity in [{unit}]"
                    y_label = "counts"
                    
            
                    data = super_fits_cuts[key]
                    specific_data = build_hist_general(
                        data_list=data,
                        edge_max=70 / vd_factor,
                        edge_min = 40 / vd_factor,
                        n_bins = 200,
                        # adjust range/binning per-quantity if needed, e.g. by checking key
                    )
            
                    # "/" in a key (e.g. "chi2/ndf_...") isn't safe in a filename
                    safe_key = key.replace("/", "_")
                    if not do_ramp_measurement:
                        strdataset_name = f"{safe_key}_{pct_ar}_{pct_co2}_{u_wire}"

                    elif do_ramp_measurement:
                        strdataset_name = dataset_name


                    #does double gauss fit to dada, worse convergence and less reliable than parabola
                    """
                    fig, ax, path, fit_results = fit_gaussian_hist(
                        specific_data=specific_data,
                        dataset_name=strdataset_name,
                        plot_save_path=plot_save_path,
                        xlabel=x_label,
                        ylabel=y_label,
                        title=title,
                        filename_suffix=suffix,
                    )
                    """
                    
                    fig, ax, path, fit_results = fit_parabola_peak(
                        specific_data=specific_data,
                        dataset_name=strdataset_name,
                        plot_save_path=plot_save_path,
                        xlabel=x_label,
                        ylabel=y_label,
                        title=title,
                        filename_suffix=suffix,
                        scale_factor = vd_factor
                    )

                    analysis_out[dataset_name] = fit_results
                    
                    #print(f"fitted mean drift velocity = {fit_results['mean']:.4g} ± {fit_results['mean_err']:.4g} {unit}")

                    if do_only_vd_peak_fit:
                        continue

                if do_only_vd_peak_fit:
                    continue


                #hist of all interesting hist metrics
                for j in range(len(good_super_fit_keys)):
                    key = good_super_fit_keys[j][0]
                    title =good_super_fit_keys[j][1] + f"\n{pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {suffix}"
                    factor = good_super_fit_keys[j][2]
                    unit = good_super_fit_keys[j][3]
                    x_label = good_super_fit_keys[j][4]
                    y_label = good_super_fit_keys[j][5]
                    
            
                    #data = super_fits_cuts[key]
                    
                    if key == "tan_alpha_free_vd_super_fit":
                        data = np.arctan(super_fits_cuts[key])
                        speckey = "alpha"

                        # export only for real (parsed) gas/voltage conditions, and only the
                        # w_cut selection (the physically meaningful target for reweighting)
                        if suffix == w_cut and all(isinstance(v, int) for v in (pct_ar, pct_co2, u_wire)):
                            export_path = measured_alpha_hist_path(pcls_path, pct_ar, pct_co2, u_wire, suffix)
                            os.makedirs(os.path.dirname(export_path), exist_ok=True)
                            np.savez(
                                export_path,
                                counts=np.asarray(specific_data["hist"]),
                                bin_edges=np.asarray(specific_data["edges"]),
                            )
                            print(f"exported measured alpha histogram -> {export_path}")
                    else:
                        speckey = None
                        data = super_fits_cuts[key]
                        speckey = None
                    specific_data = build_hist_general(
                        data_list=data,
                        n_bins = 300,
                        # adjust range/binning per-quantity if needed, e.g. by checking key
                    )
            
                    # "/" in a key (e.g. "chi2/ndf_...") isn't safe in a filename
                    safe_key = key.replace("/", "_")
            
                    fig, ax, path = plot_hist_general(
                        specific_data=specific_data,
                        dataset_name=dataset_name,
                        plot_save_path=plot_save_path,
                        filename_suffix=safe_key + "_" + suffix,
                        scale_factor = factor,
                        title = title,
                        xlabel = x_label,
                        ylabel = y_label,
                        plot_type = plot_type,
                        speckey = speckey,
                    )
            
                plt.close("all")
                # done with all hists
                # beginning hist2d plots 

                data_to_hist_2d(
                    data_x=super_fits_cuts["x0_free_vd_super_fit"],
                    data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
                    x_label="x0",
                    y_label="v_d",
                    title=f"Hist of x_0 and v_d {no_cut}",
                    save_path=plot_save_path + f"vd_vs_x0_{suffix}{plot_type}",
                )

                data_to_hist_2d(
                    data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
                    data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
                    x_label="alpha",
                    y_label="v_d",
                    title=f"Hist of alpha vs vd {suffix}",
                    save_path=plot_save_path + f"vd_vs_alpha_{suffix}{plot_type}",
                )

                data_to_hist_2d(
                    data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
                    data_y=super_fits_cuts["x0_free_vd_super_fit"],
                    x_label="alpha",
                    y_label="x_0",
                    title=f"Hist of alpha vs x_0 {suffix}",
                    save_path=plot_save_path + f"x0_vs_tanalpha_{suffix}{plot_type}",
                )

                data_to_hist_2d(
                    data_x=super_fits_cuts["x0_free_vd_super_fit"],
                    data_y=super_fits_cuts["dt0_free_vd_super_fit"] * derived_params._ts_unit,
                    x_label="x_0[mm]",
                    y_label="dt_0 [ns]",
                    title=f"Hist of x_0 vs dt_0 {suffix}",
                    save_path=plot_save_path + f"dt_0_vs_x0_{suffix}{plot_type}",
                )
                data_to_hist_2d(
                    data_x=super_fits_cuts["err_vd_free_vd_super_fit"] * vd_factor,
                    data_y=super_fits_cuts["vd_free_vd_super_fit"] * vd_factor,
                    x_label=f"err_vd [$\\mu$m/ns]",
                    y_label=f"vd [\\mu$m/ns]",
                    title=f"Hist of vd vs err vd {suffix}",
                    save_path=plot_save_path + f"vd_vs_err_vd_{suffix}{plot_type}",
                )

                data_to_hist_2d(
                    data_x=super_fits_cuts["err_vd_free_vd_super_fit"] * vd_factor,
                    data_y=super_fits_cuts["err_tan_alpha_free_vd_super_fit"],
                    x_label=f"err_vd [$\\mu$m/ns]",
                    y_label=f"err tan ($\\alpha$)",
                    title=f"Hist of err_tan_alpha vs err_vd {suffix}",
                    save_path=plot_save_path + f"err_tan_alpha_vs_err_vd_{suffix}{plot_type}",
                )

                for k in range(8):
                    data_to_hist_2d(
                        data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
                        data_y=super_fits_cuts[f"dt{k}_free_vd_super_fit"] * derived_params._ts_unit,
                        x_label="alpha [deg]",
                        y_label=f"dt_{k} [ns]",
                        title=f"Hist of dt_{k} vs alpha {suffix}",
                        save_path=plot_save_path + f"dt{k}_vs_alpha_{suffix}{plot_type}",
                    )



                detector_track(super_fits_cuts = super_fits_cuts,
                    dataset_info = dataset_info,
                    plot_idcs = np.linspace(0, len(super_fits_cuts), 5, dtype = int),
                    suffix = suffix,
                    plot_save_path = plot_save_path,
                    dataset_name = dataset_name,
                    fit_suffix="_free_vd_super_fit",
                    plot_type=plot_type,
                    zoom=True,
                    zoom_margin=20.0,
                    orient="phi",
                )


                print(len(super_fits_cuts["muon_ts"]))
                print(len(super_fits_cuts["pat_type_sl1"]))   # should match the line above
                print(np.unique(super_fits_cuts["muon_ts"])[:10])

                results = analyze_pattern_type_data(
                    data=super_fits_cuts,
                    dataset_name=dataset_name,
                    plot_save_path=plot_save_path,
                    plot_type= plot_type,
                    save_plots=True,
                    verbose=True,
                    suffix = suffix,
                    dataset_info = dataset_info,
                    fit_type = "super_fit",
                    add_title_info = "Super Fit" # fit type for title
                )


                saved_paths = muon_heatmap_from_fits(
                    fits_cuts=super_fits_cuts,
                    plot_save_path=plot_save_path,
                    dataset_name=dataset_name,
                    fit_suffix="_free_vd_super_fit",
                    plot_type = plot_type,
                    cut_suff = suffix,
                    dataset_info = dataset_info,

                )

                if suffix == no_cut:
                    plot_super_fit_errors_vs_tan_alpha(super_fits_cuts, 
                                                       base_path=base_path, 
                                                       dataset_name=dataset_name, 
                                                       suffix=suffix, 
                                                       plot_type=plot_type,
                                                       dataset_info=dataset_info)
            
                    
            
                plt.close("all")

        # -----------------------------------------------------------------
        # FIX: this block used to live OUTSIDE the per-dataset loop, after
        # it had already finished, so it referenced sl_fits_file / dataset_info
        # / pct_ar / etc. from whatever the LAST loop iteration happened to
        # set (or, if that iteration was skipped via `continue`, those names
        # were never bound at all -> UnboundLocalError). It has been moved
        # here, inside the loop, so it runs once per dataset using that
        # dataset's own files/info, matching how do_super_fit_analysis's
        # block above is already scoped.
        # -----------------------------------------------------------------
        if do_refit_full_analysis:
            #sl_fits = data_utils.load_pickle(file = sl_fits_file) #when using pcl files use this

            sl_fits = uproot.open(f"{sl_fits_file}:tree").arrays(library="np")
            print(sl_fits.keys())
            #print(f"###### Importing refits...")
            sl_refits = uproot.open(f"{sl_refits_file}:tree").arrays(library="np")
            #print("### imported refits data from file: " + sl_refits_file)
            # The analysis of refits beginns here
            for i in range(2):

                if i == 0:
                    # cuts for four cell fits only possible hists
                    sl_fits_cuts = data_utils.cut_data(
                        data=sl_fits,
                        conditions=[
                            ("impossible", "==", 0),
                            
                        ],
                        silent=True,
                    )

                    suffix = no_cut

                elif i == 1:
                    # cuts for four cell fits only possible hists
                    sl_fits_cuts = data_utils.cut_data(
                        data=sl_fits,
                        conditions=[
                            ("impossible", "==", 0),
                            ("chi2/ndf", "<", 10),


                        ],
                        silent=True,
                    )

                    suffix = w_cut 
                # for key in goood_fit_keys:...
                key = "tan_alpha"
                title = f"Distribution of tan($\\alpha$) \n{pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {w_cut}"
                factor = 1
                unit = ""
                x_label = "tan(alpha)"
                y_label = "counts"

                data = sl_refits[key]
                if key == "tan_alpha":
                    specific_data = build_hist_general(
                        data_list=data,
                        n_bins = 200,
                        # adjust range/binning per-quantity if needed, e.g. by checking key
                    )


                # "/" in a key (e.g. "chi2/ndf_...") isn't safe in a filename
                safe_key = key.replace("/", "_")

                fig, ax, path = plot_hist_general(
                    specific_data=specific_data,
                    dataset_name=dataset_name,
                    plot_save_path=plot_save_path,
                    filename_suffix=safe_key + "_" + suffix,
                    scale_factor = factor,
                    title = title,
                    xlabel = x_label,
                    ylabel = y_label,
                    plot_type = plot_type,

                )

                key = "tan_alpha"

                title = (
                    f"Distribution of $\\alpha$\n"
                    f"{pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {suffix}"
                )

                factor = 180 / np.pi      # falls du Grad darstellen möchtest
                unit = "°"
                x_label = r"tan($\alpha$)"
                y_label = "Counts"

                data = np.arctan(sl_fits_cuts[key])

                specific_data = build_hist_general(
                    data_list=data,
                    n_bins=200,
                )

                fig, ax, path = plot_hist_general(
                specific_data=specific_data,
                dataset_name=dataset_name,
                plot_save_path=plot_save_path,
                filename_suffix="alpha_" + suffix,
                scale_factor=1,   # you already convert to degrees here
                title=title,
                xlabel=x_label,
                ylabel=y_label,
                plot_type=plot_type,   # matches your `factor = 180/np.pi` conversion
                )
                results = analyze_pattern_type_data(
                data=sl_fits,
                dataset_name=dataset_name,
                plot_save_path=plot_save_path,
                plot_type= plot_type,
                save_plots=True,
                verbose=True,
                suffix = suffix,
                dataset_info = dataset_info,
                fit_type = "fit",
                add_title_info = "four hit Fit" # fit type for title
                )

                plt.close("all")

    if do_ramp_measurement:
        data_utils.store_pickle(analysis_out, f"{pcls_path}analysis_out_track_fit_ramp.pcl")


    else:
        data_utils.store_pickle(analysis_out, f"{pcls_path}analysis_out_track_fit.pcl")


    #When doing ramp measurement, this loop is used
    if do_ramp_measurement:
        def exp(t, a, b, c):
            return a * np.exp(-b * t) + c

        print("Analysis of ramp measurement begins...")

        analysis_out = data_utils.load_pickle(f"{pcls_path}analysis_out_track_fit_ramp.pcl")

        times = []
        values = []
        errors = []

        for dataset, result in analysis_out.items():
            if result["peak_err"] < 1:
                times.append(parse_start_time(dataset))
                values.append(result["peak"])
                errors.append(result["tot_err"])

        p0 = [
            values[0] - values[-1],  # amplitude
            0.6,                     # per day
            values[-1],              # equilibrium
        ]

        # numeric day values, shifted so the fit domain starts at t=0
        t_num = mdates.date2num(times)
        t0_num = t_num[0]              # remember the reference epoch to convert back later
        t_shifted = t_num - t0_num

        popt, pcov = curve_fit(exp, t_shifted, values, sigma=errors, absolute_sigma=True, p0=p0)
        a_fit, b_fit, c_fit = popt
        err_a_fit = np.sqrt(pcov[0][0])
        err_b_fit = np.sqrt(pcov[1][1])
        err_c_fit = np.sqrt(pcov[2][2])
        print(f"Exponential fit parameters: a = {a_fit:.4g} ± {err_a_fit:.2g}, "
            f"b = {b_fit:.4g} ± {err_b_fit:.2g}, c = {c_fit:.4g} ± {err_c_fit:.2g}")

        plt.figure(figsize=(10, 5))
        plt.errorbar(
            times, values, yerr=errors,
            fmt="o", capsize=4, markersize=6,
            label=r"$U_{\mathrm{wire}} = 3600\,\mathrm{V}$",
        )

        # build the fit curve in the SAME shifted-day domain used for the fit,
        # then convert that domain back to real datetimes only for plotting
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
        plt.title(r"Drift velocity over time ($U_{\mathrm{wire}}=3600$ V) Track-fit method")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base_path}plots/ramp_analysis_track_fit{plot_type}")

    if not do_ramp_measurement:

        analysis_out = data_utils.load_pickle(f"{pcls_path}analysis_out_track_fit.pcl")
        fig, ax, path = plot_vd_by_gas_mix(
        analysis_out=analysis_out,
        base_path=base_path,
        dataset_info_fn=parse_fit_name,
        plot_type=plot_type,
        fig_size=fig_size,
        method = "track_fit",
        strmethod= "Track-fit Method"
        )

    return


if __name__ == "__main__":
    main()
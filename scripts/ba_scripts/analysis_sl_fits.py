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
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    
    
    vd_factor = 1 / derived_params._drift_velocity_conversion

    def plot_statistics(*, keylist, fits, title):

        for key in keylist:
            if 'vd_free_vd_super_fit' in key:
                factor = 1 / derived_params._drift_velocity_conversion
            else:
                factor = 1

            # hist of key distrubution
            n_refits = data_utils.length(fits)
            plt.figure()
            plt.hist(fits[key]*factor, bins=100, histtype="step", color="black")
            plt.xlabel(key + " value")
            plt.xlim(min(fits[key] * factor), max(fits[key] * factor))
            plt.ylabel("counts")
            plt.title("distribution of " + key + title)
            safe_key = key.replace("/", "_")
            path = f"{plot_save_path}{dataset_name}{safe_key}{title}{plot_type}"
            plt.savefig(path, bbox_inches="tight")
            print(f"### saved plot to {path}")
            plt.close()

            if key =='vd_free_vd_super_fit':
                try:
                    def gauss_lin(x, A, mu, sigma, b, c):
                        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b*x + c

                    # --- 1. build histogram from data + err_data, same as your dt_hit pipeline ---

                    x_min, x_max = 50, 62
                    n_bins = 80
                    edges = np.linspace(x_min, x_max, n_bins + 1)

                    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = \
                        hist_utils.create_empty_histogram(edges=edges)

                    hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = \
                        hist_utils.calculate_histogram_and_shifted_histograms(data=fits[key]*factor, edges=edges, err_data=fits["err_" + key]*factor)

                    hist += hist_
                    entries += entries_
                    underflow += underflow_
                    overflow += overflow_
                    hist_err_right += hist_err_right_
                    hist_err_left += hist_err_left_

                    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(
                        hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True
                    )
                    err_hist_stat = np.sqrt(hist)

                    print(f"created histogram:")
                    print(f"  entries = {entries}, underflow = {underflow}, overflow = {overflow}")

                    # --- 2. fit gauss + lin background ---

                    yerr_fit = np.where(err_hist == 0, 1, err_hist)   # avoid zero-weight bins breaking curve_fit
                    p0 = [hist.max(), centers[np.argmax(hist)], 1.0, 10, 10]

                    popt, pcov = curve_fit(gauss_lin, centers, hist, p0=p0, sigma=yerr_fit, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))

                    # --- 3. plot ---

                    plt.figure()
                    plt.step(centers, hist, where="mid", color="black", label="Histogram")
                    plt.errorbar(centers, hist, yerr=err_hist, fmt="none", ecolor="black", elinewidth=1, capsize=0)

                    x = np.linspace(x_min, x_max, 1000)
                    plt.plot(x, gauss_lin(x, *popt), "r-", lw=2,
                            label=f"$\\mu={popt[1]:.3f}\\pm{perr[1]:.3f}$\n$\\sigma={popt[2]:.3f}\\pm{perr[2]:.3f}$")

                    plt.xlabel(key + " value")
                    plt.ylabel("counts")
                    plt.xlim(x_min, x_max)
                    plt.title("distribution of " + key + title)
                    plt.legend()

                    safe_key = key.replace("/", "_")
                    path = f"{plot_save_path}{dataset_name}{safe_key}{title}_zoom_{plot_type}"
                    plt.savefig(path, bbox_inches="tight")
                    print(f"### saved plot to {path}")
                    plt.close()
                except:
                    print("Fit failed...\nContinuing with other hists")

        return 
   
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
        ylabel = "counts",
        filename_suffix="ALL",
        start_idx=0,
        scale_factor=1,          # e.g. tu -> ns conversion; set to 1.0 to disable
        log_scale=False,
        power_limits=[-4, 4],
        bin_unit = "",
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
        title = "",
    ):
        """
        General histogram plotting function.
    
        Reads a histogram (values, edges, asymmetric errors, over/underflow) out of
        `specific_data`, plots it via hist_utils.plot_histogram, and optionally saves
        the figure to disk.
    
        This function does NOT build or mutate `specific_data` in any way -- it only
        reads from it. Use `build_hist_general` (or your own logic) to construct the
        dict beforehand, e.g.:
    
            specific_data = build_hist_general(
                data_list=my_quantity,          # any array: energy, charge, Δt, ...
                err_data_list=my_quantity_err,  # optional
                # edge_min/edge_max auto-detected from the data by default;
                # pass them explicitly here to override
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
            X-axis label.
        filename_suffix : str
            Inserted into the filename to distinguish plots, e.g. "ALL", "SUBSET_1".
        start_idx : int
            Index to slice hist/edges/errors from (drops leading bins).
        scale_factor : float
            Multiplier applied to edges (e.g. unit conversion). Use 1.0 for none.
        log_scale : bool
            Passed to hist_utils.plot_histogram.
        power_limits : list
            Passed to hist_utils.plot_histogram.
        bin_unit : str
            Passed to hist_utils.plot_histogram.
        add_info : bool
            Passed to hist_utils.plot_histogram.
        legend_font_size : int
            Currently unused by hist_utils.plot_histogram in the original snippet,
            kept here in case you want to apply it (e.g. via ax.legend(fontsize=...)).
        fig_size : tuple
            Figure size.
        xlim : tuple, None, or False
            If None, defaults to (0, max(bins)). If False, xlim is not set.
            If a tuple, used directly as ax.set_xlim(*xlim).
        hist_key, err_hist_key, err_hist_down_key, err_hist_up_key, edges_key,
        overflow_key, underflow_key : str
            Keys used to pull data out of `specific_data`, in case your dict uses
            different naming for different histograms.
        save : bool
            Whether to save the figure to disk.
        verbose : bool
            Whether to print progress messages.
    
        Returns
        -------
        fig, ax, path
            The created figure, axis, and the save path (path is None if save=False).
        """
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
    
        if xlim is None:
            ax.set_xlim(np.amin(bins), np.amax(bins))
        elif xlim is not False:
            ax.set_xlim(*xlim)

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
    
        Parameters
        ----------
        super_fits_cuts : dict of arrays
            The (already cut) super-fit dataset, e.g. the output of
            data_utils.cut_data() on the super_fits pcl. Must contain per-fit
            arrays for sl1, sl3, pat_type_sl1/sl3, wi{0-3}_sl1/sl3, ts{0-7},
            err_ts{0-7}, and the fitted parameters selected by `fit_suffix`
            (lat_id1/2, t0, x0, tan_alpha, vd, their errors, all corr_* terms,
            ref_x, ref_z, chi2/ndf).
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
    
        Returns
        -------
        saved_paths : dict
            {idx: {"ts_vs_fit": path, "local_track": path,
                "detector_track": path, "detector_track_zoom": path_or_None}}
            for every index in `plot_idcs`.
        """
        os.makedirs(plot_save_path, exist_ok=True)
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
            # fit results
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
    
            # fit function evaluated at all 8 layers
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
            fig.savefig(fname=path)
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
    
            saved_paths[idx] = paths
    
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
        ):
        #from scipy.optimize import curve_fit

        def gaussian(x, amplitude, mean, sigma):
            return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

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

        if verbose:
            print(f"Fitting double Gaussian to {dataset_name} ({filename_suffix})...")

        err_hist_sym = (err_hist_up + err_hist_down) / 2.0
        err_hist_safe = np.where(err_hist_sym <= 0, 1.0, err_hist_sym)

        if fit_range is not None:
            mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
        else:
            mask = np.ones_like(centers, dtype=bool)

        n_params = 6

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

        try:
            popt, pcov = curve_fit(
                double_gaussian,
                centers[mask],
                hist[mask],
                p0=p0,
                sigma=err_hist_safe[mask],
                absolute_sigma=True,
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"  Double Gaussian fit failed for {dataset_name} ({filename_suffix}): {e}")
            popt = np.full(n_params, np.nan)
            perr = np.full(n_params, np.nan)
            pcov = None

        amp1, mean1, sigma1, amp2, mean2, sigma2 = popt
        sigma2, sigma2 = np.abs(sigma1), np.abs(sigma2)
        amp1_err, mean1_err, sigma1_err, amp2_err, mean2_err, sigma2_err = perr

        # --- NEW: chi2 / ndf from the fitted (masked) points ---
        chi2 = np.nan
        ndf = np.nan
        chi2_ndf = np.nan
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




    list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50", "cosmic_82-18_3550-1800-1200_run1_th20_cut100", 
                    "cosmic_82-18_3575-1800-1200_run1_th20_cut100", "cosmic_82-18_3600-1800-1200_run1_th20_cut100", 
                    "cosmic_82-18_3625-1800-1200_run1_th20_cut100", 
                    "cosmic_85-15_3550-1800-1200_run1_th20_cut100", "cosmic_85-15_3575-1800-1200_run1_th20_cut100", 
                    "cosmic_85-15_3600-1800-1200_run2_th20_cut100"]

    #list_of_fits = ["cosmic_82-18_3550-1800-1200_run1_th20_cut_50"]

    do_only_gauss_fit = False #if stet to True, only double gauss fit is done, resulding in a quicker analysis




    ramp_datasets = [
    "data_mic0_start_2026-07-24_18-06-10_stop_2026-07-24_18-16-11",
    "data_mic0_start_2026-07-24_22-16-13_stop_2026-07-24_22-26-14",
    "data_mic0_start_2026-07-25_02-26-16_stop_2026-07-25_02-36-17",
    "data_mic0_start_2026-07-25_06-36-19_stop_2026-07-25_06-46-20",
    #"data_mic0_start_2026-07-25_10-46-22_stop_2026-07-25_10-56-23",
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
    do_ramp_measurement = False # when set to True, the parser function extracts time information

    if do_ramp_measurement:
        list_of_fits = ramp_datasets
        do_only_gauss_fit = True
    else: 
        list_of_fits = list_of_fits

    base_path = "data_ba/"

    plot_type = ".png"
    fig_size = (8,6)
    #dataset_name = "cosmic_82-18_3550-1800-1200_run1_th20_cut_50"
    analysis_out = {}
    
    for dataset_idx in range(len(list_of_fits)):


        dataset_name = list_of_fits[dataset_idx]

        sl_patterns_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_patterns.pcl"
        sl_fits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_fits.pcl"
        sl_refits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_refits.pcl"
        super_fits_path = base_path + f"pcls/{dataset_name}/" + dataset_name + "_super_fits.pcl"
        plot_save_path = base_path + f"plots/sl_fits/{dataset_name}/" 
        os.makedirs(plot_save_path, exist_ok=True)
    

        ### data import
        #print(f"###### Importing fits...")
        sl_fits = data_utils.load_pickle(file = sl_fits_file)

        #print(f"###### Importing refits...")
        sl_refits = data_utils.load_pickle(file = sl_refits_file)
        #print("### imported refits data from file: " + sl_refits_file)

        #print(f"###### Importing super fits...")
        super_fits = data_utils.load_pickle(file = super_fits_path)
        print("### imported super fits data from file: " + super_fits_path)

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
        


        

        #print(super_fits.keys())
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


        for i in range(2):
            if i == 1:
                # beginning with analysis of all fits that are flagged as "possible" (impossible == 0)
                super_fits_cuts = data_utils.cut_data(
                    data=super_fits,
                    conditions=[
                        ("impossible_free_vd_super_fit", "==", 0),
                        #("chi2/ndf_free_vd_super_fit", "<", 10),
                        #("vd_free_vd_super_fit", "<", 70 * derived_params._drift_velocity_conversion),
                        #("vd_free_vd_super_fit", ">", 40 * derived_params._drift_velocity_conversion),
            
                        #("dt0_refit", ">", min_td),
                        #("dt0_refit", "<", max_td),
            
                    ],
                    silent=True,
                )
                suffix = no_cut

            elif i == 0:
                # The analyisis of more restrictive cuts beginns here
                super_fits_cuts = data_utils.cut_data(
                    data=super_fits,
                    conditions=[
                        ("impossible_free_vd_super_fit", "==", 0),
                        ("chi2/ndf_free_vd_super_fit", "<", 10),
                        #("chi2/ndf_free_vd_super_fit", ">", 0.5),
                        #("vd_free_vd_super_fit", "<", 59 * derived_params._drift_velocity_conversion),
                        #("vd_free_vd_super_fit", ">", 51 * derived_params._drift_velocity_conversion),
                        ("err_t0_free_vd_super_fit", "<", 10),
                        #("err_vd_free_vd_super_fit", "<", 10),
                        #("err_tan_alpha_free_vd_super_fit", ">", 0.25*1e6),
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
                title = f"Gaussian fit to drift velocity histogram {pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {suffix}"
                factor = vd_factor
                unit = "um/ns"
                x_label = f"drift velocity in [{unit}]"
                y_label = "counts"
                
        
                data = super_fits_cuts[key]
                specific_data = build_hist_general(
                    data_list=data,
                    # adjust range/binning per-quantity if needed, e.g. by checking key
                )
        
                # "/" in a key (e.g. "chi2/ndf_...") isn't safe in a filename
                safe_key = key.replace("/", "_")
                fig, ax, path, fit_results = fit_gaussian_hist(
                    specific_data=specific_data,
                    dataset_name=f"{safe_key}_{pct_ar}_{pct_co2}_{u_wire}",
                    plot_save_path=plot_save_path,
                    xlabel=x_label,
                    ylabel=y_label,
                    title=title,
                    filename_suffix=suffix,
                )
                analysis_out[dataset_name] = fit_results

                print(f"fitted mean drift velocity = {fit_results['mean']:.4g} ± {fit_results['mean_err']:.4g} {unit}")

                if do_only_gauss_fit:
                    continue

            if do_only_gauss_fit:
                continue


            #hist of all interesting hist metrics
            for i in range(len(good_super_fit_keys)):
                key = good_super_fit_keys[i][0]
                title =good_super_fit_keys[i][1] + f" {pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {suffix}"
                factor = good_super_fit_keys[i][2]
                unit = good_super_fit_keys[i][3]
                x_label = good_super_fit_keys[i][4]
                y_label = good_super_fit_keys[i][5]
                
        
                data = super_fits_cuts[key]
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
                    plot_type = plot_type
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

            for i in range(8):
                data_to_hist_2d(
                    data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
                    data_y=super_fits_cuts[f"dt{i}_free_vd_super_fit"] * derived_params._ts_unit,
                    x_label="alpha",
                    y_label=f"dt_{i} [ns]",
                    title=f"Hist of dt_{i} vs alpha {suffix}",
                    save_path=plot_save_path + f"dt{i}_vs_alpha_{suffix}{plot_type}",
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
            
        
                
        
            plt.close("all")


    


        """
        # The analyisis of more restrictive cuts beginns here
        super_fits_cuts = data_utils.cut_data(
            data=super_fits,
            conditions=[
                ("impossible_free_vd_super_fit", "==", 0),
                ("chi2/ndf_free_vd_super_fit", "<", 10),
                #("chi2/ndf_free_vd_super_fit", ">", 0.5),
                #("vd_free_vd_super_fit", "<", 50 * derived_params._drift_velocity_conversion),
                #("err_t0_free_vd_super_fit", "<", 10),
                ("err_vd_free_vd_super_fit", "<", 10),
                #("vd_free_vd_super_fit", ">", 40 * derived_params._drift_velocity_conversion),

                #("dt0_refit", ">", min_td),
                #("dt0_refit", "<", max_td),

            ],
            silent=True,
        )



        #super_fits_cuts = data_utils.merge_dataset([super_fits_cuts1, super_fits_cuts2])

        additional_cut_suffix = ""

            #hist of all interesting plot metrics
        for i in range(len(good_super_fit_keys)):
            key = good_super_fit_keys[i][0]
            title =good_super_fit_keys[i][1] + f" {pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {w_cut} {additional_cut_suffix}"
            factor = good_super_fit_keys[i][2]
            unit = good_super_fit_keys[i][3]
            x_label = good_super_fit_keys[i][4]
            y_label = good_super_fit_keys[i][5]
            

            data = super_fits_cuts[key]
            specific_data = build_hist_general(
                data_list=data,
                # adjust range/binning per-quantity if needed, e.g. by checking key
            )

            # "/" in a key (e.g. "chi2/ndf_...") isn't safe in a filename
            safe_key = key.replace("/", "_")

            fig, ax, path = plot_hist_general(
                specific_data=specific_data,
                dataset_name=dataset_name,
                plot_save_path=plot_save_path,
                filename_suffix=safe_key + "_" + w_cut + "_" + additional_cut_suffix,
                scale_factor = factor,
                title = title,
                xlabel = x_label,
                ylabel = y_label,
                plot_type = plot_type,

            )

            plt.close("all")

        # done with all hists with further cuts

        data_to_hist_2d(
            data_x=super_fits_cuts["chi2/ndf_free_vd_super_fit"],
            data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
            x_label="chi2/ndf",
            y_label="v_d",
            title=f"Hist of chi2/ndf and v_d {no_cut}",
            save_path=plot_save_path + f"vd_vs_chi2_ndf_{w_cut}{plot_type}",
            n_bins=100,
        )

        # beginning hist2d plots with all possible flagged hists with further cuts

        data_to_hist_2d(
            data_x=super_fits_cuts["x0_free_vd_super_fit"],
            data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
            x_label="x0",
            y_label="v_d",
            title=f"Hist of x_0 and v_d {w_cut} {additional_cut_suffix}",
            save_path=plot_save_path + f"vd_vs_x0_{w_cut}_{additional_cut_suffix}{plot_type}",
        )

        data_to_hist_2d(
            data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
            data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
            x_label="alpha",
            y_label="v_d",
            title=f"Hist of alpha vs vd {w_cut} {additional_cut_suffix}",
            save_path=plot_save_path + f"vd_vs_alpha_{w_cut}_{additional_cut_suffix}{plot_type}",
        )

        data_to_hist_2d(
            data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
            data_y=super_fits_cuts["x0_free_vd_super_fit"],
            x_label="alpha",
            y_label="x_0",
            title=f"Hist of alpha vs x_0 {w_cut} {additional_cut_suffix}",
            save_path=plot_save_path + f"x0_vs_tanalpha_{w_cut}_{additional_cut_suffix}{plot_type}",
        )

        data_to_hist_2d(
            data_x=super_fits_cuts["x0_free_vd_super_fit"],
            data_y=super_fits_cuts["dt0_free_vd_super_fit"] * derived_params._ts_unit,
            x_label="x_0[mm]",
            y_label="dt_0 [ns]",
            title=f"Hist of x_0 vs dt_0 {w_cut} {additional_cut_suffix}",
            save_path=plot_save_path + f"dt_0_vs_x0_{w_cut}_{additional_cut_suffix}{plot_type}",
        )

        for i in range(8):
            data_to_hist_2d(
                data_x=super_fits_cuts[f"dt{i}_free_vd_super_fit"]* derived_params._ts_unit,
                data_y=super_fits_cuts[f"x0_free_vd_super_fit"] ,
                x_label=f"dt{i}",
                y_label=f"x0 [mm]",
                title=f"Hist of x0 vs dt{i} {w_cut} {additional_cut_suffix}",
                save_path=plot_save_path + f"dt{i}_vs_x0_{w_cut}_{additional_cut_suffix}{plot_type}",
            )


        """



        """
        # The analysis of refits beginns here
        for i in range(2):

            if i == 0:
                # cuts for four cell fits only possible hists
                sl_refits_cuts = data_utils.cut_data(
                    data=sl_refits,
                    conditions=[
                        ("impossible_refit", "==", 0),
                    ],
                    silent=True,
                )

                suffix = no_cut

            elif i == 1:
                # cuts for four cell fits only possible hists
                sl_refits_cuts = data_utils.cut_data(
                    data=sl_refits,
                    conditions=[
                        ("impossible_refit", "==", 0),
                    ],
                    silent=True,
                )

                suffix = w_cut 


            # analysis of refits 
            data_to_hist_2d(
                data_x=sl_refits_cuts["x0_refit"],
                data_y=sl_refits_cuts['vd_refit'] * vd_factor,
                x_label="x0",
                y_label="v_d",
                title=f"Hist of x_0 and v_d refit {suffix}",
                save_path=plot_save_path + f"refit_vd_vs_x0_{suffix}{plot_type}",
            )

            data_to_hist_2d(
                data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
                data_y=sl_refits_cuts['vd_refit'] * vd_factor,
                x_label="alpha",
                y_label="v_d",
                title=f"Hist of alpha vs vd refit {suffix}",
                save_path=plot_save_path + f"refit_vd_vs_alpha_{suffix}{plot_type}",
            )

            data_to_hist_2d(
                data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
                data_y=sl_refits_cuts["x0_refit"],
                x_label="alpha",
                y_label="x_0",
                title=f"Hist of alpha vs x_0 refit {suffix}",
                save_path=plot_save_path + f"refit_x0_vs_tanalpha_{suffix}{plot_type}",
            )

            data_to_hist_2d(
                data_x=sl_refits_cuts["x0_refit"],
                data_y=sl_refits_cuts["dt0_refit"] * derived_params._ts_unit,
                x_label="x_0[mm]",
                y_label="dt_0 [ns]",
                title=f"Hist of x_0 vs dt_0 refit {suffix}",
                save_path=plot_save_path + f"refit_dt_0_vs_x0_{suffix}{plot_type}",
            )

            for i in range(4):
                data_to_hist_2d(
                    data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
                    data_y=sl_refits_cuts[f"dt{i}_refit"] * derived_params._ts_unit,
                    x_label="alpha",
                    y_label=f"dt_{i} [ns]",
                    title=f"Hist of dt_{i} vs alpha refit {suffix}",
                    save_path=plot_save_path + f"refit_dt{i}_vs_alpha_{suffix}{plot_type}",
                )
        """


    if not do_ramp_measurement:
            
        plt.figure(figsize=fig_size)

        # Get all unique wire voltages
        unique_u_wires = sorted(set(
            parse_fit_name(name=dataset)["U_wire"]
            for dataset in analysis_out.keys()
        ))

        # Create one color per wire voltage
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_u_wires)))

        # Map voltage -> color
        wire_color_map = dict(zip(unique_u_wires, colors))

        for dataset, result in analysis_out.items():
            dataset_info = parse_fit_name(name=dataset)

            pct_ar = dataset_info["pct_Ar"]
            pct_co2 = dataset_info["pct_CO2"]
            u_wire = dataset_info["U_wire"]

            mean_vd = result["mean_1"]
            err_mean_vd = result["mean_1_err"]

            plt.errorbar(
                pct_ar,
                mean_vd,
                yerr=err_mean_vd,
                fmt="o",
                capsize=4,
                markersize=6,
                color=wire_color_map[u_wire],
                label=f"{pct_ar}/{pct_co2}, $U_{{wire}}={u_wire}$ V"
            )

        plt.xlabel("Ar concentration [%]")
        plt.ylabel(r"$v_d$ [$\mu$m/ns]")
        plt.title("Comparison of gas mixtures and drift velocities")
        plt.grid(True)

        # Avoid duplicate legend entries for the same voltage
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys())

        plt.tight_layout()
        plt.savefig(base_path + f"plots/vd_track_fit_comparison{plot_type}")



    #When doing ramp measurement, this loop is used
    if do_ramp_measurement:
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

        for dataset, result in analysis_out.items():
            times.append(parse_start_time(dataset))
            values.append(result["mean_1"])
            errors.append(result["mean_1_err"])

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
        plt.title(r"Drift velocity over time ($U_{\mathrm{wire}}=3600$ V) Track-fit method")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_save_path + f"ramp_analysis_track_fit{plot_type}")

    return


if __name__ == "__main__":
    main()
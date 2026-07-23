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
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    
    


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
            path = f"{plot_save_path}{dataset_name}_DIFF_SPECIFIC_{filename_suffix}{plot_type}"
            if verbose:
                print(f"store histogram plot as {path}.")
            fig.savefig(path)
            if verbose:
                print(f"Done saving hist as {path}\n")
    
        return fig, ax, path
 

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




    list_of_fits = ["cosmic_85-15_3600-1800-1200_run2_th20_cut", "cosmic_85-15_3550-1800-1200_test1", 
                    "cosmic_85-15_3000-1500-1000_test3", "cosmic_85-15_3550-1800-1200_test1", 
                    "cosmic_82-18_3600-1800-1200_test1_th20"]
    
    base_path = "data_ba/"

    plot_type = ".png"
    dataset_name = "cosmic_82-18_3550-1800-1200_run1_th20_cut_50"

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


    vd_factor = 1 / derived_params._drift_velocity_conversion



    
    
    # Dataset info from name; Use parse_fit_name to extract information from dataset name
    dataset_info = parse_fit_name(name = dataset_name)
    pct_ar = dataset_info["pct_Ar"]
    pct_co2 = dataset_info["pct_CO2"]
    u_wire = dataset_info["U_wire"]
    u_fieldshaper = dataset_info["U_Fieldshaper"]
    u_catheode = dataset_info["U_cathode"]

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
    


    

    print(super_fits.keys())














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


    #hist of all interesting hist metrics
    for i in range(len(good_super_fit_keys)):
        key = good_super_fit_keys[i][0]
        title =good_super_fit_keys[i][1] + f" {pct_ar}/{pct_co2} Ar/CO2 U_wire = {u_wire} {no_cut}"
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
            filename_suffix=safe_key + "_" + no_cut,
            scale_factor = factor,
            title = title,
            xlabel = x_label,
            ylabel = y_label,
            plot_type = plot_type,

        )

    plt.close("all")

    # done with all hists


    # beginning hist2d plots with all possible flagged hists


    

    data_to_hist_2d(
        data_x=super_fits_cuts["x0_free_vd_super_fit"],
        data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
        x_label="x0",
        y_label="v_d",
        title=f"Hist of x_0 and v_d {no_cut}",
        save_path=plot_save_path + f"vd_vs_x0_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
        data_y=super_fits_cuts['vd_free_vd_super_fit'] * vd_factor,
        x_label="alpha",
        y_label="v_d",
        title=f"Hist of alpha vs vd {no_cut}",
        save_path=plot_save_path + f"vd_vs_alpha_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
        data_y=super_fits_cuts["x0_free_vd_super_fit"],
        x_label="alpha",
        y_label="x_0",
        title=f"Hist of alpha vs x_0 {no_cut}",
        save_path=plot_save_path + f"x0_vs_tanalpha_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=super_fits_cuts["x0_free_vd_super_fit"],
        data_y=super_fits_cuts["dt0_free_vd_super_fit"] * derived_params._ts_unit,
        x_label="x_0[mm]",
        y_label="dt_0 [ns]",
        title=f"Hist of x_0 vs dt_0 {no_cut}",
        save_path=plot_save_path + f"dt_0_vs_x0_{no_cut}{plot_type}",
    )

    for i in range(8):
        data_to_hist_2d(
            data_x=np.rad2deg(np.arctan(super_fits_cuts["tan_alpha_free_vd_super_fit"])),
            data_y=super_fits_cuts[f"dt{i}_free_vd_super_fit"] * derived_params._ts_unit,
            x_label="alpha",
            y_label=f"dt_{i} [ns]",
            title=f"Hist of dt_{i} vs alpha {no_cut}",
            save_path=plot_save_path + f"dt{i}_vs_alpha_{no_cut}{plot_type}",
        )


    
    # The analysis of only possible cuts ends here











    # The analyisis of more restrictive cuts beginns here
    super_fits_cuts = data_utils.cut_data(
        data=super_fits,
        conditions=[
            ("impossible_free_vd_super_fit", "==", 0),
            ("chi2/ndf_free_vd_super_fit", "<", 10),
            #("chi2/ndf_free_vd_super_fit", ">", 0.5),
            #("vd_free_vd_super_fit", "<", 50 * derived_params._drift_velocity_conversion),
            ("err_t0_free_vd_super_fit", "<", 10),
            #("err_t0_free_vd_super_fit", ">", 0.1),
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













    # The analysis of refits beginns here
    # cuts for four cell fits only possible hists
    sl_refits_cuts = data_utils.cut_data(
        data=sl_refits,
        conditions=[
            ("impossible_refit", "==", 0),
        ],
        silent=True,
    )

    # analysis of refits 
    data_to_hist_2d(
        data_x=sl_refits_cuts["x0_refit"],
        data_y=sl_refits_cuts['vd_refit'] * vd_factor,
        x_label="x0",
        y_label="v_d",
        title=f"Hist of x_0 and v_d refit {no_cut}",
        save_path=plot_save_path + f"refit_vd_vs_x0_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
        data_y=sl_refits_cuts['vd_refit'] * vd_factor,
        x_label="alpha",
        y_label="v_d",
        title=f"Hist of alpha vs vd refit {no_cut}",
        save_path=plot_save_path + f"refit_vd_vs_alpha_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
        data_y=sl_refits_cuts["x0_refit"],
        x_label="alpha",
        y_label="x_0",
        title=f"Hist of alpha vs x_0 refit {no_cut}",
        save_path=plot_save_path + f"refit_x0_vs_tanalpha_{no_cut}{plot_type}",
    )

    data_to_hist_2d(
        data_x=sl_refits_cuts["x0_refit"],
        data_y=sl_refits_cuts["dt0_refit"] * derived_params._ts_unit,
        x_label="x_0[mm]",
        y_label="dt_0 [ns]",
        title=f"Hist of x_0 vs dt_0 refit",
        save_path=plot_save_path + f"refit_dt_0_vs_x0_{no_cut}{plot_type}",
    )

    for i in range(4):
        data_to_hist_2d(
            data_x=np.rad2deg(np.arctan(sl_refits_cuts["tan_alpha_refit"])),
            data_y=sl_refits_cuts[f"dt{i}_refit"] * derived_params._ts_unit,
            x_label="alpha",
            y_label=f"dt_{i} [ns]",
            title=f"Hist of dt_{i} vs alpha refit {no_cut}",
            save_path=plot_save_path + f"refit_dt{i}_vs_alpha_{no_cut}{plot_type}",
        )


    return


if __name__ == "__main__":
    main()
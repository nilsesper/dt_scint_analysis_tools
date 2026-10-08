#################################################################
### sl fits: plots which combine several branches
# - drift times of all four layers in one histogram
# - fit residuals (measured hit time - fitted hit time), all layers and per layer
# - drift velocity in um/ns (for fits with free drift velocity)
# - time between consecutive fits, number and rate of fits per superlayer
# Histograms of the single branches: plot_histograms.py
#
# examples:
#   python scripts/plot_sl_fits.py --sl_fits_file out/run_sl_fits.root --store_plots plots/sl_fits --cuts "impossible,==,0;chi2/ndf,<,20"
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import cut_utils, data_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

### one histogram with the cuts as title, stored as plot_name
def plot_one_histogram(data, xlabel, plot_name, cut_title, args, scale=1.0, bin_unit=None, full_range=False, log_scale=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    edges = plot_utils.choose_edges(data, n_bins=args.n_bins, full_range=full_range)
    plot_utils.draw_histogram(ax, data, edges, xlabel=xlabel, log_scale=log_scale, bin_unit=bin_unit, scale=scale)
    ax.set_title(cut_title, fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_name, args)

### title text of the cuts, e.g. "cuts: impossible == 0, chi2/ndf < 20"
def make_cut_title(cuts):
    cut_texts = []
    for cut in cuts:
        cut_texts.append(f"{cut[0]} {cut[1]} {cut[2]:g}")
    return "cuts: " + ", ".join(cut_texts)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Drift time, residual and rate plots of sl fits.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: sl fits (.root)")
    parser.add_argument("--suffix", type=str, default="", help="suffix of the fit result branches to plot, if the fit was stored with one (default: none)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\" "
                             "(default: \"impossible<suffix>,==,0\")")
    parser.add_argument("--n_bins", type=int, default=100, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    sfx = args.suffix
    name = "sl_fits" + sfx

    ### cuts
    if args.cuts is not None:
        cuts = cut_utils.parse_cuts(args.cuts)
    else:
        cuts = [("impossible" + sfx, "==", 0)]

    ### data import (only the needed branches)
    keys = ["sl", "t0" + sfx, "vd" + sfx]
    for ly in range(4):
        keys.append(f"ts{ly}")
    for ly in range(4):
        keys.append(f"err_ts{ly}")
    for ly in range(4):
        keys.append(f"dt{ly}{sfx}")
    for cut in cuts:
        keys.append(cut[0])
    unique_keys = []
    for key in keys:
        if key not in unique_keys:
            unique_keys.append(key)
    keys = sorted(unique_keys)
    log(f"###### Importing sl fits from {args.sl_fits_file}...")
    sl_fits = root_utils.read_branches(args.sl_fits_file, keys)
    n_all = root_utils.length(sl_fits)
    sl_fits = data_utils.cut_data(data=sl_fits, conditions=cuts, silent=True)
    n_sl_fits = root_utils.length(sl_fits)
    cut_title = make_cut_title(cuts)
    log(f"{cut_title}: {n_sl_fits:,} / {n_all:,} fits selected")
    if n_sl_fits == 0:
        raise RuntimeError("No fits pass the cuts, nothing to plot.")

    ### fitted drift times of all layers
    drift_times_per_layer = []
    for ly in range(4):
        drift_times_per_layer.append(sl_fits[f"dt{ly}{sfx}"])
    drift_times = np.concatenate(drift_times_per_layer)
    plot_one_histogram(drift_times, "$t_\\text{drift}$ of all layers [ns]", f"{name}_drift_time", cut_title, args,
                       scale=plot_utils.TS_UNIT_NS, bin_unit="ns", log_scale=args.log_scale)

    ### fit residuals: measured hit time - fitted hit time, where the fitted hit time is t0 + fitted drift time
    residuals = {}
    for ly in range(4):
        ts = np.asarray(sl_fits[f"ts{ly}"], dtype=np.float64)
        residuals[ly] = ts - sl_fits["t0" + sfx] - sl_fits[f"dt{ly}{sfx}"]
    residuals_per_layer = []
    for ly in range(4):
        residuals_per_layer.append(residuals[ly])
    all_residuals = np.concatenate(residuals_per_layer)
    plot_one_histogram(all_residuals, "Fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]", f"{name}_residuals", cut_title, args,
                       scale=plot_utils.TS_UNIT_NS, bin_unit="ns", log_scale=args.log_scale)

    ### fit residuals per layer, all in one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    edges = plot_utils.choose_edges(all_residuals, n_bins=args.n_bins)
    top = 1
    for ly in range(4):
        hist, _ = np.histogram(residuals[ly], bins=edges)
        mean_ns = np.mean(residuals[ly]) * plot_utils.TS_UNIT_NS
        std_ns = np.std(residuals[ly]) * plot_utils.TS_UNIT_NS
        ax.stairs(hist, edges * plot_utils.TS_UNIT_NS, linewidth=2, label=f"Layer {ly}: mean {mean_ns:.2f} ns, std {std_ns:.2f} ns")
        top = max(top, hist.max())
    if args.log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=top * np.exp(1.1))
    else:
        ax.set_ylim(bottom=0, top=top * 1.3)
    ax.set_xlabel("Fit residual $T_{ly} - T_{ly}^\\text{fit}$ [ns]")
    ax.set_ylabel("Counts")
    ax.set_title(cut_title, fontsize=12)
    ax.legend(prop={"size": 14}, fancybox=False)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{name}_residuals_per_layer", args)

    ### pull: residual divided by the hit time uncertainty
    pulls_per_layer = []
    for ly in range(4):
        pulls_per_layer.append(residuals[ly] / sl_fits[f"err_ts{ly}"])
    pulls = np.concatenate(pulls_per_layer)
    finite = np.isfinite(pulls)
    plot_one_histogram(pulls[finite], "Fit residual / hit time uncertainty", f"{name}_residual_pulls", cut_title, args,
                       log_scale=args.log_scale)

    ### drift velocity in um/ns (only meaningful if it was a free fit parameter)
    vd_um_per_ns = sl_fits["vd" + sfx] / derived_params._drift_velocity_conversion
    if np.amin(vd_um_per_ns) != np.amax(vd_um_per_ns):
        plot_one_histogram(vd_um_per_ns, "$v_\\text{drift}$ [um/ns]", f"{name}_drift_velocity", cut_title, args,
                           bin_unit="um/ns", log_scale=args.log_scale)
    else:
        log(f"drift velocity is the same for all fits ({vd_um_per_ns[0]:.2f} um/ns, fixed in the fit): no drift velocity plot")

    ### time between consecutive fits (all superlayers together, sorted by t0)
    t0_sorted = np.sort(sl_fits["t0" + sfx])
    if n_sl_fits > 1:
        delta_t0 = np.diff(t0_sorted)
        plot_one_histogram(delta_t0, "Time between consecutive fits $\\Delta T_0$ [ms]", f"{name}_delta_t0", cut_title, args,
                           scale=plot_utils.TS_UNIT_NS * 1e-6, bin_unit="ms", log_scale=True)

    ### number and rate of fits per superlayer
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"]))
    log(f"measurement duration = {duration} s")
    for sl in params._dt_chamber["sls"].keys():
        count = int(np.sum(sl_fits["sl"] == sl))
        if duration > 0:
            rate = f"{count / duration:.3f} +- {np.sqrt(count) / duration:.3f} Hz"
        else:
            rate = "n/a"
        log(f"sl={sl}: {count:,} fits, rate {rate}")

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

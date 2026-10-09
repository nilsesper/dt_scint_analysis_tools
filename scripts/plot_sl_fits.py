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

from analysis_tools.utils import data_utils, root_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import derived_params

# ---------------------------------------------------------------

### title text of the cuts, e.g. "cuts: impossible == 0, chi2/ndf < 20"
def make_cut_title(cuts):
    cut_texts = []
    for cut in cuts:
        cut_texts.append(f"{cut[0]} {cut[1]} {cut[2]:g}")
    return "cuts: " + ", ".join(cut_texts)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main():
    parser = argparse.ArgumentParser(description="Drift time, residual and rate plots of sl fits.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: sl fits (.root)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\" "
                             "(default: \"impossible,==,0\")")
    parser.add_argument("--n_bins", type=int, default=100, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)
    name = "sl_fits"
    ns = plot_utils.TS_UNIT_NS

    ### cuts
    root_utils.check_input_file(args.sl_fits_file)
    if args.cuts is not None:
        cuts = data_utils.parse_cuts(args.cuts)
    else:
        cuts = [("impossible", "==", 0)]

    ### data import (only the needed branches)
    keys = ["sl", "t0", "vd"]
    for ly in range(4):
        keys.append(f"ts{ly}")
        keys.append(f"err_ts{ly}")
        keys.append(f"dt{ly}")
    for cut in cuts:
        if cut[0] not in keys:
            keys.append(cut[0])
    log(f"###### Importing sl fits from {args.sl_fits_file}...")
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    n_all = root_utils.length(sl_fits)
    sl_fits = data_utils.cut_data(sl_fits, cuts, silent=True)
    n_sl_fits = root_utils.length(sl_fits)
    cut_title = make_cut_title(cuts)
    log(f"{cut_title}: {n_sl_fits:,} / {n_all:,} fits selected")
    if n_sl_fits == 0:
        raise RuntimeError("No fits pass the cuts, nothing to plot.")

    ### fitted drift times of all layers
    drift_times_per_layer = []
    for ly in range(4):
        drift_times_per_layer.append(sl_fits[f"dt{ly}"])
    drift_times = np.concatenate(drift_times_per_layer)
    plot_utils.plot_histogram(drift_times, f"{name}_drift_time", args, xlabel="$t_\\text{drift}$ of all layers [ns]", title=cut_title,
                              scale=ns, bin_unit="ns")

    ### fit residuals: measured hit time - fitted hit time, where the fitted hit time is t0 + fitted drift time
    residuals = []
    for ly in range(4):
        ts = np.asarray(sl_fits[f"ts{ly}"], dtype=np.float64)
        residuals.append(ts - sl_fits["t0"] - sl_fits[f"dt{ly}"])
    all_residuals = np.concatenate(residuals)
    plot_utils.plot_histogram(all_residuals, f"{name}_residuals", args, xlabel="Fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]",
                              title=cut_title, scale=ns, bin_unit="ns")

    ### fit residuals per layer, all in one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    edges = plot_utils.choose_edges(all_residuals, n_bins=args.n_bins)
    top = 1
    for ly in range(4):
        hist, _ = np.histogram(residuals[ly], bins=edges)
        mean_ns = np.mean(residuals[ly]) * ns
        std_ns = np.std(residuals[ly]) * ns
        ax.stairs(hist, edges * ns, linewidth=2, label=f"Layer {ly}: mean {mean_ns:.2f} ns, std {std_ns:.2f} ns")
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
    plot_utils.plot_histogram(pulls, f"{name}_residual_pulls", args, xlabel="Fit residual / hit time uncertainty", title=cut_title)

    ### drift velocity in um/ns (only meaningful if it was a free fit parameter)
    vd_um_per_ns = sl_fits["vd"] / derived_params._drift_velocity_conversion
    if np.amin(vd_um_per_ns) != np.amax(vd_um_per_ns):
        plot_utils.plot_histogram(vd_um_per_ns, f"{name}_drift_velocity", args, xlabel="$v_\\text{drift}$ [um/ns]", title=cut_title, bin_unit="um/ns")
    else:
        log(f"drift velocity is the same for all fits ({vd_um_per_ns[0]:.2f} um/ns, fixed in the fit): no drift velocity plot")

    ### time between consecutive fits (all superlayers together, sorted by t0)
    if n_sl_fits > 1:
        delta_t0 = np.diff(np.sort(sl_fits["t0"]))
        plot_utils.plot_histogram(delta_t0, f"{name}_delta_t0", args, xlabel="Time between consecutive fits $\\Delta T_0$ [ms]", title=cut_title,
                                  scale=ns * 1e-6, bin_unit="ms", log_scale=True)

    ### number and rate of fits per superlayer
    duration = ns * 1e-9 * float(np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"]))
    log(f"measurement duration = {duration} s")
    for sl in dt_chamber_utils.superlayers():
        count = int(np.sum(sl_fits["sl"] == sl))
        if duration > 0:
            rate = f"{count / duration:.3f} +- {np.sqrt(count) / duration:.3f} Hz"
        else:
            rate = "n/a"
        log(f"sl={sl}: {count:,} fits, rate {rate}")

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

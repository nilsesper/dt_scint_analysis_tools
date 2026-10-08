#################################################################
### sl fits: plots which combine several branches
# - drift times of all four layers in one histogram
# - fit residuals (measured hit time - fitted hit time), all layers and per layer
# - drift velocity in um/ns (for fits with free drift velocity)
# - time between consecutive fits, number and rate of fits per superlayer
# Histograms of the single branches: plot_histograms.py
#
# examples:
#   python scripts/dt_root/plot_sl_fits.py --sl_fits_file out/run_sl_fits.root --store_plots plots/sl_fits --cuts "impossible,==,0;chi2/ndf,<,20"
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import data_utils, dt_pipeline_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

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
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)
    sfx = args.suffix
    name = "sl_fits" + sfx

    ### data import (only the needed branches)
    cuts = dt_pipeline_utils.parse_cuts(args.cuts) if args.cuts is not None else [("impossible" + sfx, "==", 0)]
    keys = ["sl", "t0" + sfx, "vd" + sfx] + [f"ts{ly}" for ly in range(4)] + [f"err_ts{ly}" for ly in range(4)] + [f"dt{ly}{sfx}" for ly in range(4)]
    keys = sorted(set(keys) | {c[0] for c in cuts})
    log(f"###### Importing sl fits from {args.sl_fits_file}...")
    sl_fits = root_utils.read_branches(args.sl_fits_file, keys)
    n_all = root_utils.length(sl_fits)
    sl_fits = data_utils.cut_data(data=sl_fits, conditions=cuts, silent=True)
    n_sl_fits = root_utils.length(sl_fits)
    cut_title = "cuts: " + ", ".join(f"{c[0]} {c[1]} {c[2]:g}" for c in cuts)
    log(f"{cut_title}: {n_sl_fits:,} / {n_all:,} fits selected")
    if n_sl_fits == 0:
        raise RuntimeError("No fits pass the cuts, nothing to plot.")

    def _hist(data, xlabel, plot_name, *, scale=1.0, bin_unit=None, full_range=False, log_scale=args.log_scale):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        edges = plot_utils.choose_edges(data, n_bins=args.n_bins, full_range=full_range)
        plot_utils.draw_histogram(ax, data, edges, xlabel=xlabel, log_scale=log_scale, bin_unit=bin_unit, scale=scale)
        ax.set_title(cut_title, fontsize=12)
        fig.tight_layout()
        plot_utils.finish_figure(fig, plot_name, **out)

    ### fitted drift times of all layers
    drift_times = np.concatenate([sl_fits[f"dt{ly}{sfx}"] for ly in range(4)])
    _hist(drift_times, "$t_\\text{drift}$ of all layers [ns]", f"{name}_drift_time", scale=plot_utils.TS_UNIT_NS, bin_unit="ns")

    ### fit residuals: measured hit time - fitted hit time, where the fitted hit time is t0 + fitted drift time
    residuals = {ly: np.asarray(sl_fits[f"ts{ly}"], dtype=np.float64) - sl_fits["t0" + sfx] - sl_fits[f"dt{ly}{sfx}"] for ly in range(4)}
    all_residuals = np.concatenate([residuals[ly] for ly in range(4)])
    _hist(all_residuals, "Fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]", f"{name}_residuals", scale=plot_utils.TS_UNIT_NS, bin_unit="ns")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    edges = plot_utils.choose_edges(all_residuals, n_bins=args.n_bins)
    top = 1
    for ly in range(4):
        hist, _ = np.histogram(residuals[ly], bins=edges)
        ax.stairs(hist, edges * plot_utils.TS_UNIT_NS, linewidth=2,
                  label=f"Layer {ly}: mean {np.mean(residuals[ly]) * plot_utils.TS_UNIT_NS:.2f} ns, std {np.std(residuals[ly]) * plot_utils.TS_UNIT_NS:.2f} ns")
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
    plot_utils.finish_figure(fig, f"{name}_residuals_per_layer", **out)
    # pull: residual divided by the hit time uncertainty
    pulls = np.concatenate([residuals[ly] / sl_fits[f"err_ts{ly}"] for ly in range(4)])
    _hist(pulls[np.isfinite(pulls)], "Fit residual / hit time uncertainty", f"{name}_residual_pulls")

    ### drift velocity in um/ns (only meaningful if it was a free fit parameter)
    vd_um_per_ns = sl_fits["vd" + sfx] / derived_params._drift_velocity_conversion
    if np.amin(vd_um_per_ns) != np.amax(vd_um_per_ns):
        _hist(vd_um_per_ns, "$v_\\text{drift}$ [um/ns]", f"{name}_drift_velocity", bin_unit="um/ns")
    else:
        log(f"drift velocity is the same for all fits ({vd_um_per_ns[0]:.2f} um/ns, fixed in the fit): no drift velocity plot")

    ### time between consecutive fits (all superlayers together, sorted by t0)
    t0_sorted = np.sort(sl_fits["t0" + sfx])
    if n_sl_fits > 1:
        delta_t0 = np.diff(t0_sorted)
        _hist(delta_t0, "Time between consecutive fits $\\Delta T_0$ [ms]", f"{name}_delta_t0", scale=plot_utils.TS_UNIT_NS * 1e-6, bin_unit="ms", log_scale=True)

    ### number and rate of fits per superlayer
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(sl_fits["ts0"]) - np.amin(sl_fits["ts0"]))
    log(f"measurement duration = {duration} s")
    for sl in params._dt_chamber["sls"].keys():
        count = int(np.sum(sl_fits["sl"] == sl))
        rate = f"{count / duration:.3f} +- {np.sqrt(count) / duration:.3f} Hz" if duration > 0 else "n/a"
        log(f"sl={sl}: {count:,} fits, rate {rate}")

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

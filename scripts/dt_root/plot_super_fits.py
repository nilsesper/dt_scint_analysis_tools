#################################################################
### super fits: fit quality and comparison with the two sl fits they were built from
# - fit residuals (measured - fitted hit time) of all 8 layers together and per layer
# - drift times of all layers, drift velocity (if it was a free fit parameter)
# - super fit - sl fit: arrival time T0, slope tan(alpha) and track position in the middle of each superlayer
# - sl fit of the first phi sl - sl fit of the second phi sl: T0 and tan(alpha)
# (histograms of the single branches: plot_histograms.py, single super fits: singleplot_super_fit.py)
#
# example:
#   python scripts/dt_root/plot_super_fits.py --input_file out/run_super_fits_cut.root --store_plots plots/super_fits
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
    parser = argparse.ArgumentParser(description="Fit quality of super fits and comparison with their sl fits.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path: super fits (.root)")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\" "
                             "(default: \"impossible<suffix>,==,0\")")
    parser.add_argument("--n_bins", type=int, default=80, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)
    sfx = args.suffix
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]

    ### data import (only the needed branches)
    root_utils.check_input_file(args.input_file)
    if "t0" + sfx not in root_utils.list_branches(args.input_file):
        raise KeyError(f"No super fit results with suffix \"{sfx}\" in {args.input_file}.")
    cuts = dt_pipeline_utils.parse_cuts(args.cuts) if args.cuts is not None else [("impossible" + sfx, "==", 0)]
    keys = [k + sfx for k in ["t0", "x0", "tan_alpha", "vd", "err_vd", "ref_x", "ref_z", "ts_residual"]] + [f"dt{j}{sfx}" for j in range(8)] + [f"err_ts{j}" for j in range(8)]
    for sl in phi_sls:
        keys += [f"t0_sl{sl}", f"x0_sl{sl}", f"tan_alpha_sl{sl}", f"wi3_sl{sl}"]
    keys = sorted(set(keys) | {c[0] for c in cuts})
    log(f"###### Importing super fits from {args.input_file}...")
    fits = root_utils.read_branches(args.input_file, keys)
    n_all = root_utils.length(fits)
    fits = data_utils.cut_data(data=fits, conditions=cuts, silent=True)
    n_fits = root_utils.length(fits)
    cut_title = "cuts: " + ", ".join(f"{c[0]} {c[1]} {c[2]:g}" for c in cuts)
    log(f"{cut_title}: {n_fits} / {n_all} super fits selected")
    if n_fits == 0:
        raise RuntimeError("No super fits pass the cuts, nothing to plot.")

    def _hist(data, xlabel, plot_name, *, scale=1.0, bin_unit=None):
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        edges = plot_utils.choose_edges(data, n_bins=args.n_bins)
        plot_utils.draw_histogram(ax, data, edges, xlabel=xlabel, log_scale=args.log_scale, bin_unit=bin_unit, scale=scale)
        ax.set_title(cut_title, fontsize=12)
        fig.tight_layout()
        plot_utils.finish_figure(fig, plot_name, **out)

    def _overlay(datasets, xlabel, plot_name, *, scale=1.0, unit=""):
        # several distributions in one plot, same binning; datasets = [(label, values)]
        all_values = np.concatenate([v for _, v in datasets])
        edges = plot_utils.choose_edges(all_values[np.isfinite(all_values)], n_bins=args.n_bins)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        top = 1
        for label, values in datasets:
            hist, _ = np.histogram(values, bins=edges)
            ax.stairs(hist, edges * scale, linewidth=2, label=f"{label}: mean {np.mean(values) * scale:.2f}{unit}, std {np.std(values) * scale:.2f}{unit}")
            top = max(top, hist.max())
        if args.log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5, top=top * np.exp(1.5))
        else:
            ax.set_ylim(bottom=0, top=top * (1.1 + 0.08 * len(datasets)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(cut_title, fontsize=12)
        ax.legend(prop={"size": 13}, fancybox=False, ncols=2 if len(datasets) > 4 else 1)
        fig.tight_layout()
        plot_utils.finish_figure(fig, plot_name, **out)

    ns = plot_utils.TS_UNIT_NS
    ### fit residuals: measured - fitted hit time ("ts_residual" is fitted - measured)
    residuals = -np.asarray(fits["ts_residual" + sfx], dtype=np.float64)  # shape (n_fits, 8)
    _hist(residuals.ravel(), "Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]", "super_fits_residuals", scale=ns, bin_unit="ns")
    layer_names = [f"SL {sl} Ly {ly}" for sl in phi_sls for ly in range(4)]
    _overlay([(layer_names[j], residuals[:, j]) for j in range(8)], "Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ [ns]", "super_fits_residuals_per_layer", scale=ns, unit=" ns")
    err_ts = np.stack([fits[f"err_ts{j}"] for j in range(8)], axis=1)
    _hist((residuals / err_ts).ravel(), "Super fit residual / hit time uncertainty", "super_fits_residual_pulls")

    ### drift times of all layers
    _hist(np.concatenate([fits[f"dt{j}{sfx}"] for j in range(8)]), "$t_\\text{drift}$ of all layers [ns]", "super_fits_drift_time", scale=ns, bin_unit="ns")

    ### drift velocity in um/ns (only if it was a free fit parameter)
    vd_um_per_ns = fits["vd" + sfx] / derived_params._drift_velocity_conversion
    if np.any(fits["err_vd" + sfx] != 0):
        _hist(vd_um_per_ns, "$v_\\text{drift}$ [um/ns]", "super_fits_drift_velocity", bin_unit="um/ns")
    else:
        log(f"drift velocity was fixed in the super fit ({vd_um_per_ns[0]:.2f} um/ns): no drift velocity plot")

    ### super fit - sl fit
    # the super fit frame has the top wire of the super pattern at (0, 0), the sl fit frame the wire of ly 3
    x_ref_super = derived_params._super_pattern_x_ref + fits["ref_x" + sfx]
    z_ref_super = derived_params._super_pattern_z_ref + fits["ref_z" + sfx]
    delta_t0, delta_tan_alpha, delta_x = [], [], []
    for sl in phi_sls:
        wires = sorted(derived_params._dt_cell_coordinates[sl][3].keys())
        x_wire = np.full(max(wires) + 1, np.nan)
        for wi in wires:
            x_wire[wi] = derived_params._dt_cell_coordinates[sl][3][wi][3]
        x_ref_sl = x_wire[fits[f"wi3_sl{sl}"].astype(np.intp)]
        z_ref_sl = derived_params._dt_cell_coordinates[sl][3][wires[0]][5]
        z_mid = derived_params.sl_z_center[sl]
        x_sl = derived_params.f_x_muon(z=z_mid - z_ref_sl, x0=fits[f"x0_sl{sl}"], tan_alpha=fits[f"tan_alpha_sl{sl}"]) + x_ref_sl
        x_super = derived_params.f_x_muon(z=z_mid - z_ref_super, x0=fits["x0" + sfx], tan_alpha=fits["tan_alpha" + sfx]) + x_ref_super
        delta_t0.append((f"SL {sl}", fits[f"t0_sl{sl}"] - fits["t0" + sfx]))
        delta_tan_alpha.append((f"SL {sl}", fits[f"tan_alpha_sl{sl}"] - fits["tan_alpha" + sfx]))
        delta_x.append((f"SL {sl}", x_sl - x_super))
    _overlay(delta_t0, "$T_0$(SL fit) $-$ $T_0$(super fit) [ns]", "super_fits_vs_sl_fits_t0", scale=ns, unit=" ns")
    _overlay(delta_tan_alpha, "$\\tan\\alpha$(SL fit) $-$ $\\tan\\alpha$(super fit)", "super_fits_vs_sl_fits_tan_alpha")
    _overlay(delta_x, "Track position in the middle of the superlayer: SL fit $-$ super fit [mm]", "super_fits_vs_sl_fits_position", unit=" mm")

    ### the two sl fits against each other
    a, b = phi_sls[0], phi_sls[1]
    _hist(fits[f"t0_sl{a}"] - fits[f"t0_sl{b}"], f"$T_0$(SL {a} fit) $-$ $T_0$(SL {b} fit) [ns]", "super_fits_sl_fits_delta_t0", scale=ns, bin_unit="ns")
    _hist(fits[f"tan_alpha_sl{a}"] - fits[f"tan_alpha_sl{b}"], f"$\\tan\\alpha$(SL {a} fit) $-$ $\\tan\\alpha$(SL {b} fit)", "super_fits_sl_fits_delta_tan_alpha")

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

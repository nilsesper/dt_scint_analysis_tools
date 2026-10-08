#################################################################
### super fits: fit quality and comparison with the two sl fits they were built from
# - fit residuals (measured - fitted hit time) of all 8 layers together and per layer
# - drift times of all layers, drift velocity (if it was a free fit parameter)
# - super fit - sl fit: arrival time T0, slope tan(alpha) and track position in the middle of each superlayer
# - sl fit of the first phi sl - sl fit of the second phi sl: T0 and tan(alpha)
# (histograms of the single branches: plot_histograms.py, single super fits: singleplot_super_fit.py)
#
# example:
#   python scripts/plot_super_fits.py --super_fits_file out/run_super_fits_cut.root --store_plots plots/super_fits
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import cut_utils, data_utils, dt_chamber_utils, dt_geometry_utils as geometry, dt_pipeline_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

### one histogram of the finite values of data, with the cuts as title, stored as plot_name
def plot_one_histogram(data, xlabel, plot_name, cut_title, args, scale=1.0, bin_unit=None):
    data = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(data)
    data = data[finite]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    edges = plot_utils.choose_edges(data, n_bins=args.n_bins)
    plot_utils.draw_histogram(ax, data, edges, xlabel=xlabel, log_scale=args.log_scale, bin_unit=bin_unit, scale=scale)
    ax.set_title(cut_title, fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_name, args)

### several distributions in one plot with the same binning
# labels[i] is the legend label of the values values_list[i]
def plot_overlay(labels, values_list, xlabel, plot_name, cut_title, args, scale=1.0, unit=""):
    all_values = np.concatenate(values_list)
    finite = np.isfinite(all_values)
    edges = plot_utils.choose_edges(all_values[finite], n_bins=args.n_bins)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    top = 1
    for i in range(len(values_list)):
        values = values_list[i]
        hist, _ = np.histogram(values, bins=edges)
        mean = np.mean(values) * scale
        std = np.std(values) * scale
        ax.stairs(hist, edges * scale, linewidth=2, label=f"{labels[i]}: mean {mean:.2f}{unit}, std {std:.2f}{unit}")
        top = max(top, hist.max())
    if args.log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=top * np.exp(1.5))
    else:
        ax.set_ylim(bottom=0, top=top * (1.1 + 0.08 * len(values_list)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(cut_title, fontsize=12)
    if len(values_list) > 4:
        n_legend_columns = 2
    else:
        n_legend_columns = 1
    ax.legend(prop={"size": 13}, fancybox=False, ncols=n_legend_columns)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_name, args)

### title text of the cuts, e.g. "cuts: impossible_super_fits == 0"
def make_cut_title(cuts):
    cut_texts = []
    for cut in cuts:
        cut_texts.append(f"{cut[0]} {cut[1]} {cut[2]:g}")
    return "cuts: " + ", ".join(cut_texts)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit quality of super fits and comparison with their sl fits.")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: super fits (.root)")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\" "
                             "(default: \"impossible<suffix>,==,0\")")
    parser.add_argument("--n_bins", type=int, default=80, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    sfx = args.suffix
    phi_sls = []
    for sl in params._dt_chamber["sls"].keys():
        if params._dt_chamber["sls"][sl]["orient"] == "phi":
            phi_sls.append(sl)

    ### cuts
    root_utils.check_input_file(args.super_fits_file)
    if "t0" + sfx not in root_utils.list_branches(args.super_fits_file):
        raise KeyError(f"No super fit results with suffix \"{sfx}\" in {args.super_fits_file}.")
    if args.cuts is not None:
        cuts = cut_utils.parse_cuts(args.cuts)
    else:
        cuts = [("impossible" + sfx, "==", 0)]

    ### data import (only the needed branches)
    keys = []
    for key in ["t0", "x0", "tan_alpha", "vd", "err_vd", "ref_x", "ref_z", "ts_residual"]:
        keys.append(key + sfx)
    for j in range(8):
        keys.append(f"dt{j}{sfx}")
    for j in range(8):
        keys.append(f"err_ts{j}")
    for sl in phi_sls:
        keys += [f"t0_sl{sl}", f"x0_sl{sl}", f"tan_alpha_sl{sl}", f"wi3_sl{sl}"]
    for cut in cuts:
        keys.append(cut[0])
    unique_keys = []
    for key in keys:
        if key not in unique_keys:
            unique_keys.append(key)
    keys = sorted(unique_keys)
    log(f"###### Importing super fits from {args.super_fits_file}...")
    fits = root_utils.read_branches(args.super_fits_file, keys)
    n_all = root_utils.length(fits)
    fits = data_utils.cut_data(data=fits, conditions=cuts, silent=True)
    n_fits = root_utils.length(fits)
    cut_title = make_cut_title(cuts)
    log(f"{cut_title}: {n_fits:,} / {n_all:,} super fits selected")
    if n_fits == 0:
        raise RuntimeError("No super fits pass the cuts, nothing to plot.")

    ns = plot_utils.TS_UNIT_NS
    ### fit residuals: measured - fitted hit time ("ts_residual" is fitted - measured)
    residuals = -np.asarray(fits["ts_residual" + sfx], dtype=np.float64)  # shape (n_fits, 8)
    plot_one_histogram(residuals.ravel(), "Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]", "super_fits_residuals",
                       cut_title, args, scale=ns, bin_unit="ns")
    layer_names = []
    for sl in phi_sls:
        for ly in range(4):
            layer_names.append(f"SL {sl} Ly {ly}")
    residuals_per_layer = []
    for j in range(8):
        residuals_per_layer.append(residuals[:, j])
    plot_overlay(layer_names, residuals_per_layer, "Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ [ns]", "super_fits_residuals_per_layer",
                 cut_title, args, scale=ns, unit=" ns")
    # pull: residual divided by the hit time uncertainty
    err_ts_per_layer = []
    for j in range(8):
        err_ts_per_layer.append(fits[f"err_ts{j}"])
    err_ts = np.stack(err_ts_per_layer, axis=1)
    pulls = residuals / err_ts
    plot_one_histogram(pulls.ravel(), "Super fit residual / hit time uncertainty", "super_fits_residual_pulls", cut_title, args)

    ### drift times of all layers
    drift_times_per_layer = []
    for j in range(8):
        drift_times_per_layer.append(fits[f"dt{j}{sfx}"])
    drift_times = np.concatenate(drift_times_per_layer)
    plot_one_histogram(drift_times, "$t_\\text{drift}$ of all layers [ns]", "super_fits_drift_time", cut_title, args, scale=ns, bin_unit="ns")

    ### drift velocity in um/ns (only if it was a free fit parameter)
    vd_um_per_ns = fits["vd" + sfx] / derived_params._drift_velocity_conversion
    if np.any(fits["err_vd" + sfx] != 0):
        plot_one_histogram(vd_um_per_ns, "$v_\\text{drift}$ [um/ns]", "super_fits_drift_velocity", cut_title, args, bin_unit="um/ns")
    else:
        log(f"drift velocity was fixed in the super fit ({vd_um_per_ns[0]:.2f} um/ns): no drift velocity plot")

    ### super fit - sl fit
    # the super fit frame has the top wire of the super pattern at (0, 0), the sl fit frame the wire of ly 3
    x_ref_super = geometry.SUPER_FRAME_ORIGIN[0] + fits["ref_x" + sfx]
    z_ref_super = geometry.SUPER_FRAME_ORIGIN[1] + fits["ref_z" + sfx]
    sl_labels = []
    delta_t0 = []
    delta_tan_alpha = []
    delta_x = []
    for sl in phi_sls:
        # x position of every wire of ly 3, the array index is the wire number
        x_wire = np.full(max(dt_chamber_utils.wires(sl, 3)) + 1, np.nan)
        for wi in dt_chamber_utils.wires(sl, 3):
            x_wire[wi] = geometry.cell(sl, 3, wi)["center"][geometry.X]
        wire_numbers = fits[f"wi3_sl{sl}"].astype(np.intp)
        x_ref_sl = x_wire[wire_numbers]
        z_ref_sl = geometry.layer_z(sl, 3)
        z_mid = geometry.superlayer_box(sl)["center"][geometry.Z]
        x_sl = geometry.track_position(z=z_mid - z_ref_sl, x0=fits[f"x0_sl{sl}"], tan_alpha=fits[f"tan_alpha_sl{sl}"]) + x_ref_sl
        x_super = geometry.track_position(z=z_mid - z_ref_super, x0=fits["x0" + sfx], tan_alpha=fits["tan_alpha" + sfx]) + x_ref_super
        sl_labels.append(f"SL {sl}")
        delta_t0.append(fits[f"t0_sl{sl}"] - fits["t0" + sfx])
        delta_tan_alpha.append(fits[f"tan_alpha_sl{sl}"] - fits["tan_alpha" + sfx])
        delta_x.append(x_sl - x_super)
    plot_overlay(sl_labels, delta_t0, "$T_0$(SL fit) $-$ $T_0$(super fit) [ns]", "super_fits_vs_sl_fits_t0",
                 cut_title, args, scale=ns, unit=" ns")
    plot_overlay(sl_labels, delta_tan_alpha, "$\\tan\\alpha$(SL fit) $-$ $\\tan\\alpha$(super fit)", "super_fits_vs_sl_fits_tan_alpha",
                 cut_title, args)
    plot_overlay(sl_labels, delta_x, "Track position in the middle of the superlayer: SL fit $-$ super fit [mm]", "super_fits_vs_sl_fits_position",
                 cut_title, args, unit=" mm")

    ### the two sl fits against each other
    a = phi_sls[0]
    b = phi_sls[1]
    plot_one_histogram(fits[f"t0_sl{a}"] - fits[f"t0_sl{b}"], f"$T_0$(SL {a} fit) $-$ $T_0$(SL {b} fit) [ns]", "super_fits_sl_fits_delta_t0",
                       cut_title, args, scale=ns, bin_unit="ns")
    plot_one_histogram(fits[f"tan_alpha_sl{a}"] - fits[f"tan_alpha_sl{b}"], f"$\\tan\\alpha$(SL {a} fit) $-$ $\\tan\\alpha$(SL {b} fit)",
                       "super_fits_sl_fits_delta_tan_alpha", cut_title, args)

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

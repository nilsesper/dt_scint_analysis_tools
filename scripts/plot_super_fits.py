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

from analysis_tools.utils import data_utils, dt_fit_utils, root_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import derived_params

# ---------------------------------------------------------------

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
def main():
    parser = argparse.ArgumentParser(description="Fit quality of super fits and comparison with their sl fits.")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: super fits (.root)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\" "
                             "(default: \"impossible_super_fits,==,0\")")
    parser.add_argument("--n_bins", type=int, default=80, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)
    sfx = dt_fit_utils.SUPER_FIT_SUFFIX
    phi_sls = dt_chamber_utils.phi_superlayers()
    ns = plot_utils.TS_UNIT_NS

    ### cuts
    root_utils.check_input_file(args.super_fits_file)
    if "t0" + sfx not in root_utils.branch_names(args.super_fits_file, root_utils.DEFAULT_TREE):
        raise KeyError(f"No super fit results (\"t0{sfx}\") in {args.super_fits_file}.")
    if args.cuts is not None:
        cuts = data_utils.parse_cuts(args.cuts)
    else:
        cuts = [("impossible" + sfx, "==", 0)]

    ### data import (only the needed branches)
    keys = []
    for key in ["t0", "x0", "tan_alpha", "vd", "err_vd", "ref_x", "ref_z", "ts_residual"]:
        keys.append(key + sfx)
    for j in range(8):
        keys.append(f"dt{j}{sfx}")
        keys.append(f"err_ts{j}")
    for sl in phi_sls:
        keys += [f"t0_sl{sl}", f"x0_sl{sl}", f"tan_alpha_sl{sl}", f"wi3_sl{sl}"]
    for cut in cuts:
        if cut[0] not in keys:
            keys.append(cut[0])
    log(f"###### Importing super fits from {args.super_fits_file}...")
    fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    n_all = root_utils.length(fits)
    fits = data_utils.cut_data(fits, cuts, silent=True)
    n_fits = root_utils.length(fits)
    cut_title = make_cut_title(cuts)
    log(f"{cut_title}: {n_fits:,} / {n_all:,} super fits selected")
    if n_fits == 0:
        raise RuntimeError("No super fits pass the cuts, nothing to plot.")

    ### fit residuals: measured - fitted hit time ("ts_residual" is fitted - measured), shape (n_fits, 8)
    residuals = -np.asarray(fits["ts_residual" + sfx], dtype=np.float64)
    plot_utils.plot_histogram(residuals.ravel(), "super_fits_residuals", args, xlabel="Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]",
                              title=cut_title, scale=ns, bin_unit="ns")
    layer_names = []
    residuals_per_layer = []
    for i in range(len(phi_sls)):
        for ly in range(4):
            layer_names.append(f"SL {phi_sls[i]} Ly {ly}")
            residuals_per_layer.append(residuals[:, 4 * i + ly])
    plot_overlay(layer_names, residuals_per_layer, "Super fit residual $T_{ly} - T_{ly}^\\text{fit}$ [ns]", "super_fits_residuals_per_layer",
                 cut_title, args, scale=ns, unit=" ns")

    ### pull: residual divided by the hit time uncertainty
    pulls = []
    for j in range(8):
        pulls.append(residuals[:, j] / fits[f"err_ts{j}"])
    plot_utils.plot_histogram(np.concatenate(pulls), "super_fits_residual_pulls", args, xlabel="Super fit residual / hit time uncertainty", title=cut_title)

    ### drift times of all layers
    drift_times_per_layer = []
    for j in range(8):
        drift_times_per_layer.append(fits[f"dt{j}{sfx}"])
    drift_times = np.concatenate(drift_times_per_layer)
    plot_utils.plot_histogram(drift_times, "super_fits_drift_time", args, xlabel="$t_\\text{drift}$ of all layers [ns]", title=cut_title,
                              scale=ns, bin_unit="ns")

    ### drift velocity in um/ns (only if it was a free fit parameter)
    vd_um_per_ns = fits["vd" + sfx] / derived_params._drift_velocity_conversion
    if np.any(fits["err_vd" + sfx] != 0):
        plot_utils.plot_histogram(vd_um_per_ns, "super_fits_drift_velocity", args, xlabel="$v_\\text{drift}$ [um/ns]", title=cut_title, bin_unit="um/ns")
    else:
        log(f"drift velocity was fixed in the super fit ({vd_um_per_ns[0]:.2f} um/ns): no drift velocity plot")

    ### super fit - sl fit: T0, tan(alpha) and track position in the middle of the superlayer (chamber frame)
    sl_labels = []
    delta_t0 = []
    delta_tan_alpha = []
    delta_x = []
    for sl in phi_sls:
        # chamber position of the reference wire (layer 3, wire wi3) of every sl fit
        h_ref_sl = np.zeros(n_fits)
        z_ref_sl = np.zeros(n_fits)
        for i in range(n_fits):
            h_ref_sl[i], z_ref_sl[i] = dt_chamber_utils.wire_position(sl, 3, int(fits[f"wi3_sl{sl}"][i]))
        z_mid = dt_chamber_utils.superlayer_box(sl)["center"][dt_chamber_utils.Z]
        x_sl = dt_chamber_utils.track_position_in_chamber(h_ref_sl, z_ref_sl, fits[f"x0_sl{sl}"], fits[f"tan_alpha_sl{sl}"], z_mid)
        x_super = dt_chamber_utils.track_position_in_chamber(fits["ref_x" + sfx], fits["ref_z" + sfx], fits["x0" + sfx], fits["tan_alpha" + sfx], z_mid)
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
    plot_utils.plot_histogram(fits[f"t0_sl{a}"] - fits[f"t0_sl{b}"], "super_fits_sl_fits_delta_t0", args,
                              xlabel=f"$T_0$(SL {a} fit) $-$ $T_0$(SL {b} fit) [ns]", title=cut_title, scale=ns, bin_unit="ns")
    plot_utils.plot_histogram(fits[f"tan_alpha_sl{a}"] - fits[f"tan_alpha_sl{b}"], "super_fits_sl_fits_delta_tan_alpha", args,
                              xlabel=f"$\\tan\\alpha$(SL {a} fit) $-$ $\\tan\\alpha$(SL {b} fit)", title=cut_title)

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

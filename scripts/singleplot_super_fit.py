#################################################################
### event display of single super fits, together with the two sl fits they were built from
# for every selected super fit two figures:
#   - the 8 hit timestamps (4 layers of each phi superlayer) with the timestamps fitted by the super fit and by
#     the two sl fits, and the fit residuals below
#   - the cells of both phi superlayers around the track, with the hit positions (from the drift times of the
#     super fit), the two sl fit track segments and the super fit track
# The super fits file holds everything needed (hits, sl fit results, super fit results).
#
# Super fits are selected by their row in the file (--rows), or the first ones passing --cuts are taken (--n_fits).
#
# examples:
#   python scripts/singleplot_super_fit.py --super_fits_file out/run_super_fits_cut.root --rows 0,10 --store_plots plots/single_super_fits
#   python scripts/singleplot_super_fit.py --super_fits_file out/run_super_fits.root --n_fits 5 \
#          --cuts "chi2/ndf_super_fits,>,10" --store_plots plots/single_super_fits_bad
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import cut_utils, data_utils, dt_chamber_utils, dt_geometry_utils as geometry, dt_pipeline_utils, geoplot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

### rows of the first n super fits which pass the cuts
def select_rows(input_file, cuts, n_fits):
    cut_keys = []
    for cut in cuts:
        if cut[0] not in cut_keys:
            cut_keys.append(cut[0])
    cut_data = root_utils.read_branches(input_file, sorted(cut_keys))
    cut_data["__row"] = np.arange(root_utils.length(cut_data))
    passing_rows = data_utils.cut_data(data=cut_data, conditions=cuts, silent=True)["__row"]
    log(f"{len(passing_rows):,} super fits pass the cuts {cuts}, taking the first {min(n_fits, len(passing_rows)):,}")
    rows = []
    for row in passing_rows[:n_fits]:
        rows.append(int(row))
    return rows

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Event display of single super fits with their two sl fits.")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: super fits (.root)")
    parser.add_argument("--rows", type=str, default=None, help="rows of the super fits to plot, separated by \",\"")
    parser.add_argument("--n_fits", type=int, default=5, help="if --rows is not given: number of super fits to plot (the first ones passing --cuts)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="if --rows is not given: selection of the super fits, format \"key1,operator1,value1;...\" "
                             "(default: \"impossible<suffix>,==,0\")")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--zoom_width", type=float, default=300, help="width of the cell view in mm")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    sfx = args.suffix

    ### which super fits
    root_utils.check_input_file(args.super_fits_file)
    if "t0" + sfx not in root_utils.list_branches(args.super_fits_file):
        raise KeyError(f"No super fit results with suffix \"{sfx}\" in {args.super_fits_file}.")
    if args.rows is not None:
        rows = []
        for text in args.rows.split(","):
            if text.strip() != "":
                rows.append(int(text))
    else:
        if args.cuts is not None:
            cuts = cut_utils.parse_cuts(args.cuts)
        else:
            cuts = [("impossible" + sfx, "==", 0)]
        rows = select_rows(args.super_fits_file, cuts, args.n_fits)
    if len(rows) == 0:
        raise RuntimeError("No super fits selected, nothing to plot.")
    super_fits = root_utils.read_rows(args.super_fits_file, rows)

    ### the two phi superlayers, their colours and the labels of the 8 layers
    phi_sls = []
    for sl in params._dt_chamber["sls"].keys():
        if params._dt_chamber["sls"][sl]["orient"] == "phi":
            phi_sls.append(sl)
    pat_names = list(params._dt_sl_patterns.keys())
    sl_colors = {phi_sls[0]: "tab:red", phi_sls[1]: "tab:orange"}
    layer_labels = []
    for sl in phi_sls:
        for ly in range(4):
            layer_labels.append(f"SL {sl}\nLy {ly}")
    idx8 = np.arange(8)

    for i in range(len(rows)):
        row = rows[i]
        fit = {}
        for key in super_fits.keys():
            fit[key] = super_fits[key][i]
        ### hits: index 0-3 = four layers of the first phi sl, 4-7 = four layers of the second phi sl
        ts = np.zeros(8, dtype=np.float64)
        err_ts = np.zeros(8, dtype=np.float64)
        for j in range(8):
            ts[j] = fit[f"ts{j}"]
            err_ts[j] = fit[f"err_ts{j}"]
        wires = {}
        for sl in phi_sls:
            wires[sl] = []
            for ly in range(4):
                wires[sl].append(int(fit[f"wi{ly}_sl{sl}"]))
        x_cell_list = []
        z_cell_list = []
        for sl in phi_sls:
            for ly in range(4):
                x_wire, z_wire = geometry.wire_position(sl, ly, wires[sl][ly])
                x_cell_list.append(x_wire)
                z_cell_list.append(z_wire)
        x_cell = np.array(x_cell_list)
        z_cell = np.array(z_cell_list)
        ### super fit results; the fit frame has the top wire of the super pattern at (0, 0)
        t0 = fit["t0" + sfx]
        x0 = fit["x0" + sfx]
        tan_alpha = fit["tan_alpha" + sfx]
        vd = fit["vd" + sfx]
        err_t0 = fit["err_t0" + sfx]
        err_x0 = fit["err_x0" + sfx]
        err_tan_alpha = fit["err_tan_alpha" + sfx]
        err_vd = fit["err_vd" + sfx]
        corr_x0_tan_alpha = fit["corr_x0_tan_alpha" + sfx]
        chi2ndf = fit["chi2/ndf" + sfx]
        vd_um_per_ns = vd / derived_params._drift_velocity_conversion
        err_vd_um_per_ns = err_vd / derived_params._drift_velocity_conversion
        vd_is_free = err_vd != 0
        x_ref = geometry.SUPER_FRAME_ORIGIN[0] + fit["ref_x" + sfx]
        z_ref = geometry.SUPER_FRAME_ORIGIN[1] + fit["ref_z" + sfx]
        pat_type = {}
        for sl in phi_sls:
            pat_type[sl] = int(fit[f"pat_type_sl{sl}"])
        lat_ids = {phi_sls[0]: int(fit["lat_id1" + sfx]), phi_sls[1]: int(fit["lat_id2" + sfx])}
        laterality_list = []
        for sl in phi_sls:
            sl_laterality = params._dt_sl_patterns[pat_names[pat_type[sl]]]["laterality"][lat_ids[sl]]
            for ly in range(4):
                laterality_list.append(sl_laterality[ly])
        laterality = np.array(laterality_list, dtype=np.float64)
        # timestamps fitted by the super fit ("ts_residual" is fitted - measured timestamp)
        super_fit_ts = ts + np.asarray(fit["ts_residual" + sfx], dtype=np.float64)
        ### sl fit results (each in its own frame with the wire of ly 3 at (0, 0))
        sl_fit = {}
        for n in range(len(phi_sls)):
            sl = phi_sls[n]
            ref_wi = wires[sl][3]
            fit_ts_list = []
            for ly in range(4):
                fit_ts_list.append(fit[f"t0_sl{sl}"] + fit[f"dt{ly}_sl{sl}"])
            sl_fit[sl] = {
                "t0": fit[f"t0_sl{sl}"],
                "x0": fit[f"x0_sl{sl}"],
                "tan_alpha": fit[f"tan_alpha_sl{sl}"],
                "chi2/ndf": fit[f"chi2/ndf_sl{sl}"],
                "err_x0": fit[f"err_x0_sl{sl}"],
                "err_tan_alpha": fit[f"err_tan_alpha_sl{sl}"],
                "corr_x0_tan_alpha": fit[f"corr_x0_tan_alpha_sl{sl}"],
                "x_ref": geometry.wire_position(sl, 3, ref_wi)[0],
                "z_ref": geometry.wire_position(sl, 3, ref_wi)[1],
                "fit_ts": np.array(fit_ts_list),
                "laterality": params._dt_sl_patterns[pat_names[pat_type[sl]]]["laterality"][int(fit[f"laterality_sl{sl}"])],
                "idx": idx8[4 * n: 4 * n + 4],
            }
            # distance sl fit - super fit in the middle of the superlayer
            z_mid = geometry.superlayer_box(sl)["center"][geometry.Z]
            x_sl = geometry.track_position(z=z_mid - sl_fit[sl]["z_ref"], x0=sl_fit[sl]["x0"], tan_alpha=sl_fit[sl]["tan_alpha"]) + sl_fit[sl]["x_ref"]
            x_super = geometry.track_position(z=z_mid - z_ref, x0=x0, tan_alpha=tan_alpha) + x_ref
            sl_fit[sl]["dx"] = x_sl - x_super
        ### print
        impossible_text = ""
        if fit["impossible" + sfx]:
            impossible_text = " (flagged impossible)"
        if vd_is_free:
            err_vd_text = f" +- {err_vd_um_per_ns:.2f}"
        else:
            err_vd_text = " (fixed)"
        log(f"super fit, row {row}{impossible_text}:")
        log(f"  super fit: t0 = {t0:.2f} +- {err_t0:.2f} TU, tan_alpha = {tan_alpha:.4f} +- {err_tan_alpha:.4f}, "
              f"vd = {vd_um_per_ns:.2f}{err_vd_text} um/ns, chi2/ndf = {chi2ndf:.3f}")
        for sl in phi_sls:
            f_sl = sl_fit[sl]
            if f"row_sl{sl}" in fit:
                sl_fit_row_text = str(int(fit[f"row_sl{sl}"]))
            else:
                sl_fit_row_text = "?"
            sl_fit_laterality = []
            for lat in f_sl["laterality"]:
                sl_fit_laterality.append(int(lat))
            super_fit_laterality = []
            for j in f_sl["idx"]:
                super_fit_laterality.append(int(laterality[j]))
            log(f"  sl {sl} fit (row {sl_fit_row_text} of the sl fits file): pattern {pat_type[sl]}, wires {wires[sl]}, "
                  f"laterality sl fit {sl_fit_laterality} / super fit {super_fit_laterality}")
            log(f"      t0 = {f_sl['t0']:.2f} TU (sl - super: {f_sl['t0'] - t0:+.2f}), tan_alpha = {f_sl['tan_alpha']:.4f} (sl - super: {f_sl['tan_alpha'] - tan_alpha:+.4f}), "
                  f"chi2/ndf = {f_sl['chi2/ndf']:.3f}, position sl - super = {f_sl['dx']:+.2f} mm")
        log(f"  super fit residuals measured - fitted [TU] = {np.round(ts - super_fit_ts, 2)}")
        title = f"Super fit row {row}: SL {phi_sls[0]} pattern {pat_type[phi_sls[0]]} + SL {phi_sls[1]} pattern {pat_type[phi_sls[1]]}"
        super_label = f"""Super fit (8 hits):
$T_0=({t0 - np.floor(np.amin(ts)):.0f}\\pm{err_t0:.0f})$ {params._key_units['t0']}
$\\tan\\alpha=({tan_alpha:.4f}\\pm{err_tan_alpha:.4f})$
$v_d={vd_um_per_ns:.1f}$ um/ns"""
        if vd_is_free:
            super_label += f" $\\pm{err_vd_um_per_ns:.1f}$"
        else:
            super_label += " (fixed)"
        super_label += f"\n$\\chi^2/N_{{df}}={chi2ndf:.2f}$"

        ################################
        ###### timestamps of the 8 layers with super fit and sl fits, and residuals
        fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True, height_ratios=(3, 1))
        ts_ref = np.floor(np.amin(ts))
        ax[0].errorbar(x=idx8, y=ts - ts_ref, yerr=err_ts, color="black", marker="o", markersize=7, linestyle="", label="Hit timestamps", zorder=5)
        ax[0].plot(idx8 + 0.12, super_fit_ts - ts_ref, color="tab:blue", marker="v", markersize=8, linestyle="", label=super_label)
        ax[0].axhline(y=t0 - ts_ref, color="tab:blue", linestyle="--", linewidth=1.5)
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(x=idx8 + 0.12, y=ts - super_fit_ts, yerr=err_ts, color="tab:blue", marker="v", markersize=7, linestyle="")
        for sl in phi_sls:
            f_sl = sl_fit[sl]
            sl_label = f"SL {sl} fit (4 hits):\n$T_0={f_sl['t0'] - ts_ref:.0f}$ {params._key_units['t0']}, $\\tan\\alpha={f_sl['tan_alpha']:.3f}$\n$\\chi^2/N_{{df}}={f_sl['chi2/ndf']:.2f}$"
            sl_t0_line = f_sl["t0"] - ts_ref
            ax[0].plot(f_sl["idx"] - 0.12, f_sl["fit_ts"] - ts_ref, color=sl_colors[sl], marker="^", markersize=8, linestyle="", label=sl_label)
            ax[0].plot([f_sl["idx"][0] - 0.4, f_sl["idx"][-1] + 0.4], [sl_t0_line, sl_t0_line], color=sl_colors[sl], linestyle=":", linewidth=1.5)
            ax[1].errorbar(x=f_sl["idx"] - 0.12, y=ts[f_sl["idx"]] - f_sl["fit_ts"], yerr=err_ts[f_sl["idx"]], color=sl_colors[sl], marker="^", markersize=7, linestyle="")
        ax[0].axvline(x=3.5, color="gray", linewidth=1)
        ax[1].axvline(x=3.5, color="gray", linewidth=1)
        ax[0].set_ylabel(f"Timestamp $T - {ts_ref:.0f}$ [TU]")
        ax[0].set_title(title + " (dashed / dotted lines: fitted $T_0$)", fontsize=15)
        ax[0].legend(prop={"size": 13}, fancybox=False, framealpha=params._legend_alpha, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax[1].set_ylabel("Residuals [TU]")
        ax[1].set_xticks(idx8)
        ax[1].set_xticklabels(layer_labels)
        fig.align_ylabels(ax)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.08)
        plot_utils.save_figure(fig, f"super_fit_row{row}_timestamps", args)

        ################################
        ###### cells of both phi superlayers with hit positions, sl fits and super fit (global chamber frame)
        dt_cell_data = dt_chamber_utils.cell_display_map()
        for sl in phi_sls:
            for ly in range(4):
                dt_cell_data[sl][ly][wires[sl][ly]]["color"] = "aqua"
        z_range = np.linspace(np.amin(z_cell) - params._plot_z_margin * 1.5, np.amax(z_cell) + params._plot_z_margin * 1.5, 600)
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        ax = geoplot_utils.chamber_ax(ax=ax, orient="phi", cell_data=dt_cell_data, wire=True)
        # hit positions from the drift times of the super fit: x = x_wire + laterality * (T - T0) * vd
        x_hits = x_cell + laterality * (ts - t0) * vd
        err_x_hits = np.sqrt((vd * err_ts) ** 2 + (vd * err_t0) ** 2)
        ax.errorbar(x=x_hits, y=z_cell, xerr=err_x_hits, color="black", marker="o", markersize=6, linestyle="", label="Hit positions (super fit $T_0$, $v_d$)", zorder=8)
        # super fit track with uncertainty band
        track = geometry.track_position(z=z_range - z_ref, x0=x0, tan_alpha=tan_alpha) + x_ref
        err_track = geometry.err_track_position(z=z_range - z_ref, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        ax.plot(track, z_range, linewidth=2, color="tab:blue", label=super_label, zorder=7)
        ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:blue", alpha=0.2, zorder=6)
        # sl fit segments with uncertainty band
        for sl in phi_sls:
            f_sl = sl_fit[sl]
            sl_z_range = np.linspace(geometry.layer_z(sl, 0) - params._plot_z_margin, geometry.layer_z(sl, 3) + params._plot_z_margin, 200)
            sl_track = geometry.track_position(z=sl_z_range - f_sl["z_ref"], x0=f_sl["x0"], tan_alpha=f_sl["tan_alpha"]) + f_sl["x_ref"]
            err_sl_track = geometry.err_track_position(z=sl_z_range - f_sl["z_ref"], x0=f_sl["x0"], tan_alpha=f_sl["tan_alpha"], err_x0=f_sl["err_x0"],
                                                       err_tan_alpha=f_sl["err_tan_alpha"], corr_x0_tan_alpha=f_sl["corr_x0_tan_alpha"])
            sl_label = f"SL {sl} fit: $\\tan\\alpha={f_sl['tan_alpha']:.3f}$, $\\chi^2/N_{{df}}={f_sl['chi2/ndf']:.2f}$\nSL fit $-$ super fit: {f_sl['dx']:+.2f} mm"
            ax.plot(sl_track, sl_z_range, color=sl_colors[sl], linewidth=3, linestyle="--", label=sl_label, zorder=5)
            ax.fill_betweenx(x1=sl_track - err_sl_track, x2=sl_track + err_sl_track, y=sl_z_range, color=sl_colors[sl], alpha=0.2, zorder=4)
        x_center = np.mean(track)
        ax.set_xlim(x_center - args.zoom_width / 2, x_center + args.zoom_width / 2)
        ax.set_ylim(np.amin(z_range), np.amax(z_range))
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_title(title + ", $x$-$z$-plane", fontsize=15)
        ax.legend(prop={"size": 13}, fancybox=False, framealpha=0.9, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        plot_utils.save_figure(fig, f"super_fit_row{row}_track", args)

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

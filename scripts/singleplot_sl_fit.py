#################################################################
### event display of single sl fits
# for every selected fit two figures:
#   - hit timestamps and fitted timestamps per layer, with the fit residuals below
#   - the pattern cells with the hit positions (from the drift times) and the fitted track
#
# Fits are selected by their row in the file (--rows), or the first fits passing --cuts are taken (--n_fits).
#
# examples:
#   python scripts/singleplot_sl_fit.py --sl_fits_file out/run_sl_fits.root --rows 500,600,700 --store_plots plots/single_fits
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import data_utils, dt_chamber_utils, dt_fit_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

### rows of the first n fits which pass the cuts
def select_rows(input_file, cuts, n_fits):
    cut_keys = []
    for cut in cuts:
        if cut[0] not in cut_keys:
            cut_keys.append(cut[0])
    cut_data = root_utils.read_tree(input_file, root_utils.DEFAULT_TREE, branches=sorted(cut_keys))
    cut_data["__row"] = np.arange(root_utils.length(cut_data))
    passing_rows = data_utils.cut_data(data=cut_data, conditions=cuts, silent=True)["__row"]
    log(f"{len(passing_rows):,} fits pass the cuts {cuts}, taking the first {min(n_fits, len(passing_rows)):,}")
    rows = []
    for row in passing_rows[:n_fits]:
        rows.append(int(row))
    return rows

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main():
    parser = argparse.ArgumentParser(description="Event display and fit residuals of single sl fits.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: sl fits (.root)")
    parser.add_argument("--rows", type=str, default=None, help="rows of the fits to plot, separated by \",\"")
    parser.add_argument("--n_fits", type=int, default=5, help="if --rows is not given: number of fits to plot (the first ones passing --cuts)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="if --rows is not given: selection of the fits, format \"key1,operator1,value1;...\" "
                             "(default: \"impossible,==,0\")")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    ### which fits
    root_utils.check_input_file(args.sl_fits_file)
    if args.rows is not None:
        rows = []
        for text in args.rows.split(","):
            if text.strip() != "":
                rows.append(int(text))
    else:
        if args.cuts is not None:
            cuts = data_utils.parse_cuts(args.cuts)
        else:
            cuts = [("impossible", "==", 0)]
        rows = select_rows(args.sl_fits_file, cuts, args.n_fits)
    if len(rows) == 0:
        raise RuntimeError("No fits selected, nothing to plot.")

    for row in rows:
        sl_fit = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, row, row + 1)
        fit = {}
        for key in sl_fit.keys():
            fit[key] = sl_fit[key][0]
        ### data
        lys = np.arange(0, 4)
        ts = np.zeros(4, dtype=np.float64)
        err_ts = np.zeros(4, dtype=np.float64)
        wires = []
        for ly in range(4):
            ts[ly] = fit[f"ts{ly}"]
            err_ts[ly] = fit[f"err_ts{ly}"]
            wires.append(int(fit[f"wi{ly}"]))
        sl = int(fit["sl"])
        ### pattern
        pat_type = int(fit["pat_type"])
        pat_name = list(params._dt_sl_patterns.keys())[pat_type]  # pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"]  # list of [lat for ly0,1,2,3] laterality lists
        lat_idx = int(fit["laterality"])
        laterality = np.array(lats[lat_idx])
        laterality_list = []
        for lat in laterality:
            laterality_list.append(int(lat))
        ### fit results (vd in mm / timestamp unit, as used by the fit function)
        t0 = fit["t0"]
        x0 = fit["x0"]
        tan_alpha = fit["tan_alpha"]
        vd = fit["vd"]
        err_t0 = fit["err_t0"]
        err_x0 = fit["err_x0"]
        err_tan_alpha = fit["err_tan_alpha"]
        err_vd = fit["err_vd"]
        corr_t0_x0 = fit["corr_t0_x0"]
        corr_t0_tan_alpha = fit["corr_t0_tan_alpha"]
        corr_t0_vd = fit["corr_t0_vd"]
        corr_x0_tan_alpha = fit["corr_x0_tan_alpha"]
        corr_x0_vd = fit["corr_x0_vd"]
        corr_tan_alpha_vd = fit["corr_tan_alpha_vd"]
        chi2ndf = fit["chi2/ndf"]
        impossible = bool(fit["impossible"])
        vd_um_per_ns = vd / derived_params._drift_velocity_conversion
        err_vd_um_per_ns = err_vd / derived_params._drift_velocity_conversion
        ### wire positions of the pattern in the track frame (wire of layer 3 at (0, 0))
        wi3 = wires[3]
        z_arr = np.zeros(4, dtype=np.float64)
        x_cell = np.zeros(4, dtype=np.float64)
        for ly in lys:
            x_cell[ly], z_arr[ly] = dt_chamber_utils.position_in_track_frame(sl, ly, wires[ly], sl, wi3)
        ### fitted timestamps
        fit_ts = np.zeros(4)
        err_fit_ts = np.zeros(4)
        for ly in lys:
            fit_ts[ly] = dt_fit_utils.hit_time(h_wire=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z_wire=z_arr[ly], laterality=laterality[ly], vd=vd)
            err_fit_ts[ly] = dt_fit_utils.err_hit_time(
                h_wire=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z_wire=z_arr[ly], laterality=laterality[ly], vd=vd,
                err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd,
                corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=corr_t0_vd,
                corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=corr_x0_vd, corr_tan_alpha_vd=corr_tan_alpha_vd,
            )
        residuals = ts - fit_ts
        ### print
        impossible_text = ""
        if impossible:
            impossible_text = " (flagged impossible)"
        log(f"sl fit, row {row}{impossible_text}:")
        log(f"  sl = {sl}, pat_type = {pat_type} ({pat_name}), wires = {wires}, laterality = {laterality_list} (idx {lat_idx})")
        log(f"  t0 = {t0:.2f} +- {err_t0:.2f} TU, x0 = {x0:.3f} +- {err_x0:.3f} mm, tan_alpha = {tan_alpha:.4f} +- {err_tan_alpha:.4f}")
        log(f"  vd = {vd_um_per_ns:.2f} +- {err_vd_um_per_ns:.2f} um/ns, chi2/ndf = {chi2ndf:.3f}")
        log(f"  residuals [TU] = {np.round(residuals, 2)}")
        title = f"SL {sl} ({params._dt_chamber['sls'][sl]['orient']}), Pattern {pat_type}, Laterality {laterality_list}, row {row}"

        ################################
        ###### timestamps per layer with fit, and residuals
        fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, height_ratios=(3, 1))
        ts_ref = np.floor(np.amin(ts))  # plot relative to the first hit, absolute timestamps are too large to read
        ts_label = "Hit timestamps"
        ax[0].errorbar(x=lys - 0.02, y=ts - ts_ref, yerr=err_ts, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        fit_label = f"""Track fit:
$T_0=({t0 - ts_ref:.0f}\\pm{err_t0:.0f})$ {params._key_units['t0']}
$x_0=({x0:.1f}\\pm{err_x0:.1f})$ {params._key_units['x0']}
$\\tan\\alpha=({tan_alpha:.2f}\\pm{err_tan_alpha:.2f})$
$v_d=({vd_um_per_ns:.1f}\\pm{err_vd_um_per_ns:.1f})$ um/ns
$\\chi^2/N_{{df}}={chi2ndf:.2f}$"""
        ax[0].errorbar(x=lys + 0.02, y=fit_ts - ts_ref, yerr=err_fit_ts, color="tab:red", marker="v", markersize=7, linestyle="", label=fit_label)
        ax[0].axhline(y=t0 - ts_ref, color="tab:green", linestyle="--", linewidth=1.5, label="Arrival time $T_0$")
        ax[0].set_ylabel(f"Timestamp $T_{{ly}} - {ts_ref:.0f}$ [TU]")
        ax[0].legend(prop={"size": 14}, fancybox=False, framealpha=params._legend_alpha, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax[0].set_title(title, fontsize=16)
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(x=lys, y=residuals, yerr=err_ts, color="black", marker="o", markersize=7, linestyle="")
        res_lim = max(np.amax(np.abs(residuals) + err_ts) * 1.1, 1e-9)
        ax[1].set_ylim(-res_lim, res_lim)
        ax[1].set_ylabel("Residuals [TU]")
        fig.align_ylabels(ax)
        ax[1].set_xlabel("Layer $ly$")
        ax[1].set_xticks([0, 1, 2, 3])
        ax[1].set_xticklabels(["0", "1", "2", "3"])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        plot_utils.save_figure(fig, f"sl_fit_row{row}_timestamps", args)

        ################################
        ###### pattern cells with hit positions and fitted track (local frame)
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        plot_utils.draw_pattern_cells(ax, sl, wi3, wires)
        # hit positions from the measured drift times
        x_hits = x_cell + laterality * (ts - t0) * vd
        err_x_hits = np.sqrt(np.clip(
              (laterality * vd) ** 2 * err_ts ** 2
            + (-laterality * vd) ** 2 * err_t0 ** 2
            + (laterality * (ts - t0)) ** 2 * err_vd ** 2
            + 2 * (-laterality * vd) * (laterality * (ts - t0)) * corr_t0_vd,
            0, None,
        ))
        ax.errorbar(x=x_hits, y=z_arr, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label="Hit positions", zorder=5)
        # fitted track with uncertainty band
        z_range = np.linspace(np.amin(z_arr) - params._plot_z_margin, np.amax(z_arr) + params._plot_z_margin, 1000)
        track = dt_chamber_utils.track_position(z=z_range, x0=x0, tan_alpha=tan_alpha)
        err_track = dt_chamber_utils.err_track_position(z=z_range, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        ax.plot(track, z_range, linewidth=2, color="tab:red", label=fit_label, zorder=4)
        ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:red", alpha=0.2, zorder=3)
        ax.legend(prop={"size": 14}, loc="center left", bbox_to_anchor=(1.01, 0.5), fancybox=False, framealpha=params._legend_alpha)
        axis_name = "x" if params._dt_chamber["sls"][sl]["orient"] == "phi" else "y"
        ax.set_xlabel(f"${axis_name}-{axis_name}_\\text{{wire,ly=3}}$ [mm]")
        ax.set_ylabel("$z-z_\\text{wire,ly=3}$ [mm]")
        ax.set_ylim(np.amin(z_range), np.amax(z_range))
        half_width = 2.6 * params._dt_cell_width
        ax.set_xlim(-half_width, half_width)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        plot_utils.save_figure(fig, f"sl_fit_row{row}_track", args)

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

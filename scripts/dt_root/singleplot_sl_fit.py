#################################################################
### event display of single sl fits
# for every selected fit two figures:
#   - hit timestamps and fitted timestamps per layer, with the fit residuals below
#   - the pattern cells with the hit positions (from the drift times) and the fitted track
#
# Fits are selected by their row in the file (--rows), or the first fits passing --cuts are taken (--n_fits).
#
# examples:
#   python scripts/dt_root/singleplot_sl_fit.py --input_file out/run_sl_fits.root --rows 500,600,700 --store_plots plots/single_fits
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import data_utils, dt_pipeline_utils, geoplot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

### rows of the first n fits which pass the cuts
def _select_rows(input_file, cuts, n_fits):
    cut_data = root_utils.read_branches(input_file, sorted({c[0] for c in cuts}))
    cut_data["__row"] = np.arange(root_utils.length(cut_data))
    rows = data_utils.cut_data(data=cut_data, conditions=cuts, silent=True)["__row"]
    log(f"{len(rows)} fits pass the cuts {cuts}, taking the first {min(n_fits, len(rows))}")
    return [int(r) for r in rows[:n_fits]]

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Event display and fit residuals of single sl fits.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path: sl fits (.root)")
    parser.add_argument("--rows", type=str, default=None, help="rows of the fits to plot, separated by \",\"")
    parser.add_argument("--n_fits", type=int, default=5, help="if --rows is not given: number of fits to plot (the first ones passing --cuts)")
    parser.add_argument("--cuts", type=str, default=None,
                        help="if --rows is not given: selection of the fits, format \"key1,operator1,value1;...\" "
                             "(default: \"impossible<suffix>,==,0\")")
    parser.add_argument("--suffix", type=str, default="", help="suffix of the fit result branches to plot, if the fit was stored with one (default: none)")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)
    sfx = args.suffix

    ### which fits
    root_utils.check_input_file(args.input_file)
    if args.rows is not None:
        rows = [int(s) for s in args.rows.split(",") if s.strip() != ""]
    else:
        cuts = dt_pipeline_utils.parse_cuts(args.cuts) if args.cuts is not None else [("impossible" + sfx, "==", 0)]
        rows = _select_rows(args.input_file, cuts, args.n_fits)
    if len(rows) == 0:
        raise RuntimeError("No fits selected, nothing to plot.")
    sl_fits = root_utils.read_rows(args.input_file, rows)
    if "t0" + sfx not in sl_fits:
        raise KeyError(f"No fit results with suffix \"{sfx}\" in {args.input_file}.")

    for i, row in enumerate(rows):
        fit = {k: sl_fits[k][i] for k in sl_fits.keys()}
        ### data
        lys = np.arange(0, 4)
        ts = np.array([fit[f"ts{ly}"] for ly in range(4)], dtype=np.float64)
        err_ts = np.array([fit[f"err_ts{ly}"] for ly in range(4)], dtype=np.float64)
        sl = int(fit["sl"])
        ### pattern
        pat_type = int(fit["pat_type"])
        pat_name = list(params._dt_sl_patterns.keys())[pat_type]  # pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"]  # list of [lat for ly0,1,2,3] laterality lists
        lat_idx = int(fit["laterality" + sfx])
        laterality = np.array(lats[lat_idx])
        ### fit results (vd in mm / timestamp unit, as used by the fit function)
        t0, x0, tan_alpha, vd = fit["t0" + sfx], fit["x0" + sfx], fit["tan_alpha" + sfx], fit["vd" + sfx]
        err_t0, err_x0, err_tan_alpha, err_vd = fit["err_t0" + sfx], fit["err_x0" + sfx], fit["err_tan_alpha" + sfx], fit["err_vd" + sfx]
        corr = {k: fit[k + sfx] for k in ["corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd", "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd"]}
        chi2ndf = fit["chi2/ndf" + sfx]
        impossible = bool(fit["impossible" + sfx])
        vd_um_per_ns, err_vd_um_per_ns = vd / derived_params._drift_velocity_conversion, err_vd / derived_params._drift_velocity_conversion
        ### geometry of the pattern: z of the layers, x of the wires (local frame: wire of ly 3 at (0, 0))
        z_arr, x_cell = np.full(4, 0, dtype=np.float64), np.full(4, 0, dtype=np.float64)
        for ly in lys:
            z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3]
            rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
            x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2]
        ### fitted timestamps
        fit_ts, err_fit_ts = np.zeros(4), np.zeros(4)
        for ly in lys:
            fit_ts[ly] = derived_params.f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd)
            err_fit_ts[ly] = derived_params.err_f_ts_fit(
                x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd,
                err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd, **corr,
            )
        residuals = ts - fit_ts
        ### print
        log(f"sl fit, row {row}{' (flagged impossible)' if impossible else ''}:")
        log(f"  sl = {sl}, pat_type = {pat_type} ({pat_name}), wires = {[int(fit[f'wi{ly}']) for ly in range(4)]}, laterality = {[int(l) for l in laterality]} (idx {lat_idx})")
        log(f"  t0 = {t0:.2f} +- {err_t0:.2f} TU, x0 = {x0:.3f} +- {err_x0:.3f} mm, tan_alpha = {tan_alpha:.4f} +- {err_tan_alpha:.4f}")
        log(f"  vd = {vd_um_per_ns:.2f} +- {err_vd_um_per_ns:.2f} um/ns, chi2/ndf = {chi2ndf:.3f}")
        log(f"  residuals [TU] = {np.round(residuals, 2)}")
        title = f"SL {sl} ({params._dt_chamber['sls'][sl]['orient']}), Pattern {pat_type}, Laterality {[int(l) for l in laterality]}, row {row}"

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
        ax[1].set_xticks([j for j in range(4)])
        ax[1].set_xticklabels([f"{j}" for j in range(4)])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        plot_utils.finish_figure(fig, f"sl_fit{sfx}_row{row}_timestamps", **out)

        ################################
        ###### pattern cells with hit positions and fitted track (local frame)
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax = geoplot_utils.empty_sl_pattern_ax(ax, pat_name, wire=True)
        # hit positions from the measured drift times
        x_hits = x_cell + laterality * (ts - t0) * vd
        err_x_hits = np.sqrt(np.clip(
              (laterality * vd) ** 2 * err_ts ** 2
            + (-laterality * vd) ** 2 * err_t0 ** 2
            + (laterality * (ts - t0)) ** 2 * err_vd ** 2
            + 2 * (-laterality * vd) * (laterality * (ts - t0)) * corr["corr_t0_vd"],
            0, None,
        ))
        ax.errorbar(x=x_hits, y=z_arr, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label="Hit positions", zorder=5)
        # fitted track with uncertainty band
        z_range = np.linspace(np.amin(z_arr) - params._cell_height, np.amax(z_arr) + params._cell_height, 1000)
        track = derived_params.f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha)
        err_track = derived_params.err_f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr["corr_x0_tan_alpha"])
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
        plot_utils.finish_figure(fig, f"sl_fit{sfx}_row{row}_track", **out)

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

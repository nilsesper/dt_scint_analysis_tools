#################################################################
### event display of single dt muons
# one figure per muon: x-z view (phi superlayers) and y-z view (theta superlayer), each as full chamber
# and zoomed to the track, with
#   - the hit cells of the three sl fits the muon was built from
#   - the sl fit track segments with uncertainty band
#   - the super fit of the two phi superlayers (x-z view)
#   - the global muon track with uncertainty band
#   - the distance between sl fit and global track in each superlayer (printed and in the legend)
#
# Needs the three files of the dt muon chain: the cut sl fits, the cut super fits and the dt muons.
# Muons are selected by their row in the dt muons file (--rows), default are the first --n_muons.
#
# example:
#   python scripts/singleplot_dt_muon.py --sl_fits_file out/run_sl_fits_cut.root \
#          --super_fits_file out/run_super_fits_cut.root --dt_muons_file out/run_dt_muons.root --rows 0,5 --store_plots plots/single_muons
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry, dt_pipeline_utils, geoplot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

SL_FIT_LINE_WIDTH = 3
MUON_LINE_WIDTH = 2

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Event display of single dt muons.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: cut super fits the muons were made from (.root)")
    parser.add_argument("--suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--rows", type=str, default=None, help="rows of the muons to plot in the dt muons file, separated by \",\"")
    parser.add_argument("--n_muons", type=int, default=5, help="if --rows is not given: number of muons to plot (the first ones in the file)")
    parser.add_argument("--zoom_width", type=float, default=400, help="width of the zoomed view in mm")
    parser.add_argument("--simulation", action="store_true", help="also draw the simulated muon track (simulation only)")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)

    ### which muons
    for path in (args.sl_fits_file, args.super_fits_file, args.dt_muons_file):
        root_utils.check_input_file(path)
    n_dt_muons = root_utils.n_entries(args.dt_muons_file)
    if args.rows is not None:
        rows = [int(s) for s in args.rows.split(",") if s.strip() != ""]
    else:
        rows = list(range(min(args.n_muons, n_dt_muons)))
    if len(rows) == 0:
        raise RuntimeError("No dt muons selected, nothing to plot.")
    dt_muons = root_utils.read_rows(args.dt_muons_file, rows)
    sls = list(params._dt_chamber["sls"].keys())
    layer_z = [geometry.layer_z(sl, ly) for sl in sls for ly in dt_chamber_utils.layers(sl)]
    z_range = np.linspace(min(layer_z) - params._plot_z_margin * 2, max(layer_z) + params._plot_z_margin * 2, 1000)

    for i, row in enumerate(rows):
        muon = {k: dt_muons[k][i] for k in dt_muons.keys()}
        ### sl fits and super fit of this muon
        if "super_fit_row" not in muon:
            raise KeyError(f"{args.dt_muons_file} has no branch \"super_fit_row\": it was not made by super_fits_to_dt_muons.py.")
        fit_rows = [int(muon[f"sl{sl}_fit_row"]) for sl in sls]
        sl_fits = root_utils.read_rows(args.sl_fits_file, fit_rows)
        n_fits = root_utils.length(sl_fits)
        for j in range(n_fits):
            if int(sl_fits["sl"][j]) != sls[j]:
                raise RuntimeError(f"Row {fit_rows[j]} of {args.sl_fits_file} is not a fit of sl {sls[j]}. Is this the file the muons were made from?")
        super_fit = {k: v[0] for k, v in root_utils.read_rows(args.super_fits_file, [int(muon["super_fit_row"])]).items()}
        sfx = args.suffix
        if "t0" + sfx not in super_fit:
            raise KeyError(f"No super fit results with suffix \"{sfx}\" in {args.super_fits_file}.")
        # super fit track in the global frame: x(z) = x_ref + x0 + (z - z_ref) * tan_alpha
        super_x_ref = geometry.SUPER_FRAME_ORIGIN[0] + super_fit["ref_x" + sfx]
        super_z_ref = geometry.SUPER_FRAME_ORIGIN[1] + super_fit["ref_z" + sfx]
        super_vd = super_fit["vd" + sfx] / derived_params._drift_velocity_conversion
        super_err_vd = super_fit["err_vd" + sfx] / derived_params._drift_velocity_conversion

        ### muon parameters
        x0, err_x0 = muon["x0"], muon["err_x0"]
        y0, err_y0 = muon["y0"], muon["err_y0"]
        z0, err_z0 = muon["z0"], muon["err_z0"]
        ts, err_ts = muon["ts"], muon["err_ts"]
        phi, err_phi = muon["phi"], muon["err_phi"]
        theta, err_theta = muon["theta"], muon["err_theta"]
        log(f"dt muon, row {row}:")
        log(f"  ts = {ts:.2f} +- {err_ts:.2f} TU")
        log(f"  theta = {theta * params.rad_to_deg:.2f} +- {err_theta * params.rad_to_deg:.2f} deg, phi = {phi * params.rad_to_deg:.2f} +- {err_phi * params.rad_to_deg:.2f} deg")
        log(f"  (x0, y0, z0) = ({x0:.1f} +- {err_x0:.1f}, {y0:.1f} +- {err_y0:.1f}, {z0:.1f}) mm")
        log(f"  super fit row {int(muon['super_fit_row'])}: t0 = {super_fit['t0' + sfx]:.1f} TU, tan_alpha = {super_fit['tan_alpha' + sfx]:.3f}, "
              f"vd = {super_vd:.1f} +- {super_err_vd:.1f} um/ns, chi2/ndf = {super_fit['chi2/ndf' + sfx]:.2f}")
        log(f"  theta sl fit - super fit t0 = {muon['delta_t0']:.1f} TU, theta candidates in time window = {int(muon['n_theta_candidates'])}")

        ### hit cells
        dt_cell_data = dt_chamber_utils.cell_display_map()
        for k in range(n_fits):
            sl = int(sl_fits["sl"][k])
            for ly in range(4):
                dt_cell_data[sl][ly][int(sl_fits[f"wi{ly}"][k])]["color"] = "aqua"

        fig, axes = plt.subplots(2, 2, figsize=(20, 10), width_ratios=(2.2, 1))
        for i_orient, orient in enumerate(["phi", "theta"]):
            ### track of the muon in this projection
            track = geometry.muon_track_position(orient=orient, z=z_range, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi)
            err_track = geometry.err_muon_track_position(orient=orient, z=z_range, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi, err_x0=err_x0, err_y0=err_y0, err_z0=err_z0, err_phi=err_phi, err_theta=err_theta)
            muon_label = f"""Global track (row {row}):
$T_0=({ts:.1f}\\pm{err_ts:.1f})$ {params._key_units['t0']}
$\\theta=({theta * params.rad_to_deg:.1f}\\pm{err_theta * params.rad_to_deg:.1f})^\\circ$
$\\phi=({phi * params.rad_to_deg:.1f}\\pm{err_phi * params.rad_to_deg:.1f})^\\circ$"""
            ### sl fit segments in this projection: (sl, z values, x values, x uncertainties, distance to the global track)
            segments = []
            for k in range(n_fits):
                sl = int(sl_fits["sl"][k])
                if params._dt_chamber["sls"][sl]["orient"] != orient:
                    continue
                sl_z_range = np.linspace(geometry.layer_z(sl, 0) - params._plot_z_margin, geometry.layer_z(sl, 3) + params._plot_z_margin, 200)
                x_ref_cell, z_ref_cell = geometry.wire_position(sl, 3, int(sl_fits["wi3"][k]))  # origin of the pattern frame of the fit
                sl_track = geometry.track_position(z=sl_z_range - z_ref_cell, x0=sl_fits["x0"][k], tan_alpha=sl_fits["tan_alpha"][k]) + x_ref_cell
                err_sl_track = geometry.err_track_position(z=sl_z_range - z_ref_cell, x0=sl_fits["x0"][k], tan_alpha=sl_fits["tan_alpha"][k], err_x0=sl_fits["err_x0"][k], err_tan_alpha=sl_fits["err_tan_alpha"][k], corr_x0_tan_alpha=sl_fits["corr_x0_tan_alpha"][k])
                # distance sl fit - global track in the middle of the superlayer
                z_mid = geometry.superlayer_box(sl).center[geometry.Z]
                x_sl_mid = geometry.track_position(z=z_mid - z_ref_cell, x0=sl_fits["x0"][k], tan_alpha=sl_fits["tan_alpha"][k]) + x_ref_cell
                x_glob_mid = geometry.muon_track_position(orient=orient, z=z_mid, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi)
                segments.append((sl, sl_z_range, sl_track, err_sl_track, x_sl_mid - x_glob_mid))
                log(f"  sl {sl} ({orient}) fit row {fit_rows[k]}: t0 = {sl_fits['t0'][k]:.1f} TU, tan_alpha = {sl_fits['tan_alpha'][k]:.3f}, "
                      f"chi2/ndf = {sl_fits['chi2/ndf'][k]:.2f}, sl fit - global track = {x_sl_mid - x_glob_mid:.2f} mm")
            residual_text = ", ".join(f"SL {sl}: {res:+.1f} mm" for sl, _, _, _, res in segments)
            sl_fit_label = "SL pattern fit\n(fit $-$ global track: " + residual_text + ")"

            for i_view, ax in enumerate(axes[i_orient]):
                zoom = (i_view == 1)
                ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=zoom)
                ax.plot(track, z_range, linewidth=MUON_LINE_WIDTH, color="tab:green", label=muon_label, zorder=6)
                ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:green", alpha=0.2, zorder=5)
                if args.simulation:
                    sim_label = f"""Simulated track:
$T_0={muon['sim_ts']:.1f}$ {params._key_units['t0']}
$\\theta={muon['sim_theta'] * params.rad_to_deg:.1f}^\\circ$
$\\phi={muon['sim_phi'] * params.rad_to_deg:.1f}^\\circ$"""
                    sim_track = geometry.muon_track_position(orient=orient, z=z_range, x0=muon["sim_x0"], y0=muon["sim_y0"], z0=muon["sim_z0"], theta=muon["sim_theta"], phi=muon["sim_phi"])
                    ax.plot(sim_track, z_range, linewidth=1, color="black", label=sim_label, linestyle="--", zorder=7)
                if orient == "phi":
                    super_track = geometry.track_position(z=z_range - super_z_ref, x0=super_fit["x0" + sfx], tan_alpha=super_fit["tan_alpha" + sfx]) + super_x_ref
                    super_label = f"Super fit (SL 1 + SL 3):\n$v_d=({super_vd:.1f}\\pm{super_err_vd:.1f})$ um/ns, $\\chi^2/N_{{df}}={super_fit['chi2/ndf' + sfx]:.2f}$"
                    ax.plot(super_track, z_range, color="tab:blue", linewidth=1.5, linestyle=":", label=super_label, zorder=8)
                for j, (sl, sl_z_range, sl_track, err_sl_track, _) in enumerate(segments):
                    ax.plot(sl_track, sl_z_range, color="tab:red", linewidth=SL_FIT_LINE_WIDTH, label=sl_fit_label if j == 0 else None, zorder=4)
                    ax.fill_betweenx(x1=sl_track - err_sl_track, x2=sl_track + err_sl_track, y=sl_z_range, color="tab:red", alpha=0.2, zorder=3)
                ax.set_ylim(np.amin(z_range), np.amax(z_range))
                axis_name = "x" if orient == "phi" else "y"
                ax.set_xlabel(f"${axis_name}$ [mm]")
                ax.set_ylabel("$z$ [mm]")
                if zoom:
                    x_center = np.mean(track)
                    ax.set_xlim(x_center - args.zoom_width / 2, x_center + args.zoom_width / 2)
                    ax.set_title(f"zoom to the track ({args.zoom_width:g} mm wide)")
                    ax.legend(prop={"size": 11}, fancybox=False, framealpha=0.9, loc="center left", bbox_to_anchor=(1.01, 0.5))
                else:
                    lo = geometry.chamber_box().low[geometry.X] if orient == "phi" else geometry.chamber_box().low[geometry.Y]
                    hi = geometry.chamber_box().high[geometry.X] if orient == "phi" else geometry.chamber_box().high[geometry.Y]
                    ax.set_xlim(lo - 100, hi + 100)
                    view = "SL-$\\phi$ view" if orient == "phi" else "SL-$\\theta$ view"
                    ax.set_title(f"DT track, row {row}: ${axis_name}$-$z$-plane ({view})")
        fig.tight_layout()
        plot_utils.finish_figure(fig, f"dt_muon_row{row}", **out)

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

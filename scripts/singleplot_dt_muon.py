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

from analysis_tools.utils import dt_chamber_utils, dt_fit_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

SL_FIT_LINE_WIDTH = 3
MUON_LINE_WIDTH = 2

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14})
def main():
    parser = argparse.ArgumentParser(description="Event display of single dt muons.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: cut super fits the muons were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--rows", type=str, default=None, help="rows of the muons to plot in the dt muons file, separated by \",\"")
    parser.add_argument("--n_muons", type=int, default=5, help="if --rows is not given: number of muons to plot (the first ones in the file)")
    parser.add_argument("--zoom_width", type=float, default=400, help="width of the zoomed view in mm")
    parser.add_argument("--simulation", action="store_true", help="also draw the simulated muon track (simulation only)")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)
    sfx = dt_fit_utils.SUPER_FIT_SUFFIX

    ### which muons
    root_utils.check_input_file(args.sl_fits_file)
    root_utils.check_input_file(args.super_fits_file)
    root_utils.check_input_file(args.dt_muons_file)
    n_dt_muons = root_utils.number_of_rows(args.dt_muons_file, root_utils.DEFAULT_TREE)
    if args.rows is not None:
        rows = []
        for text in args.rows.split(","):
            if text.strip() != "":
                rows.append(int(text))
    else:
        rows = list(range(min(args.n_muons, n_dt_muons)))
    if len(rows) == 0:
        raise RuntimeError("No dt muons selected, nothing to plot.")

    ### z range of the plots: all layers of the chamber
    sls = list(params._dt_chamber["sls"].keys())
    layer_z = []
    for sl in sls:
        for ly in dt_chamber_utils.layers(sl):
            layer_z.append(dt_chamber_utils.layer_z(sl, ly))
    z_range = np.linspace(min(layer_z) - params._plot_z_margin * 2, max(layer_z) + params._plot_z_margin * 2, 1000)

    for row in rows:
        dt_muon = root_utils.read_tree(args.dt_muons_file, root_utils.DEFAULT_TREE, row, row + 1)
        muon = {}
        for key in dt_muon.keys():
            muon[key] = dt_muon[key][0]
        ### sl fits and super fit of this muon
        if "super_fit_row" not in muon:
            raise KeyError(f"{args.dt_muons_file} has no branch \"super_fit_row\": it was not made by super_fits_to_dt_muons.py.")
        fit_rows = []
        for sl in sls:
            fit_rows.append(int(muon[f"sl{sl}_fit_row"]))
        # one dict per sl fit: {key: value}
        sl_fits = []
        for fit_row in fit_rows:
            sl_fit_data = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, fit_row, fit_row + 1)
            sl_fit = {}
            for key in sl_fit_data.keys():
                sl_fit[key] = sl_fit_data[key][0]
            sl_fits.append(sl_fit)
        n_fits = len(sl_fits)
        for j in range(n_fits):
            if int(sl_fits[j]["sl"]) != sls[j]:
                raise RuntimeError(f"Row {fit_rows[j]} of {args.sl_fits_file} is not a fit of sl {sls[j]}. Is this the file the muons were made from?")
        super_fit_row = int(muon["super_fit_row"])
        super_fit_data = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, super_fit_row, super_fit_row + 1)
        super_fit = {}
        for key in super_fit_data.keys():
            super_fit[key] = super_fit_data[key][0]
        if "t0" + sfx not in super_fit:
            raise KeyError(f"No super fit results with suffix \"{sfx}\" in {args.super_fits_file}.")
        # reference wire of the super fit in the chamber frame
        super_x_ref = super_fit["ref_x" + sfx]
        super_z_ref = super_fit["ref_z" + sfx]
        super_vd = super_fit["vd" + sfx] / derived_params._drift_velocity_conversion
        super_err_vd = super_fit["err_vd" + sfx] / derived_params._drift_velocity_conversion

        ### muon parameters
        x0 = muon["x0"]
        err_x0 = muon["err_x0"]
        y0 = muon["y0"]
        err_y0 = muon["err_y0"]
        z0 = muon["z0"]
        err_z0 = muon["err_z0"]
        ts = muon["ts"]
        err_ts = muon["err_ts"]
        phi = muon["phi"]
        err_phi = muon["err_phi"]
        theta = muon["theta"]
        err_theta = muon["err_theta"]
        log(f"dt muon, row {row}:")
        log(f"  ts = {ts:.2f} +- {err_ts:.2f} TU")
        log(f"  theta = {theta * params.rad_to_deg:.2f} +- {err_theta * params.rad_to_deg:.2f} deg, phi = {phi * params.rad_to_deg:.2f} +- {err_phi * params.rad_to_deg:.2f} deg")
        log(f"  (x0, y0, z0) = ({x0:.1f} +- {err_x0:.1f}, {y0:.1f} +- {err_y0:.1f}, {z0:.1f}) mm")
        log(f"  super fit row {int(muon['super_fit_row'])}: t0 = {super_fit['t0' + sfx]:.1f} TU, tan_alpha = {super_fit['tan_alpha' + sfx]:.3f}, "
              f"vd = {super_vd:.1f} +- {super_err_vd:.1f} um/ns, chi2/ndf = {super_fit['chi2/ndf' + sfx]:.2f}")
        log(f"  theta sl fit - super fit t0 = {muon['delta_t0']:.1f} TU, theta candidates in time window = {int(muon['n_theta_candidates'])}")

        ### hit cells
        cell_colors = {}
        for k in range(n_fits):
            sl = int(sl_fits[k]["sl"])
            for ly in range(4):
                cell_colors[(sl, ly, int(sl_fits[k][f"wi{ly}"]))] = "aqua"

        fig, axes = plt.subplots(2, 2, figsize=(20, 10), width_ratios=(2.2, 1))
        orients = ["phi", "theta"]
        for i_orient in range(len(orients)):
            orient = orients[i_orient]
            ### track of the muon in this projection
            track = dt_chamber_utils.muon_track_position(orient=orient, z=z_range, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi)
            err_track = dt_chamber_utils.err_muon_track_position(orient=orient, z=z_range, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi,
                                                        err_x0=err_x0, err_y0=err_y0, err_z0=err_z0, err_phi=err_phi, err_theta=err_theta)
            muon_label = f"""Global track (row {row}):
$T_0=({ts:.1f}\\pm{err_ts:.1f})$ {params._key_units['t0']}
$\\theta=({theta * params.rad_to_deg:.1f}\\pm{err_theta * params.rad_to_deg:.1f})^\\circ$
$\\phi=({phi * params.rad_to_deg:.1f}\\pm{err_phi * params.rad_to_deg:.1f})^\\circ$"""
            ### sl fit segments in this projection, with the distance sl fit - global track
            segments = []
            for k in range(n_fits):
                sl = int(sl_fits[k]["sl"])
                if params._dt_chamber["sls"][sl]["orient"] != orient:
                    continue
                sl_z_range = np.linspace(dt_chamber_utils.layer_z(sl, 0) - params._plot_z_margin, dt_chamber_utils.layer_z(sl, 3) + params._plot_z_margin, 200)
                h_ref, z_ref = dt_chamber_utils.wire_position(sl, 3, int(sl_fits[k]["wi3"]))  # origin of the track frame of the fit
                sl_track = dt_chamber_utils.track_position_in_chamber(h_ref, z_ref, sl_fits[k]["x0"], sl_fits[k]["tan_alpha"], sl_z_range)
                err_sl_track = dt_chamber_utils.err_track_position_in_chamber(z_ref, sl_z_range, sl_fits[k]["err_x0"], sl_fits[k]["err_tan_alpha"], sl_fits[k]["corr_x0_tan_alpha"])
                # distance sl fit - global track in the middle of the superlayer
                z_mid = dt_chamber_utils.superlayer_box(sl)["center"][dt_chamber_utils.Z]
                x_sl_mid = dt_chamber_utils.track_position_in_chamber(h_ref, z_ref, sl_fits[k]["x0"], sl_fits[k]["tan_alpha"], z_mid)
                x_glob_mid = dt_chamber_utils.muon_track_position(orient=orient, z=z_mid, x0=x0, y0=y0, z0=z0, theta=theta, phi=phi)
                segment = {"sl": sl, "z": sl_z_range, "x": sl_track, "err_x": err_sl_track, "residual": x_sl_mid - x_glob_mid}
                segments.append(segment)
                log(f"  sl {sl} ({orient}) fit row {fit_rows[k]}: t0 = {sl_fits[k]['t0']:.1f} TU, tan_alpha = {sl_fits[k]['tan_alpha']:.3f}, "
                      f"chi2/ndf = {sl_fits[k]['chi2/ndf']:.2f}, sl fit - global track = {x_sl_mid - x_glob_mid:.2f} mm")
            residual_texts = []
            for segment in segments:
                residual_texts.append(f"SL {segment['sl']}: {segment['residual']:+.1f} mm")
            sl_fit_label = "SL pattern fit\n(fit $-$ global track: " + ", ".join(residual_texts) + ")"

            ### full chamber view (left) and zoom to the track (right)
            for i_view in range(2):
                ax = axes[i_orient][i_view]
                zoom = (i_view == 1)
                plot_utils.draw_chamber(ax, orient, cell_colors=cell_colors, wires=zoom)
                ax.plot(track, z_range, linewidth=MUON_LINE_WIDTH, color="tab:green", label=muon_label, zorder=6)
                ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:green", alpha=0.2, zorder=5)
                if args.simulation:
                    sim_label = f"""Simulated track:
$T_0={muon['sim_ts']:.1f}$ {params._key_units['t0']}
$\\theta={muon['sim_theta'] * params.rad_to_deg:.1f}^\\circ$
$\\phi={muon['sim_phi'] * params.rad_to_deg:.1f}^\\circ$"""
                    sim_track = dt_chamber_utils.muon_track_position(orient=orient, z=z_range, x0=muon["sim_x0"], y0=muon["sim_y0"], z0=muon["sim_z0"],
                                                            theta=muon["sim_theta"], phi=muon["sim_phi"])
                    ax.plot(sim_track, z_range, linewidth=1, color="black", label=sim_label, linestyle="--", zorder=7)
                if orient == "phi":
                    super_track = dt_chamber_utils.track_position_in_chamber(super_x_ref, super_z_ref, super_fit["x0" + sfx], super_fit["tan_alpha" + sfx], z_range)
                    super_label = f"Super fit (SL 1 + SL 3):\n$v_d=({super_vd:.1f}\\pm{super_err_vd:.1f})$ um/ns, $\\chi^2/N_{{df}}={super_fit['chi2/ndf' + sfx]:.2f}$"
                    ax.plot(super_track, z_range, color="tab:blue", linewidth=1.5, linestyle=":", label=super_label, zorder=8)
                for j in range(len(segments)):
                    segment = segments[j]
                    # only the first segment gets a legend entry
                    if j == 0:
                        segment_label = sl_fit_label
                    else:
                        segment_label = None
                    ax.plot(segment["x"], segment["z"], color="tab:red", linewidth=SL_FIT_LINE_WIDTH, label=segment_label, zorder=4)
                    ax.fill_betweenx(x1=segment["x"] - segment["err_x"], x2=segment["x"] + segment["err_x"], y=segment["z"], color="tab:red", alpha=0.2, zorder=3)
                ax.set_ylim(np.amin(z_range), np.amax(z_range))
                if orient == "phi":
                    axis_name = "x"
                    lo = dt_chamber_utils.chamber_box()["low"][dt_chamber_utils.X]
                    hi = dt_chamber_utils.chamber_box()["high"][dt_chamber_utils.X]
                    view = "SL-$\\phi$ view"
                else:
                    axis_name = "y"
                    lo = dt_chamber_utils.chamber_box()["low"][dt_chamber_utils.Y]
                    hi = dt_chamber_utils.chamber_box()["high"][dt_chamber_utils.Y]
                    view = "SL-$\\theta$ view"
                ax.set_xlabel(f"${axis_name}$ [mm]")
                ax.set_ylabel("$z$ [mm]")
                if zoom:
                    x_center = np.mean(track)
                    ax.set_xlim(x_center - args.zoom_width / 2, x_center + args.zoom_width / 2)
                    ax.set_title(f"zoom to the track ({args.zoom_width:g} mm wide)")
                    ax.legend(prop={"size": 11}, fancybox=False, framealpha=0.9, loc="center left", bbox_to_anchor=(1.01, 0.5))
                else:
                    ax.set_xlim(lo - 100, hi + 100)
                    ax.set_title(f"DT track, row {row}: ${axis_name}$-$z$-plane ({view})")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muon_row{row}", args)

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

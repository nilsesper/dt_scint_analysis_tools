#################################################################
### dt muons: track maps, projections and timing
# - x-y map of the track positions in the plane of each superlayer
# - x-z and y-z projections of all tracks through the chamber, with the chamber geometry
# - 3d view of the tracks through the three superlayers
# - polar and azimuthal angle, arrival times, time between consecutive muons, muon rate
# (histograms of the single branches: plot_histograms.py)
#
# example:
#   python scripts/dt_root/plot_dt_muons.py --dt_muons_file out/run_dt_muons.root --store_plots plots/dt_muons
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import data_utils, dt_pipeline_utils, dt_utils, geoplot_utils, hist_utils, muon_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

SLS = [1, 2, 3]
MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"]

def _cell_rectangle(sl, ly, wi, **kwargs):
    # derived_params._dt_cell_coordinates = {sl: {ly: {wi: [[xmin, xmax], [ymin, ymax], [zmin, zmax], x_center, y_center, z_center]}}}
    c = derived_params._dt_cell_coordinates[sl][ly][wi]
    return pat.Rectangle((c[0][0], c[1][0]), width=c[0][1] - c[0][0], height=c[1][1] - c[1][0], **kwargs)

def _colorbar(fig, ax, im_obj, label):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3])
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, format=formatter)
    cbar.set_label(label)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Track maps, projections, 3d view and timing plots of dt muons.")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--mark_cells", type=str, default=None,
                        help="cells to mark in the maps (e.g. dead or noisy ones) as \"sl:ly:wi,sl:ly:wi,...\", or \"none\" "
                             "(default: the cells listed in params._dt_wire_mask and params._dt_dead_wires)")
    parser.add_argument("--xy_bin_width", type=float, default=20, help="bin width of the x-y maps in mm")
    parser.add_argument("--xz_bin_width", type=float, default=5, help="bin width of the x-z and y-z projections in mm")
    parser.add_argument("--n_tracks_3d", type=int, default=150, help="max number of tracks drawn in the 3d view")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)

    ### cells to mark
    if args.mark_cells is None:
        marked_cells = plot_utils.masked_and_dead_cells()
    elif args.mark_cells.strip().lower() == "none":
        marked_cells = []
    else:
        marked_cells = plot_utils.parse_cells(args.mark_cells)
    for sl, ly, wi in marked_cells:
        if wi not in derived_params._dt_cell_coordinates.get(sl, {}).get(ly, {}):
            raise ValueError(f"--mark_cells: cell sl={sl}, ly={ly}, wi={wi} does not exist in the chamber.")

    ### data import
    cuts = dt_pipeline_utils.parse_cuts(args.cuts)
    log(f"###### Importing dt muons from {args.dt_muons_file}...")
    dt_muons = root_utils.read_branches(args.dt_muons_file, sorted(set(MUON_KEYS) | {c[0] for c in cuts}))
    n_all = root_utils.length(dt_muons)
    if len(cuts) > 0:
        dt_muons = data_utils.cut_data(data=dt_muons, conditions=cuts, silent=True)
        log(f"cuts {cuts}: {root_utils.length(dt_muons):,} / {n_all:,} muons selected")
    dt_muons = {k: dt_muons[k] for k in MUON_KEYS}
    n_dt_muons = root_utils.length(dt_muons)
    if n_dt_muons == 0:
        raise RuntimeError("No dt muons to plot.")

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"]))
    log(f"measurement duration = {duration} s")
    log(f"dt muon count: {n_dt_muons:,}")
    if duration > 0:
        log(f"dt muon rate: {n_dt_muons / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")

    ####### X-Y maps: track positions in the plane of each superlayer
    for sl in SLS:
        margin = 100  # mm
        x_edges = np.arange(start=derived_params.sl_x_min[sl] - margin, stop=derived_params.sl_x_max[sl] + margin + args.xy_bin_width, step=args.xy_bin_width)
        y_edges = np.arange(start=derived_params.sl_y_min[sl] - margin, stop=derived_params.sl_y_max[sl] + margin + args.xy_bin_width, step=args.xy_bin_width)
        dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=derived_params.sl_z_center[sl])
        pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
        ax.add_patch(pat.Rectangle(
            (derived_params.sl_x_min[sl], derived_params.sl_y_min[sl]),
            width=(derived_params.sl_x_max[sl] - derived_params.sl_x_min[sl]), height=(derived_params.sl_y_max[sl] - derived_params.sl_y_min[sl]),
            edgecolor="white", facecolor="None", label="Superlayer position",
        ))
        first_label = True
        for cell_sl, ly, wi in marked_cells:
            if cell_sl != sl:
                continue
            ax.add_patch(_cell_rectangle(sl, ly, wi, edgecolor="red", facecolor="None", label="Marked cells" if first_label else None))
            first_label = False
        ax.set_title(f"DT tracks in SL {sl} ($z={derived_params.sl_z_center[sl]:.0f}$ mm)")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
        _colorbar(fig, ax, im_obj, "Counts")
        entries = int(np.sum(pos_muons_hist2d))
        info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
        hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
        fig.tight_layout()
        plot_utils.finish_figure(fig, f"dt_muons_xy_sl{sl}", **out)

    ####### X-Z and Y-Z projections of the tracks through the chamber
    marked_cell_data = dt_utils._chamber_data()
    for sl, ly, wi in marked_cells:
        marked_cell_data[sl][ly][wi]["color"] = "tab:red"
    sl_z_coord = (np.amin([derived_params.sl_z_min[sl] for sl in SLS]), np.amax([derived_params.sl_z_max[sl] for sl in SLS]))
    for orient, slice_name in [("phi", "xz"), ("theta", "yz")]:
        if slice_name == "xz":
            sl_x_coord = (np.amin([derived_params.sl_x_min[sl] for sl in SLS]), np.amax([derived_params.sl_x_max[sl] for sl in SLS]))
        else:
            sl_x_coord = (np.amin([derived_params.sl_y_min[sl] for sl in SLS]), np.amax([derived_params.sl_y_max[sl] for sl in SLS]))
        margin = 100  # mm
        x_edges = np.arange(start=sl_x_coord[0] - margin, stop=sl_x_coord[1] + margin + args.xz_bin_width, step=args.xz_bin_width)
        z_edges = np.arange(start=sl_z_coord[0] - margin, stop=sl_z_coord[1] + margin + args.xz_bin_width, step=args.xz_bin_width)
        z_bins = hist_utils.centers_from_edges(z_edges)
        # track position at every z bin
        x_muons, z_muons = [], []
        for z_pos in z_bins:
            dt_muons_moved = muon_utils.change_muon_base_point(muons=dt_muons, z_new=z_pos)
            x_muons.append(dt_muons_moved["x0"] if slice_name == "xz" else dt_muons_moved["y0"])
            z_muons.append(dt_muons_moved["z0"])
        pos_muons_hist2d, _, _ = np.histogram2d(x=np.concatenate(z_muons), y=np.concatenate(x_muons), bins=(z_edges, x_edges))
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]], aspect="auto")
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=marked_cell_data, wire=False, transparent=True)
        ax.set_title(f"DT tracks, ${slice_name[0]}$-$z$ projection ({n_dt_muons:,} tracks)")
        ax.set_xlabel(f"${slice_name[0]}$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        other = "y" if slice_name == "xz" else "x"
        _colorbar(fig, ax, im_obj, f"Tracks per bin\n(summed over ${other}$)")
        legend_entries = {"Chamber geometry": pat.Patch(edgecolor="white", facecolor="none")}
        if len(marked_cells) > 0:
            legend_entries["Marked cells"] = pat.Patch(edgecolor="tab:red", facecolor="none")
        legend = ax.legend(legend_entries.values(), legend_entries.keys(), prop={"size": 11}, loc="lower center", ncols=2, fancybox=False, framealpha=params._legend_alpha, facecolor="gray")
        fig.tight_layout()
        plot_utils.finish_figure(fig, f"dt_muons_{slice_name}_chamber", **out)

    ####### 3d view of the tracks through the superlayers
    n_3d = min(args.n_tracks_3d, n_dt_muons)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")
    for sl in SLS:  # superlayer boxes
        x0, x1 = derived_params.sl_x_min[sl], derived_params.sl_x_max[sl]
        y0, y1 = derived_params.sl_y_min[sl], derived_params.sl_y_max[sl]
        z0, z1 = derived_params.sl_z_min[sl], derived_params.sl_z_max[sl]
        color = "tab:blue" if params._dt_chamber["sls"][sl]["orient"] == "phi" else "tab:orange"
        for z in (z0, z1):
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z, z, z, z, z], color=color, linewidth=1.2)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            ax.plot([x, x], [y, y], [z0, z1], color=color, linewidth=1.2)
    z_lo, z_hi = sl_z_coord[0] - 150, sl_z_coord[1] + 150
    tan_alpha_x = np.tan(dt_muons["theta"]) * np.cos(dt_muons["phi"])
    tan_alpha_y = np.tan(dt_muons["theta"]) * np.sin(dt_muons["phi"])
    for i in range(n_3d):
        z = np.array([z_lo, z_hi])
        x = dt_muons["x0"][i] + tan_alpha_x[i] * (z - dt_muons["z0"][i])
        y = dt_muons["y0"][i] + tan_alpha_y[i] * (z - dt_muons["z0"][i])
        ax.plot(x, y, z, color="tab:green", linewidth=0.8, alpha=0.6)
    ax.set_xlim(sl_x_coord_all(0) - 200, sl_x_coord_all(1) + 200)
    ax.set_ylim(sl_y_coord_all(0) - 200, sl_y_coord_all(1) + 200)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("$x$ [mm]", labelpad=14)
    ax.set_ylabel("$y$ [mm]", labelpad=14)
    ax.set_zlabel("$z$ [mm]", labelpad=10)
    x_span = sl_x_coord_all(1) - sl_x_coord_all(0) + 400
    y_span = sl_y_coord_all(1) - sl_y_coord_all(0) + 400
    ax.set_box_aspect((x_span, y_span, 0.45 * max(x_span, y_span)))  # z is stretched, the chamber is flat
    ax.set_title(f"DT tracks ({n_3d:,} of {n_dt_muons:,} shown, $z$ axis stretched)")
    legend_entries = {
        "Phi superlayers (SL 1, SL 3)": pat.Patch(edgecolor="tab:blue", facecolor="none"),
        "Theta superlayer (SL 2)": pat.Patch(edgecolor="tab:orange", facecolor="none"),
        "DT tracks": pat.Patch(edgecolor="tab:green", facecolor="none"),
    }
    ax.legend(legend_entries.values(), legend_entries.keys(), prop={"size": 12}, loc="upper left", fancybox=False)
    fig.tight_layout()
    plot_utils.finish_figure(fig, "dt_muons_3d", **out)

    ####### 1d histograms
    with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}):
        def _hist(data, xlabel, plot_name, *, scale=1.0, bin_unit=None, log_scale=False, full_range=True):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            edges = plot_utils.choose_edges(data, n_bins=args.n_bins, full_range=full_range)
            plot_utils.draw_histogram(ax, data, edges, xlabel=xlabel, log_scale=log_scale, bin_unit=bin_unit, scale=scale)
            fig.tight_layout()
            plot_utils.finish_figure(fig, plot_name, **out)
        ### angles
        _hist(np.rad2deg(dt_muons["theta"]), "Polar angle $\\theta$ [deg]", "dt_muons_theta_deg", bin_unit="deg")
        _hist(np.rad2deg(dt_muons["phi"]), "Azimuthal angle $\\phi$ [deg]", "dt_muons_phi_deg", bin_unit="deg")
        # projected angles in the x-z and y-z planes
        _hist(np.rad2deg(np.arctan(tan_alpha_x)), "Projected angle in the $x$-$z$ plane [deg]", "dt_muons_angle_xz_deg", bin_unit="deg")
        _hist(np.rad2deg(np.arctan(tan_alpha_y)), "Projected angle in the $y$-$z$ plane [deg]", "dt_muons_angle_yz_deg", bin_unit="deg")
        ### arrival times since the first muon
        ts_sorted = np.sort(np.asarray(dt_muons["ts"], dtype=np.float64))
        _hist(ts_sorted - ts_sorted[0], "$T_0$ since first muon [s]", "dt_muons_ts", scale=plot_utils.TS_UNIT_NS * 1e-9, bin_unit="s")
        ### time between consecutive muons
        if n_dt_muons > 1:
            _hist(np.diff(ts_sorted), "$\\Delta T_0$ [ms]", "dt_muons_delta_ts", scale=plot_utils.TS_UNIT_NS * 1e-6, bin_unit="ms", log_scale=True)

    plot_utils.show_figures(args.show_plots)

def sl_x_coord_all(i):
    return (np.amin([derived_params.sl_x_min[sl] for sl in SLS]), np.amax([derived_params.sl_x_max[sl] for sl in SLS]))[i]

def sl_y_coord_all(i):
    return (np.amin([derived_params.sl_y_min[sl] for sl in SLS]), np.amax([derived_params.sl_y_max[sl] for sl in SLS]))[i]

if __name__ == "__main__":
    main()
    log("###### Done.")

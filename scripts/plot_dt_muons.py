#################################################################
### dt muons: track maps, projections and timing
# - x-y map of the track positions in the plane of each superlayer
# - x-z and y-z projections of all tracks through the chamber, with the chamber geometry
# - 3d view of the tracks through the three superlayers
# - polar and azimuthal angle, arrival times, time between consecutive muons, muon rate
# (histograms of the single branches: plot_histograms.py)
#
# example:
#   python scripts/plot_dt_muons.py --dt_muons_file out/run_dt_muons.root --store_plots plots/dt_muons
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import data_utils, hist_utils, muon_utils, root_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import params

# ---------------------------------------------------------------

SLS = dt_chamber_utils.superlayers()
MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"]

### rectangle patch of one cell in the x-y plane
def cell_rectangle(sl, ly, wi, **kwargs):
    c = dt_chamber_utils.cell(sl, ly, wi)
    return patches.Rectangle((c["low"][dt_chamber_utils.X], c["low"][dt_chamber_utils.Y]), width=dt_chamber_utils.box_size(c, dt_chamber_utils.X), height=dt_chamber_utils.box_size(c, dt_chamber_utils.Y), **kwargs)

### colorbar with scientific notation for large numbers
def add_colorbar(fig, ax, im_obj, label):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3])
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, format=formatter)
    cbar.set_label(label)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="Track maps, projections, 3d view and timing plots of dt muons.")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--mark_cells", type=str, default=None,
                        help="cells to mark in the maps (e.g. dead or noisy ones) as \"sl:ly:wi,sl:ly:wi,...\", or \"none\" "
                             "(default: the cells listed in params._dt_wire_mask and params._dt_dead_wires)")
    parser.add_argument("--xy_bin_width", type=float, default=20, help="bin width of the x-y maps in mm")
    parser.add_argument("--xz_bin_width", type=float, default=5, help="bin width of the x-z and y-z projections in mm")
    parser.add_argument("--n_tracks_3d", type=int, default=1_000, help="max number of tracks drawn in the 3d view")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    ### cells to mark
    if args.mark_cells is None:
        marked_cells = sorted(dt_chamber_utils.excluded_cells())
    elif args.mark_cells.strip().lower() == "none":
        marked_cells = []
    else:
        marked_cells = plot_utils.parse_cells(args.mark_cells)
    for sl, ly, wi in marked_cells:
        if (sl, ly, wi) not in dt_chamber_utils.chamber_cells():
            raise ValueError(f"--mark_cells: cell sl={sl}, ly={ly}, wi={wi} does not exist in the chamber.")

    ### data import
    cuts = data_utils.parse_cuts(args.cuts)
    keys = []
    for key in MUON_KEYS:
        keys.append(key)
    for cut in cuts:
        if cut[0] not in keys:
            keys.append(cut[0])
    log(f"###### Importing dt muons from {args.dt_muons_file}...")
    dt_muons = root_utils.read_tree(args.dt_muons_file, root_utils.DEFAULT_TREE, branches=keys)
    n_all = root_utils.length(dt_muons)
    if len(cuts) > 0:
        dt_muons = data_utils.cut_data(dt_muons, cuts, silent=True)
        log(f"cuts {cuts}: {root_utils.length(dt_muons):,} / {n_all:,} muons selected")
    # keep only the muon branches (not the extra branches of the cuts)
    muon_branches = {}
    for key in MUON_KEYS:
        muon_branches[key] = dt_muons[key]
    dt_muons = muon_branches
    n_dt_muons = root_utils.length(dt_muons)
    if n_dt_muons == 0:
        raise RuntimeError("No dt muons to plot.")

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"]))
    log(f"measurement duration = {duration} s")
    log(f"dt muon count: {n_dt_muons:,}")
    if duration > 0:
        log(f"dt muon rate: {n_dt_muons / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")

    ### X-Y maps: track positions in the plane of each superlayer
    for sl in SLS:
        margin = 100  # mm
        box = dt_chamber_utils.superlayer_box(sl)
        x_edges = np.arange(start=box["low"][dt_chamber_utils.X] - margin, stop=box["high"][dt_chamber_utils.X] + margin + args.xy_bin_width, step=args.xy_bin_width)
        y_edges = np.arange(start=box["low"][dt_chamber_utils.Y] - margin, stop=box["high"][dt_chamber_utils.Y] + margin + args.xy_bin_width, step=args.xy_bin_width)
        dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=box["center"][dt_chamber_utils.Z])
        pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
        ax.add_patch(patches.Rectangle(
            (box["low"][dt_chamber_utils.X], box["low"][dt_chamber_utils.Y]), width=(box["high"][dt_chamber_utils.X] - box["low"][dt_chamber_utils.X]), height=(box["high"][dt_chamber_utils.Y] - box["low"][dt_chamber_utils.Y]),
            edgecolor="white", facecolor="None", label="Superlayer position",
        ))
        # marked cells of this superlayer, only the first one gets a legend label
        first_label = True
        for cell_sl, ly, wi in marked_cells:
            if cell_sl != sl:
                continue
            if first_label:
                label = "Marked cells"
            else:
                label = None
            ax.add_patch(cell_rectangle(sl, ly, wi, edgecolor="red", facecolor="None", label=label))
            first_label = False
        ax.set_title(f"DT tracks in SL {sl} ($z={box["center"][dt_chamber_utils.Z]:.0f}$ mm)")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
        add_colorbar(fig, ax, im_obj, "Counts")
        entries = int(np.sum(pos_muons_hist2d))
        info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
        hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muons_xy_sl{sl}", args)

    ### X-Z and Y-Z projections of the tracks through the chamber
    marked_cell_colors = {}
    for cell in marked_cells:
        marked_cell_colors[cell] = "tab:red"
    sl_z_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.Z)
    for orient, slice_name in [("phi", "xz"), ("theta", "yz")]:
        if slice_name == "xz":
            sl_x_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.X)
            other = "y"
        else:
            sl_x_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)
            other = "x"
        margin = 100  # mm
        x_edges = np.arange(start=sl_x_coord[0] - margin, stop=sl_x_coord[1] + margin + args.xz_bin_width, step=args.xz_bin_width)
        z_edges = np.arange(start=sl_z_coord[0] - margin, stop=sl_z_coord[1] + margin + args.xz_bin_width, step=args.xz_bin_width)
        z_bins = (z_edges[:-1] + z_edges[1:]) / 2
        # track position at every z bin
        x_muons = []
        z_muons = []
        for z_pos in z_bins:
            dt_muons_moved = muon_utils.change_muon_base_point(muons=dt_muons, z_new=z_pos)
            if slice_name == "xz":
                x_muons.append(dt_muons_moved["x0"])
            else:
                x_muons.append(dt_muons_moved["y0"])
            z_muons.append(dt_muons_moved["z0"])
        pos_muons_hist2d, _, _ = np.histogram2d(x=np.concatenate(z_muons), y=np.concatenate(x_muons), bins=(z_edges, x_edges))
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
        im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]], aspect="auto")
        plot_utils.draw_chamber(ax, orient, cell_colors=marked_cell_colors, wires=False, transparent=True)
        ax.set_title(f"DT tracks, ${slice_name[0]}$-$z$ projection ({n_dt_muons:,} tracks)")
        ax.set_xlabel(f"${slice_name[0]}$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        add_colorbar(fig, ax, im_obj, f"Tracks per bin\n(summed over ${other}$)")
        legend_handles = [patches.Patch(edgecolor="white", facecolor="none")]
        legend_labels = ["Chamber geometry"]
        if len(marked_cells) > 0:
            legend_handles.append(patches.Patch(edgecolor="tab:red", facecolor="none"))
            legend_labels.append("Marked cells")
        ax.legend(legend_handles, legend_labels, prop={"size": 11}, loc="lower center", ncols=2, fancybox=False, framealpha=params._legend_alpha, facecolor="gray")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muons_{slice_name}_chamber", args)

    ### 3d view of the tracks through the superlayers
    n_3d = min(args.n_tracks_3d, n_dt_muons)
    #n_3d = n_dt_muons
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")
    ### superlayer boxes: top and bottom rectangle and the four vertical edges
    for sl in SLS:
        box = dt_chamber_utils.superlayer_box(sl)
        x0, y0, z0 = box["low"]
        x1, y1, z1 = box["high"]
        if dt_chamber_utils.orientation(sl) == "phi":
            color = "tab:blue"
        else:
            color = "tab:orange"
        for z in [z0, z1]:
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z, z, z, z, z], color=color, linewidth=1.2)
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            ax.plot([x, x], [y, y], [z0, z1], color=color, linewidth=1.2)
    ### tracks as straight lines from z_lo to z_hi
    z_lo = sl_z_coord[0] - 150
    z_hi = sl_z_coord[1] + 150
    tan_alpha_x = np.tan(dt_muons["theta"]) * np.cos(dt_muons["phi"])
    tan_alpha_y = np.tan(dt_muons["theta"]) * np.sin(dt_muons["phi"])
    for i in range(n_3d):
        z = np.array([z_lo, z_hi])
        x = dt_muons["x0"][i] + tan_alpha_x[i] * (z - dt_muons["z0"][i])
        y = dt_muons["y0"][i] + tan_alpha_y[i] * (z - dt_muons["z0"][i])
        ax.plot(x, y, z, color="tab:green", linewidth=0.8, alpha=0.6)
    ax.set_xlim(dt_chamber_utils.superlayers_range(dt_chamber_utils.X)[0] - 200, dt_chamber_utils.superlayers_range(dt_chamber_utils.X)[1] + 200)
    ax.set_ylim(dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)[0] - 200, dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)[1] + 200)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("$x$ [mm]", labelpad=14)
    ax.set_ylabel("$y$ [mm]", labelpad=14)
    ax.set_zlabel("$z$ [mm]", labelpad=10)
    x_span = dt_chamber_utils.superlayers_range(dt_chamber_utils.X)[1] - dt_chamber_utils.superlayers_range(dt_chamber_utils.X)[0] + 400
    y_span = dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)[1] - dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)[0] + 400
    ax.set_box_aspect((x_span, y_span, 0.45 * max(x_span, y_span)))  # z is stretched, the chamber is flat
    ax.set_title(f"DT tracks ({n_3d:,} of {n_dt_muons:,} shown, $z$ axis stretched)")
    legend_handles = [
        patches.Patch(edgecolor="tab:blue", facecolor="none"),
        patches.Patch(edgecolor="tab:orange", facecolor="none"),
        patches.Patch(edgecolor="tab:green", facecolor="none"),
    ]
    legend_labels = ["Phi superlayers (SL 1, SL 3)", "Theta superlayer (SL 2)", "DT tracks"]
    ax.legend(legend_handles, legend_labels, prop={"size": 12}, loc="upper left", fancybox=False)
    fig.tight_layout()
    plot_utils.save_figure(fig, "dt_muons_3d", args)

    ### 1d histograms
    with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}):
        ### positions
        plot_utils.plot_histogram(dt_muons["x0"], "dt_muons_x0", args, xlabel="Position $x_0$ [mm]",
                                          full_range=True, log_scale=False, bin_unit="mm")
        plot_utils.plot_histogram(dt_muons["y0"], "dt_muons_y0", args, xlabel="Position $y_0$ [mm]",
                                                  full_range=True, log_scale=False, bin_unit="mm")
        plot_utils.plot_histogram(dt_muons["z0"], "dt_muons_z0", args, xlabel="Position $z_0$ [mm]",
                                                  full_range=True, log_scale=False, bin_unit="mm")
        ### angles
        plot_utils.plot_histogram(np.rad2deg(dt_muons["theta"]), "dt_muons_theta_deg", args, xlabel="Polar angle $\\theta$ [deg]",
                                  full_range=True, log_scale=False, bin_unit="deg")
        plot_utils.plot_histogram(np.rad2deg(dt_muons["phi"]), "dt_muons_phi_deg", args, xlabel="Azimuthal angle $\\phi$ [deg]",
                                  full_range=True, log_scale=False, bin_unit="deg")
        # weighted with 1 / sin theta
        theta_edges = np.linspace(0-0.5,70+0.5,51)
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        sin_theta_weights = 1/np.sin(np.deg2rad(theta_centers))
        plot_utils.plot_histogram(np.rad2deg(dt_muons["theta"]), "dt_muons_theta_weighted_deg", args, xlabel="Polar angle $\\theta$ weighted by $1/\\sin\\theta$ [deg]",
                                  full_range=True, log_scale=False, bin_unit="deg",
                                  edges=theta_edges, weights=sin_theta_weights)
        ### projected angles in the x-z and y-z planes
        plot_utils.plot_histogram(np.rad2deg(np.arctan(tan_alpha_x)), "dt_muons_angle_xz_deg", args, xlabel="Projected angle in the $x$-$z$ plane [deg]",
                                  full_range=True, log_scale=False, bin_unit="deg")
        plot_utils.plot_histogram(np.rad2deg(np.arctan(tan_alpha_y)), "dt_muons_angle_yz_deg", args, xlabel="Projected angle in the $y$-$z$ plane [deg]",
                                  full_range=True, log_scale=False, bin_unit="deg")
        ### arrival times since the first muon
        ts_sorted = np.sort(np.asarray(dt_muons["ts"], dtype=np.float64))
        plot_utils.plot_histogram(ts_sorted - ts_sorted[0], "dt_muons_ts", args, xlabel="$T_0$ since first muon [s]",
                                  full_range=True, log_scale=False, scale=plot_utils.TS_UNIT_NS * 1e-9, bin_unit="s")
        ### time between consecutive muons
        if n_dt_muons > 1:
            plot_utils.plot_histogram(np.diff(ts_sorted), "dt_muons_delta_ts", args, xlabel="$\\Delta T_0$ [ms]",
                                      full_range=True, log_scale=True, scale=plot_utils.TS_UNIT_NS * 1e-6, bin_unit="ms")

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

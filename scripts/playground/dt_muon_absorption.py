#################################################################
### analysis of muon absorption
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
def add_colorbar(fig, ax, im_obj, label, cmap=None):
    if cmap is not None:
        im_obj.set_cmap(cmap)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits([-3, 3])
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05, format=formatter)
    cbar.set_label(label)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons without absorber (.root)")
    parser.add_argument("--dt_muons_file_absorber", type=str, required=True, help="input file path: dt muons with absorber (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--xy_bin_width", type=float, default=20, help="bin width of the x-y maps in mm")
    parser.add_argument("--xz_bin_width", type=float, default=5, help="bin width of the x-z and y-z projections in mm")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    ### data import
    cuts = data_utils.parse_cuts(args.cuts)
    keys = []
    for key in MUON_KEYS:
        keys.append(key)
    for cut in cuts:
        if cut[0] not in keys:
            keys.append(cut[0])
    log(f"###### Importing dt muons without absorber from {args.dt_muons_file}...")
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
        raise RuntimeError("No dt muons with absorber to plot.")
    log(f"###### Importing dt muons with absorber from {args.dt_muons_file_absorber}...")
    dt_muons_absorber = root_utils.read_tree(args.dt_muons_file_absorber, root_utils.DEFAULT_TREE, branches=keys)
    n_all_absorber = root_utils.length(dt_muons_absorber)
    if len(cuts) > 0:
        dt_muons_absorber = data_utils.cut_data(dt_muons_absorber, cuts, silent=True)
        log(f"cuts {cuts}: {root_utils.length(dt_muons_absorber):,} / {n_all_absorber:,} muons selected")
    # keep only the muon branches (not the extra branches of the cuts)
    muon_branches_absorber = {}
    for key in MUON_KEYS:
        muon_branches_absorber[key] = dt_muons_absorber[key]
    dt_muons_absorber = muon_branches_absorber
    n_dt_muons_absorber = root_utils.length(dt_muons_absorber)
    if n_dt_muons_absorber == 0:
        raise RuntimeError("No dt muons with absorber to plot.")

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"]))
    log(f"measurement duration without absorber = {duration} s")
    log(f"dt muon count without absorber: {n_dt_muons:,}")
    if duration > 0:
        log(f"dt muon rate without absorber: {n_dt_muons / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")
    duration_absorber = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons_absorber["ts"]) - np.amin(dt_muons_absorber["ts"]))
    log(f"measurement duration with absorber = {duration_absorber} s")
    log(f"dt muon count with absorber: {n_dt_muons_absorber:,}")
    if duration > 0:
        log(f"dt muon rate with absorber: {n_dt_muons_absorber / duration_absorber:.3f} +- {np.sqrt(n_dt_muons_absorber) / duration_absorber:.3f} Hz")

    # ### X-Y maps: track positions in the plane of each superlayer
    # for sl in SLS:
    #     margin = 200  # mm
    #     box = dt_chamber_utils.superlayer_box(sl)
    #     x_edges = np.arange(start=box["low"][dt_chamber_utils.X] - margin, stop=box["high"][dt_chamber_utils.X] + margin + args.xy_bin_width, step=args.xy_bin_width)
    #     y_edges = np.arange(start=box["low"][dt_chamber_utils.Y] - margin, stop=box["high"][dt_chamber_utils.Y] + margin + args.xy_bin_width, step=args.xy_bin_width)
    #     dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=box["center"][dt_chamber_utils.Z])
    #     pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    #     im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
    #     ax.add_patch(patches.Rectangle(
    #         (box["low"][dt_chamber_utils.X], box["low"][dt_chamber_utils.Y]), width=(box["high"][dt_chamber_utils.X] - box["low"][dt_chamber_utils.X]), height=(box["high"][dt_chamber_utils.Y] - box["low"][dt_chamber_utils.Y]),
    #         edgecolor="white", facecolor="None", label="Superlayer position",
    #     ))
    #     # marked cells of this superlayer, only the first one gets a legend label
    #     first_label = True
    #     for cell_sl, ly, wi in marked_cells:
    #         if cell_sl != sl:
    #             continue
    #         if first_label:
    #             label = "Marked cells"
    #         else:
    #             label = None
    #         ax.add_patch(cell_rectangle(sl, ly, wi, edgecolor="red", facecolor="None", label=label))
    #         first_label = False
    #     ax.set_title(f"DT tracks in SL {sl} ($z={box["center"][dt_chamber_utils.Z]:.0f}$ mm)")
    #     ax.set_xlabel("$x$ [mm]")
    #     ax.set_ylabel("$y$ [mm]")
    #     ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
    #     add_colorbar(fig, ax, im_obj, "Counts")
    #     entries = int(np.sum(pos_muons_hist2d))
    #     info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
    #     hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
    #     fig.tight_layout()
    #     plot_utils.save_figure(fig, f"dt_muon_absorption_xy_sl{sl}", args)

    # ### X-Y maps: track positions at specified height
    new_z_planes = {
        273.95+53.5 + 20 # mm
    }
    # without absorber
    for new_z_plane in new_z_planes:
        margin = 200  # mm
        box = dt_chamber_utils.superlayer_box(sl=3)
        x_edges = np.arange(start=box["low"][dt_chamber_utils.X] - margin, stop=box["high"][dt_chamber_utils.X] + margin + args.xy_bin_width, step=args.xy_bin_width)
        y_edges = np.arange(start=box["low"][dt_chamber_utils.Y] - margin, stop=box["high"][dt_chamber_utils.Y] + margin + args.xy_bin_width, step=args.xy_bin_width)
        dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=new_z_plane)
        pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
        pos_muons_hist2d_rate = pos_muons_hist2d / duration
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        im_obj = ax.imshow(X=pos_muons_hist2d_rate, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
        # ax.add_patch(patches.Rectangle(
        #     (box["low"][dt_chamber_utils.X], box["low"][dt_chamber_utils.Y]), width=(box["high"][dt_chamber_utils.X] - box["low"][dt_chamber_utils.X]), height=(box["high"][dt_chamber_utils.Y] - box["low"][dt_chamber_utils.Y]),
        #     edgecolor="white", facecolor="None", label="Superlayer position",
        # ))
        ax.set_title(f"DT tracks without absorber at $z={new_z_plane}$ mm")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        # ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
        add_colorbar(fig, ax, im_obj, "Rate [Hz]")
        entries = int(np.sum(pos_muons_hist2d))
        info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
        hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muon_absorption_xy_z{new_z_plane}_without_absorber", args)
    # with absorber
    for new_z_plane in new_z_planes:
        margin = 200  # mm
        box = dt_chamber_utils.superlayer_box(sl=3)
        x_edges = np.arange(start=box["low"][dt_chamber_utils.X] - margin, stop=box["high"][dt_chamber_utils.X] + margin + args.xy_bin_width, step=args.xy_bin_width)
        y_edges = np.arange(start=box["low"][dt_chamber_utils.Y] - margin, stop=box["high"][dt_chamber_utils.Y] + margin + args.xy_bin_width, step=args.xy_bin_width)
        dt_muons_sl_absorber = muon_utils.change_muon_base_point(muons=dt_muons_absorber, z_new=new_z_plane)
        pos_muons_hist2d_absorber, _, _ = np.histogram2d(x=dt_muons_sl_absorber["y0"], y=dt_muons_sl_absorber["x0"], bins=(y_edges, x_edges))
        pos_muons_hist2d_absorber_rate = pos_muons_hist2d_absorber / duration_absorber
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        im_obj = ax.imshow(X=pos_muons_hist2d_absorber_rate, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
        # ax.add_patch(patches.Rectangle(
        #     (box["low"][dt_chamber_utils.X], box["low"][dt_chamber_utils.Y]), width=(box["high"][dt_chamber_utils.X] - box["low"][dt_chamber_utils.X]), height=(box["high"][dt_chamber_utils.Y] - box["low"][dt_chamber_utils.Y]),
        #     edgecolor="white", facecolor="None", label="Superlayer position",
        # ))
        ax.set_title(f"DT tracks with absorber at $z={new_z_plane}$ mm")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        # ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
        add_colorbar(fig, ax, im_obj, "Rate [Hz]")
        entries = int(np.sum(pos_muons_hist2d_absorber))
        info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
        hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muon_absorption_xy_z{new_z_plane}_with_absorber", args)
    # difference: without absorber - with absorber
    for new_z_plane in new_z_planes:
        margin = 200  # mm
        box = dt_chamber_utils.superlayer_box(sl=3)
        x_edges = np.arange(start=box["low"][dt_chamber_utils.X] - margin, stop=box["high"][dt_chamber_utils.X] + margin + args.xy_bin_width, step=args.xy_bin_width)
        y_edges = np.arange(start=box["low"][dt_chamber_utils.Y] - margin, stop=box["high"][dt_chamber_utils.Y] + margin + args.xy_bin_width, step=args.xy_bin_width)
        dt_muons_sl = muon_utils.change_muon_base_point(muons=dt_muons, z_new=new_z_plane)
        pos_muons_hist2d, _, _ = np.histogram2d(x=dt_muons_sl["y0"], y=dt_muons_sl["x0"], bins=(y_edges, x_edges))
        pos_muons_hist2d_rate = pos_muons_hist2d / duration
        dt_muons_sl_absorber = muon_utils.change_muon_base_point(muons=dt_muons_absorber, z_new=new_z_plane)
        pos_muons_hist2d_absorber, _, _ = np.histogram2d(x=dt_muons_sl_absorber["y0"], y=dt_muons_sl_absorber["x0"], bins=(y_edges, x_edges))
        pos_muons_hist2d_absorber_rate = pos_muons_hist2d_absorber / duration_absorber
        pos_muons_hist2d_difference_rate = pos_muons_hist2d_rate - pos_muons_hist2d_absorber_rate
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        im_obj = ax.imshow(X=pos_muons_hist2d_difference_rate, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="equal")
        # ax.add_patch(patches.Rectangle(
        #     (box["low"][dt_chamber_utils.X], box["low"][dt_chamber_utils.Y]), width=(box["high"][dt_chamber_utils.X] - box["low"][dt_chamber_utils.X]), height=(box["high"][dt_chamber_utils.Y] - box["low"][dt_chamber_utils.Y]),
        #     edgecolor="white", facecolor="None", label="Superlayer position",
        # ))
        ax.set_title(f"DT tracks difference at $z={new_z_plane}$ mm")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        # ax.legend(prop={"size": 12}, loc="upper left", fancybox=False, framealpha=params._legend_alpha)
        add_colorbar(fig, ax, im_obj, "Rate without absorber $-$ rate with absorber [Hz]", cmap="bwr")
        # entries = int(np.sum(pos_muons_hist2d_difference_rate))
        # info_str = f"entries = {entries:,}\nnot shown = {n_dt_muons - entries:,}\ntotal = {n_dt_muons:,}\nbin width = {args.xy_bin_width:g} mm $\\times$ {args.xy_bin_width:g} mm"
        # hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="bottom left")
        fig.tight_layout()
        plot_utils.save_figure(fig, f"dt_muon_absorption_xy_z{new_z_plane}_absorber_difference", args)

    # ### X-Z and Y-Z projections of the tracks through the chamber
    # marked_cell_colors = {}
    # for cell in marked_cells:
    #     marked_cell_colors[cell] = "tab:red"
    # sl_z_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.Z)
    # for orient, slice_name in [("phi", "xz"), ("theta", "yz")]:
    #     if slice_name == "xz":
    #         sl_x_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.X)
    #         other = "y"
    #     else:
    #         sl_x_coord = dt_chamber_utils.superlayers_range(dt_chamber_utils.Y)
    #         other = "x"
    #     margin_x, margin_z = 100, 300  # mm
    #     x_edges = np.arange(start=sl_x_coord[0] - margin_x, stop=sl_x_coord[1] + margin_x + args.xz_bin_width, step=args.xz_bin_width)
    #     z_edges = np.arange(start=sl_z_coord[0] - margin_z, stop=sl_z_coord[1] + margin_z + args.xz_bin_width, step=args.xz_bin_width)
    #     z_bins = (z_edges[:-1] + z_edges[1:]) / 2
    #     # track position at every z bin
    #     x_muons = []
    #     z_muons = []
    #     for z_pos in z_bins:
    #         dt_muons_moved = muon_utils.change_muon_base_point(muons=dt_muons, z_new=z_pos)
    #         if slice_name == "xz":
    #             x_muons.append(dt_muons_moved["x0"])
    #         else:
    #             x_muons.append(dt_muons_moved["y0"])
    #         z_muons.append(dt_muons_moved["z0"])
    #     pos_muons_hist2d, _, _ = np.histogram2d(x=np.concatenate(z_muons), y=np.concatenate(x_muons), bins=(z_edges, x_edges))
    #     fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    #     im_obj = ax.imshow(X=pos_muons_hist2d, origin="lower", extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]], aspect="auto")
    #     plot_utils.draw_chamber(ax, orient, cell_colors=marked_cell_colors, wires=False, transparent=True)
    #     ax.set_title(f"DT tracks, ${slice_name[0]}$-$z$ projection ({n_dt_muons:,} tracks)")
    #     ax.set_xlabel(f"${slice_name[0]}$ [mm]")
    #     ax.set_ylabel("$z$ [mm]")
    #     add_colorbar(fig, ax, im_obj, f"Tracks per bin\n(summed over ${other}$)")
    #     legend_handles = [patches.Patch(edgecolor="white", facecolor="none")]
    #     legend_labels = ["Chamber geometry"]
    #     if len(marked_cells) > 0:
    #         legend_handles.append(patches.Patch(edgecolor="tab:red", facecolor="none"))
    #         legend_labels.append("Marked cells")
    #     ax.legend(legend_handles, legend_labels, prop={"size": 11}, loc="lower center", ncols=2, fancybox=False, framealpha=params._legend_alpha, facecolor="gray")
    #     fig.tight_layout()
    #     plot_utils.save_figure(fig, f"dt_muon_absorption_{slice_name}_chamber", args)

    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

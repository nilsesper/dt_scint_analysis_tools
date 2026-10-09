###########################################
### PLOTTING: histograms, the chamber, figure output
###########################################
# Used by the plotting scripts in scripts/.
#
# One histogram as one plot file:
#   plot_utils.plot_histogram(values, "t0", args, xlabel="$T_0$ [TU]")
# This chooses the bins, counts the entries (hist_utils.calculate_histogram), draws bars with error bars and an info box
# (hist_utils.plot_histogram), and stores the figure as <args.store_plots>/t0.png.
#
# The chamber in the x-z ("phi") or y-z ("theta") view:
#   plot_utils.draw_chamber(ax, "phi", cell_colors={(sl, ly, wi): "red"}, wires=True)

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils import hist_utils, root_utils

# -----------------------------------------

TS_UNIT_NS = 0.78  # ns per timestamp unit (TU)

# -----------------------------------------
# script arguments and figure output
# -----------------------------------------

### choose the matplotlib backend: no window is needed unless plots are shown; call before importing matplotlib.pyplot
def setup_backend(show_plots):
    if not show_plots:
        mpl.use("Agg")

### arguments of all plotting scripts
def add_plot_arguments(parser):
    parser.add_argument("--store_plots", type=str, default=None, help="output directory for the plots (created if needed)")
    parser.add_argument("--show_plots", action="store_true", help="show the plots in windows")
    parser.add_argument("--format", type=str, default="png", help="file format of the stored plots, e.g. png or pdf")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")

def check_plot_arguments(parser, args):
    if args.store_plots is None and not args.show_plots:
        parser.error("give --store_plots <directory> and/or --show_plots")

### store a figure as <args.store_plots>/<name>.<args.format> and close it (unless the plots are shown)
def save_figure(fig, name, args):
    import matplotlib.pyplot as plt
    if args.store_plots is not None:
        os.makedirs(args.store_plots, exist_ok=True)
        path = os.path.join(args.store_plots, f"{name}.{args.format}")
        fig.savefig(path, dpi=120)
        root_utils.log(f"stored {path}")
    if not args.show_plots:
        plt.close(fig)

### show all open figures at the end of a script (only with --show_plots)
def show_figures(args):
    if args.show_plots:
        import matplotlib.pyplot as plt
        plt.show()

# -----------------------------------------
# labels and cell lists
# -----------------------------------------

### file-name-safe version of a branch name ("chi2/ndf" -> "chi2ndf")
def safe_name(key):
    return re.sub(r"[^A-Za-z0-9_.\-]+", "", key.replace("/", ""))

### axis label of a branch: symbol and unit from params._key_symbols / params._key_units if known, else the branch name
# "t0_super_fits" -> symbol of t0 + " (super fits)", "t0_sl1" -> symbol of t0 + " (SL 1)"
def key_label(key):
    base, suffix_text = key, ""
    if key not in params._key_symbols:
        if key.endswith("_super_fits") and key[:-len("_super_fits")] in params._key_symbols:
            base = key[:-len("_super_fits")]
            suffix_text = " (super fits)"
        elif len(key) > 4 and key[-4:-1] == "_sl" and key[-1].isdigit() and key[:-4] in params._key_symbols:
            base = key[:-4]
            suffix_text = f" (SL {key[-1]})"
    if base not in params._key_symbols:
        if "$" in key:
            return key.replace("_", "\\_")
        return key
    label = params._key_symbols[base] + suffix_text
    unit = ""
    if base in params._key_units:
        unit = params._key_units[base]
    if unit == "":
        return label
    return f"{label} [{unit}]"

### cell list "sl:ly:wi,sl:ly:wi,..." -> [(sl, ly, wi)]
def parse_cells(cells_text):
    cells = []
    if cells_text is None or cells_text.strip() == "":
        return cells
    for item in cells_text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Cannot read cell \"{item}\". Expected format: sl:ly:wi")
        cells.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return cells

# -----------------------------------------
# histograms
# -----------------------------------------

### bin edges for the values of a histogram
# - whole numbers over a small range: one bin per value
# - else n_bins equal bins over the full range, or (full_range=False) over the central 99% of the values;
#   values outside are counted as underflow / overflow
def choose_edges(values, n_bins=50, full_range=False):
    low, high = float(np.amin(values)), float(np.amax(values))
    whole_numbers = values.dtype.kind in "iub" or bool(np.all(values == np.round(values)))
    if whole_numbers and (high - low) <= 300:
        return np.arange(np.floor(low) - 0.5, np.ceil(high) + 1.5, 1.0)
    if not full_range:
        percentile_low, percentile_high = np.percentile(values, (0.5, 99.5))
        if percentile_high > percentile_low:
            low, high = float(percentile_low), float(percentile_high)
    if high == low:
        low, high = low - 0.5, high + 0.5
    # 1% more on both sides, so that values exactly on the limits are inside
    pad = 0.01 * (high - low)
    return np.linspace(low - pad, high + pad, n_bins + 1)

### draw already counted bins into ax: bars with error bars (sqrt(N), at least 1, none for empty bins) and an info box
# with entries, underflow, overflow, total, bin count and bin width (drawing: hist_utils.plot_histogram)
# weights (optional): one factor per bin, the bar heights are multiplied by it
def draw_histogram(ax, counts, edges, entries, underflow, overflow, xlabel="", log_scale=False, bin_unit=None, scale=1.0,
                   weights=None, info_loc="top right"):
    centers = hist_utils.centers_from_edges(edges)
    err_counts, _, _ = hist_utils.calculate_hist_uncertainty(hist=counts, do_stat_err=True)
    err_counts = np.where(counts > 0, err_counts, 0)
    hist_utils.plot_histogram(ax=ax, hist=counts, centers=centers * scale, err_hist=err_counts, log_scale=log_scale, add_info=True,
                              entries=int(entries), overflow=int(overflow), underflow=int(underflow), bin_unit=bin_unit,
                              info_loc=info_loc, power_limits=[-3, 4], weights=weights)
    ax.set_xlabel(xlabel)

### one histogram of values as one plot file <args.store_plots>/<name>.<args.format>
# edges: bin edges (default: choose_edges with args.n_bins bins); scale: factor for the x axis (e.g. TS_UNIT_NS for ns)
# NaN / inf values are left out
def plot_histogram(values, name, args, xlabel="", title="", edges=None, full_range=False, log_scale=None, bin_unit=None, scale=1.0, weights=None):
    import matplotlib.pyplot as plt
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if log_scale is None:
        log_scale = args.log_scale
    if edges is None:
        edges = choose_edges(values, n_bins=args.n_bins, full_range=full_range)
    counts, edges, centers, entries, underflow, overflow = hist_utils.calculate_histogram(data=values, edges=edges)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_histogram(ax, counts, edges, entries, underflow, overflow, xlabel=xlabel, log_scale=log_scale, bin_unit=bin_unit, scale=scale, weights=weights)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    save_figure(fig, name, args)

# -----------------------------------------
# the chamber
# -----------------------------------------

### rectangle of a box in the view (h_axis horizontal, v_axis vertical)
def box_rectangle(box, h_axis, v_axis, edgecolor, facecolor, zorder=1):
    return patches.Rectangle((box["low"][h_axis], box["low"][v_axis]), width=dt_chamber_utils.box_size(box, h_axis), height=dt_chamber_utils.box_size(box, v_axis),
                             edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)

### draw the chamber in the view of orient: "phi" (x horizontal, z vertical) or "theta" (y horizontal, z vertical)
# Superlayers which measure the horizontal axis are drawn cell by cell (wires=True: with the wire as dot). In the other
# superlayer the cells of a layer lie behind each other; only the first one is drawn (wires=True: wire as line).
# cell_colors: {(sl, ly, wi): colour} of coloured cells, the other cells are grey
# transparent: only the outlines (in the colour of the cell), for drawing on top of a 2d histogram
def draw_chamber(ax, orient, cell_colors=None, wires=False, transparent=False):
    if cell_colors is None:
        cell_colors = {}
    h_axis = dt_chamber_utils.MEASURED_AXIS[orient]
    v_axis = dt_chamber_utils.Z
    default_color = params._color_info["cell"][None]
    for sl in dt_chamber_utils.superlayers():
        # superlayer outline
        sl_pos, sl_size = params._dt_chamber["sls"][sl]["pos"], params._dt_chamber["sls"][sl]["size"]
        if transparent:
            ax.add_patch(patches.Rectangle((sl_pos[h_axis], sl_pos[v_axis]), width=sl_size[h_axis], height=sl_size[v_axis], facecolor="none", edgecolor="white"))
        else:
            ax.add_patch(patches.Rectangle((sl_pos[h_axis], sl_pos[v_axis]), width=sl_size[h_axis], height=sl_size[v_axis],
                                           facecolor=params._color_info["sl"]["fill"], edgecolor=params._color_info["sl"]["edge"]))
        for ly in dt_chamber_utils.layers(sl):
            for wi in dt_chamber_utils.wires(sl, ly):
                box = dt_chamber_utils.cell(sl, ly, wi)
                color = default_color
                if (sl, ly, wi) in cell_colors:
                    color = cell_colors[(sl, ly, wi)]
                if dt_chamber_utils.orientation(sl) == orient:
                    # cell seen along its wire
                    zorder = 1
                    if color != default_color:
                        zorder = 2  # coloured cells on top
                    if transparent:
                        ax.add_patch(box_rectangle(box, h_axis, v_axis, edgecolor=color, facecolor="none", zorder=zorder))
                    else:
                        ax.add_patch(box_rectangle(box, h_axis, v_axis, edgecolor=params._color_info["cell"]["edge"], facecolor=color, zorder=zorder))
                    if wires:
                        wire_color = params._color_info["cell"]["wire"]
                        if transparent:
                            wire_color = "white"
                        ax.add_patch(patches.Circle((box["center"][h_axis], box["center"][v_axis]), radius=params._wire_draw_radius, edgecolor=None, facecolor=wire_color))
                elif wi == dt_chamber_utils.wires(sl, ly)[0]:
                    # layer seen from the side: one cell for the whole layer
                    if transparent:
                        ax.add_patch(box_rectangle(box, h_axis, v_axis, edgecolor=color, facecolor="none"))
                    else:
                        ax.add_patch(box_rectangle(box, h_axis, v_axis, edgecolor=params._color_info["cell"]["edge"], facecolor=params._color_info["cell"]["side_view"]))
                    if wires and not transparent:
                        ax.add_patch(patches.Polygon([(box["low"][h_axis], box["center"][v_axis]), (box["high"][h_axis], box["center"][v_axis])],
                                                     linewidth=params._wire_draw_linewidth, edgecolor=params._color_info["cell"]["wire"],
                                                     facecolor=None, closed=False, visible=True))

### draw the cells around an sl pattern in its track frame (reference: wire wi3 of layer 3 of superlayer sl);
# the 4 cells of the pattern (pattern_wires: wire of layer 0-3) in colour
def draw_pattern_cells(ax, sl, wi3, pattern_wires, wires=True):
    # wires relative to wi3 of the cells to draw, per layer (enough for every pattern shape)
    cells_to_draw = {3: [0], 2: [-1, 0], 1: [-1, 0, 1], 0: [-2, -1, 0, 1]}
    h_axis = dt_chamber_utils.measured_axis(sl)
    h_ref, z_ref = dt_chamber_utils.wire_position(sl, 3, wi3)
    for ly in cells_to_draw:
        for relative_wire in cells_to_draw[ly]:
            wi = wi3 + relative_wire
            if not dt_chamber_utils.is_cell(sl, ly, wi):
                continue
            box = dt_chamber_utils.cell(sl, ly, wi)
            color = params._color_info["cell"][None]
            if pattern_wires[ly] == wi:
                color = "aqua"
            low = (box["low"][h_axis] - h_ref, box["low"][dt_chamber_utils.Z] - z_ref)
            width = dt_chamber_utils.box_size(box, h_axis)
            height = dt_chamber_utils.box_size(box, dt_chamber_utils.Z)
            ax.add_patch(patches.Rectangle(low, width=width, height=height, edgecolor=params._color_info["cell"]["edge"], facecolor=color))
            if wires:
                center = (box["center"][h_axis] - h_ref, box["center"][dt_chamber_utils.Z] - z_ref)
                ax.add_patch(patches.Circle(center, radius=params._wire_draw_radius, edgecolor=None, facecolor=params._color_info["cell"]["wire"]))

#################################################################
### plots of a testpulse timing calibration (output of dumpfile_to_dt_tp_corrections.py)
# - testpulse time per wire (after the offset correction), one panel per SL, colours = layers
# - timing correction per wire in ns, one panel per SL
# - chamber map of the corrections (SL / layer vs wire), cells without testpulse peak are white
# - distribution of the corrections per SL
# - time inside the orbit of all testpulse hits per SL, around the testpulse peaks (from the histograms in the file),
#   with the mean first peak and the testpulse offset correction (params._tp_time_offset) of every group of frontend connectors
# - optionally (--dt_tp_hits_file + --cells): the timing histogram of single cells with the selected first peak
# Also works for calibration files converted from the old .pcl (pcl_to_root.py), then only the correction plots are made.
#
# examples:
#   python scripts/plot_dt_tp_corrections.py --dt_tp_corrections_file calib/tp_corrections.root --store_plots plots/tp
#   python scripts/plot_dt_tp_corrections.py --dt_tp_corrections_file calib/tp_corrections.root --store_plots plots/tp \
#          --dt_tp_hits_file calib/tp_dt_hits.root --cells "1:0:10,2:3:40" --ignore_cells "1:1:49,2:1:57,3:1:49"
#################################################################

import argparse
import sys
import numpy as np
import uproot

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

SLS = (1, 2, 3)

### one panel per SL, one series per layer: value (+- error) vs wire
def plot_per_wire(cal, select, key, err_key, ylabel, title, scale=1.0):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i_sl in range(len(SLS)):
        sl = SLS[i_sl]
        ax = axes[i_sl]
        for ly in range(4):
            sel = select & (cal["sl"] == sl) & (cal["ly"] == ly)
            if not sel.any():
                continue
            err = None
            if err_key is not None:
                err = cal[err_key][sel] * scale
            ax.errorbar(x=cal["wi"][sel] + 0.15 * (ly - 1.5), y=cal[key][sel] * scale, yerr=err, color=derived_params.color_wheel(ly),
                        linestyle="", marker="o", markersize=5, label=f"Ly {ly}")
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(f"SL {sl}", fontsize=16)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Wire")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", prop={"size": 14})
    fig.suptitle(title)
    fig.tight_layout()
    return fig

### chamber map: one row per (SL, layer), one column per wire
def plot_chamber_map(cal, select, key, label, title, scale=1.0):
    n_wires = int(cal["wi"].max()) + 1
    matrix = np.full((12, n_wires), np.nan)
    for i_cell in range(len(select)):
        if not select[i_cell]:
            continue
        row = 4 * (cal["sl"][i_cell] - 1) + cal["ly"][i_cell]
        matrix[row][cal["wi"][i_cell]] = cal[key][i_cell] * scale
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    limit = 1
    if np.isfinite(matrix).any():
        limit = np.nanmax(np.abs(matrix))
    im_obj = ax.imshow(X=matrix, origin="lower", extent=[-0.5, n_wires - 0.5, -0.5, 11.5], cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xlabel("Wire")
    ax.set_yticks(list(range(12)))
    ytick_labels = []
    for sl in SLS:
        for ly in range(4):
            ytick_labels.append(f"SL {sl}, Ly {ly}")
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.set_aspect("auto")
    ax.set_title(title)
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05)
    cbar.set_label(label)
    fig.tight_layout()
    return fig

### index of the first bin edge which is >= value (len(edges) if there is none), edges have to be sorted
def first_edge_at_or_above(edges, value):
    index = 0
    while index < len(edges) and edges[index] < value:
        index += 1
    return index

### sort key for frontend connector names like "10a": number first, then the letter
def fe_name_sort_key(fe_name):
    return (int(fe_name[:-1]), fe_name[-1])

### names of the frontend connectors of the selected cells, without duplicates, sorted
def frontend_names(cal, group):
    fe_names = []
    for fe_id in cal["fe_id"][group]:
        fe_name = params._fe_idx_list[fe_id]
        if fe_name not in fe_names:
            fe_names.append(fe_name)
    return sorted(fe_names, key=fe_name_sort_key)

### row of a cell in the calibration tree (None if the cell is not in the tree)
def find_cell_row(cal, sl, ly, wi):
    for i_row in range(root_utils.length(cal)):
        if cal["sl"][i_row] == sl and cal["ly"][i_row] == ly and cal["wi"][i_row] == wi:
            return i_row
    return None

### distribution of the corrections per SL
def plot_correction_histogram(cal, valid):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    values_ns = cal["ts_corr"][valid] * plot_utils.TS_UNIT_NS
    edges = np.linspace(values_ns.min() - 0.5, values_ns.max() + 0.5, 41)
    for sl in SLS:
        sel = valid & (cal["sl"] == sl)
        hist, _ = np.histogram(cal["ts_corr"][sel] * plot_utils.TS_UNIT_NS, bins=edges)
        mean_ns = np.mean(cal['ts_corr'][sel]) * plot_utils.TS_UNIT_NS
        rms_ns = np.std(cal['ts_corr'][sel]) * plot_utils.TS_UNIT_NS
        ax.stairs(hist, edges, linewidth=2, color=derived_params.color_wheel(sl), label=f"SL {sl}: mean {mean_ns:.2f} ns, rms {rms_ns:.2f} ns")
    ax.set_xlabel(r"$T_\mathrm{corr}$ [ns]")
    ax.set_ylabel("Cells")
    ax.legend(prop={"size": 14})
    fig.tight_layout()
    return fig

### time inside the orbit of all hits per SL, around the testpulse peaks
# per group of frontend connectors with the same testpulse offset (params._tp_time_offset, e.g. phi / theta OBDT,
# old cables): solid line = mean raw first peak, dash-dotted line = after subtracting the offset (arrow in between);
# dotted black line = alignment target (mean testpulse time after offset correction)
def plot_ts_orbit_per_sl(cal, valid, dt_tp_corrections_file, window_after_peak):
    corrected = cal["tp_ts_mean_raw"] - cal["tp_offset"]
    lo = min(cal["peak_ts_min"][valid].min(), corrected[valid].min()) - 20
    hi = max(cal["peak_ts_max"][valid].max(), corrected[valid].max()) + window_after_peak
    all_offsets = np.round(cal["tp_offset"], 6)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    with uproot.open(dt_tp_corrections_file) as f:
        for i_sl in range(len(SLS)):
            sl = SLS[i_sl]
            ax = axes[i_sl]
            name = f"ts_orbit_sl{sl}"
            if name not in f:
                continue
            hist, edges = f[name].to_numpy()
            # bins first ... last - 1 are shown
            first = first_edge_at_or_above(edges, lo)
            last = first_edge_at_or_above(edges, hi)
            shown_hist = hist[first:last]
            shown_edges = edges[first:last + 1]
            ax.stairs(shown_hist, shown_edges, linewidth=1.5, color="dimgray")
            ax.set_yscale("log")
            ax.set_ylabel("Hits", fontsize=16)
            ax.set_title(f"SL {sl}: {int(hist.sum()):,} hits, {int(np.sum(shown_hist))} shown", fontsize=16)
            sl_valid = valid & (cal["sl"] == sl)
            if not sl_valid.any():
                continue
            y_arrow = max(shown_hist.max(), 1) * 3
            offsets = np.unique(np.round(cal["tp_offset"][sl_valid], 6))
            for i_group in range(len(offsets)):
                offset = offsets[i_group]
                group = sl_valid & (all_offsets == offset)
                n_group_cells = int(np.sum(group))
                fe_names = frontend_names(cal, group)
                color = derived_params.color_wheel(i_group)
                raw_mean = np.mean(cal["tp_ts_mean_raw"][group])
                ax.axvline(raw_mean, color=color, linestyle="-", linewidth=1.5,
                           label=f"raw peak {raw_mean:.1f} TU, FE {','.join(fe_names)} ({n_group_cells:,} cells)")
                if offset != 0:
                    ax.axvline(raw_mean - offset, color=color, linestyle="-.", linewidth=1.5,
                               label=f"offset correction $-${offset:.1f} TU ({offset * plot_utils.TS_UNIT_NS:.1f} ns)")
                    y_group_arrow = y_arrow * (1.8 ** i_group)
                    ax.annotate("", xy=(raw_mean - offset, y_group_arrow), xytext=(raw_mean, y_group_arrow),
                                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))
            target = np.mean(cal["ts_target"][sl_valid])
            ax.axvline(target, color="black", linestyle=":", linewidth=1.5, label=f"alignment target {target:.1f} TU")
            ax.set_ylim(top=max(shown_hist.max(), 1) * 30)
            ax.legend(prop={"size": 11}, loc="upper right")
    axes[-1].set_xlabel(r"$T_\mathrm{orbit}$ = tdc + 32 bx [TU]")
    fig.suptitle("Testpulse hits: time inside the orbit and testpulse offset correction")
    fig.tight_layout()
    return fig

### timing histogram of one cell with the selected first peak
# r: row of the cell in the calibration tree, ts_orbit: times of all hits of the cell
def plot_single_cell(cal, r, ts_orbit, sl, ly, wi, window_after_peak):
    if cal["valid"][r]:
        lo = cal["peak_ts_min"][r] - 20
        hi = cal["peak_ts_max"][r] + window_after_peak
    else:
        lo = ts_orbit.min() - 20
        hi = ts_orbit.min() + window_after_peak
    edges = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    hist, _ = np.histogram(ts_orbit, bins=edges)
    ax.stairs(hist, edges, linewidth=1.5, color="black", label=f"{len(ts_orbit):,} hits ({int(hist.sum()):,} shown)")
    if cal["valid"][r]:
        ax.axvspan(cal["peak_ts_min"][r] - 0.5, cal["peak_ts_max"][r] + 0.5, color="tab:orange", alpha=0.3,
                   label=f"first peak: {cal['n_peak_hits'][r]:,} hits")
        ax.axvline(cal["tp_ts_mean_raw"][r], color="tab:red", linestyle="--",
                   label=f"mean {cal['tp_ts_mean_raw'][r]:.2f} $\\pm$ {cal['tp_ts_err_raw'][r]:.2f} TU")
    ax.set_xlabel(r"$T_\mathrm{orbit}$ = tdc + 32 bx [TU]")
    ax.set_ylabel("Hits")
    if cal["valid"][r]:
        title_end = f"$T_\\mathrm{{corr}}$ = {cal['ts_corr'][r]:.2f} TU"
    else:
        title_end = "no peak"
    ax.set_title(f"Testpulse timing SL {sl}, Ly {ly}, Wi {wi}: " + title_end)
    ax.legend(prop={"size": 14})
    fig.tight_layout()
    return fig

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 18})
def main():
    parser = argparse.ArgumentParser(description="Plots of a testpulse timing calibration.")
    parser.add_argument("--dt_tp_corrections_file", type=str, required=True, help="input file path: timing calibration (.root) from dumpfile_to_dt_tp_corrections.py")
    parser.add_argument("--dt_tp_hits_file", type=str, default=None,
                        help="optional input file path: testpulse dt hits (.root) written with --dt_tp_hits_file by dumpfile_to_dt_tp_corrections.py")
    parser.add_argument("--cells", type=str, default=None,
                        help="cells for the single cell timing histograms (needs --dt_tp_hits_file), format \"sl:ly:wi,sl:ly:wi,...\"")
    parser.add_argument("--ignore_cells", type=str, default=None, help="cells left out of all plots, format \"sl:ly:wi,sl:ly:wi,...\"")
    parser.add_argument("--window_after_peak", type=float, default=300,
                        help="ts_orbit range shown after the testpulse peaks in the timing histograms, in ts units")
    parser.add_argument("--prefix", type=str, default="tp", help="prefix of the plot file names")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)
    if args.cells is not None and args.dt_tp_hits_file is None:
        parser.error("--cells needs --dt_tp_hits_file")

    ### read the calibration, select the cells to show
    root_utils.check_input_file(args.dt_tp_corrections_file)
    cal = root_utils.read_tree(args.dt_tp_corrections_file, root_utils.DEFAULT_TREE)
    full = "tp_ts_mean" in cal  # made by dumpfile_to_dt_tp_corrections.py (not only converted corrections)
    n_cells = root_utils.length(cal)
    keep = np.full(n_cells, True)
    for sl, ly, wi in plot_utils.parse_cells(args.ignore_cells):
        keep &= ~((cal["sl"] == sl) & (cal["ly"] == ly) & (cal["wi"] == wi))
    valid = keep
    if full:
        valid = keep & (cal["valid"] == 1)
    log(f"###### {args.dt_tp_corrections_file}: {n_cells:,} cells, {int(valid.sum()):,} with correction shown")

    ### testpulse time per wire
    if full:
        fig = plot_per_wire(cal, valid, "tp_ts_mean", "tp_ts_err", r"$\left\langle T_\mathrm{orbit} \right\rangle$ [TU]",
                            "Testpulse time per cell (after offset correction)")
        plot_utils.save_figure(fig, f"{args.prefix}_tp_time_per_wire", args)
    ### correction per wire
    fig = plot_per_wire(cal, valid, "ts_corr", "err_ts_corr", r"$T_\mathrm{corr}$ [ns]", "Timing correction per cell", scale=plot_utils.TS_UNIT_NS)
    plot_utils.save_figure(fig, f"{args.prefix}_ts_corr_per_wire", args)
    ### chamber map
    fig = plot_chamber_map(cal, valid, "ts_corr", r"$T_\mathrm{corr}$ [ns]", "Timing correction", scale=plot_utils.TS_UNIT_NS)
    plot_utils.save_figure(fig, f"{args.prefix}_ts_corr_map", args)
    ### distribution of the corrections
    fig = plot_correction_histogram(cal, valid)
    plot_utils.save_figure(fig, f"{args.prefix}_ts_corr_hist", args)

    ### time inside the orbit of all hits per SL
    if full:
        fig = plot_ts_orbit_per_sl(cal, valid, args.dt_tp_corrections_file, args.window_after_peak)
        plot_utils.save_figure(fig, f"{args.prefix}_ts_orbit_per_sl", args)

    ### timing histograms of single cells
    cells = plot_utils.parse_cells(args.cells)
    if len(cells) > 0:
        hits = root_utils.read_tree(args.dt_tp_hits_file, root_utils.DT_HITS_TREE, branches=["sl", "ly", "wi", "ts_orbit"])
        for sl, ly, wi in cells:
            r = find_cell_row(cal, sl, ly, wi)
            ts_orbit = hits["ts_orbit"][(hits["sl"] == sl) & (hits["ly"] == ly) & (hits["wi"] == wi)]
            if r is None or len(ts_orbit) == 0:
                log(f"cell {sl}:{ly}:{wi}: no hits / no calibration row, skipped")
                continue
            fig = plot_single_cell(cal, r, ts_orbit, sl, ly, wi, args.window_after_peak)
            plot_utils.save_figure(fig, f"{args.prefix}_cell_{sl}_{ly}_{wi}", args)
    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

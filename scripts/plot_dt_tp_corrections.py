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
def _plot_per_wire(cal, select, key, err_key, ylabel, title, scale=1.0):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i_sl, sl in enumerate(SLS):
        ax = axes[i_sl]
        for ly in range(4):
            sel = select & (cal["sl"] == sl) & (cal["ly"] == ly)
            if not sel.any():
                continue
            err = cal[err_key][sel] * scale if err_key is not None else None
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
def _plot_chamber_map(cal, select, key, label, title, scale=1.0):
    n_wires = int(cal["wi"].max()) + 1
    matrix = np.full((12, n_wires), np.nan)
    rows = 4 * (cal["sl"][select] - 1) + cal["ly"][select]
    matrix[rows, cal["wi"][select]] = cal[key][select] * scale
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1
    im_obj = ax.imshow(X=matrix, origin="lower", extent=[-0.5, n_wires - 0.5, -0.5, 11.5], cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xlabel("Wire")
    ax.set_yticks(list(range(12)))
    ax.set_yticklabels([f"SL {sl}, Ly {ly}" for sl in SLS for ly in range(4)], fontsize=12)
    ax.set_aspect("auto")
    ax.set_title(title)
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05)
    cbar.set_label(label)
    fig.tight_layout()
    return fig

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 18})
def main(argv=None):
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
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)
    if args.cells is not None and args.dt_tp_hits_file is None:
        parser.error("--cells needs --dt_tp_hits_file")

    root_utils.check_input_file(args.dt_tp_corrections_file)
    cal = root_utils.read_tree(args.dt_tp_corrections_file, root_utils.DEFAULT_TREE)
    full = "tp_ts_mean" in cal  # made by dumpfile_to_dt_tp_corrections.py (not only converted corrections)
    n_cells = root_utils.length(cal)
    keep = np.full(n_cells, True)
    for sl, ly, wi in plot_utils.parse_cells(args.ignore_cells):
        keep &= ~((cal["sl"] == sl) & (cal["ly"] == ly) & (cal["wi"] == wi))
    valid = keep & (cal["valid"] == 1 if full else np.full(n_cells, True))
    log(f"###### {args.dt_tp_corrections_file}: {n_cells:,} cells, {int(valid.sum()):,} with correction shown")

    ### testpulse time per wire
    if full:
        fig = _plot_per_wire(cal, valid, "tp_ts_mean", "tp_ts_err", r"$\left\langle T_\mathrm{orbit} \right\rangle$ [TU]",
                             "Testpulse time per cell (after offset correction)")
        plot_utils.finish_figure(fig, f"{args.prefix}_tp_time_per_wire", **out)
    ### correction per wire
    fig = _plot_per_wire(cal, valid, "ts_corr", "err_ts_corr", r"$T_\mathrm{corr}$ [ns]", "Timing correction per cell", scale=plot_utils.TS_UNIT_NS)
    plot_utils.finish_figure(fig, f"{args.prefix}_ts_corr_per_wire", **out)
    ### chamber map
    fig = _plot_chamber_map(cal, valid, "ts_corr", r"$T_\mathrm{corr}$ [ns]", "Timing correction", scale=plot_utils.TS_UNIT_NS)
    plot_utils.finish_figure(fig, f"{args.prefix}_ts_corr_map", **out)
    ### distribution of the corrections
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    values_ns = cal["ts_corr"][valid] * plot_utils.TS_UNIT_NS
    edges = np.linspace(values_ns.min() - 0.5, values_ns.max() + 0.5, 41)
    for sl in SLS:
        sel = valid & (cal["sl"] == sl)
        hist, _ = np.histogram(cal["ts_corr"][sel] * plot_utils.TS_UNIT_NS, bins=edges)
        ax.stairs(hist, edges, linewidth=2, color=derived_params.color_wheel(sl),
                  label=f"SL {sl}: mean {np.mean(cal['ts_corr'][sel]) * plot_utils.TS_UNIT_NS:.2f} ns, rms {np.std(cal['ts_corr'][sel]) * plot_utils.TS_UNIT_NS:.2f} ns")
    ax.set_xlabel(r"$T_\mathrm{corr}$ [ns]")
    ax.set_ylabel("Cells")
    ax.legend(prop={"size": 14})
    fig.tight_layout()
    plot_utils.finish_figure(fig, f"{args.prefix}_ts_corr_hist", **out)

    ### time inside the orbit of all hits per SL, around the testpulse peaks
    # per group of frontend connectors with the same testpulse offset (params._tp_time_offset, e.g. phi / theta OBDT,
    # old cables): solid line = mean raw first peak, dash-dotted line = after subtracting the offset (arrow in between);
    # dotted black line = alignment target (mean testpulse time after offset correction)
    if full:
        corrected = cal["tp_ts_mean_raw"] - cal["tp_offset"]
        lo = min(cal["peak_ts_min"][valid].min(), corrected[valid].min()) - 20
        hi = max(cal["peak_ts_max"][valid].max(), corrected[valid].max()) + args.window_after_peak
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        with uproot.open(args.dt_tp_corrections_file) as f:
            for i_sl, sl in enumerate(SLS):
                ax = axes[i_sl]
                name = f"ts_orbit_sl{sl}"
                if name not in f:
                    continue
                hist, edges = f[name].to_numpy()
                first, last = int(np.searchsorted(edges, lo)), int(np.searchsorted(edges, hi))  # bins first ... last - 1
                ax.stairs(hist[first:last], edges[first:last + 1], linewidth=1.5, color="dimgray")
                ax.set_yscale("log")
                ax.set_ylabel("Hits", fontsize=16)
                ax.set_title(f"SL {sl}: {int(hist.sum()):,} hits, {int(np.sum(hist[first:last]))} shown", fontsize=16)
                sl_valid = valid & (cal["sl"] == sl)
                if not sl_valid.any():
                    continue
                y_arrow = max(hist[first:last].max(), 1) * 3
                offsets = np.round(cal["tp_offset"][sl_valid], 6)
                for i_group, offset in enumerate(np.unique(offsets)):
                    group = np.flatnonzero(sl_valid)[offsets == offset]
                    fe_names = sorted({params._fe_idx_list[fe_id] for fe_id in cal["fe_id"][group]}, key=lambda n: (int(n[:-1]), n[-1]))
                    color = derived_params.color_wheel(i_group)
                    raw_mean = np.mean(cal["tp_ts_mean_raw"][group])
                    ax.axvline(raw_mean, color=color, linestyle="-", linewidth=1.5,
                               label=f"raw peak {raw_mean:.1f} TU, FE {','.join(fe_names)} ({len(group):,} cells)")
                    if offset != 0:
                        ax.axvline(raw_mean - offset, color=color, linestyle="-.", linewidth=1.5,
                                   label=f"offset correction $-${offset:.1f} TU ({offset * plot_utils.TS_UNIT_NS:.1f} ns)")
                        ax.annotate("", xy=(raw_mean - offset, y_arrow * (1.8 ** i_group)), xytext=(raw_mean, y_arrow * (1.8 ** i_group)),
                                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5))
                target = np.mean(cal["ts_target"][sl_valid])
                ax.axvline(target, color="black", linestyle=":", linewidth=1.5, label=f"alignment target {target:.1f} TU")
                ax.set_ylim(top=max(hist[first:last].max(), 1) * 30)
                ax.legend(prop={"size": 11}, loc="upper right")
        axes[-1].set_xlabel(r"$T_\mathrm{orbit}$ = tdc + 32 bx [TU]")
        fig.suptitle("Testpulse hits: time inside the orbit and testpulse offset correction")
        fig.tight_layout()
        plot_utils.finish_figure(fig, f"{args.prefix}_ts_orbit_per_sl", **out)

    ### timing histograms of single cells
    cells = plot_utils.parse_cells(args.cells)
    if len(cells) > 0:
        hits = root_utils.read_branches(args.dt_tp_hits_file, ["sl", "ly", "wi", "ts_orbit"], root_utils.DT_HITS_TREE)
        for sl, ly, wi in cells:
            row = np.flatnonzero((cal["sl"] == sl) & (cal["ly"] == ly) & (cal["wi"] == wi))
            ts_orbit = hits["ts_orbit"][(hits["sl"] == sl) & (hits["ly"] == ly) & (hits["wi"] == wi)]
            if len(row) == 0 or len(ts_orbit) == 0:
                log(f"cell {sl}:{ly}:{wi}: no hits / no calibration row, skipped")
                continue
            r = row[0]
            lo = (cal["peak_ts_min"][r] if cal["valid"][r] else ts_orbit.min()) - 20
            hi = (cal["peak_ts_max"][r] if cal["valid"][r] else ts_orbit.min()) + args.window_after_peak
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
            ax.set_title(f"Testpulse timing SL {sl}, Ly {ly}, Wi {wi}: " + (f"$T_\\mathrm{{corr}}$ = {cal['ts_corr'][r]:.2f} TU" if cal["valid"][r] else "no peak"))
            ax.legend(prop={"size": 14})
            fig.tight_layout()
            plot_utils.finish_figure(fig, f"{args.prefix}_cell_{sl}_{ly}_{wi}", **out)
    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

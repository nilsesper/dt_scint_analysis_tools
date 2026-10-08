#################################################################
### dt hits: occupancy and rate of every cell
# - occupancy map and rate map of the full chamber (superlayer / layer vs wire)
# - rate per wire as bar plots, one figure per superlayer
# - list of cells with low or high occupancy, average cell rates per superlayer
# - optionally the hit time difference histogram made by dt_hits_to_hit_diff_hist.py
# (histograms of the single branches: plot_histograms.py)
#
# example:
#   python scripts/dt_root/plot_dt_hits.py --dt_hits_file out/run_dt_hits.root --store_plots plots/dt_hits \
#          --hit_diff_hist_file out/run_hit_diff_hist.root
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import dt_pipeline_utils, hist_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

def _step_size(value):
    return int(value) if value.strip().isdigit() else value

def _wires(sl, ly):
    return range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] + 1)

### 2d map of the chamber: one row per (superlayer, layer), one column per wire
def _plot_chamber_matrix(values, colorbar_label, title):
    n_wires = max(max(_wires(sl, ly)) for sl in range(1, 4) for ly in range(4)) + 1
    chamber_matrix = np.full((12, n_wires), np.nan)  # nan: no cell
    for sl in range(1, 4):
        for ly in range(4):
            for wi in _wires(sl, ly):
                chamber_matrix[4 * (sl - 1) + ly][wi] = values[sl][ly][wi]
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    im_obj = ax.imshow(X=chamber_matrix, origin="lower", extent=[0 - 0.5, n_wires - 1 + 0.5, 0 - 0.5, 11 + 0.5], vmin=0)
    ax.set_xlabel("Wire")
    ax.set_yticks(list(range(12)))
    ax.set_yticklabels([f"SL {sl}, Ly {ly}" for sl in range(1, 4) for ly in range(4)])
    ax.set_aspect("auto")
    ax.set_title(title)
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main(argv=None):
    parser = argparse.ArgumentParser(description="Occupancy and rate plots of dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--hit_diff_hist_file", type=str, default=None,
                        help="optional input file path: hit time difference histogram from dt_hits_to_hit_diff_hist.py (.root)")
    parser.add_argument("--low_fraction", type=float, default=0.5, help="cells below this fraction of the mean count are listed as low occupancy")
    parser.add_argument("--high_fraction", type=float, default=1.5, help="cells above this fraction of the mean count are listed as high occupancy")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE,
                        help="how much of the input file is read at once: memory size like \"200 MB\" or a number of rows")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args(argv)
    plot_utils.check_plot_arguments(parser, args)
    out = dict(store_plots=args.store_plots, show_plots=args.show_plots, file_format=args.format)

    ### count hits per cell
    log(f"###### Counting hits per cell in {args.dt_hits_file}...")
    cell_counts, ts_min, ts_max, n_hits = dt_pipeline_utils.count_dt_cells(args.dt_hits_file, step_size=_step_size(args.step_size))
    if n_hits == 0:
        raise RuntimeError(f"No dt hits in {args.dt_hits_file}.")
    duration_seconds = plot_utils.TS_UNIT_NS * 1e-9 * float(ts_max - ts_min)
    log(f"{n_hits:,} dt hits, measurement duration = {duration_seconds} s")
    cell_rates = {sl: {ly: {wi: c / duration_seconds for wi, c in cell_counts[sl][ly].items()} for ly in cell_counts[sl]} for sl in cell_counts}

    ### occupancy and rate maps
    fig = _plot_chamber_matrix(cell_counts, "Hits", f"DT hits per cell ({n_hits} hits)")
    plot_utils.finish_figure(fig, "dt_hits_occupancy_map", **out)
    fig = _plot_chamber_matrix(cell_rates, "Rate [Hz]", f"DT hit rate per cell ({duration_seconds:.2f} s)")
    plot_utils.finish_figure(fig, "dt_hits_rate_map", **out)

    ### low and high occupancy cells
    all_counts = [cell_counts[sl][ly][wi] for sl in range(1, 4) for ly in range(4) for wi in _wires(sl, ly)]
    n_cells, total_count = len(all_counts), int(np.sum(all_counts))
    mean_count = total_count / n_cells
    log(f"total count all cells: {total_count:,} +- {np.sqrt(total_count):.1f}")
    log(f"mean count all cells: {mean_count:.2f} +- {np.sqrt(total_count) / n_cells:.2f}")
    log(f"mean rate all cells: {mean_count / duration_seconds:.3f} +- {np.sqrt(total_count) / n_cells / duration_seconds:.3f} Hz")
    log(f"cells below {args.low_fraction} x mean or above {args.high_fraction} x mean:")
    low_cells, high_cells = [], []
    for sl in range(1, 4):
        for ly in range(4):
            for wi in _wires(sl, ly):
                count = cell_counts[sl][ly][wi]
                if count < args.low_fraction * mean_count or count > args.high_fraction * mean_count:
                    kind = "low " if count < args.low_fraction * mean_count else "high"
                    (low_cells if kind == "low " else high_cells).append((sl, ly, wi))
                    readout = derived_params._dt_inverted_remap_table.get(sl, {}).get(ly, {}).get(wi)
                    readout_str = f" (ro_ch={readout['ro_ch']:2}, ch={readout['ch']:3})" if readout is not None else ""
                    log(f"  {kind} occupancy in sl={sl:1}, ly={ly:1}, wi={wi:2}{readout_str}: {count:,} hits, {count / duration_seconds:.2f} Hz")
    log("as cell list for --mark_cells of the other plotting scripts:")
    log("  low:  " + ",".join(f"{sl}:{ly}:{wi}" for sl, ly, wi in low_cells))
    log("  high: " + ",".join(f"{sl}:{ly}:{wi}" for sl, ly, wi in high_cells))

    ### average rates without the low occupancy cells
    def _average(sls):
        counts = [cell_counts[sl][ly][wi] for sl in sls for ly in range(4) for wi in _wires(sl, ly) if (sl, ly, wi) not in low_cells]
        total = np.sum(counts)
        return total / len(counts) / duration_seconds, np.sqrt(total) / len(counts) / duration_seconds
    log("average cell rates (low occupancy cells not considered):")
    for label, sls in [("sl 1 (phi)", [1]), ("sl 2 (theta)", [2]), ("sl 3 (phi)", [3]), ("sl 1 & 3 (phi)", [1, 3]), ("chamber", [1, 2, 3])]:
        rate, err_rate = _average(sls)
        log(f"  {label:15s}: {rate:.3f} +- {err_rate:.3f} Hz")

    ### rate per wire, one figure per superlayer
    for sl in range(1, 4):
        fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        for ly in range(4):
            wires = np.array(list(_wires(sl, ly)))
            wire_hits = np.array([cell_counts[sl][ly][wi] for wi in wires])
            wire_rates = wire_hits / duration_seconds
            err_wire_rates = np.sqrt(wire_hits) / duration_seconds
            ax[ly].bar(wires, wire_rates, width=1, align="center")
            ax[ly].bar(wires, bottom=wire_rates - err_wire_rates, height=2 * err_wire_rates, width=1, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
            ax[ly].set_ylim(bottom=0, top=max(np.amax(wire_rates + err_wire_rates), 1e-9) * 1.1)
            if ly == 3:
                ax[ly].set_xlabel("Wire")
            ax[ly].set_ylabel("Rate [Hz]")
            ax[ly].set_title(f"SL {sl}, Ly {ly}", fontsize=16)
        fig.tight_layout()
        plot_utils.finish_figure(fig, f"dt_hits_rate_sl{sl}", **out)

    ### hit time difference histogram (made before by dt_hits_to_hit_diff_hist.py)
    if args.hit_diff_hist_file is not None:
        root_utils.check_input_file(args.hit_diff_hist_file)
        bins = root_utils.read_tree(args.hit_diff_hist_file, root_utils.DEFAULT_TREE)
        summary = root_utils.read_tree(args.hit_diff_hist_file, root_utils.SUMMARY_TREE)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        hist_utils.plot_histogram(
            ax=ax, hist=bins["hist"], centers=bins["center"] * plot_utils.TS_UNIT_NS, err_hist=np.where(bins["hist"] > 0, bins["err_hist"], 0), log_scale=True, add_info=True,
            entries=int(summary["entries"][0]), overflow=int(summary["overflow"][0]), underflow=int(summary["underflow"][0]), bin_unit="ns", power_limits=[-3, 4],
        )
        ax.set_xlabel("Time between consecutive hits of the same cell [ns]")
        fig.tight_layout()
        plot_utils.finish_figure(fig, "dt_hits_hit_diff", **out)

    plot_utils.show_figures(args.show_plots)

if __name__ == "__main__":
    main()
    log("###### Done.")

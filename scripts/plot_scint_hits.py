#################################################################
### scint hits: rates of every SiPM and strip, SiPM time differences
# - raw scint hits (optional, --raw_scint_hits_file): rate of every SiPM (layer, strip, SiPM 0 / 1)
# - scint hits: rate of every strip, ts SiPM 0 - ts SiPM 1, number of SiPM hits in the group
#
# example:
#   python scripts/plot_scint_hits.py --scint_hits_file out/run_scint_hits.root \
#          --raw_scint_hits_file out/run_raw_scint_hits.root --store_plots plots/scint
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib.pyplot as plt

from analysis_tools.utils import root_utils, scint_hit_utils

# ---------------------------------------------------------------

TS_TO_SECONDS = 0.78e-9

### duration of a table of hits in seconds (first to last hit)
def duration_seconds(hits):
    return TS_TO_SECONDS * (np.amax(hits["ts"]) - np.amin(hits["ts"]))

### bar plot of the rate of every strip of both layers; counts: {(ly, st): count} or {(ly, st, sipm): count} with sipms
def plot_strip_rates(counts, duration, name, title, args, sipms=None):
    fig, axes = plt.subplots(1, len(scint_hit_utils.layers()), figsize=(14, 5), sharey=True)
    for k in range(len(scint_hit_utils.layers())):
        ly = scint_hit_utils.layers()[k]
        ax = axes[k]
        st_list = np.array(scint_hit_utils.strips(ly))
        if sipms is None:
            rates = []
            for st in st_list:
                rates.append(counts[(ly, st)] / duration)
            ax.bar(st_list, rates, width=0.8, color="tab:blue", edgecolor="black")
        else:
            for sipm in sipms:
                rates = []
                for st in st_list:
                    rates.append(counts[(ly, st, sipm)] / duration)
                ax.bar(st_list - 0.2 + 0.4 * sipm, rates, width=0.4, edgecolor="black", label=f"SiPM {sipm}")
            ax.legend()
        ax.set_title(f"Layer {ly} ({scint_hit_utils.orientation(ly)})")
        ax.set_xlabel("Strip")
        ax.set_xticks(st_list)
    axes[0].set_ylabel("Rate [Hz]")
    fig.suptitle(title)
    fig.tight_layout()
    plot_utils.save_figure(fig, name, args)

def main():
    parser = argparse.ArgumentParser(description="Plots of scint hits (and raw scint hits).")
    parser.add_argument("--scint_hits_file", type=str, required=True, help="input file path: scint hits (.root)")
    parser.add_argument("--raw_scint_hits_file", type=str, default=None, help="optional input file path: raw scint hits (.root)")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the histograms")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis for the histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.scint_hits_file)
    scint_hits = root_utils.read_tree(args.scint_hits_file, root_utils.SCINT_HITS_TREE)
    n_scint_hits = root_utils.length(scint_hits)
    log(f"[plot scint hits] {n_scint_hits:,} scint hits from \"{args.scint_hits_file}\"")

    if args.raw_scint_hits_file is not None:
        root_utils.check_input_file(args.raw_scint_hits_file)
        raw_hits = root_utils.read_tree(args.raw_scint_hits_file, root_utils.RAW_SCINT_HITS_TREE, branches=["ly", "st", "sipm", "ts"])
        counts = {}
        for ly in scint_hit_utils.layers():
            for st in scint_hit_utils.strips(ly):
                for sipm in (0, 1):
                    counts[(ly, st, sipm)] = 0
        for i in range(len(raw_hits["ts"])):
            counts[(int(raw_hits["ly"][i]), int(raw_hits["st"][i]), int(raw_hits["sipm"][i]))] += 1
        raw_duration = duration_seconds(raw_hits)
        log(f"[plot scint hits] {len(raw_hits['ts']):,} raw scint hits in {raw_duration:.1f} s")
        plot_strip_rates(counts, raw_duration, "raw_scint_hit_rates", "Raw scint hits (single SiPMs)", args, sipms=(0, 1))

    if n_scint_hits == 0:
        log("[plot scint hits] no scint hits, nothing more to plot")
        plot_utils.show_figures(args)
        return
    counts = {}
    for ly in scint_hit_utils.layers():
        for st in scint_hit_utils.strips(ly):
            counts[(ly, st)] = 0
    for i in range(n_scint_hits):
        counts[(int(scint_hits["ly"][i]), int(scint_hits["st"][i]))] += 1
    duration = duration_seconds(scint_hits)
    log(f"[plot scint hits] {n_scint_hits:,} scint hits in {duration:.1f} s: {n_scint_hits / duration:.2f} Hz")
    plot_strip_rates(counts, duration, "scint_hit_rates", "Scint hits (both SiPMs of a strip)", args)

    plot_utils.plot_histogram(scint_hits["sipm_delta_ts_signed"], "scint_hits_sipm_delta_ts_signed", args,
                              xlabel=plot_utils.key_label("sipm_delta_ts_signed"), full_range=True, bin_unit="TU")
    plot_utils.plot_histogram(scint_hits["n_raw_hits_group"], "scint_hits_n_raw_hits_group", args,
                              xlabel="SiPM channels with a hit in the group", full_range=True)
    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

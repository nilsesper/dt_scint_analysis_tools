#################################################################
### correlation of dt muons and scint hits (file of correlate_dt_muons_and_scint_hits.py)
# - delta_ts = ts scint - ts dt: full range, and a close-up with a Gaussian fit of the peak
# - distance of the dt track to the hit strip, per layer
# - dt muons at the height of the scintillator (x-y), matched ones and (with --dt_muons_file) all of them
#
# example:
#   python scripts/plot_scint_correlation.py --corr_file out/run_corr_dt_muons_scint_hits.root \
#          --dt_muons_file out/run_dt_muons.root --store_plots plots/corr
#################################################################

import argparse
import sys
import numpy as np
from scipy.optimize import curve_fit

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib.pyplot as plt

from analysis_tools.utils import dt_chamber_utils, hist_utils, muon_utils, root_utils, scint_hit_utils

# ---------------------------------------------------------------

def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

### close-up histogram of delta_ts (in ns) with a Gaussian fit in fit_range (ns)
def plot_delta_ts_fit(delta_ts_ns, closeup_range, fit_range, args):
    edges = np.arange(closeup_range[0], closeup_range[1] + plot_utils.TS_UNIT_NS, plot_utils.TS_UNIT_NS)  # 1 TU bins
    counts, edges, centers, entries, underflow, overflow = hist_utils.calculate_histogram(data=delta_ts_ns, edges=edges)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_utils.draw_histogram(ax, counts, edges, entries, underflow, overflow, xlabel="$T_\\text{scint} - T_\\text{DT}$ [ns]", bin_unit="ns")

    in_fit = (centers >= fit_range[0]) & (centers <= fit_range[1])
    fit_centers, fit_counts = centers[in_fit], counts[in_fit]
    err_fit_counts = np.sqrt(np.maximum(fit_counts, 1))
    if np.sum(fit_counts) < 20 or len(fit_counts) < 4:
        log(f"[plot correlation] only {int(np.sum(fit_counts))} entries in the fit range {fit_range} ns, no Gaussian fit")
    else:
        p0 = (np.amax(fit_counts), fit_centers[np.argmax(fit_counts)], (fit_range[1] - fit_range[0]) / 4)
        try:
            popt, pcov = curve_fit(gaussian, fit_centers, fit_counts, p0=p0, sigma=err_fit_counts, absolute_sigma=True)
            err = np.sqrt(np.diag(pcov))
            chi2 = np.sum((fit_counts - gaussian(fit_centers, *popt))**2 / err_fit_counts**2)
            ndf = len(fit_counts) - 3
            fwhm_factor = 2 * np.sqrt(2 * np.log(2))
            text = (f"Gaussian fit $a \\cdot \\exp(-(t-b)^2 / 2c^2)$\n$b = ({popt[1]:.2f} \\pm {err[1]:.2f})$ ns\n"
                    f"$c = ({abs(popt[2]):.2f} \\pm {err[2]:.2f})$ ns\n$\\chi^2 / N_{{df}} = {chi2:.1f} / {ndf}$")
            x = np.linspace(fit_range[0], fit_range[1], 200)
            ax.plot(x, gaussian(x, *popt), color="tab:red", label=text)
            ax.axvline(popt[1], color="tab:red", linestyle="--")
            ax.legend(loc="upper left", fontsize=11)
            log(f"[plot correlation] peak position b = {popt[1]:.2f} +- {err[1]:.2f} ns, width c = {abs(popt[2]):.2f} +- {err[2]:.2f} ns, "
                f"FWHM = {fwhm_factor * abs(popt[2]):.2f} +- {fwhm_factor * err[2]:.2f} ns, chi2/ndf = {chi2:.1f} / {ndf}")
        except RuntimeError as error:
            log(f"[plot correlation] Gaussian fit failed: {error}")
    ax.set_title("Matched dt muons and scint hits")
    fig.tight_layout()
    plot_utils.save_figure(fig, "corr_delta_ts_fit", args)

### dt muons at the height of the scintillator (x-y), with the outlines of the strips
def plot_muons_at_scintillator(dt_muons, name, title, args):
    scint_box = scint_hit_utils.scintillator_box()
    muons_at_scint = muon_utils.change_muon_base_point(muons=dt_muons, z_new=scint_box["center"][dt_chamber_utils.Z], silent=True)
    margin, bin_width = 200, 10
    x_edges = np.arange(scint_box["low"][0] - margin, scint_box["high"][0] + margin + bin_width, bin_width)
    y_edges = np.arange(scint_box["low"][1] - margin, scint_box["high"][1] + margin + bin_width, bin_width)
    counts, _, _ = np.histogram2d(muons_at_scint["y0"], muons_at_scint["x0"], bins=(y_edges, x_edges))
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    im_obj = ax.imshow(counts, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    for ly in scint_hit_utils.layers():
        color = ["white", "tab:orange"][ly % 2]
        for st in scint_hit_utils.strips(ly):
            ax.add_patch(plot_utils.box_rectangle(scint_hit_utils.strip_box(ly, st), 0, 1, edgecolor=color, facecolor="none"))
        ax.plot([], [], color=color, label=f"strips layer {ly}")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title(f"{title} ($z = {scint_box['center'][2]:.0f}$ mm)")
    cbar = fig.colorbar(im_obj, ax=ax, fraction=0.05)
    cbar.set_label("Counts")
    entries = int(np.sum(counts))
    hist_utils.add_infobox(ax=ax, info_str=f"entries = {entries}\nnot shown = {root_utils.length(dt_muons) - entries}\nbin width = {bin_width} mm",
                           info_loc="bottom right")
    fig.tight_layout()
    plot_utils.save_figure(fig, name, args)

def main():
    parser = argparse.ArgumentParser(description="Plots of the correlation of dt muons and scint hits.")
    parser.add_argument("--corr_file", type=str, required=True, help="input file path: correlation (.root, from correlate_dt_muons_and_scint_hits.py)")
    parser.add_argument("--dt_muons_file", type=str, default=None,
                        help="optional: the dt muons file the correlation was made from (for the x-y maps of the dt muons)")
    parser.add_argument("--closeup_range", type=str, default="-100,-10", help="range of the delta_ts close-up in ns: \"low,high\"")
    parser.add_argument("--fit_range", type=str, default="-70,-49", help="range of the Gaussian fit of the delta_ts peak in ns: \"low,high\"")
    parser.add_argument("--n_bins", type=int, default=100, help="number of bins of the histograms")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis for the histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)
    closeup_range = [float(v) for v in args.closeup_range.split(",")]
    fit_range = [float(v) for v in args.fit_range.split(",")]

    root_utils.check_input_file(args.corr_file)
    corr = root_utils.read_tree(args.corr_file, root_utils.DEFAULT_TREE)
    n_pairs = root_utils.length(corr)
    log(f"[plot correlation] {n_pairs:,} pairs of dt muon and scint hit from \"{args.corr_file}\"")
    if n_pairs == 0:
        return

    delta_ts_ns = corr["delta_ts"] * plot_utils.TS_UNIT_NS
    plot_utils.plot_histogram(delta_ts_ns, "corr_delta_ts", args, xlabel="$T_\\text{scint} - T_\\text{DT}$ [ns]",
                              full_range=True, log_scale=True, bin_unit="ns")
    plot_delta_ts_fit(delta_ts_ns, closeup_range, fit_range, args)
    for ly in scint_hit_utils.layers():
        in_layer = corr["ly"] == ly
        if np.sum(in_layer) > 0:
            plot_utils.plot_histogram(corr["dist_to_strip"][in_layer], f"corr_dist_to_strip_ly{ly}", args,
                                      xlabel=f"{plot_utils.key_label('dist_to_strip')} (layer {ly})", full_range=True, bin_unit="mm")

    if args.dt_muons_file is not None:
        root_utils.check_input_file(args.dt_muons_file)
        dt_muons = root_utils.read_tree(args.dt_muons_file, root_utils.DEFAULT_TREE)
        matched = {}
        for key in dt_muons:
            matched[key] = dt_muons[key][corr["dt_muon_row"]]
        plot_muons_at_scintillator(matched, "corr_matched_dt_muons_xy", "Matched dt muons", args)
        plot_muons_at_scintillator(dt_muons, "corr_all_dt_muons_xy", "All dt muons", args)
    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

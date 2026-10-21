#################################################################
### analysis of sl fit residuals
#################################################################

import argparse
import sys
import numpy as np
from tqdm import tqdm
import scipy

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import data_utils, hist_utils, muon_utils, root_utils, timestamp_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import params

# ---------------------------------------------------------------

theta_sl = dt_chamber_utils.theta_superlayer()
phi_sl_1, phi_sl_2 = dt_chamber_utils.phi_superlayers()
MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"] + [f"super_fit_row", f"sl{theta_sl}_fit_row", "sl1_t0", "sl2_t0", "sl3_t0", "sl1_t0_sig_prop_corr", "sl2_t0_sig_prop_corr", "sl3_t0_sig_prop_corr"]

script_prefix = "dt_residuals"

###############################

### mean and its uncertainty of the residual in bins of the position
def binned_mean(residual, position, edges):
    n_bins = len(edges) - 1
    mean = np.zeros(n_bins)
    sigma = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((position >= edges[i]) & (position < edges[i+1]))
        mean[i] = np.mean( residual[muon_idcs] )
        sigma[i] = np.std( residual[muon_idcs], ddof=1 ) / np.sqrt( len(residual[muon_idcs]) )
    return mean, sigma

### one superlayer: residual (reference time - t0 of this superlayer, in TU) vs. the track position along its wires
# residual, residual_corr: before / after the signal propagation correction (in TU, converted to ns here)
# position: track position along the wires at the height of the superlayer (mm)
# name: used in the plot file names and the log, e.g. "theta" or "phi1"; position_name: e.g. "x0"
# position_label, residual_label: axis labels
# min_/max_position: fit range of bin centers, can be smaller than position array
def analyze_residuals(residual, position, position_label, residual_label, n_bins, args, min_position=-np.inf, max_position=np.inf):
    residual = residual * 0.78 # convert to ns
    plot_name = f"{script_prefix}_mean_residual_by_position"

    ### calculate binned distribution: position along the wire vs. residual
    edges = np.linspace(np.amin(position)-1, np.amax(position)+1, n_bins+1)
    bin_width = np.diff(edges)[0]
    bins = (edges[:-1] + edges[1:]) / 2
    mean, sigma = binned_mean(residual, position, edges)

    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(bins, mean, yerr=sigma, xerr=np.diff(edges)/4, linestyle="", marker=".")
    ax.set_xlabel(position_label)
    ax.set_ylabel(residual_label)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_name, args)

###############################

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: cut super fits the muons were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--n_bins", type=int, default=100, help="number of bins")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.sl_fits_file)
    root_utils.check_input_file(args.super_fits_file)
    root_utils.check_input_file(args.dt_muons_file)

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
        raise RuntimeError("No dt muons found in the input file.")

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"]))
    log(f"measurement duration = {duration} s")
    log(f"dt muon count: {n_dt_muons:,}")
    if duration > 0:
        log(f"dt muon rate: {n_dt_muons / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")

    ### load sl fits and super fits
    log("loading sl super fits...")
    keys = [f"row_sl{phi_sl_1}", f"row_sl{phi_sl_2}"]
    super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading sl fits...")
    keys = ["t0", "tan_alpha", "x0", "vd"] + [f"{key}{ly}" for ly in range(4) for key in ["dt", "ts", "wi"]]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

    ### find respective sl fit and super fit rows
    theta_sl_fit_rows = dt_muons[f"sl{theta_sl}_fit_row"]
    phi_superfit_rows = dt_muons[f"super_fit_row"]
    phi_sl_fit_rows = np.concatenate([super_fits[f"row_sl{phi_sl_1}"], super_fits[f"row_sl{phi_sl_2}"]])
    sl_fit_rows = np.concatenate([theta_sl_fit_rows, phi_sl_fit_rows])
    # sl_fits = data_utils.slice_data(data=sl_fits, slice_indices=sl_fit_rows)

    n_bins = 20 #50

    ### fit residuals: measured hit time - fitted hit time, where the fitted hit time is t0 + fitted drift time
    residuals = []
    positions = []
    for ly in range(4):
        ts = np.asarray(sl_fits[f"ts{ly}"][sl_fit_rows], dtype=np.float64)
        residuals.append(ts - sl_fits["t0"][sl_fit_rows] - sl_fits[f"dt{ly}"][sl_fit_rows])
        positions.append(sl_fits[f"dt{ly}"][sl_fit_rows] * sl_fits["vd"][sl_fit_rows])
        #positions.append(sl_fits[f"dt{ly}"][sl_fit_rows])
    all_residuals = np.concatenate(residuals)
    all_positions = np.concatenate(positions)
    plot_utils.plot_histogram(all_residuals, f"{script_prefix}_residuals", args, xlabel="Fit residual $T_{ly} - T_{ly}^\\text{fit}$ of all layers [ns]",
                                title=None, scale=plot_utils.TS_UNIT_NS, bin_unit="ns")

    analyze_residuals(
        residual=all_residuals,
        position=all_positions,
        position_label="reco position in cell [mm]",
        residual_label="residual from linear drift fit [ns]",
        n_bins=n_bins, args=args)


if __name__ == "__main__":
    main()
    log("###### Done.")
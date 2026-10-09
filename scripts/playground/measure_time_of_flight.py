#################################################################
### analysis of signal propagation along the wire
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
MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"] + [f"super_fit_row", f"sl{theta_sl}_fit_row"]

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="Track maps, projections, 3d view and timing plots of dt muons.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: cut super fits the muons were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
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

    ### find respective sl fit and super fit rows
    theta_sl_fit_rows = dt_muons[f"sl{theta_sl}_fit_row"]
    phi_superfit_rows = dt_muons[f"super_fit_row"]

    ### load sl fits and super fits
    log("loading sl fits...")
    keys = ["t0"]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading sl super fits...")
    keys = ["t0_super_fits", f"t0_sl{phi_sl_1}", f"t0_sl{phi_sl_2}", "tan_alpha_super_fits"]
    super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

    #################################
    ### MEASURE MUON TOF
    #################################

    ### calculate time difference between consecutive sls
    sl12_t0_diff = sl_fits["t0"][theta_sl_fit_rows] - super_fits[f"t0_sl{phi_sl_1}"][phi_superfit_rows]
    sl23_t0_diff = super_fits[f"t0_sl{phi_sl_2}"][phi_superfit_rows] - sl_fits["t0"][theta_sl_fit_rows]
    sl13_t0_diff = super_fits[f"t0_sl{phi_sl_2}"][phi_superfit_rows] - super_fits[f"t0_sl{phi_sl_1}"][phi_superfit_rows]

    sl12_t0_diff *= 0.78 # convert to ns
    sl23_t0_diff *= 0.78 # convert to ns
    sl13_t0_diff *= 0.78 # convert to ns
    log(f"sl12_t0_diff: mean={np.mean(sl12_t0_diff)} ns, std={np.std(sl12_t0_diff, ddof=1)} ns")
    log(f"sl23_t0_diff: mean={np.mean(sl23_t0_diff)} ns, std={np.std(sl23_t0_diff, ddof=1)} ns")
    log(f"sl13_t0_diff: mean={np.mean(sl13_t0_diff)} ns, std={np.std(sl13_t0_diff, ddof=1)} ns")

    plot_utils.plot_histogram(sl12_t0_diff, "sl12_t0_diff", args, xlabel="time difference SL2 $-$ SL1 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl23_t0_diff, "sl23_t0_diff", args, xlabel="time difference SL3 $-$ SL2 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl13_t0_diff, "sl13_t0_diff", args, xlabel="time difference SL3 $-$ SL1 [ns]",
            full_range=True, log_scale=False, bin_unit="ns")

    ### calculate binned distribution: reco muon angle vs. time difference
    #angle = dt_muons["theta"]
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_theta = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_theta = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_theta[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_theta[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_theta, yerr=sigma_sl13_t0_diff_by_theta, xerr=np.diff(np.rad2deg(angle_edges))/4, linestyle="")
    ax.set_xlabel(f"reco theta [deg]")
    ax.set_ylabel(f"mean time difference SL3 $-$ SL1 [ns]")
    ax.set_ylim(mean_sl13_t0_diff_by_theta[n_bins//2]-5 , mean_sl13_t0_diff_by_theta[n_bins//2]+5)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_sl13_t0_diff_by_theta", args)

if __name__ == "__main__":
    main()
    log("###### Done.")

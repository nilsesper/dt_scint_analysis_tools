#################################################################
### measure dt scint time of flight
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
    parser.add_argument("--corr_file", type=str, required=True, help="input file path: correlation (.root, from correlate_dt_muons_and_scint_hits.py)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.corr_file)
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
    log(f"###### Importing dt scint correlations from {args.corr_file}...")
    corr = root_utils.read_tree(args.corr_file, root_utils.DEFAULT_TREE)
    n_pairs = root_utils.length(corr)
    log(f"[plot correlation] {n_pairs:,} pairs of dt muon and scint hit from \"{args.corr_file}\"")
    if n_pairs == 0:
        return

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(dt_muons["ts"]) - np.amin(dt_muons["ts"]))
    log(f"measurement duration = {duration} s")
    log(f"dt muon count: {n_dt_muons:,}")
    log(f"correlation count: {n_pairs:,}")
    if duration > 0:
        log(f"dt muon rate: {n_dt_muons / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")
        log(f"correlation rate: {n_pairs / duration:.3f} +- {np.sqrt(n_dt_muons) / duration:.3f} Hz")

    ### find respective sl fit and super fit rows
    dt_muons_rows = corr[f"dt_muon_row"]
    scint_hits_rows = corr[f"scint_hit_row"]

    #################################
    ### MEASURE MUON TOF
    #################################

    ### calculate time difference between consecutive sls
    dt_scint_ts_diff = corr["ts_scint"] - corr["ts_dt"]
    dt_theta = dt_muons["theta"][dt_muons_rows]
    dist_to_strip = corr["dist_to_strip"]

    dt_scint_ts_diff *= 0.78 # convert to ns

    ### restrict to peak
    muon_idcs = np.where((dt_scint_ts_diff >= -80) & (dt_scint_ts_diff <= -30) & (np.abs(dist_to_strip) < 20))
    dt_scint_ts_diff = dt_scint_ts_diff[muon_idcs]
    dt_theta = dt_theta[muon_idcs]

    log(f"dt_scint_ts_diff: mean={np.mean(dt_scint_ts_diff)} ns, std={np.std(dt_scint_ts_diff, ddof=1)} ns")

    plot_utils.plot_histogram(dt_scint_ts_diff, "dt_scint_ts_diff", args, xlabel="time difference Scint $-$ DT [ns]",
        full_range=True, log_scale=False, bin_unit="ns")

    ### calculate binned distribution: reco muon angle vs. time difference
    angle = dt_theta
    n_bins = 10
    angle_edges = np.linspace(np.deg2rad(2), np.deg2rad(50), n_bins+1)
    bin_width = angle_edges[1]-angle_edges[0]
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_theta = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_theta = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_theta[i] = np.mean( dt_scint_ts_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_theta[i] = np.std( dt_scint_ts_diff[muon_idcs], ddof=1 ) / np.sqrt( len(dt_scint_ts_diff[muon_idcs]) )
    # remove nan
    good_bins = ~(np.isnan(mean_sl13_t0_diff_by_theta) | np.isnan(sigma_sl13_t0_diff_by_theta))
    mean_sl13_t0_diff_by_theta = mean_sl13_t0_diff_by_theta[good_bins]
    sigma_sl13_t0_diff_by_theta = sigma_sl13_t0_diff_by_theta[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges)

    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_theta, yerr=sigma_sl13_t0_diff_by_theta, xerr=bin_width/2, linestyle="")
    ax.set_xlabel(f"reco DT $\\theta$ [deg]")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    #ax.set_ylim(mean_sl13_t0_diff_by_theta[n_bins//2]-5 , mean_sl13_t0_diff_by_theta[n_bins//2]+5)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_dt_scint_ts_diff_by_theta", args)

    ### more plots
    fig, ax = plt.subplots()
    ax.errorbar(1/np.cos(angle_bins), mean_sl13_t0_diff_by_theta, yerr=sigma_sl13_t0_diff_by_theta, linestyle="")
    ax.set_xlabel(f"reco DT $1/\\cos\\theta$")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_dt_scint_ts_diff_by_1_cos_theta", args)

    ### perform linear fit
    cos_angle_bins = 1/np.cos(angle_bins)
    mean_sl13_t0_diff_by_theta = mean_sl13_t0_diff_by_theta
    sigma_sl13_t0_diff_by_theta = sigma_sl13_t0_diff_by_theta

    fig, ax = plt.subplots()
    ax.errorbar(cos_angle_bins, mean_sl13_t0_diff_by_theta, yerr=sigma_sl13_t0_diff_by_theta, linestyle="")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=cos_angle_bins, ydata=mean_sl13_t0_diff_by_theta, sigma=sigma_sl13_t0_diff_by_theta, p0=[10,-50], absolute_sigma=True)
    a_val, b_val = popt
    log("linear fit to mean_phi1_t0_residual_by_y0")
    log(f"  a_val = {a_val}")
    log(f"  b_val = {b_val}")
    x_arr = np.linspace(np.amin(cos_angle_bins), np.amax(cos_angle_bins), num=5000)
    ax.plot(x_arr, f_fit(x_arr, a_val, b_val), color="tab:red", label=f"linear fit:\n$a={a_val:.1f}$ ns\n$b={b_val:.1f}$ ns")
    ax.set_xlabel(f"reco DT $1/\\cos\\theta$")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    ax.legend()
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_dt_scint_ts_diff_by_1_cos_theta_fit", args)

    log(f"a_val = h / v_muon ~ 1m / v_muon")
    log(f"v_muon ~ {1/(a_val*1e-9):.4e} m/s")

if __name__ == "__main__":
    main()
    log("###### Done.")

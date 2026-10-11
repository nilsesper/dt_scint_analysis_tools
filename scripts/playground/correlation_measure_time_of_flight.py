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

script_prefix = f"corr_tof"

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="Track maps, projections, 3d view and timing plots of dt muons.")
    parser.add_argument("--corr_file", type=str, required=True, help="input file path: correlation (.root, from correlate_dt_muons_and_scint_hits.py)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: cut super fits the muons were made from (.root)")
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

    ### find respective sl fit and super fit rows
    theta_sl_fit_rows = dt_muons[f"sl{theta_sl}_fit_row"]
    phi_superfit_rows = dt_muons[f"super_fit_row"]

    ### load sl fits and super fits
    log("loading sl fits...")
    keys = ["t0", "tan_alpha"]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading sl super fits...")
    keys = ["t0_super_fits", f"t0_sl{phi_sl_1}", f"t0_sl{phi_sl_2}", "tan_alpha_super_fits"]
    super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

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
    ### measure tof depending on theta angle (muon path length difference)
    #################################

    ### calculate time difference between consecutive sls
    dt_scint_ts_diff = corr["ts_scint"] - corr["ts_dt"]
    dt_theta = dt_muons["theta"][dt_muons_rows]
    dt_phi = dt_muons["phi"][dt_muons_rows]
    dist_to_strip = corr["dist_to_strip"]

    dt_scint_ts_diff *= 0.78 # convert to ns

    ### restrict to peak
    muon_idcs = np.where((dt_scint_ts_diff >= -80) & (dt_scint_ts_diff <= -30) & (np.abs(dist_to_strip) < 20))
    dt_scint_ts_diff = dt_scint_ts_diff[muon_idcs]
    dt_theta = dt_theta[muon_idcs]
    dt_phi = dt_phi[muon_idcs]
    dt_alpha_x = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows[muon_idcs]])
    # dt_alpha_y = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows[muon_idcs]])

    log(f"dt_scint_ts_diff: mean={np.mean(dt_scint_ts_diff)} ns, std={np.std(dt_scint_ts_diff, ddof=1)} ns")

    plot_utils.plot_histogram(dt_scint_ts_diff, f"{script_prefix}_dt_scint_ts_diff", args, xlabel="time difference Scint $-$ DT [ns]",
        full_range=True, log_scale=False, bin_unit="ns")

    angle = dt_alpha_x
    # angle = dt_alpha_y
    # angle = dt_theta
    # angle = dt_phi

    ### calculate binned distribution
    n_bins = 50
    # angle_edges = np.linspace(-np.deg2rad(60), np.deg2rad(60), n_bins+1)
    angle_edges = np.linspace(np.amin(angle)-0.1, np.amax(angle)+0.1, n_bins+1)
    bin_width = angle_edges[1]-angle_edges[0]
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_alpha_x = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_alpha_x = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_alpha_x[i] = np.mean( dt_scint_ts_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_alpha_x[i] = np.std( dt_scint_ts_diff[muon_idcs], ddof=1 ) / np.sqrt( len(dt_scint_ts_diff[muon_idcs]) )
    # remove nan
    good_bins = ~(np.isnan(mean_sl13_t0_diff_by_alpha_x) | np.isnan(sigma_sl13_t0_diff_by_alpha_x))
    mean_sl13_t0_diff_by_alpha_x = mean_sl13_t0_diff_by_alpha_x[good_bins]
    sigma_sl13_t0_diff_by_alpha_x = sigma_sl13_t0_diff_by_alpha_x[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges)
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_alpha_x, yerr=sigma_sl13_t0_diff_by_alpha_x, xerr=bin_width/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco DT $\\alpha_x$ [deg]")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_dt_scint_ts_diff_by_alpha_x", args)

    ### calculate binned distribution: abs reco muon angle vs. time difference sl13_t0_diff
    abs_angle = np.abs(angle)
    n_bins = 50
    angle_edges = np.linspace(np.amin(abs_angle)-1, np.amax(abs_angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_alpha_x = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_alpha_x = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((abs_angle >= angle_edges[i]) & (abs_angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_alpha_x[i] = np.mean( dt_scint_ts_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_alpha_x[i] = np.std( dt_scint_ts_diff[muon_idcs], ddof=1 ) / np.sqrt( len(dt_scint_ts_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_alpha_x, yerr=sigma_sl13_t0_diff_by_alpha_x, xerr=np.diff(np.rad2deg(angle_edges))/4, linestyle="", marker=".")
    ax.set_xlabel(f"reco DT $|\\alpha_x|$ [deg]")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_dt_scint_ts_diff_by_abs_alpha_x", args)

    ### perform linear fit
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl13_t0_diff_by_alpha_x) | np.isnan(sigma_sl13_t0_diff_by_alpha_x)) & ((angle_bins >= np.deg2rad(0)) & (angle_bins < np.deg2rad(60)))
    mean_sl13_t0_diff_by_alpha_x = mean_sl13_t0_diff_by_alpha_x[good_bins]
    sigma_sl13_t0_diff_by_alpha_x = sigma_sl13_t0_diff_by_alpha_x[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges)  
    # perform fit
    cos_angle_bins = 1/np.cos(angle_bins)
    fig, ax = plt.subplots()
    ax.errorbar(1/np.cos(angle_bins), mean_sl13_t0_diff_by_alpha_x, yerr=sigma_sl13_t0_diff_by_alpha_x, linestyle="", marker=".")
    ax.set_xlabel(f"reco DT $1/\\cos|\\alpha_x|$")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_dt_scint_ts_diff_by_1_cos_abs_alpha_x", args)

    fig, ax = plt.subplots()
    ax.errorbar(cos_angle_bins, mean_sl13_t0_diff_by_alpha_x, yerr=sigma_sl13_t0_diff_by_alpha_x, linestyle="", marker=".")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=cos_angle_bins, ydata=mean_sl13_t0_diff_by_alpha_x, sigma=sigma_sl13_t0_diff_by_alpha_x, p0=[10,-50], absolute_sigma=True)
    a_val, b_val = popta_val, b_val = popt
    err_a_val, err_b_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    v_muon = 1.0/(a_val*1e-9)
    err_v_muon = 1.0/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(cos_angle_bins), np.amax(cos_angle_bins), num=5000)
    ax.plot(x_arr, f_fit(x_arr, a_val, b_val), color="tab:red", label=f"linear fit $ax+b$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns\n$v_\\mu = h / a=1.0\\text{{ m }}/a$\n$=({v_muon:.3e}\\pm{err_v_muon:.3e})$ m/s")
    ax.set_xlabel(f"reco DT $1/\\cos\\alpha_x$")
    ax.set_ylabel(f"mean time difference Scint $-$ DT [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_dt_scint_ts_diff_by_1_cos_alpha_x_fit", args)

    log(f"a_val = h / v_muon ~ 0.23m / v_muon")
    log(f"v_muon = h / a_val ~ {v_muon:.4e} m/s")

    #################################
    ### apply timing correction to dt timestamp
    #################################

    dt_muon_sl1_t0 = super_fits[f"t0_sl{phi_sl_1}"][phi_superfit_rows]
    dt_muon_sl2_t0 = sl_fits["t0"][theta_sl_fit_rows]
    dt_muon_sl3_t0 = super_fits[f"t0_sl{phi_sl_2}"][phi_superfit_rows]
    ### correct timing of the different sl timestamps with quadratic correction function
    # f_corr( alpha_x or alpha_y ) = corr for t0_sl_fit
    def f_corr(x, a, b, c):
        return a*x**2+b*x+c


if __name__ == "__main__":
    main()
    log("###### Done.")

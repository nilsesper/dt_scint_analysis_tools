#################################################################
### measure dt muon time of flight
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

script_prefix = "dt_tof"

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
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
    keys = ["t0", "tan_alpha"]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading sl super fits...")
    keys = ["t0_super_fits", f"t0_sl{phi_sl_1}", f"t0_sl{phi_sl_2}", "tan_alpha_super_fits"]
    super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

    #################################
    ### sl time differences by angles
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

    plot_utils.plot_histogram(sl12_t0_diff, f"{script_prefix}_sl12_t0_diff", args, xlabel="time diff SL2$-$SL1 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl23_t0_diff, f"{script_prefix}_sl23_t0_diff", args, xlabel="time diff SL3$-$SL2 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl13_t0_diff, f"{script_prefix}_sl13_t0_diff", args, xlabel="time diff SL3$-$SL1 [ns]",
            full_range=True, log_scale=False, bin_unit="ns")
    
    ####################
    ### by angle_x i.e. angle in sl 1, 3 (phi)

    ### calculate binned distribution: reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_x", args)

    ### calculate binned distribution: reco muon angle vs. time difference sl12_t0_diff
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl12_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl12_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl12_t0_diff_by_angle[i] = np.mean( sl12_t0_diff[muon_idcs] )
        sigma_sl12_t0_diff_by_angle[i] = np.std( sl12_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl12_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl12_t0_diff_by_angle, yerr=sigma_sl12_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL2$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl12_t0_diff_by_angle_x", args)

    ### perform quadratic fit
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl12_t0_diff_by_angle) | np.isnan(sigma_sl12_t0_diff_by_angle)) & ((np.abs(angle_bins) <= np.deg2rad(20)))
    mean_sl12_t0_diff_by_angle = mean_sl12_t0_diff_by_angle[good_bins]
    sigma_sl12_t0_diff_by_angle = sigma_sl12_t0_diff_by_angle[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges) 
    # perform fit
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl12_t0_diff_by_angle, yerr=sigma_sl12_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b, c):
        return a*x**2+b*x+c
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=angle_bins, ydata=mean_sl12_t0_diff_by_angle, sigma=sigma_sl12_t0_diff_by_angle, p0=[1,1,1], absolute_sigma=True)
    a_val, b_val, c_val = popt
    err_a_val, err_b_val, err_c_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    log(f"  c_val = {c_val} +- {err_c_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(angle_bins), np.amax(angle_bins), num=5000)
    ax.plot(np.rad2deg(x_arr), f_fit(x_arr, a_val, b_val, c_val), color="tab:red", label=f"linear fit $ax^2+bx+c$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns/$\\sqrt{{\\text{{deg}}}}$\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns/deg\n$c=({c_val:.3f}\\pm{err_c_val:.3f})$ ns")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL2$-$SL1 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl12_t0_diff_by_angle_x_fit", args)

    ### calculate binned distribution: reco muon angle vs. time difference sl23_t0_diff
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl23_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl23_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl23_t0_diff_by_angle[i] = np.mean( sl23_t0_diff[muon_idcs] )
        sigma_sl23_t0_diff_by_angle[i] = np.std( sl23_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl23_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl23_t0_diff_by_angle, yerr=sigma_sl23_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL2 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl23_t0_diff_by_angle_x", args)

    ### perform quadratic fit
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl23_t0_diff_by_angle) | np.isnan(sigma_sl23_t0_diff_by_angle)) & ((np.abs(angle_bins) <= np.deg2rad(20)))
    mean_sl23_t0_diff_by_angle = mean_sl23_t0_diff_by_angle[good_bins]
    sigma_sl23_t0_diff_by_angle = sigma_sl23_t0_diff_by_angle[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges) 
    # perform fit
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl23_t0_diff_by_angle, yerr=sigma_sl23_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b, c):
        return a*x**2+b*x+c
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=angle_bins, ydata=mean_sl23_t0_diff_by_angle, sigma=sigma_sl23_t0_diff_by_angle, p0=[1,1,1], absolute_sigma=True)
    a_val, b_val, c_val = popt
    err_a_val, err_b_val, err_c_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    log(f"  c_val = {c_val} +- {err_c_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(angle_bins), np.amax(angle_bins), num=5000)
    ax.plot(np.rad2deg(x_arr), f_fit(x_arr, a_val, b_val, c_val), color="tab:red", label=f"linear fit $ax^2+bx+c$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns/$\\sqrt{{\\text{{deg}}}}$\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns/deg\n$c=({c_val:.3f}\\pm{err_c_val:.3f})$ ns")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL2 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl23_t0_diff_by_angle_x_fit", args)

    ####################
    ### by angle_y i.e. angle in sl 2 (theta)

    ### calculate binned distribution: reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_y", args)

    ### calculate binned distribution: reco muon angle vs. time difference sl12_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl12_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl12_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl12_t0_diff_by_angle[i] = np.mean( sl12_t0_diff[muon_idcs] )
        sigma_sl12_t0_diff_by_angle[i] = np.std( sl12_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl12_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl12_t0_diff_by_angle, yerr=sigma_sl12_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL2$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl12_t0_diff_by_angle_y", args)

    ### perform quadratic fit
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl12_t0_diff_by_angle) | np.isnan(sigma_sl12_t0_diff_by_angle)) & ((np.abs(angle_bins) <= np.deg2rad(20)))
    mean_sl12_t0_diff_by_angle = mean_sl12_t0_diff_by_angle[good_bins]
    sigma_sl12_t0_diff_by_angle = sigma_sl12_t0_diff_by_angle[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges) 
    # perform fit
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl12_t0_diff_by_angle, yerr=sigma_sl12_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b, c):
        return a*x**2+b*x+c
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=angle_bins, ydata=mean_sl12_t0_diff_by_angle, sigma=sigma_sl12_t0_diff_by_angle, p0=[1,1,1], absolute_sigma=True)
    a_val, b_val, c_val = popt
    err_a_val, err_b_val, err_c_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    log(f"  c_val = {c_val} +- {err_c_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(angle_bins), np.amax(angle_bins), num=5000)
    ax.plot(np.rad2deg(x_arr), f_fit(x_arr, a_val, b_val, c_val), color="tab:red", label=f"linear fit $ax^2+bx+c$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns/$\\sqrt{{\\text{{deg}}}}$\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns/deg\n$c=({c_val:.3f}\\pm{err_c_val:.3f})$ ns")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL2$-$SL1 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl12_t0_diff_by_angle_y_fit", args)

    ### calculate binned distribution: reco muon angle vs. time difference sl23_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl23_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl23_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl23_t0_diff_by_angle[i] = np.mean( sl23_t0_diff[muon_idcs] )
        sigma_sl23_t0_diff_by_angle[i] = np.std( sl23_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl23_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl23_t0_diff_by_angle, yerr=sigma_sl23_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL2 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl23_t0_diff_by_angle_y", args)

    ### perform quadratic fit
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl23_t0_diff_by_angle) | np.isnan(sigma_sl23_t0_diff_by_angle)) & ((np.abs(angle_bins) <= np.deg2rad(20)))
    mean_sl23_t0_diff_by_angle = mean_sl23_t0_diff_by_angle[good_bins]
    sigma_sl23_t0_diff_by_angle = sigma_sl23_t0_diff_by_angle[good_bins]
    angle_bins = angle_bins[good_bins]
    angle_edges = []
    for i in range(len(angle_bins)):
        angle_edges.append(angle_bins[i]-bin_width/2)
        if angle_bins[i]+bin_width/2 != angle_edges[-1]:
            angle_edges.append(angle_bins[i]+bin_width/2)
    angle_edges = np.array(angle_edges) 
    # perform fit
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl23_t0_diff_by_angle, yerr=sigma_sl23_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b, c):
        return a*x**2+b*x+c
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=angle_bins, ydata=mean_sl23_t0_diff_by_angle, sigma=sigma_sl23_t0_diff_by_angle, p0=[1,1,1], absolute_sigma=True)
    a_val, b_val, c_val = popt
    err_a_val, err_b_val, err_c_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    log(f"  c_val = {c_val} +- {err_c_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(angle_bins), np.amax(angle_bins), num=5000)
    ax.plot(np.rad2deg(x_arr), f_fit(x_arr, a_val, b_val, c_val), color="tab:red", label=f"linear fit $ax^2+bx+c$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns/$\\sqrt{{\\text{{deg}}}}$\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns/deg\n$c=({c_val:.3f}\\pm{err_c_val:.3f})$ ns")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL2 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl23_t0_diff_by_angle_y_fit", args)

    ### calculate binned distribution: reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_y", args)

    ##########################
    ### tof analysis with sl1 sl3 time difference for different theta angles
    ##########################
    print("tof analysis with sl1 sl3 time difference for different theta angles")
    min_angle_phi, max_angle_phi = 0, 2 # deg

    ### calculate binned distribution: reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    angle_phi = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]) & ((np.abs(angle_phi) >= np.deg2rad(min_angle_phi)) & (np.abs(angle_phi) < np.deg2rad(max_angle_phi))))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_y", args)

    ### perform signal prop correction
    # define corr function
    v_sig_prop = 2.7e8 # m/s
    def ts_sig_prop_corr(alpha):
        # ts_corr = delta_x / v_sig_prop
        # delta_x = h * tan_alpha
        ts_corr = 0.23*np.tan(alpha)/v_sig_prop
        ts_corr *= 1e9 # convert to ns
        return ts_corr
    ### recalculate with applied corr
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]) & (np.abs(angle_phi) < np.deg2rad(max_angle_phi)))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle[muon_idcs]) )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle[muon_idcs]) , ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_y$ (theta sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_y_corr", args)

    ### calculate binned distribution: abs reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    angle_phi = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    abs_angle = np.abs(angle)
    n_bins = 50
    #angle_edges = np.linspace(np.amin(abs_angle)-1, np.amax(abs_angle)+1, n_bins+1)
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        #muon_idcs = np.where((abs_angle >= angle_edges[i]) & (abs_angle < angle_edges[i+1]) & (np.abs(angle_phi) < np.deg2rad(max_angle_phi)))
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]) & (np.abs(angle_phi) < np.deg2rad(max_angle_phi)))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle[muon_idcs]) )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle[muon_idcs]) , ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.abs(np.rad2deg(angle_bins)), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $|\\alpha_y|$ (theta sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_abs_angle_y", args)

    ### perform linear fit
    angle_min, angle_max = 0, 55 # deg
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl13_t0_diff_by_angle) | np.isnan(sigma_sl13_t0_diff_by_angle)) & ((np.abs(angle_bins) >= np.deg2rad(angle_min)) & (np.abs(angle_bins) < np.deg2rad(angle_max)))
    mean_sl13_t0_diff_by_angle = mean_sl13_t0_diff_by_angle[good_bins]
    sigma_sl13_t0_diff_by_angle = sigma_sl13_t0_diff_by_angle[good_bins]
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
    ax.errorbar(cos_angle_bins, mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=cos_angle_bins, ydata=mean_sl13_t0_diff_by_angle, sigma=sigma_sl13_t0_diff_by_angle, p0=[10,-50], absolute_sigma=True)
    a_val, b_val = popt
    err_a_val, err_b_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(cos_angle_bins), np.amax(cos_angle_bins), num=5000)
    ax.plot(x_arr, f_fit(x_arr, a_val, b_val), color="tab:red", label=f"linear fit $ax+b$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns\n$v_\\mu = h / a=0.23\\text{{ m }}/a$\n$=({v_muon:.3e}\\pm{err_v_muon:.3e})$ m/s")
    ax.set_xlabel(f"reco $1/\\cos|\\alpha_y|$ (theta sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_abs_angle_y_fit", args)

    log(f"a_val = h / v_muon ~ 0.23m / v_muon")
    log(f"v_muon = h / a_val ~ {v_muon:.4e} m/s")

    ##########################
    ### tof analysis with sl1 sl3 time difference for different phi angles
    ##########################
    print("tof analysis with sl1 sl3 time difference for different phi angles")
    min_angle_theta, max_angle_theta = 0, 2 # deg

    ### calculate binned distribution: reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    angle_theta = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where(((angle >= angle_edges[i]) & (angle < angle_edges[i+1])) & ((np.abs(angle_theta) >= np.deg2rad(min_angle_theta)) & (np.abs(angle_theta) < np.deg2rad(max_angle_theta))))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs], ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_x", args)

    ### perform signal prop correction
    # define corr function
    v_sig_prop = 2.7e8 # m/s
    def ts_sig_prop_corr(alpha):
        # ts_corr = delta_x / v_sig_prop
        # delta_x = h * tan_alpha
        ts_corr = 0.23*np.tan(alpha)/v_sig_prop
        ts_corr *= 1e9 # convert to ns
        return ts_corr
    ### recalculate with applied corr
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    angle_theta = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    n_bins = 50
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]) & (np.abs(angle_theta) < np.deg2rad(max_angle_theta)))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle_theta[muon_idcs]) )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle_theta[muon_idcs]) , ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.rad2deg(angle_bins), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $\\alpha_x$ (phi sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_angle_x_corr", args)

    ### calculate binned distribution: abs reco muon angle vs. time difference sl13_t0_diff
    angle = np.arctan(super_fits["tan_alpha_super_fits"][phi_superfit_rows])
    angle_theta = np.arctan(sl_fits["tan_alpha"][theta_sl_fit_rows])
    #abs_angle = np.abs(angle)
    n_bins = 50
    #angle_edges = np.linspace(np.amin(abs_angle)-1, np.amax(abs_angle)+1, n_bins+1)
    angle_edges = np.linspace(np.amin(angle)-1, np.amax(angle)+1, n_bins+1)
    angle_bins = (angle_edges[:-1] + angle_edges[1:]) / 2
    mean_sl13_t0_diff_by_angle = np.zeros(n_bins)
    sigma_sl13_t0_diff_by_angle = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        #muon_idcs = np.where((abs_angle >= angle_edges[i]) & (abs_angle < angle_edges[i+1]) & (np.abs(angle_theta) < np.deg2rad(max_angle_theta)))
        muon_idcs = np.where((angle >= angle_edges[i]) & (angle < angle_edges[i+1]) & (np.abs(angle_theta) < np.deg2rad(max_angle_theta)))
        mean_sl13_t0_diff_by_angle[i] = np.mean( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle_theta[muon_idcs]) )
        sigma_sl13_t0_diff_by_angle[i] = np.std( sl13_t0_diff[muon_idcs] - ts_sig_prop_corr(angle_theta[muon_idcs]) , ddof=1 ) / np.sqrt( len(sl13_t0_diff[muon_idcs]) )
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(np.abs(np.rad2deg(angle_bins)), mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, xerr=np.diff(np.rad2deg(angle_edges))/2, linestyle="", marker=".")
    ax.set_xlabel(f"reco $|\\alpha_x|$ (phi sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_abs_angle_x", args)

    ### perform linear fit
    angle_min, angle_max = 0, 55 # deg
    # remove nan
    bin_width = angle_edges[1]-angle_edges[0]
    good_bins = ~(np.isnan(mean_sl13_t0_diff_by_angle) | np.isnan(sigma_sl13_t0_diff_by_angle)) & ((np.abs(angle_bins) >= np.deg2rad(angle_min)) & (np.abs(angle_bins) <= np.deg2rad(angle_max)))
    mean_sl13_t0_diff_by_angle = mean_sl13_t0_diff_by_angle[good_bins]
    sigma_sl13_t0_diff_by_angle = sigma_sl13_t0_diff_by_angle[good_bins]
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
    ax.errorbar(cos_angle_bins, mean_sl13_t0_diff_by_angle, yerr=sigma_sl13_t0_diff_by_angle, linestyle="", marker=".", label="data")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, xdata=cos_angle_bins, ydata=mean_sl13_t0_diff_by_angle, sigma=sigma_sl13_t0_diff_by_angle, p0=[10,-50], absolute_sigma=True)
    a_val, b_val = popt
    err_a_val, err_b_val = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    log("linear fit")
    log(f"  a_val = {a_val} +- {err_a_val}")
    log(f"  b_val = {b_val} +- {err_b_val}")
    v_muon = 0.23/(a_val*1e-9)
    err_v_muon = 0.23/(a_val*1e-9)**2 * (err_a_val*1e-9)
    x_arr = np.linspace(np.amin(cos_angle_bins), np.amax(cos_angle_bins), num=5000)
    ax.plot(x_arr, f_fit(x_arr, a_val, b_val), color="tab:red", label=f"linear fit $ax+b$:\n$a=({a_val:.3f}\\pm{err_a_val:.3f})$ ns\n$b=({b_val:.3f}\\pm{err_b_val:.3f})$ ns\n$v_\\mu = h / a=0.23\\text{{ m }}/a$\n$=({v_muon:.3e}\\pm{err_v_muon:.3e})$ m/s")
    ax.set_xlabel(f"reco $1/\\cos|\\alpha_x|$ (phi sl) [deg]")
    ax.set_ylabel(f"corr mean time diff SL3$-$SL1 [ns]")
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{script_prefix}_mean_sl13_t0_diff_by_abs_angle_x_fit", args)

    log(f"a_val = h / v_muon ~ 0.23m / v_muon")
    log(f"v_muon = h / a_val ~ {v_muon:.4e} m/s")

if __name__ == "__main__":
    main()
    log("###### Done.")

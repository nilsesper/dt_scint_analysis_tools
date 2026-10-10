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
    keys = ["t0_super_fits", f"t0_sl{phi_sl_1}", f"t0_sl{phi_sl_2}"]
    super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

    n_bins = 20 #50

    #################################
    ### ANALYZE FOR THETA SL
    #################################

    ### calculate residual of theta from ts reference
    theta_t0_residual = np.zeros(n_dt_muons)
    # calculate residual of theta from phi superfit ts
    theta_t0_residual = super_fits["t0_super_fits"][phi_superfit_rows] - sl_fits["t0"][theta_sl_fit_rows]

    theta_sl_central_z = np.mean([dt_chamber_utils.layer_z(sl=theta_sl, ly=0), dt_chamber_utils.layer_z(sl=theta_sl, ly=3) ])
    dt_muons_theta_base_point = muon_utils.change_muon_base_point(muons=dt_muons, z_new=theta_sl_central_z)

    ### calculate binned distribution: phi-sensitive coordinate vs. theta residual
    x0_edges = np.linspace(np.amin(dt_muons["x0"])-1, np.amax(dt_muons["x0"])+1, n_bins+1)
    x0_bins = (x0_edges[:-1] + x0_edges[1:]) / 2
    mean_theta_t0_residual_by_x0 = np.zeros(n_bins)
    sigma_theta_t0_residual_by_x0 = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((dt_muons_theta_base_point["x0"] >= x0_edges[i]) & (dt_muons_theta_base_point["x0"] < x0_edges[i+1]))
        mean_theta_t0_residual_by_x0[i] = np.mean( theta_t0_residual[muon_idcs] )
        sigma_theta_t0_residual_by_x0[i] = np.std( theta_t0_residual[muon_idcs], ddof=1 ) / np.sqrt( len(theta_t0_residual[muon_idcs]) )
    mean_theta_t0_residual_by_x0 *= 0.78 # conversion to ns
    sigma_theta_t0_residual_by_x0 *= 0.78 # conversion to ns
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(x0_bins, mean_theta_t0_residual_by_x0, yerr=sigma_theta_t0_residual_by_x0, xerr=np.diff(x0_edges)/4, linestyle="")
    ax.set_xlabel(f"reco position (along theta wire) $x_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ theta sl fit $T_0$ [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_theta_t0_residual_by_x0", args)
    ### perform linear fit
    fig, ax = plt.subplots()
    ax.errorbar(x0_bins, mean_theta_t0_residual_by_x0, yerr=sigma_theta_t0_residual_by_x0, xerr=np.diff(x0_edges)/4, linestyle="", label="data")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, x0_bins, mean_theta_t0_residual_by_x0, p0=[0.01,0], absolute_sigma=True)
    inv_v_prop, t_offset = popt
    v_prop = 1/inv_v_prop
    v_prop_conv = v_prop*1e6 # conversion to m/s
    log("linear fit to mean_theta_t0_residual_by_x0")
    log(f"  t_offset = {t_offset}")
    log(f"  inv_v_prop = {inv_v_prop}")
    log(f"  v_prop = {v_prop}")
    x_arr = np.linspace(np.amin(x0_edges), np.amax(x0_edges), num=5000)
    ax.plot(x_arr, f_fit(x_arr, inv_v_prop, t_offset), color="tab:red", label=f"linear fit:\n$v_\\text{{prop}}={v_prop_conv:.2e}$ m/s\n$t_\\text{{offset}}={t_offset:.1f}$ ns")
    ax.set_xlabel(f"reco position (along theta wire) $x_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ theta sl fit $T_0$ [ns]")
    ax.legend()
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_theta_t0_residual_by_x0_fit", args)

    #################################
    ### ANALYZE FOR PHI 1 SL
    #################################

    ### calculate residual of theta from ts reference
    phi1_t0_residual = np.zeros(n_dt_muons)
    # calculate residual of theta from phi superfit ts
    phi1_t0_residual = sl_fits["t0"][theta_sl_fit_rows] - super_fits[f"t0_sl{phi_sl_1}"][phi_superfit_rows]

    phi1_sl_central_z = np.mean([dt_chamber_utils.layer_z(sl=phi_sl_1, ly=0), dt_chamber_utils.layer_z(sl=phi_sl_1, ly=3) ])
    dt_muons_phi1_base_point = muon_utils.change_muon_base_point(muons=dt_muons, z_new=phi1_sl_central_z)

    ### calculate binned distribution: phi-sensitive coordinate vs. theta residual
    y0_edges = np.linspace(np.amin(dt_muons["y0"])-1, np.amax(dt_muons["y0"])+1, n_bins+1)
    y0_bins = (y0_edges[:-1] + y0_edges[1:]) / 2
    mean_phi1_t0_residual_by_y0 = np.zeros(n_bins)
    sigma_phi1_t0_residual_by_y0 = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((dt_muons_phi1_base_point["y0"] >= y0_edges[i]) & (dt_muons_phi1_base_point["y0"] < y0_edges[i+1]))
        mean_phi1_t0_residual_by_y0[i] = np.mean( phi1_t0_residual[muon_idcs] )
        sigma_phi1_t0_residual_by_y0[i] = np.std( phi1_t0_residual[muon_idcs], ddof=1 ) / np.sqrt( len(phi1_t0_residual[muon_idcs]) )
    mean_phi1_t0_residual_by_y0 *= 0.78 # conversion to ns
    sigma_phi1_t0_residual_by_y0 *= 0.78 # conversion to ns
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(y0_bins, mean_phi1_t0_residual_by_y0, yerr=sigma_phi1_t0_residual_by_y0, xerr=np.diff(y0_edges)/4, linestyle="")
    ax.set_xlabel(f"reco position (along phi wire) $y_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ phi1 sl fit $T_0$ [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_phi1_t0_residual_by_y0", args)
    ### perform linear fit
    fig, ax = plt.subplots()
    ax.errorbar(y0_bins, mean_phi1_t0_residual_by_y0, yerr=sigma_phi1_t0_residual_by_y0, xerr=np.diff(y0_edges)/4, linestyle="", label="data")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, y0_bins, mean_phi1_t0_residual_by_y0, p0=[0.01,0], absolute_sigma=True)
    inv_v_prop, t_offset = popt
    v_prop = 1/inv_v_prop
    v_prop_conv = v_prop*1e6 # conversion to m/s
    log("linear fit to mean_phi1_t0_residual_by_y0")
    log(f"  t_offset = {t_offset}")
    log(f"  inv_v_prop = {inv_v_prop}")
    log(f"  v_prop = {v_prop}")
    x_arr = np.linspace(np.amin(y0_edges), np.amax(y0_edges), num=5000)
    ax.plot(x_arr, f_fit(x_arr, inv_v_prop, t_offset), color="tab:red", label=f"linear fit:\n$v_\\text{{prop}}={v_prop_conv:.2e}$ m/s\n$t_\\text{{offset}}={t_offset:.1f}$ ns")
    ax.set_xlabel(f"reco position (along phi wire) $y_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ phi1 sl fit $T_0$ [ns]")
    ax.legend()
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_phi1_t0_residual_by_y0_fit", args)

    #################################
    ### ANALYZE FOR PHI 2 SL
    #################################

    ### calculate residual of theta from ts reference
    phi2_t0_residual = np.zeros(n_dt_muons)
    # calculate residual of theta from phi superfit ts
    phi2_t0_residual = sl_fits["t0"][theta_sl_fit_rows] - super_fits[f"t0_sl{phi_sl_2}"][phi_superfit_rows]

    phi2_sl_central_z = np.mean([dt_chamber_utils.layer_z(sl=phi_sl_1, ly=0), dt_chamber_utils.layer_z(sl=phi_sl_1, ly=3) ])
    dt_muons_phi2_base_point = muon_utils.change_muon_base_point(muons=dt_muons, z_new=phi2_sl_central_z)

    ### calculate binned distribution: phi-sensitive coordinate vs. theta residual
    y0_edges = np.linspace(np.amin(dt_muons["y0"])-1, np.amax(dt_muons["y0"])+1, n_bins+1)
    y0_bins = (y0_edges[:-1] + y0_edges[1:]) / 2
    mean_phi2_t0_residual_by_y0 = np.zeros(n_bins)
    sigma_phi2_t0_residual_by_y0 = np.zeros(n_bins)
    for i in tqdm(range(n_bins)):
        muon_idcs = np.where((dt_muons_phi2_base_point["y0"] >= y0_edges[i]) & (dt_muons_phi2_base_point["y0"] < y0_edges[i+1]))
        mean_phi2_t0_residual_by_y0[i] = np.mean( phi2_t0_residual[muon_idcs] )
        sigma_phi2_t0_residual_by_y0[i] = np.std( phi2_t0_residual[muon_idcs], ddof=1 ) / np.sqrt( len(phi2_t0_residual[muon_idcs]) )
    mean_phi2_t0_residual_by_y0 *= 0.78 # conversion to ns
    sigma_phi2_t0_residual_by_y0 *= 0.78 # conversion to ns
    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(y0_bins, mean_phi2_t0_residual_by_y0, yerr=sigma_phi2_t0_residual_by_y0, xerr=np.diff(y0_edges)/4, linestyle="")
    ax.set_xlabel(f"reco position (along phi wire) $y_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ phi2 sl fit $T_0$ [ns]")
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_phi2_t0_residual_by_y0", args)
    ### perform linear fit
    fig, ax = plt.subplots()
    ax.errorbar(y0_bins, mean_phi2_t0_residual_by_y0, yerr=sigma_phi2_t0_residual_by_y0, xerr=np.diff(y0_edges)/4, linestyle="", label="data")
    def f_fit(x, a, b):
        return a*x+b
    popt, pcov = scipy.optimize.curve_fit(f_fit, y0_bins, mean_phi2_t0_residual_by_y0, p0=[0.01,0], absolute_sigma=True)
    inv_v_prop, t_offset = popt
    v_prop = 1/inv_v_prop
    v_prop_conv = v_prop*1e6 # conversion to m/s
    log("linear fit to mean_phi2_t0_residual_by_y0")
    log(f"  t_offset = {t_offset}")
    log(f"  inv_v_prop = {inv_v_prop}")
    log(f"  v_prop = {v_prop}")
    x_arr = np.linspace(np.amin(y0_edges), np.amax(y0_edges), num=5000)
    ax.plot(x_arr, f_fit(x_arr, inv_v_prop, t_offset), color="tab:red", label=f"linear fit:\n$v_\\text{{prop}}={v_prop_conv:.2e}$ m/s\n$t_\\text{{offset}}={t_offset:.1f}$ ns")
    ax.set_xlabel(f"reco position (along phi wire) $y_0$ [mm]")
    ax.set_ylabel(f"reco reference $T$ $-$ phi2 sl fit $T_0$ [ns]")
    ax.legend()
    fig.tight_layout()
    plot_utils.save_figure(fig, f"mean_phi2_t0_residual_by_y0_fit", args)


if __name__ == "__main__":
    main()
    log("###### Done.")

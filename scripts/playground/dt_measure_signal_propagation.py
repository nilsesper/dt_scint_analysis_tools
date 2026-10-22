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

from analysis_tools.utils import data_utils, hist_utils, muon_utils, root_utils, timestamp_utils, dt_correction_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import params

# ---------------------------------------------------------------

theta_sl = dt_chamber_utils.theta_superlayer()
phi_sl_1, phi_sl_2 = dt_chamber_utils.phi_superlayers()
MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"] + [f"sl{theta_sl}_fit_row", f"sl{phi_sl_1}_fit_row", f"sl{phi_sl_2}_fit_row", "sl1_t0", "sl2_t0", "sl3_t0"]

script_prefix = "dt_wireprop"

###############################

### linear fit function: t = 1/v_prop * position + t_offset
def f_fit(x, a, b):
    return a*x+b

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
def analyze_superlayer(residual, residual_corr, position, name, position_name, position_label, residual_label, n_bins, args, min_position=-np.inf, max_position=np.inf):

    plot_name = f"{script_prefix}_mean_{name}_t0_residual_by_{position_name}"

    ### calculate binned distribution: position along the wire vs. residual
    edges = np.linspace(np.amin(position)-1, np.amax(position)+1, n_bins+1)
    bin_width = np.diff(edges)[0]
    bins = (edges[:-1] + edges[1:]) / 2
    mean, sigma = binned_mean(residual, position, edges)
    mean_corr, sigma_corr = binned_mean(residual_corr, position, edges)

    ### plot distribution
    fig, ax = plt.subplots()
    ax.errorbar(bins, mean, yerr=sigma, xerr=np.diff(edges)/2, linestyle="", marker=".")
    ax.set_xlabel(position_label)
    ax.set_ylabel(residual_label)
    fig.tight_layout()
    plot_utils.save_figure(fig, plot_name, args)

    ### perform linear fit (bins without a value, e.g. empty bins, are left out)
    good_bins = ~(np.isnan(mean) | np.isnan(sigma) | (sigma == 0)) & (bins > min_position) & (bins < max_position)
    fig, ax = plt.subplots()
    ax.errorbar(bins, mean, yerr=sigma, xerr=np.diff(edges)/2, linestyle="", label="data", marker=".")
    popt, pcov = scipy.optimize.curve_fit(f_fit, bins[good_bins], mean[good_bins], sigma=sigma[good_bins], p0=[0.01,0], absolute_sigma=True)
    inv_v_prop, t_offset = popt
    err_inv_vprop, err_t_offset = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    v_prop = 1/inv_v_prop
    err_v_prop = 1/inv_v_prop**2 * err_inv_vprop
    v_prop_conv = v_prop*1e6 # conversion to m/s
    err_v_prop_conv = err_v_prop*1e6 # conversion to m/s
    log(f"linear fit to mean_{name}_t0_residual_by_{position_name}")
    log(f"  t_offset = {t_offset} +- {err_t_offset}")
    log(f"  inv_v_prop = {inv_v_prop} +- {err_inv_vprop}")
    log(f"  v_prop = {v_prop} +- {err_v_prop}")
    pos_arr = np.linspace(np.amin(bins[good_bins])-bin_width/2, np.amax(bins[good_bins])+bin_width/2, num=5000)
    ax.plot(pos_arr, f_fit(pos_arr, inv_v_prop, t_offset), color="tab:red", label=f"linear fit $1/v_\\text{{prop}} x+t_\\text{{offset}}$:\n$v_\\text{{prop}}=({v_prop_conv:.2e}\\pm{err_v_prop_conv:.2e})$ m/s\n$t_\\text{{offset}}=({t_offset:.3f}\\pm{err_t_offset:.3f})$ ns")
    ax.set_xlabel(position_label)
    ax.set_ylabel(residual_label)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{plot_name}_fit", args)

    ### plot distribution before and after the correction
    fig, ax = plt.subplots()
    ax.errorbar(bins, mean, yerr=sigma, xerr=np.diff(edges)/2, linestyle="", marker=".", color="tab:blue", label="before")
    ax.errorbar(bins, mean_corr, yerr=sigma_corr, xerr=np.diff(edges)/2, linestyle="", marker=".", color="tab:orange", label="after")
    ax.set_xlabel(position_label)
    ax.set_ylabel(residual_label)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plot_utils.save_figure(fig, f"{plot_name}_corr", args)

###############################

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.sl_fits_file)
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
    phi1_sl_fit_rows = dt_muons[f"sl{phi_sl_1}_fit_row"]
    phi2_sl_fit_rows = dt_muons[f"sl{phi_sl_2}_fit_row"]

    ### load sl fits and super fits
    log("loading sl fits...")
    keys = ["t0", "tan_alpha"]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    log("loading done")

    #################################
    ### sl time differences
    #################################

    n_bins = 30

    ### determine times and angles
    sl1_t0 = dt_muons["sl1_t0"]
    sl2_t0 = dt_muons["sl2_t0"]
    sl3_t0 = dt_muons["sl3_t0"]
    tan_angle_theta = sl_fits["tan_alpha"][theta_sl_fit_rows]
    tan_angle_phi = ( sl_fits["tan_alpha"][phi1_sl_fit_rows] + sl_fits["tan_alpha"][phi2_sl_fit_rows] ) / 2

    ### calculate time difference between consecutive sls
    sl12_t0_diff = sl2_t0 - sl1_t0
    sl23_t0_diff = sl3_t0 - sl2_t0
    sl13_t0_diff = sl3_t0 - sl1_t0
    sl12_t0_diff = sl2_t0 - sl1_t0
    sl23_t0_diff = sl3_t0 - sl2_t0
    sl13_t0_diff = sl3_t0 - sl1_t0
    sl12_t0_diff *= 0.78 # convert to ns
    sl23_t0_diff *= 0.78 # convert to ns
    sl13_t0_diff *= 0.78 # convert to ns

    ### timing correction due to signal prop along wire
    (_,sl1_wire_pos,_) = muon_utils.propagate_muons(dt_muons, z=dt_chamber_utils.SL1_CENTRAL_Z)
    (sl2_wire_pos,_,_) = muon_utils.propagate_muons(dt_muons, z=dt_chamber_utils.SL2_CENTRAL_Z)
    (_,sl3_wire_pos,_) = muon_utils.propagate_muons(dt_muons, z=dt_chamber_utils.SL3_CENTRAL_Z)
    # sl fit corrected timing info
    sl1_t0_sp = sl1_t0 - dt_correction_utils.ts_sig_prop_corr(x_wire=sl1_wire_pos)
    sl2_t0_sp = sl2_t0 - dt_correction_utils.ts_sig_prop_corr(x_wire=dt_chamber_utils.SL2_WIRE_LENGTH-sl2_wire_pos)
    sl3_t0_sp = sl3_t0 - dt_correction_utils.ts_sig_prop_corr(x_wire=sl3_wire_pos)

    ### calculate time difference between consecutive sls
    sl12_t0_diff_sp = sl2_t0_sp - sl1_t0_sp
    sl23_t0_diff_sp = sl3_t0_sp - sl2_t0_sp
    sl13_t0_diff_sp = sl3_t0_sp - sl1_t0_sp
    sl12_t0_diff_sp = sl2_t0_sp - sl1_t0_sp
    sl23_t0_diff_sp = sl3_t0_sp - sl2_t0_sp
    sl13_t0_diff_sp = sl3_t0_sp - sl1_t0_sp
    sl12_t0_diff_sp *= 0.78 # convert to ns
    sl23_t0_diff_sp *= 0.78 # convert to ns
    sl13_t0_diff_sp *= 0.78 # convert to ns

    ########################################

    ### THETA SL: reference = t0 of sl phi 1, position along the theta wires = x at the height of SL 2
    analyze_superlayer(
        residual=sl12_t0_diff,
        residual_corr=sl12_t0_diff_sp,
        position=muon_utils.change_muon_base_point(muons=dt_muons, z_new=dt_chamber_utils.SL2_CENTRAL_Z)["x0"],
        min_position=250,
        max_position=1750,
        name="theta_phi1", position_name="x0",
        position_label="reco position (along theta wire) $x_0$ [mm]",
        residual_label="reco time SL2$-$SL1 [ns]",
        n_bins=n_bins, args=args)

    ### THETA SL: reference = t0 of sl phi 1, position along the theta wires = x at the height of SL 2
    analyze_superlayer(
        residual=sl23_t0_diff,
        residual_corr=sl23_t0_diff_sp,
        position=muon_utils.change_muon_base_point(muons=dt_muons, z_new=dt_chamber_utils.SL2_CENTRAL_Z)["x0"],
        min_position=250,
        max_position=1750,
        name="theta_phi2", position_name="x0",
        position_label="reco position (along theta wire) $x_0$ [mm]",
        residual_label="reco time SL3$-$SL2 [ns]",
        n_bins=n_bins, args=args)

    ### PHI 1 SL: reference = t0 of the theta sl, position along the phi wires = y at the height of SL 1
    analyze_superlayer(
        residual=sl12_t0_diff,
        residual_corr=sl12_t0_diff_sp,
        position=muon_utils.change_muon_base_point(muons=dt_muons, z_new=dt_chamber_utils.SL1_CENTRAL_Z)["y0"],
        min_position=250,
        max_position=2000,
        name="phi1", position_name="y0",
        position_label="reco position (along phi wire) $y_0$ [mm]",
        residual_label="reco time SL2$-$SL1 [ns]",
        n_bins=n_bins, args=args)

    ### PHI 2 SL: reference = t0 of the theta sl, position along the phi wires = y at the height of SL 3
    analyze_superlayer(
        residual=sl23_t0_diff,
        residual_corr=sl23_t0_diff_sp,
        position=muon_utils.change_muon_base_point(muons=dt_muons, z_new=dt_chamber_utils.SL3_CENTRAL_Z)["y0"],
        min_position=250,
        max_position=2000,
        name="phi2", position_name="y0",
        position_label="reco position (along phi wire) $y_0$ [mm]",
        residual_label="reco time SL3$-$SL2 [ns]",
        n_bins=n_bins, args=args)


if __name__ == "__main__":
    main()
    log("###### Done.")
#################################################################
### investigate effects of different timing corrections
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

script_prefix = "dt_timing_corrections"

###############################

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the muons were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
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
    ### sl time differences - before any correction
    #################################

    n_bins = 50

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

    ### plot histograms of time differences
    log(f"sl12_t0_diff: mean={np.mean(sl12_t0_diff)} ns, std={np.std(sl12_t0_diff, ddof=1)} ns")
    log(f"sl23_t0_diff: mean={np.mean(sl23_t0_diff)} ns, std={np.std(sl23_t0_diff, ddof=1)} ns")
    log(f"sl13_t0_diff: mean={np.mean(sl13_t0_diff)} ns, std={np.std(sl13_t0_diff, ddof=1)} ns")
    plot_utils.plot_histogram(sl12_t0_diff, f"{script_prefix}_sl12_t0_diff", args, xlabel="time diff SL2$-$SL1 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl23_t0_diff, f"{script_prefix}_sl23_t0_diff", args, xlabel="time diff SL3$-$SL2 [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl13_t0_diff, f"{script_prefix}_sl13_t0_diff", args, xlabel="time diff SL3$-$SL1 [ns]",
            full_range=True, log_scale=False, bin_unit="ns")

    #################################
    ### sl time differences - after signal prop along wire correction
    #################################

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

    ### plot histograms of time differences
    log(f"sl12_t0_diff_sp: mean={np.mean(sl12_t0_diff_sp)} ns, std={np.std(sl12_t0_diff_sp, ddof=1)} ns")
    log(f"sl23_t0_diff_sp: mean={np.mean(sl23_t0_diff_sp)} ns, std={np.std(sl23_t0_diff_sp, ddof=1)} ns")
    log(f"sl13_t0_diff_sp: mean={np.mean(sl13_t0_diff_sp)} ns, std={np.std(sl13_t0_diff_sp, ddof=1)} ns")
    plot_utils.plot_histogram(sl12_t0_diff_sp, f"{script_prefix}_sl12_t0_diff_sp", args, xlabel="time diff SL2$-$SL1 (sp corr) [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl23_t0_diff_sp, f"{script_prefix}_sl23_t0_diff_sp", args, xlabel="time diff SL3$-$SL2 (sp corr) [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl13_t0_diff_sp, f"{script_prefix}_sl13_t0_diff_sp", args, xlabel="time diff SL3$-$SL1 (sp corr) [ns]",
            full_range=True, log_scale=False, bin_unit="ns")

    #################################
    ### sl time differences - after angular timing correction
    #################################

    ### timing correction due to track angle
    sl1_t0_sp_ang = sl1_t0_sp - dt_correction_utils.ts_fit_angle_corr(tan_alpha=tan_angle_phi)
    sl2_t0_sp_ang = sl2_t0_sp - dt_correction_utils.ts_fit_angle_corr(tan_alpha=tan_angle_theta)
    sl3_t0_sp_ang = sl3_t0_sp - dt_correction_utils.ts_fit_angle_corr(tan_alpha=tan_angle_phi)

    ### calculate time difference between consecutive sls
    sl12_t0_diff_sp_ang = sl2_t0_sp_ang - sl1_t0_sp_ang
    sl23_t0_diff_sp_ang = sl3_t0_sp_ang - sl2_t0_sp_ang
    sl13_t0_diff_sp_ang = sl3_t0_sp_ang - sl1_t0_sp_ang
    sl12_t0_diff_sp_ang = sl2_t0_sp_ang - sl1_t0_sp_ang
    sl23_t0_diff_sp_ang = sl3_t0_sp_ang - sl2_t0_sp_ang
    sl13_t0_diff_sp_ang = sl3_t0_sp_ang - sl1_t0_sp_ang
    sl12_t0_diff_sp_ang *= 0.78 # convert to ns
    sl23_t0_diff_sp_ang *= 0.78 # convert to ns
    sl13_t0_diff_sp_ang *= 0.78 # convert to ns

    ### plot histograms of time differences
    log(f"sl12_t0_diff_sp_ang: mean={np.mean(sl12_t0_diff_sp_ang)} ns, std={np.std(sl12_t0_diff_sp_ang, ddof=1)} ns")
    log(f"sl23_t0_diff_sp_ang: mean={np.mean(sl23_t0_diff_sp_ang)} ns, std={np.std(sl23_t0_diff_sp_ang, ddof=1)} ns")
    log(f"sl13_t0_diff_sp_ang: mean={np.mean(sl13_t0_diff_sp_ang)} ns, std={np.std(sl13_t0_diff_sp_ang, ddof=1)} ns")
    plot_utils.plot_histogram(sl12_t0_diff_sp, f"{script_prefix}_sl12_t0_diff_sp_ang", args, xlabel="time diff SL2$-$SL1 (sp+ang corr) [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl23_t0_diff_sp, f"{script_prefix}_sl23_t0_diff_sp_ang", args, xlabel="time diff SL3$-$SL2 (sp+ang corr) [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    plot_utils.plot_histogram(sl13_t0_diff_sp, f"{script_prefix}_sl13_t0_diff_sp_ang", args, xlabel="time diff SL3$-$SL1 (sp+ang corr) [ns]",
            full_range=True, log_scale=False, bin_unit="ns")


if __name__ == "__main__":
    main()
    log("###### Done.")

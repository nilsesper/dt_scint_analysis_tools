#################################################################
### analysis of multiple muons arriving close in time
#################################################################

import argparse
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
import matplotlib as mpl

from analysis_tools.utils import data_utils, hist_utils, muon_utils, root_utils, timestamp_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import params

# ---------------------------------------------------------------

MUON_KEYS = ["x0", "y0", "z0", "theta", "phi", "ts"]

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="Track maps, projections, 3d view and timing plots of dt muons.")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--cuts", type=str, default=None, help="cuts applied before plotting, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins of the 1d histograms")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

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

    ### check for muons close in time
    # sort muons in time
    dt_muons = timestamp_utils.sort_by_timestamp(dt_muons, silent=True)
    # check time difference between muons
    ts_tolerance = 50
    ts_diff_muons = np.diff(dt_muons["ts"])
    closeby_muon_tsdiff_indices = np.argwhere(ts_diff_muons < ts_tolerance).flatten()
    # identify indices of closeby muons
    closeby_muon_groups = [] # [(idcs) of closeby muons]
    start = None
    last_tsdiff_idx = None
    for i in range(len(closeby_muon_tsdiff_indices)):
        tsdiff_idx = closeby_muon_tsdiff_indices[i]
        if start == None:
            start = tsdiff_idx
        else:
            if tsdiff_idx - start > 1:
                closeby_muon_groups.append(list(range(start, (last_tsdiff_idx+1)+1)))
                start = tsdiff_idx
        last_tsdiff_idx = tsdiff_idx
    if start != None:
        closeby_muon_groups.append(list(range(start, (tsdiff_idx+1)+1)))
        start = None
    n_closeby_muon_groups = len(closeby_muon_groups)

    log(f"found {n_closeby_muon_groups} groups of muons that are consecutively closer than {ts_tolerance} ts units in time")
    # print more info on all of them
    # for i, idcs in enumerate(closeby_muon_groups):
    #     print(f"  closeby muon group {i}: contains {len(idcs)} consecutive muons")
    #     for idx in idcs:
    #         print(f"    dt_muon_row={idx}, ts={dt_muons['ts'][idx]}, x0={dt_muons['x0'][idx]}, y0={dt_muons['y0'][idx]}, theta={dt_muons['theta'][idx]}, phi={dt_muons['phi'][idx]}")

    ### select groups with exactly 2 muons
    closeby_dimuon_groups = []
    for i in range(n_closeby_muon_groups):
        if len(closeby_muon_groups[i]) == 2:
            closeby_dimuon_groups.append(closeby_muon_groups[i])
    log(f"found {n_closeby_muon_groups} groups of exactly two muons that are consecutively closer than {ts_tolerance} ts units in time")
    n_closeby_dimuon_groups = len(closeby_dimuon_groups)

    ### analyze dimuons
    delta_theta, delta_phi = np.zeros(n_closeby_dimuon_groups), np.zeros(n_closeby_dimuon_groups)
    delta_x0, delta_y0, delta_z0 = np.zeros(n_closeby_dimuon_groups), np.zeros(n_closeby_dimuon_groups), np.zeros(n_closeby_dimuon_groups)
    delta_ts = np.zeros(n_closeby_dimuon_groups)
    for i in range(n_closeby_dimuon_groups):
        idx1, idx2 = closeby_dimuon_groups[i]
        delta_theta[i] = dt_muons["theta"][idx2] - dt_muons["theta"][idx1]
        delta_phi[i] = dt_muons["phi"][idx2] - dt_muons["phi"][idx1]
        delta_x0[i] = dt_muons["x0"][idx2] - dt_muons["x0"][idx1]
        delta_y0[i] = dt_muons["y0"][idx2] - dt_muons["y0"][idx1]
        delta_z0[i] = dt_muons["z0"][idx2] - dt_muons["z0"][idx1]
        delta_ts[i] = dt_muons["ts"][idx2] - dt_muons["ts"][idx1]
    delta_ts *= 0.78 # convert to ns

    # create plots
    plot_utils.plot_histogram(np.rad2deg(delta_theta), "dimuon_delta_theta", args, xlabel="$\\Delta \\theta$ [deg]",
        full_range=True, log_scale=False, bin_unit="deg")
    plot_utils.plot_histogram(np.rad2deg(delta_phi), "dimuon_delta_phi", args, xlabel="$\\Delta \\Phi$ [deg]",
        full_range=True, log_scale=False, bin_unit="deg")
    plot_utils.plot_histogram(delta_x0, "dimuon_delta_x0", args, xlabel="$\\Delta x_0$ [mm]",
        full_range=True, log_scale=False, bin_unit="mm")
    plot_utils.plot_histogram(delta_y0, "dimuon_delta_y0", args, xlabel="$\\Delta y_0$ [mm]",
        full_range=True, log_scale=False, bin_unit="mm")
    plot_utils.plot_histogram(delta_z0, "dimuon_delta_z0", args, xlabel="$\\Delta z_0$ [mm]",
        full_range=True, log_scale=False, bin_unit="mm")
    plot_utils.plot_histogram(delta_ts, "dimuon_delta_ts", args, xlabel="$\\Delta T$ [ns]",
        full_range=True, log_scale=False, bin_unit="ns")
    


if __name__ == "__main__":
    main()
    log("###### Done.")

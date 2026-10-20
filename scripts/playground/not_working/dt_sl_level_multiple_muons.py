#################################################################
### analysis of multi muon events using the superlayer fits
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

from analysis_tools.utils import data_utils, hist_utils, muon_utils, root_utils, timestamp_utils, dt_fit_utils
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.params import params

# ---------------------------------------------------------------

theta_sl = dt_chamber_utils.theta_superlayer()
phi_sl_1, phi_sl_2 = dt_chamber_utils.phi_superlayers()

script_prefix = "dt_wireprop"

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 16})
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits (.root)")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.sl_fits_file)

    ### load sl fits and super fits
    log("loading sl fits...")
    keys = ["t0", "sl", "wi3", "tan_alpha", "x0"]
    sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, branches=keys)
    n_sl_fits = data_utils.length(sl_fits)
    log("loading done")
    if n_sl_fits == 0:
        raise RuntimeError("No sl fits found in the input file.")

    ### divide by superlayer
    sl_fits_per_sl = { sl: data_utils.cut_data(data=sl_fits, conditions=[("sl","==",sl)], silent=True) for sl in range(1,4) }
    n_sl_fits_per_sl = { sl: data_utils.length(sl_fits_per_sl[sl]) for sl in range(1,4) }

    ### measurement duration and muon rate
    duration = plot_utils.TS_UNIT_NS * 1e-9 * float(np.amax(sl_fits["t0"]) - np.amin(sl_fits["t0"]))
    log(f"sl fit count (all sls together): {n_sl_fits:,}")
    for sl in range(1,4):
        log(f"sl fit count (in sl {sl}): {n_sl_fits_per_sl[sl]:,}")
    if duration > 0:
        log(f"sl fit rate (all sls together): {n_sl_fits / duration:.3f} +- {np.sqrt(n_sl_fits) / duration:.3f} Hz")
        for sl in range(1,4):
            log(f"sl fit rate (in sl {sl}): {n_sl_fits_per_sl[sl] / duration:.3f} +- {np.sqrt(n_sl_fits_per_sl[sl]) / duration:.3f} Hz")

    ########################################
    ### SELECTION OF MULTI SL FIT EVENTS IN PHI SUPERLAYERS
    ### first select the sl fits in the top phi superlayer which are not ambiguous (not multiple sl fits close in time at the same position)
    ### then try to group together the sl fits in the top phi superlayer which are close in time (but not the same particle, as we have kept only non-ambiguous fits)
    ### then try to correlate sl fits in top phi superlayer with the bottom phi superlayer
    ########################################
    # here "_1" means the top phi superlayer (sl3)
    # and "_2" means the bottom phi superlayer (sl1)

    ### select only isolated sl fits in all superlayers
    ts_tolerance = 25 / 0.78 # max time difference between sl fits of same superlayer to consider as group
    wi_min_tolerance = 4 # min wi index difference between sl fits of same superlayer to consider as isolated / not the same object
    ###
    t0_differences = {}
    closeby_tsdiff_indices = {}
    closeby_indices = {}
    n_closeby = {}
    isolated_closeby_indices = {}
    n_isolated_closeby = {}
    for sl in range(1,4):
        ### find all sl fits in one sl which are close in time
        sl_fits_per_sl[sl] = data_utils.sort_by_key(data=sl_fits_per_sl[sl], sort_key="t0", silent=True)
        t0_differences[sl] = np.diff(sl_fits_per_sl[sl]["t0"])
        closeby_tsdiff_indices[sl] = np.argwhere(t0_differences[sl] < ts_tolerance).flatten()
        closeby_indices[sl] = [] # [(idcs) of closeby sl fits]
        start = None
        last_tsdiff_idx = None
        for i in range(len(closeby_tsdiff_indices[sl])):
            tsdiff_idx = closeby_tsdiff_indices[sl][i]
            if start == None:
                start = tsdiff_idx
            else:
                if tsdiff_idx - last_tsdiff_idx > 1:
                    closeby_indices[sl].append(list(range(start, (last_tsdiff_idx+1)+1)))
                    start = tsdiff_idx
            last_tsdiff_idx = tsdiff_idx
        if start != None:
            closeby_indices[sl].append(list(range(start, (tsdiff_idx+1)+1)))
            start = None
        n_closeby[sl] = len(closeby_indices[sl])
        ### remove ambiguous sl fits, i.e. those sl fits which are close together geometrically
        isolated_closeby_indices[sl] = []
        for i in range(n_closeby[sl]):
            min_wi_abs_distance = 255
            for row1 in closeby_indices[sl][i]:
                for row2 in closeby_indices[sl][i]:
                        if row1 != row2:
                            _wi_abs_distance = np.abs( np.int32(sl_fits_per_sl[sl]["wi3"][row1]) - np.int32(sl_fits_per_sl[sl]["wi3"][row2]) )
                            if _wi_abs_distance < min_wi_abs_distance:
                                min_wi_abs_distance = _wi_abs_distance
            if min_wi_abs_distance > wi_min_tolerance:
                isolated_closeby_indices[sl].append(list(closeby_indices[sl][i]))
        n_isolated_closeby[sl] = len(isolated_closeby_indices[sl])

        log(f"for superlayer {sl}:")
        log(f"  groups of > 1 sl fits within {ts_tolerance} tdc units:   {n_closeby[sl]:,}")
        log(f"  isolated groups of > 1 sl fits within {ts_tolerance} tdc units and with wi separation > {wi_min_tolerance}:   {n_isolated_closeby[sl]:,}")

    sl_top = 3
    sl_mid = 2
    sl_bottom = 1
    log(f"sl top = {sl_top}, sl_mid = {sl_mid}, sl_bottom = sl_bottom")
    # log(f"ratios between superlayers for isolated groups:")
    # log(f"  sl bottom / sl top: {n_isolated_closeby[sl_bottom] / n_isolated_closeby[sl_top]}")
    # log(f"  sl mid / sl top: {n_isolated_closeby[sl_mid] / n_isolated_closeby[sl_top]}")
    # log(f"  sl bottom / sl mid: {n_isolated_closeby[sl_bottom] / n_isolated_closeby[sl_mid]}")

    ### try to match isolated sl fits in top phi superlayer with sl fits in bottom phi superlayer
    t0_tolerance = params._muon_tgroup_tolerance
    # output format:
    matched_multi_fit_events = [] # [( (rows of sl_fits_per_sl[sl_top]), (rows of sl_fits_per_sl[sl_bottom] or None) ) ]
    ###
    used_bottom = np.zeros(n_sl_fits_per_sl[sl_bottom], dtype=bool)
    first_candidate = 0  # position in sl_bottom of the first fit of sl which is not too early
    for i in range(n_isolated_closeby[sl_top]):
        _pairs = []
        top_rows = []
        bottom_rows = []
        for row in isolated_closeby_indices[sl_top][i]:
            # check only those sl fits which anyway lie close enough in time with their t0 values
            while first_candidate < n_sl_fits_per_sl[sl_bottom] and sl_fits_per_sl[sl_bottom]["t0"][first_candidate] < sl_fits_per_sl[sl_top]["t0"][row] - t0_tolerance:
                first_candidate += 1
            closest, x_at_reco_z0_distance = None, None
            k = first_candidate
            while k < n_sl_fits_per_sl[sl_bottom] and sl_fits_per_sl[sl_bottom]["t0"][k] <= sl_fits_per_sl[sl_top]["t0"][row] + t0_tolerance:
                j = k
                k += 1
                if used_bottom[j]:
                    continue
                # check difference in tan_alpha small
                if np.abs(sl_fits_per_sl[sl_bottom]["tan_alpha"][j] - sl_fits_per_sl[sl_top]["tan_alpha"][row]) > params._muon_slphi_tan_alpha_tolerance:
                    continue
                # distance in x at reco z0 between phi sl fits
                x_top = dt_fit_utils.sl_fit_position_at_reco_z0(sl_fits_per_sl[sl_top], row)
                x_bottom = dt_fit_utils.sl_fit_position_at_reco_z0(sl_fits_per_sl[sl_bottom], j)
                x_at_reco_z0_distance = np.abs(x_top - x_bottom)
                # check distance in x small
                if x_at_reco_z0_distance > params._muon_slphi_xproj_tolerance:
                    continue
                # select "closest" sl fit by x distance (not by t0, in order to not spoil multi-muon events)
                if closest is None or x_at_reco_z0_distance < closest_x_at_reco_z0_distance:
                    closest, closest_x_at_reco_z0_distance = j, x_at_reco_z0_distance
            
            top_rows.append(row)
            if closest is None:
                bottom_rows.append(None)
                continue
            used_bottom[closest] = True
            bottom_rows.append(closest)

        matched_multi_fit_events.append( (top_rows, bottom_rows) )
    n_matched_multi_fit_events = len(matched_multi_fit_events)

    ### try to analyze how many tracks arrive at bottom and which ones are only in top
    n_2_top_and_0_bottom = 0
    n_2_top_and_1_bottom = 0
    n_2_top_and_2_bottom = 0
    ###
    for i in range(n_matched_multi_fit_events):
        top_rows, bottom_rows = matched_multi_fit_events[i]
        n_rows = len(top_rows)
        _n_top = 0
        _n_top_only = 0
        _n_top_and_bottom = 0
        for row in range(n_rows):
            if top_rows[row] is not None:
                _n_top += 1
                if bottom_rows[row] is not None:
                    _n_top_and_bottom += 1
                else:
                    _n_top_only += 1
        if _n_top == 2 and _n_top_and_bottom == 0:
            n_2_top_and_0_bottom += 1
        elif _n_top == 2 and _n_top_and_bottom == 1:
            n_2_top_and_1_bottom += 1
        elif _n_top == 2 and _n_top_and_bottom == 2:
            n_2_top_and_2_bottom += 1
    
    log(f"classification of multi track events in top-only and top-and-bottom sl fits:")
    log(f"  n_2_top_and_0_bottom = {n_2_top_and_0_bottom:,}")
    log(f"  n_2_top_and_1_bottom = {n_2_top_and_1_bottom:,}")
    log(f"  n_2_top_and_2_bottom = {n_2_top_and_2_bottom:,}")


if __name__ == "__main__":
    main()
    log("###### Done.")

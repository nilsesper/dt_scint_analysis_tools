###########################################
### SL FIT: straight track through the 4 hits of a superlayer pattern
###########################################
# Pattern frame: wire of layer 3 at (0, 0), see dt_geometry_utils.
# Every laterality of the pattern shape is fitted; the best one gives the result branches, all of them are also
# stored as "lat<i>_<key>". Patterns whose hit times cannot come from one track get impossible = 1.

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_geometry_utils as geometry, dt_track_fit_utils as track_fit

# -----------------------------------------

### fit one pattern with all its lateralities
# ts, err_ts: hit times of layer 0-3; returns the list of fit results (one per laterality), or None if the hit times
# cannot come from one track
def fit_pattern(ts, err_ts, pat_type, *, fit_vd, verbose=False):
    pat_name = list(params._dt_sl_patterns.keys())[pat_type]
    rel_wis = params._dt_sl_patterns[pat_name]["rel_wis"]

    # cells of the pattern in the pattern frame
    x_cell = np.zeros(4, dtype=np.float64)
    z = np.zeros(4, dtype=np.float64)
    for ly in range(4):
        x_cell[ly] = geometry.pattern_cell(ly, rel_wis[ly])["center"][0]
        z[ly] = geometry.pattern_layer_z(ly)

    # hit times relative to the earliest hit, and the allowed t0 range
    ts_offset = np.amin(ts)
    t0_bounds = track_fit.t0_range(np.amax(ts) - ts_offset, np.amin(ts) - ts_offset, fit_vd)
    if t0_bounds[0] >= t0_bounds[1]:
        return None

    alpha_min, alpha_max = params._dt_pattern_alpha_range[pat_type][0], params._dt_pattern_alpha_range[pat_type][1]
    tan_alpha_bounds = (np.tan(alpha_min), np.tan(alpha_max))
    reference_cell = geometry.pattern_cell(3, 0)

    fits = []
    lateralities = params._dt_sl_patterns[pat_name]["laterality"]
    for lat_id in range(len(lateralities)):
        laterality = np.array(lateralities[lat_id], dtype=np.float64)

        # x0 = track position at the wire height of layer 3: inside the half of the reference cell given by the laterality
        if laterality[3] == -1:
            x0_bounds = (reference_cell["low"][0], reference_cell["center"][0])
        elif laterality[3] == 1:
            x0_bounds = (reference_cell["center"][0], reference_cell["high"][0])
        else:
            x0_bounds = (reference_cell["center"][0], reference_cell["center"][0])

        bounds = track_fit.parameter_bounds(t0_bounds, x0_bounds, tan_alpha_bounds, fit_vd)
        start = [np.mean([bounds[0][0], bounds[1][0]]), np.mean([bounds[0][1], bounds[1][1]]), np.tan(np.mean([alpha_min, alpha_max]))]
        if fit_vd:
            start.append(track_fit.NOMINAL_VD)
        parameters, covariance = track_fit.fit_track(ts - ts_offset, err_ts, x_cell, z, laterality, bounds, np.float64(start), fit_vd)
        result, residuals = track_fit.track_fit_result(parameters, covariance, ts - ts_offset, err_ts, x_cell, z, laterality, ts_offset, ndf=1, fit_vd=fit_vd)
        result["impossible"] = 0
        result["laterality"] = lat_id
        fits.append(result)
        if verbose:
            print(f"  pattern {pat_name}, laterality {lat_id}: chi2/ndf = {result['chi2/ndf']:.3f}, t0 = {result['t0']:.2f}, "
                  f"x0 = {result['x0']:.3f}, tan_alpha = {result['tan_alpha']:.4f}, vd = {result['vd']:.5f}, residuals = {residuals}")
    return fits

### fit every row of an sl patterns table; returns the table with the fit result branches (names + suffix) added
def fit_sl_patterns(patterns, fit_vd=False, suffix="", verbose=False):
    n_patterns = len(patterns["sl"])
    sl_fits = {}
    for key in patterns:
        sl_fits[key] = patterns[key].copy()
    for key in params._sl_fit_keys:
        sl_fits[key + suffix] = np.full(n_patterns, 0, dtype=params._sl_fit_keys[key])
    for key in params._sl_fit_other_keys:
        sl_fits[key + suffix] = np.full(n_patterns, 0, dtype=params._sl_fit_other_keys[key])

    for i in range(n_patterns):
        ts = np.zeros(4, dtype=params._ts_float_type)
        err_ts = np.zeros(4, dtype=params._ts_float_type)
        for ly in range(4):
            ts[ly] = np.float64(patterns[f"ts{ly}"][i])
            err_ts[ly] = np.float64(patterns[f"err_ts{ly}"][i])
        if verbose:
            print(f"sl pattern {i}: ts = {ts}")
        fits = fit_pattern(ts, err_ts, int(patterns["pat_type"][i]), fit_vd=fit_vd, verbose=verbose)
        if fits is None:
            sl_fits["impossible" + suffix][i] = 1
            continue

        # results of the best laterality
        best = fits[track_fit.choose_best_laterality(fits)]
        for key in params._sl_fit_keys:
            sl_fits[key + suffix][i] = best[key]
        # results of all lateralities
        for lat_id in range(len(fits)):
            for key in params._sl_fit_keys:
                if key != "laterality":
                    sl_fits[f"lat{lat_id}_{key}" + suffix][i] = fits[lat_id][key]
    return sl_fits

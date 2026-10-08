###########################################
### SL FIT: straight track through the 4 hits of a superlayer pattern
###########################################
# Pattern frame: wire of layer 3 at (0, 0), see dt_geometry_utils.
# Every laterality of the pattern shape is fitted; the best one gives the result branches, all of them are also
# stored as "lat<i>_<key>". Patterns whose hit times cannot come from one track get impossible = 1.

import functools
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_geometry_utils as geometry, dt_track_fit_utils as track_fit

# -----------------------------------------

LATERALITY_RESULT_KEYS = [k for k in params._sl_fit_keys.keys() if k != "laterality"]

### z of the 4 layers in the pattern frame
def layer_z():
    return np.array([geometry.pattern_layer_z(ly) for ly in range(4)], dtype=np.float64)

### x of the 4 cells of a pattern shape in the pattern frame, and the range of the track angle the shape allows
@functools.cache
def shape_geometry(pat_type):
    pat_name = list(params._dt_sl_patterns.keys())[pat_type]
    x_cell = np.array([geometry.pattern_cell(ly, rel_wi).center[0] for ly, rel_wi in enumerate(params._dt_sl_patterns[pat_name]["rel_wis"])],
                      dtype=np.float64)
    alpha_range = (params._dt_pattern_alpha_range[pat_type][0], params._dt_pattern_alpha_range[pat_type][1])
    return pat_name, x_cell, alpha_range

### x0 = track position at the wire height of layer 3: inside the half of the reference cell given by the laterality
def x0_range(laterality_layer3):
    cell = geometry.pattern_cell(3, 0)
    return (cell.low[0] if laterality_layer3 == -1 else cell.center[0], cell.high[0] if laterality_layer3 == 1 else cell.center[0])

### all lateralities of one pattern; returns the list of fit results or None if the hit times are impossible
def fit_pattern(ts, err_ts, pat_type, *, fit_vd, verbose=False):
    pat_name, x_cell, alpha_range = shape_geometry(pat_type)
    z = layer_z()
    ts_offset = np.amin(ts)
    t0_bounds = track_fit.t0_range(np.amax(ts) - ts_offset, np.amin(ts) - ts_offset, fit_vd)
    if t0_bounds[0] >= t0_bounds[1]:
        return None
    fits = []
    for lat_id, lat in enumerate(params._dt_sl_patterns[pat_name]["laterality"]):
        laterality = np.array(lat, dtype=np.float64)
        bounds = track_fit.parameter_bounds(t0_bounds, x0_range(laterality[3]), np.tan(alpha_range), fit_vd)
        start = [np.mean([bounds[0][0], bounds[1][0]]), np.mean([bounds[0][1], bounds[1][1]]), np.tan(np.mean(alpha_range))]
        if fit_vd:
            start.append(track_fit.NOMINAL_VD)
        parameters, covariance = track_fit.fit_track(ts - ts_offset, err_ts, x_cell, z, laterality, bounds, np.float64(start), fit_vd)
        result, residuals = track_fit.track_fit_result(parameters, covariance, ts - ts_offset, err_ts, x_cell, z, laterality, ts_offset, ndf=1, fit_vd=fit_vd)
        fits.append(result | {"impossible": 0, "laterality": lat_id})
        if verbose:
            print(f"  pattern {pat_name}, laterality {lat_id}: chi2/ndf = {result['chi2/ndf']:.3f}, t0 = {result['t0']:.2f}, "
                  f"x0 = {result['x0']:.3f}, tan_alpha = {result['tan_alpha']:.4f}, vd = {result['vd']:.5f}, residuals = {residuals}")
    return fits

### fit every row of an sl patterns table; returns the table with the fit result branches (names + suffix) added
def fit_sl_patterns(patterns, *, fit_vd=False, suffix="", verbose=False):
    n_patterns = len(patterns["sl"])
    sl_fits = {k: v.copy() for k, v in patterns.items()}
    sl_fits |= {k + suffix: np.full(n_patterns, 0, dtype=v) for k, v in (params._sl_fit_keys | params._sl_fit_other_keys).items()}
    for i in range(n_patterns):
        ts = np.array([np.float64(patterns[f"ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type)
        err_ts = np.array([np.float64(patterns[f"err_ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type)
        if verbose:
            print(f"sl pattern {i}: ts = {ts}")
        fits = fit_pattern(ts, err_ts, int(patterns["pat_type"][i]), fit_vd=fit_vd, verbose=verbose)
        if fits is None:
            sl_fits["impossible" + suffix][i] = 1
            continue
        best = fits[track_fit.choose_best_laterality(fits)]
        for k in params._sl_fit_keys.keys():
            sl_fits[k + suffix][i] = best[k]
        for lat_id, fit in enumerate(fits):
            for k in LATERALITY_RESULT_KEYS:
                sl_fits[f"lat{lat_id}_{k}" + suffix][i] = fit[k]
    return sl_fits

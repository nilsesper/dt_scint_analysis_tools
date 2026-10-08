###########################################
### SUPER FIT: one straight track through the 8 hits of the two phi superlayers
###########################################
# 1. super patterns: the sl fits of the two phi superlayers are paired (close in t0, similar angle, same track position
#    at z = params._muon_reco_z0); a super pattern holds the 8 hits of both sl patterns.
# 2. super fit: one track through the 8 hits, in the super pattern frame (dt_geometry_utils: topmost wire of the
#    chamber at (0, 0)), shifted for each fit so that its own topmost wire is at (0, 0) ("ref_x", "ref_z" are stored).
#    All combinations of the lateralities of the two sl patterns are fitted, the best one is kept.

import functools
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry, dt_matching_utils, dt_track_fit_utils as track_fit

# -----------------------------------------

SIM_KEYS = ["sim_id", "sim_ts", "sim_phi", "sim_theta", "sim_x0", "sim_y0", "sim_z0"]

RESULT_DTYPES = {
    "impossible": np.int64, "lat_id1": np.int64, "lat_id2": np.int64,
    "t0": np.float64, "x0": np.float64, "tan_alpha": np.float64, "vd": np.float64, "chi2/ndf": np.float64,
    **{f"dt{ly}": np.float64 for ly in range(8)},
    "err_t0": np.float64, "err_x0": np.float64, "err_tan_alpha": np.float64, "err_vd": np.float64,
    "corr_t0_x0": np.float64, "corr_t0_tan_alpha": np.float64, "corr_t0_vd": np.float64,
    "corr_x0_tan_alpha": np.float64, "corr_x0_vd": np.float64, "corr_tan_alpha_vd": np.float64,
    "ref_x": np.float64, "ref_z": np.float64,
}

# -----------------------------------------
# super patterns
# -----------------------------------------

### sl fits which may be combined: possible fit, chi2/ndf below max_chi2ndf, |track angle| below max_alpha
def combinable_sl_fits(sl_fits, max_chi2ndf, max_alpha):
    max_tan_alpha = np.tan(max_alpha)
    return ((sl_fits["impossible"] == 0) & (sl_fits["chi2/ndf"] < max_chi2ndf)
            & (sl_fits["tan_alpha"] > -max_tan_alpha) & (sl_fits["tan_alpha"] < max_tan_alpha))

### pairs of sl fits (rows of fits_1, rows of fits_2) of the two phi superlayers which belong to the same track
def pair_phi_sl_fits(fits_1, fits_2, sl_1, sl_2):
    def similar_angle(i, j):
        return not (np.abs(fits_2["tan_alpha"][j] - fits_1["tan_alpha"][i]) > params._muon_slphi_tan_alpha_tolerance)

    def same_position(i, j):
        if not (geometry.is_cell(sl_1, 3, fits_1["wi3"][i]) and geometry.is_cell(sl_2, 3, fits_2["wi3"][j])):
            return False
        x_1 = geometry.sl_track_position(sl_1, fits_1["wi3"][i], fits_1["x0"][i], fits_1["tan_alpha"][i], params._muon_reco_z0)
        x_2 = geometry.sl_track_position(sl_2, fits_2["wi3"][j], fits_2["x0"][j], fits_2["tan_alpha"][j], params._muon_reco_z0)
        return not (np.abs(x_1 - x_2) > params._muon_slphi_xproj_tolerance)

    matches = dt_matching_utils.match_nearest_in_time(fits_1["t0"], fits_2["t0"], params._muon_tgroup_tolerance, sort_kind=None,
                                                      is_candidate=similar_angle, accept_pair=same_position)
    return [(i, j) for i, j, _ in matches]

### super patterns of a table of sl fits (rows of one chunk)
# branches: hit times ts0..ts7 / err_ts0..err_ts7 (layers 0-3 of the first phi sl, then of the second),
# per phi sl n: wires "wi<ly>_sl<n>", pattern type "pat_type_sl<n>", fit results "<key>_sl<n>", "idx_sl<n>" (index among
# the combinable fits of sl n) and "row_sl<n>" (row in sl_fits); simulation truth of the first sl and "sim_id_mismatch"
def build_phi_super_patterns(sl_fits, *, max_chi2ndf=10, max_alpha=np.deg2rad(60)):
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    combinable = combinable_sl_fits(sl_fits, max_chi2ndf, max_alpha)
    rows_1, rows_2 = np.flatnonzero(combinable & (sl_fits["sl"] == sl_1)), np.flatnonzero(combinable & (sl_fits["sl"] == sl_2))
    fits_1, fits_2 = ({k: v[rows] for k, v in sl_fits.items()} for rows in (rows_1, rows_2))
    pairs = pair_phi_sl_fits(fits_1, fits_2, sl_1, sl_2)
    idx_1, idx_2 = np.array([i for i, _ in pairs], dtype=np.int64), np.array([j for _, j in pairs], dtype=np.int64)
    n = len(pairs)

    patterns = {f"sl{sl_1}": np.full(n, sl_1, dtype=np.int64), f"sl{sl_2}": np.full(n, sl_2, dtype=np.int64),
                "sim_id_mismatch": np.full(n, 0, dtype=np.int64)}
    for sl, fits, idx, rows, first_layer in ((sl_1, fits_1, idx_1, rows_1, 0), (sl_2, fits_2, idx_2, rows_2, 4)):
        patterns[f"pat_type_sl{sl}"] = fits["pat_type"][idx].astype(np.int64)
        patterns[f"idx_sl{sl}"] = idx
        patterns[f"row_sl{sl}"] = rows[idx].astype(np.int64)
        for ly in range(4):
            patterns[f"ts{first_layer + ly}"] = fits[f"ts{ly}"][idx].astype(params._ts_float_type)
            patterns[f"err_ts{first_layer + ly}"] = fits[f"err_ts{ly}"][idx].astype(params._ts_float_type)
            patterns[f"wi{ly}_sl{sl}"] = fits[f"wi{ly}"][idx].astype(np.int64)
        for k in params._sl_fit_keys.keys():
            patterns[f"{k}_sl{sl}"] = fits[k][idx].astype(np.float64)
    for k in SIM_KEYS:
        if k in fits_1:
            patterns[k] = fits_1[k][idx_1].astype(np.float64)
    if "sim_id" in fits_1:
        patterns["sim_id_mismatch"] = (fits_1["sim_id"][idx_1] != fits_2["sim_id"][idx_2]).astype(np.int64)
    return patterns

# -----------------------------------------
# super fit
# -----------------------------------------

### allowed track angle of a super pattern: the union of the ranges of the two pattern shapes (None if empty)
@functools.cache
def alpha_range(pat_type_1, pat_type_2):
    alpha_min = min(params._dt_pattern_alpha_range[pat_type_1][0], params._dt_pattern_alpha_range[pat_type_2][0])
    alpha_max = max(params._dt_pattern_alpha_range[pat_type_1][1], params._dt_pattern_alpha_range[pat_type_2][1])
    return (alpha_min, alpha_max) if alpha_min < alpha_max else None

### one super pattern (row i): all laterality combinations; returns the list of fit results, or None if not fittable
def fit_super_pattern(super_patterns, i, *, fit_vd, verbose=False):
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    pat_name = list(params._dt_sl_patterns.keys())
    pat_type_1, pat_type_2 = int(super_patterns[f"pat_type_sl{sl_1}"][i]), int(super_patterns[f"pat_type_sl{sl_2}"][i])
    wires_1 = [int(super_patterns[f"wi{ly}_sl{sl_1}"][i]) for ly in range(4)]
    wires_2 = [int(super_patterns[f"wi{ly}_sl{sl_2}"][i]) for ly in range(4)]

    cells = [geometry.super_frame_position(sl_1, ly, wires_1[ly]) for ly in range(4)] + [geometry.super_frame_position(sl_2, ly, wires_2[ly]) for ly in range(4)]
    x_cell, z = np.array([c[0] for c in cells], dtype=np.float64), np.array([c[1] for c in cells], dtype=np.float64)
    top = np.argmax(z)
    ref_x, ref_z = x_cell[top], z[top]
    x_cell, z = x_cell - ref_x, z - ref_z

    ts = np.array([np.float64(super_patterns[f"ts{ly}"][i]) for ly in range(8)], dtype=params._ts_float_type)
    err_ts = np.array([np.float64(super_patterns[f"err_ts{ly}"][i]) for ly in range(8)], dtype=params._ts_float_type)
    ts_offset = np.amin(ts)
    t0_bounds = track_fit.t0_range(np.amax(ts) - ts_offset, np.amin(ts) - ts_offset, fit_vd)
    alphas = alpha_range(pat_type_1, pat_type_2)
    if t0_bounds[0] >= t0_bounds[1] or alphas is None:
        return None
    tan_alpha_bounds = (np.tan(alphas[0]), np.tan(alphas[1]))

    # start values from the two sl fits
    t0_start = np.clip(np.mean([super_patterns[f"t0_sl{sl_1}"][i], super_patterns[f"t0_sl{sl_2}"][i]]) - ts_offset, *t0_bounds)
    tan_alpha_start = np.clip(np.mean([super_patterns[f"tan_alpha_sl{sl_1}"][i], super_patterns[f"tan_alpha_sl{sl_2}"][i]]), *tan_alpha_bounds)
    top_sl_is_1 = sl_1 == geometry.TOP_PHI_SUPERLAYER
    x0_start = super_patterns[f"x0_sl{sl_1 if top_sl_is_1 else sl_2}"][i]

    fits = []
    for lat_id1, lat1 in enumerate(params._dt_sl_patterns[pat_name[pat_type_1]]["laterality"]):
        for lat_id2, lat2 in enumerate(params._dt_sl_patterns[pat_name[pat_type_2]]["laterality"]):
            laterality = np.array(list(lat1) + list(lat2), dtype=np.float64)
            x0_min, x0_max = geometry.super_frame_x0_range(*((wires_1[3], lat1[3]) if top_sl_is_1 else (wires_2[3], lat2[3])))
            x0_bounds = (x0_min - ref_x, x0_max - ref_x)
            bounds = track_fit.parameter_bounds(t0_bounds, x0_bounds, tan_alpha_bounds, fit_vd)
            start = [t0_start, np.clip(x0_start - ref_x, *x0_bounds), tan_alpha_start] + ([track_fit.NOMINAL_VD] if fit_vd else [])
            try:
                parameters, covariance = track_fit.fit_track(ts - ts_offset, err_ts, x_cell, z, laterality, bounds, np.float64(start), fit_vd)
            except Exception as e:
                if verbose:
                    print(f"    fit failed for lat1={lat_id1}, lat2={lat_id2}: {e}")
                continue
            result, residuals = track_fit.track_fit_result(parameters, covariance, ts - ts_offset, err_ts, x_cell, z, laterality, ts_offset,
                                                           ndf=8 - (4 if fit_vd else 3), fit_vd=fit_vd)
            fits.append(result | {"impossible": 0, "lat_id1": lat_id1, "lat_id2": lat_id2, "ref_x": ref_x, "ref_z": ref_z, "ts_residual": residuals})
            if verbose:
                print(f"    lat1={lat_id1}, lat2={lat_id2}: chi2/ndf={result['chi2/ndf']:.3f}, t0={result['t0']:.2f}, x0={result['x0']:.2f}, "
                      f"tan_alpha={result['tan_alpha']:.4f}, vd={result['vd']:.5f}")
    return fits if len(fits) > 0 else None

### fit every super pattern; returns the super patterns with the result branches (names + suffix) added
def fit_super_sl_patterns(super_patterns, *, fit_vd=True, suffix="", verbose=False):
    n_patterns = len(super_patterns["ts0"])
    fits = {k: v.copy() for k, v in super_patterns.items()}
    fits |= {k + suffix: np.full(n_patterns, 0, dtype=dtype) for k, dtype in RESULT_DTYPES.items()}
    fits["ts_residual" + suffix] = np.full((n_patterns, 8), 0, dtype=np.float64)
    for i in range(n_patterns):
        if verbose:
            print(f"\n super pattern {i}:")
        lateralities = fit_super_pattern(super_patterns, i, fit_vd=fit_vd, verbose=verbose)
        if lateralities is None:
            fits["impossible" + suffix][i] = 1
            continue
        best = lateralities[track_fit.choose_best_laterality(lateralities)]
        for k in list(RESULT_DTYPES.keys()) + ["ts_residual"]:
            fits[k + suffix][i] = best[k]
    return fits

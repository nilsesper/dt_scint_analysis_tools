###########################################
### SUPER FIT: one straight track through the 8 hits of the two phi superlayers
###########################################
# 1. super patterns: the sl fits of the two phi superlayers are paired (close in t0, similar angle, same track position
#    at z = params._muon_reco_z0); a super pattern holds the 8 hits of both sl patterns.
# 2. super fit: one track through the 8 hits, in the super pattern frame (dt_geometry_utils: topmost wire of the
#    chamber at (0, 0)), shifted for each fit so that its own topmost wire is at (0, 0) ("ref_x", "ref_z" are stored).
#    All combinations of the lateralities of the two sl patterns are fitted, the best one is kept.

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry, dt_track_fit_utils as track_fit

# -----------------------------------------

SIM_KEYS = ["sim_id", "sim_ts", "sim_phi", "sim_theta", "sim_x0", "sim_y0", "sim_z0"]

### result branches of a super fit and their data types
RESULT_DTYPES = {"impossible": np.int64, "lat_id1": np.int64, "lat_id2": np.int64,
                 "t0": np.float64, "x0": np.float64, "tan_alpha": np.float64, "vd": np.float64, "chi2/ndf": np.float64}
for ly in range(8):
    RESULT_DTYPES[f"dt{ly}"] = np.float64
for key in ["err_t0", "err_x0", "err_tan_alpha", "err_vd", "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd",
            "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd", "ref_x", "ref_z"]:
    RESULT_DTYPES[key] = np.float64

# -----------------------------------------
# super patterns
# -----------------------------------------

### rows of the sl fits of superlayer sl which may be combined: possible fit, chi2/ndf below max_chi2ndf,
# |track angle| below max_alpha
def combinable_rows(sl_fits, sl, max_chi2ndf, max_alpha):
    max_tan_alpha = np.tan(max_alpha)
    rows = []
    for i in range(len(sl_fits["sl"])):
        if sl_fits["sl"][i] != sl:
            continue
        if sl_fits["impossible"][i] != 0:
            continue
        if not sl_fits["chi2/ndf"][i] < max_chi2ndf:
            continue
        if not (sl_fits["tan_alpha"][i] > -max_tan_alpha and sl_fits["tan_alpha"][i] < max_tan_alpha):
            continue
        rows.append(i)
    return np.array(rows, dtype=np.int64)

### True if the two sl fits (row i of fits_1, row j of fits_2) give the same track position at z = params._muon_reco_z0
def same_track_position(fits_1, i, sl_1, fits_2, j, sl_2):
    if not geometry.is_cell(sl_1, 3, fits_1["wi3"][i]) or not geometry.is_cell(sl_2, 3, fits_2["wi3"][j]):
        return False
    x_1 = geometry.sl_track_position(sl_1, fits_1["wi3"][i], fits_1["x0"][i], fits_1["tan_alpha"][i], params._muon_reco_z0)
    x_2 = geometry.sl_track_position(sl_2, fits_2["wi3"][j], fits_2["x0"][j], fits_2["tan_alpha"][j], params._muon_reco_z0)
    return not (np.abs(x_1 - x_2) > params._muon_slphi_xproj_tolerance)

### pairs (i, j) of sl fits of the two phi superlayers (row i of fits_1, row j of fits_2) which belong to the same track
# The fits of sl 1 are taken in the order of their t0. Each is paired with the closest unused fit of sl 2 in time
# (|t0 difference| <= params._muon_tgroup_tolerance) with a similar angle. The pair is kept if both give the same track
# position, otherwise the fit of sl 1 stays unpaired.
def pair_phi_sl_fits(fits_1, fits_2, sl_1, sl_2):
    t0_1, t0_2 = fits_1["t0"], fits_2["t0"]
    n_2 = len(t0_2)
    tolerance = params._muon_tgroup_tolerance
    order_1 = np.argsort(t0_1)
    order_2 = np.argsort(t0_2)
    used_2 = np.zeros(n_2, dtype=bool)
    pairs = []
    first_candidate = 0  # position in order_2 of the first fit of sl 2 which is not too early
    for i in order_1:
        while first_candidate < n_2 and t0_2[order_2[first_candidate]] < t0_1[i] - tolerance:
            first_candidate += 1
        closest, closest_distance = None, None
        k = first_candidate
        while k < n_2 and t0_2[order_2[k]] <= t0_1[i] + tolerance:
            j = order_2[k]
            k += 1
            if used_2[j]:
                continue
            if np.abs(fits_2["tan_alpha"][j] - fits_1["tan_alpha"][i]) > params._muon_slphi_tan_alpha_tolerance:
                continue
            distance = np.abs(t0_2[j] - t0_1[i])
            if closest is None or distance < closest_distance:
                closest, closest_distance = j, distance
        if closest is None:
            continue
        if not same_track_position(fits_1, i, sl_1, fits_2, closest, sl_2):
            continue
        used_2[closest] = True
        pairs.append((i, closest))
    return pairs

### super patterns of a table of sl fits (rows of one chunk)
# branches: hit times ts0..ts7 / err_ts0..err_ts7 (layers 0-3 of the first phi sl, then of the second),
# per phi sl n: wires "wi<ly>_sl<n>", pattern type "pat_type_sl<n>", fit results "<key>_sl<n>", "idx_sl<n>" (index among
# the combinable fits of sl n) and "row_sl<n>" (row in sl_fits); simulation truth of the first sl and "sim_id_mismatch"
def build_phi_super_patterns(sl_fits, *, max_chi2ndf=10, max_alpha=np.deg2rad(60)):
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    rows_1 = combinable_rows(sl_fits, sl_1, max_chi2ndf, max_alpha)
    rows_2 = combinable_rows(sl_fits, sl_2, max_chi2ndf, max_alpha)
    fits_1, fits_2 = {}, {}
    for key in sl_fits:
        fits_1[key] = sl_fits[key][rows_1]
        fits_2[key] = sl_fits[key][rows_2]
    pairs = pair_phi_sl_fits(fits_1, fits_2, sl_1, sl_2)
    n = len(pairs)

    patterns = {f"sl{sl_1}": np.full(n, sl_1, dtype=np.int64), f"sl{sl_2}": np.full(n, sl_2, dtype=np.int64),
                "sim_id_mismatch": np.full(n, 0, dtype=np.int64)}
    for sl, fits, rows, first_layer, pair_position in [(sl_1, fits_1, rows_1, 0, 0), (sl_2, fits_2, rows_2, 4, 1)]:
        patterns[f"pat_type_sl{sl}"] = np.zeros(n, dtype=np.int64)
        patterns[f"idx_sl{sl}"] = np.zeros(n, dtype=np.int64)
        patterns[f"row_sl{sl}"] = np.zeros(n, dtype=np.int64)
        for ly in range(4):
            patterns[f"ts{first_layer + ly}"] = np.zeros(n, dtype=params._ts_float_type)
            patterns[f"err_ts{first_layer + ly}"] = np.zeros(n, dtype=params._ts_float_type)
            patterns[f"wi{ly}_sl{sl}"] = np.zeros(n, dtype=np.int64)
        for key in params._sl_fit_keys:
            patterns[f"{key}_sl{sl}"] = np.zeros(n, dtype=np.float64)

        for p in range(n):
            idx = pairs[p][pair_position]
            patterns[f"pat_type_sl{sl}"][p] = fits["pat_type"][idx]
            patterns[f"idx_sl{sl}"][p] = idx
            patterns[f"row_sl{sl}"][p] = rows[idx]
            for ly in range(4):
                patterns[f"ts{first_layer + ly}"][p] = fits[f"ts{ly}"][idx]
                patterns[f"err_ts{first_layer + ly}"][p] = fits[f"err_ts{ly}"][idx]
                patterns[f"wi{ly}_sl{sl}"][p] = fits[f"wi{ly}"][idx]
            for key in params._sl_fit_keys:
                patterns[f"{key}_sl{sl}"][p] = fits[key][idx]

    # simulation truth
    for key in SIM_KEYS:
        if key in fits_1:
            patterns[key] = np.zeros(n, dtype=np.float64)
            for p in range(n):
                patterns[key][p] = fits_1[key][pairs[p][0]]
    if "sim_id" in fits_1:
        for p in range(n):
            if fits_1["sim_id"][pairs[p][0]] != fits_2["sim_id"][pairs[p][1]]:
                patterns["sim_id_mismatch"][p] = 1
    return patterns

# -----------------------------------------
# super fit
# -----------------------------------------

### one super pattern (row i): all laterality combinations; returns the list of fit results, or None if not fittable
def fit_super_pattern(super_patterns, i, *, fit_vd, verbose=False):
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    pattern_names = list(params._dt_sl_patterns.keys())
    pat_type_1 = int(super_patterns[f"pat_type_sl{sl_1}"][i])
    pat_type_2 = int(super_patterns[f"pat_type_sl{sl_2}"][i])
    wires_1, wires_2 = [], []
    for ly in range(4):
        wires_1.append(int(super_patterns[f"wi{ly}_sl{sl_1}"][i]))
        wires_2.append(int(super_patterns[f"wi{ly}_sl{sl_2}"][i]))

    # wire positions of the 8 cells in the super pattern frame, shifted so that the topmost wire is at (0, 0)
    x_cell = np.zeros(8, dtype=np.float64)
    z = np.zeros(8, dtype=np.float64)
    for ly in range(4):
        x_cell[ly], z[ly] = geometry.super_frame_position(sl_1, ly, wires_1[ly])
        x_cell[4 + ly], z[4 + ly] = geometry.super_frame_position(sl_2, ly, wires_2[ly])
    top = np.argmax(z)
    ref_x, ref_z = x_cell[top], z[top]
    x_cell = x_cell - ref_x
    z = z - ref_z

    # hit times relative to the earliest hit, and the allowed parameter ranges
    ts = np.zeros(8, dtype=params._ts_float_type)
    err_ts = np.zeros(8, dtype=params._ts_float_type)
    for ly in range(8):
        ts[ly] = np.float64(super_patterns[f"ts{ly}"][i])
        err_ts[ly] = np.float64(super_patterns[f"err_ts{ly}"][i])
    ts_offset = np.amin(ts)
    t0_bounds = track_fit.t0_range(np.amax(ts) - ts_offset, np.amin(ts) - ts_offset, fit_vd)
    if t0_bounds[0] >= t0_bounds[1]:
        return None
    # allowed track angle: the union of the ranges of the two pattern shapes
    alpha_min = min(params._dt_pattern_alpha_range[pat_type_1][0], params._dt_pattern_alpha_range[pat_type_2][0])
    alpha_max = max(params._dt_pattern_alpha_range[pat_type_1][1], params._dt_pattern_alpha_range[pat_type_2][1])
    if not alpha_min < alpha_max:
        return None
    tan_alpha_bounds = (np.tan(alpha_min), np.tan(alpha_max))

    # start values from the two sl fits
    t0_start = np.mean([super_patterns[f"t0_sl{sl_1}"][i], super_patterns[f"t0_sl{sl_2}"][i]]) - ts_offset
    t0_start = np.clip(t0_start, t0_bounds[0], t0_bounds[1])
    tan_alpha_start = np.mean([super_patterns[f"tan_alpha_sl{sl_1}"][i], super_patterns[f"tan_alpha_sl{sl_2}"][i]])
    tan_alpha_start = np.clip(tan_alpha_start, tan_alpha_bounds[0], tan_alpha_bounds[1])
    top_sl_is_1 = (sl_1 == geometry.TOP_PHI_SUPERLAYER)
    if top_sl_is_1:
        x0_start = super_patterns[f"x0_sl{sl_1}"][i]
    else:
        x0_start = super_patterns[f"x0_sl{sl_2}"][i]

    fits = []
    lateralities_1 = params._dt_sl_patterns[pattern_names[pat_type_1]]["laterality"]
    lateralities_2 = params._dt_sl_patterns[pattern_names[pat_type_2]]["laterality"]
    for lat_id1 in range(len(lateralities_1)):
        for lat_id2 in range(len(lateralities_2)):
            lat1, lat2 = lateralities_1[lat_id1], lateralities_2[lat_id2]
            laterality = np.array(list(lat1) + list(lat2), dtype=np.float64)

            # x0 = track position at the height of the topmost wire: inside the half of the topmost cell given by its laterality
            if top_sl_is_1:
                x0_min, x0_max = geometry.super_frame_x0_range(wires_1[3], lat1[3])
            else:
                x0_min, x0_max = geometry.super_frame_x0_range(wires_2[3], lat2[3])
            x0_bounds = (x0_min - ref_x, x0_max - ref_x)

            bounds = track_fit.parameter_bounds(t0_bounds, x0_bounds, tan_alpha_bounds, fit_vd)
            start = [t0_start, np.clip(x0_start - ref_x, x0_bounds[0], x0_bounds[1]), tan_alpha_start]
            if fit_vd:
                start.append(track_fit.NOMINAL_VD)
            try:
                parameters, covariance = track_fit.fit_track(ts - ts_offset, err_ts, x_cell, z, laterality, bounds, np.float64(start), fit_vd)
            except Exception as e:
                if verbose:
                    print(f"    fit failed for lat1={lat_id1}, lat2={lat_id2}: {e}")
                continue
            if fit_vd:
                ndf = 8 - 4
            else:
                ndf = 8 - 3
            result, residuals = track_fit.track_fit_result(parameters, covariance, ts - ts_offset, err_ts, x_cell, z, laterality, ts_offset,
                                                           ndf=ndf, fit_vd=fit_vd)
            result["impossible"] = 0
            result["lat_id1"] = lat_id1
            result["lat_id2"] = lat_id2
            result["ref_x"] = ref_x
            result["ref_z"] = ref_z
            result["ts_residual"] = residuals
            fits.append(result)
            if verbose:
                print(f"    lat1={lat_id1}, lat2={lat_id2}: chi2/ndf={result['chi2/ndf']:.3f}, t0={result['t0']:.2f}, x0={result['x0']:.2f}, "
                      f"tan_alpha={result['tan_alpha']:.4f}, vd={result['vd']:.5f}")
    if len(fits) == 0:
        return None
    return fits

### fit every super pattern; returns the super patterns with the result branches (names + suffix) added
def fit_super_sl_patterns(super_patterns, fit_vd=True, suffix="", verbose=False):
    n_patterns = len(super_patterns["ts0"])
    super_fits = {}
    for key in super_patterns:
        super_fits[key] = super_patterns[key].copy()
    for key in RESULT_DTYPES:
        super_fits[key + suffix] = np.full(n_patterns, 0, dtype=RESULT_DTYPES[key])
    super_fits["ts_residual" + suffix] = np.full((n_patterns, 8), 0, dtype=np.float64)

    for i in range(n_patterns):
        if verbose:
            print(f"\n super pattern {i}:")
        fits = fit_super_pattern(super_patterns, i, fit_vd=fit_vd, verbose=verbose)
        if fits is None:
            super_fits["impossible" + suffix][i] = 1
            continue
        best = fits[track_fit.choose_best_laterality(fits)]
        for key in RESULT_DTYPES:
            super_fits[key + suffix][i] = best[key]
        super_fits["ts_residual" + suffix][i] = best["ts_residual"]
    return super_fits

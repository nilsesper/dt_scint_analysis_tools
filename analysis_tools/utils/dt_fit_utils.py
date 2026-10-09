###########################################
### TRACK FITS: sl fits and super fits
###########################################
# sl fit:    one straight track through the 4 hits of an sl pattern
# super fit: one straight track through the 8 hits of two sl patterns, one in each phi superlayer ("super pattern")
#
# Both fits work in the track frame of dt_chamber_utils (origin = a reference wire, h along the measured axis):
#     track:     h(z) = x0 + z * tan_alpha
#     hit time:  ts = t0 + laterality * (x0 + z_wire * tan_alpha - h_wire) / vd          (hit_time)
# Fit parameters: t0, x0, tan_alpha, and vd if the drift velocity is fitted as well (fit_vd).
# The laterality of the hits is not known: every laterality combination allowed by the pattern shape
# (params._dt_sl_patterns) is fitted and the one with the lowest chi2/ndf is kept (choose_best_laterality).
#
# How the fit is done:
#   free vd:  scipy.optimize.curve_fit
#   fixed vd: the hit time is a linear function of (t0, x0, tan_alpha), so the best parameters follow directly from
#             a linear least squares problem (linear_fit). This gives the same result as curve_fit, about 15 times faster.

import numpy as np
from scipy.optimize import curve_fit, lsq_linear

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils

# -----------------------------------------

NOMINAL_VD = derived_params._drift_velocity_mm_per_timestamp
SUPER_FIT_SUFFIX = "_super_fits"  # the result branches of the super fits are called "t0_super_fits", "chi2/ndf_super_fits", ...
SIM_KEYS = ["sim_id", "sim_ts", "sim_phi", "sim_theta", "sim_x0", "sim_y0", "sim_z0"]

# -----------------------------------------
# fit model
# -----------------------------------------

### time of the hit in the cell with the wire at (h_wire, z_wire)
def hit_time(h_wire, t0, x0, tan_alpha, z_wire, laterality, vd):
    return (x0 + z_wire * tan_alpha - h_wire) * laterality / vd + t0

### uncertainty of hit_time from the uncertainties and correlations (covariances) of the fit parameters
def err_hit_time(h_wire, t0, x0, tan_alpha, z_wire, laterality, vd, *, err_t0, err_x0, err_tan_alpha, err_vd, corr_t0_x0, corr_t0_tan_alpha,
                 corr_x0_tan_alpha, corr_t0_vd, corr_x0_vd, corr_tan_alpha_vd):
    # derivatives of hit_time by the parameters
    d_t0 = 1
    d_x0 = laterality / vd
    d_tan_alpha = z_wire * laterality / vd
    d_vd = -(x0 + z_wire * tan_alpha - h_wire) * laterality / vd**2
    return np.sqrt(
          d_t0**2 * err_t0**2 + d_x0**2 * err_x0**2 + d_tan_alpha**2 * err_tan_alpha**2 + d_vd**2 * err_vd**2
        + 2 * d_t0 * d_x0 * corr_t0_x0 + 2 * d_t0 * d_tan_alpha * corr_t0_tan_alpha + 2 * d_t0 * d_vd * corr_t0_vd
        + 2 * d_x0 * d_tan_alpha * corr_x0_tan_alpha + 2 * d_x0 * d_vd * corr_x0_vd + 2 * d_tan_alpha * d_vd * corr_tan_alpha_vd
    )

### hit_time for curve_fit: cells = [h_wire, z_wire, laterality] (one array per row, one entry per hit)
def hit_time_free_vd(cells, t0, x0, tan_alpha, vd):
    return hit_time(cells[0], t0, x0, tan_alpha, cells[1], cells[2], vd)

# -----------------------------------------
# fit of one track
# -----------------------------------------

### weighted linear least squares with the parameter bounds, for fixed vd
# ts = t0 * 1 + x0 * (laterality / vd) + tan_alpha * (z_wire * laterality / vd) - laterality * h_wire / vd
# Each hit is one row of the equation system "matrix @ (t0, x0, tan_alpha) = right side", divided by its uncertainty.
# Returns the parameters and their covariance matrix (as curve_fit with absolute_sigma=True).
def linear_fit(ts, err_ts, h_wire, z_wire, laterality, bounds):
    n_hits = len(ts)
    matrix = np.zeros((n_hits, 3), dtype=np.float64)
    right_side = np.zeros(n_hits, dtype=np.float64)
    for i in range(n_hits):
        weight = 1.0 / err_ts[i]
        matrix[i][0] = 1 * weight
        matrix[i][1] = (laterality[i] / NOMINAL_VD) * weight
        matrix[i][2] = (z_wire[i] * laterality[i] / NOMINAL_VD) * weight
        right_side[i] = (ts[i] - (-laterality[i] * h_wire[i] / NOMINAL_VD)) * weight
    result = lsq_linear(matrix, right_side, bounds=(bounds[0], bounds[1]), method="bvls")
    covariance = np.linalg.pinv(matrix.T @ matrix)
    return result.x, covariance

### fit one track to hits with known lateralities
# ts: hit times relative to the earliest hit; bounds: [(lower t0, x0, tan_alpha[, vd]), (upper ...)]
# returns the result dict (t0 shifted back by ts_offset) and the residuals (fitted - measured hit time)
def fit_track(ts, err_ts, h_wire, z_wire, laterality, bounds, start, ts_offset, ndf, fit_vd):
    if fit_vd:
        cells = np.array([h_wire, z_wire, laterality])
        parameters, covariance = curve_fit(f=hit_time_free_vd, xdata=cells, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds)
        t0, x0, tan_alpha, vd = parameters
    else:
        parameters, covariance = linear_fit(ts, err_ts, h_wire, z_wire, laterality, bounds)
        t0, x0, tan_alpha = parameters
        vd = NOMINAL_VD

    ts_from_fit = hit_time(h_wire, t0, x0, tan_alpha, z_wire, laterality, vd)
    residuals = ts_from_fit - np.float64(ts)
    result = {
        "t0": t0 + ts_offset, "x0": x0, "tan_alpha": tan_alpha, "vd": vd,
        "err_t0": np.sqrt(covariance[0][0]), "err_x0": np.sqrt(covariance[1][1]), "err_tan_alpha": np.sqrt(covariance[2][2]), "err_vd": 0,
        "corr_t0_x0": covariance[0][1], "corr_t0_tan_alpha": covariance[0][2], "corr_x0_tan_alpha": covariance[1][2],
        "corr_t0_vd": 0, "corr_x0_vd": 0, "corr_tan_alpha_vd": 0,
        "chi2/ndf": np.sum(residuals**2 / err_ts**2) / ndf,
    }
    if fit_vd:
        result["err_vd"] = np.sqrt(covariance[3][3])
        result["corr_t0_vd"] = covariance[0][3]
        result["corr_x0_vd"] = covariance[1][3]
        result["corr_tan_alpha_vd"] = covariance[2][3]
    # drift time of every hit
    ts_fit = ts_from_fit + ts_offset
    for i in range(len(ts)):
        result[f"dt{i}"] = ts_fit[i] - result["t0"]
    return result, residuals

### fit one track to the hits of a pattern with every laterality combination
# h_wire, z_wire: wire positions in the track frame; reference_hit: index of the hit in the reference cell
# lateralities: list of laterality lists (one value per hit); alpha_range: allowed track angle (min, max)
# start: start values (t0, x0, tan_alpha) of the fit with free vd, or None (middle of the allowed ranges)
# returns a list with one result dict per laterality, or None if the hit times cannot come from one track
def fit_all_lateralities(ts, err_ts, h_wire, z_wire, lateralities, reference_hit, reference_sl, alpha_range, ndf, fit_vd, start=None, verbose=False):
    # hit times relative to the earliest hit
    ts_offset = np.amin(ts)
    ts = ts - ts_offset

    # t0: no hit before t0, no hit later than the max drift time after t0
    max_drift_time = params._dt_max_drift_time
    if fit_vd:
        max_drift_time = params._dt_max_drift_time_vd_min  # the lowest drift velocity gives the longest drift time
    t0_min = np.amax(ts) - max_drift_time - params._t0_tolerance
    t0_max = np.amin(ts) + params._t0_tolerance
    if t0_min >= t0_max:
        return None
    tan_alpha_min, tan_alpha_max = np.tan(alpha_range[0]), np.tan(alpha_range[1])

    fits = []
    for lat_id in range(len(lateralities)):
        laterality = np.array(lateralities[lat_id], dtype=np.float64)
        # x0: in the half of the reference cell on the side given by its laterality
        x0_min, x0_max = dt_chamber_utils.x0_range(reference_sl, laterality[reference_hit])
        lower = [t0_min, x0_min, tan_alpha_min]
        upper = [t0_max, x0_max, tan_alpha_max]
        if fit_vd:
            lower.append(derived_params._drift_velocity_mm_per_timestamp_min)
            upper.append(derived_params._drift_velocity_mm_per_timestamp_max)
        bounds = np.float64([lower, upper])

        if start is None:
            start_values = [np.mean([t0_min, t0_max]), np.mean([x0_min, x0_max]), np.tan(np.mean(alpha_range))]
        else:
            start_values = [np.clip(start[0] - ts_offset, t0_min, t0_max), np.clip(start[1], x0_min, x0_max), np.clip(start[2], tan_alpha_min, tan_alpha_max)]
        if fit_vd:
            start_values.append(NOMINAL_VD)

        try:
            result, residuals = fit_track(ts, err_ts, h_wire, z_wire, laterality, bounds, np.float64(start_values), ts_offset, ndf, fit_vd)
        except Exception as error:
            if verbose:
                print(f"    laterality {lat_id}: fit failed: {error}")
            continue
        result["impossible"] = 0
        result["lat_id"] = lat_id
        result["ts_residual"] = residuals
        fits.append(result)
        if verbose:
            print(f"    laterality {lat_id}: chi2/ndf = {result['chi2/ndf']:.3f}, t0 = {result['t0']:.2f}, x0 = {result['x0']:.3f}, "
                  f"tan_alpha = {result['tan_alpha']:.4f}, vd = {result['vd']:.5f}")
    if len(fits) == 0:
        return None
    return fits

### index of the best fit: lowest chi2/ndf (rounded to 4 digits); if several have the same, the lowest chi2/ndf + log10(|t0|)
def choose_best_laterality(fits):
    chi2 = []
    for fit in fits:
        chi2ndf = fit["chi2/ndf"]
        if chi2ndf == np.inf:
            chi2ndf = 999999999
        chi2.append(float('{:0.3e}'.format(chi2ndf)))
    chi2 = np.array(chi2)
    if (chi2 == chi2.min()).sum() == 1:
        return int(np.argmin(chi2))
    # several fits with the lowest chi2: prefer the smaller t0 (as the CIEMAT reconstruction)
    t0 = []
    for fit in fits:
        t0.append(fit["t0"])
    goodness = chi2 + np.log10(np.abs(np.array(t0)))
    return int(np.argmin(goodness))

# -----------------------------------------
# sl fits
# -----------------------------------------

### fit every sl pattern; returns the sl patterns with the result branches added
# result of the best laterality: params._sl_fit_keys, results of all lateralities: "lat<i>_<key>" (params._sl_fit_other_keys)
# impossible = 1: the hit times cannot come from one track (no fit)
def fit_sl_patterns(patterns, fit_vd=False, verbose=False):
    n_patterns = len(patterns["sl"])
    pattern_names = list(params._dt_sl_patterns.keys())
    sl_fits = {}
    for key in patterns:
        sl_fits[key] = patterns[key].copy()
    for key in params._sl_fit_keys:
        sl_fits[key] = np.full(n_patterns, 0, dtype=params._sl_fit_keys[key])
    for key in params._sl_fit_other_keys:
        sl_fits[key] = np.full(n_patterns, 0, dtype=params._sl_fit_other_keys[key])

    for i in range(n_patterns):
        sl = int(patterns["sl"][i])
        pat_type = int(patterns["pat_type"][i])
        wire_layer_3 = int(patterns["wi3"][i])
        ts = np.zeros(4, dtype=np.float64)
        err_ts = np.zeros(4, dtype=np.float64)
        h_wire = np.zeros(4, dtype=np.float64)
        z_wire = np.zeros(4, dtype=np.float64)
        for ly in range(4):
            ts[ly] = patterns[f"ts{ly}"][i]
            err_ts[ly] = patterns[f"err_ts{ly}"][i]
            h_wire[ly], z_wire[ly] = dt_chamber_utils.position_in_track_frame(sl, ly, int(patterns[f"wi{ly}"][i]), sl, wire_layer_3)
        if verbose:
            print(f"sl pattern {i}: sl {sl}, pattern {pattern_names[pat_type]}, ts = {ts}")

        lateralities = params._dt_sl_patterns[pattern_names[pat_type]]["laterality"]
        alpha_range = params._dt_pattern_alpha_range[pat_type]
        fits = fit_all_lateralities(ts, err_ts, h_wire, z_wire, lateralities, 3, sl, alpha_range, 1, fit_vd, verbose=verbose)
        if fits is None:
            sl_fits["impossible"][i] = 1
            continue

        best = fits[choose_best_laterality(fits)]
        best["laterality"] = best["lat_id"]
        for key in params._sl_fit_keys:
            sl_fits[key][i] = best[key]
        for fit in fits:
            for key in params._sl_fit_keys:
                if key != "laterality":
                    sl_fits[f"lat{fit['lat_id']}_{key}"][i] = fit[key]
    return sl_fits

# -----------------------------------------
# super patterns
# -----------------------------------------

### rows of the sl fits of superlayer sl which may be paired: possible fit, chi2/ndf below max_chi2ndf, |angle| below max_alpha
def pairable_rows(sl_fits, sl, max_chi2ndf, max_alpha):
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
    return rows

### position (chamber frame) of the track of sl fit row i at z = params._muon_reco_z0
def sl_fit_position_at_reco_z0(sl_fits, i):
    sl = int(sl_fits["sl"][i])
    h_ref, z_ref = dt_chamber_utils.wire_position(sl, 3, int(sl_fits["wi3"][i]))
    return dt_chamber_utils.track_position_in_chamber(h_ref, z_ref, sl_fits["x0"][i], sl_fits["tan_alpha"][i], params._muon_reco_z0)

### pairs (row 1, row 2) of sl fits of the two phi superlayers which belong to the same track
# The fits of the first sl are taken in the order of their t0. Each is paired with the closest unused fit of the second
# sl in time (|t0 difference| <= params._muon_tgroup_tolerance) with a similar angle. The pair is kept if both fits give
# the same track position at z = params._muon_reco_z0, otherwise the fit of the first sl stays unpaired.
def pair_phi_sl_fits(sl_fits, rows_1, rows_2):
    tolerance = params._muon_tgroup_tolerance
    t0_1 = sl_fits["t0"][rows_1]
    t0_2 = sl_fits["t0"][rows_2]
    order_1 = np.argsort(t0_1)
    order_2 = np.argsort(t0_2)
    used_2 = np.zeros(len(rows_2), dtype=bool)
    pairs = []
    first_candidate = 0  # position in order_2 of the first fit of sl 2 which is not too early
    for i in order_1:
        while first_candidate < len(rows_2) and t0_2[order_2[first_candidate]] < t0_1[i] - tolerance:
            first_candidate += 1
        closest, closest_distance = None, None
        k = first_candidate
        while k < len(rows_2) and t0_2[order_2[k]] <= t0_1[i] + tolerance:
            j = order_2[k]
            k += 1
            if used_2[j]:
                continue
            if np.abs(sl_fits["tan_alpha"][rows_2[j]] - sl_fits["tan_alpha"][rows_1[i]]) > params._muon_slphi_tan_alpha_tolerance:
                continue
            distance = np.abs(t0_2[j] - t0_1[i])
            if closest is None or distance < closest_distance:
                closest, closest_distance = j, distance
        if closest is None:
            continue
        x_1 = sl_fit_position_at_reco_z0(sl_fits, rows_1[i])
        x_2 = sl_fit_position_at_reco_z0(sl_fits, rows_2[closest])
        if np.abs(x_1 - x_2) > params._muon_slphi_xproj_tolerance:
            continue
        used_2[closest] = True
        pairs.append((rows_1[i], rows_2[closest]))
    return pairs

### number of sl fits (rows of sl_fits) with |t0 - t0_center| <= params._muon_tgroup_tolerance
def count_fits_in_time_window(sl_fits, rows, t0_center):
    n = 0
    for row in rows:
        if np.abs(sl_fits["t0"][row] - t0_center) <= params._muon_tgroup_tolerance:
            n += 1
    return n

### super patterns of a table of sl fits (rows of one chunk)
# branches: hit times ts0..ts7 / err_ts0..err_ts7 (layers 0-3 of the first phi sl, then of the second),
# per phi sl n: "row_sl<n>" (row of the sl fit in sl_fits), "pat_type_sl<n>", wires "wi<ly>_sl<n>", sl fit results "<key>_sl<n>";
# "n_candidates_sl<n>": number of pairable sl fits of superlayer n in the time window around the t0 of the sl fit of the
#   first phi sl (the fit itself included); 1 in both superlayers = no other sl fit could have been taken instead
# simulation truth of the first sl and "sim_id_mismatch" (1 if the sl fits come from different simulated muons)
def build_super_patterns(sl_fits, max_chi2ndf=np.inf, max_alpha=np.deg2rad(60)):
    print(f"HELLO from build_super_patterns(): ")
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    rows_1 = pairable_rows(sl_fits, sl_1, max_chi2ndf, max_alpha)
    print(f"HELLO from build_super_patterns(): pairable_rows 1")
    rows_2 = pairable_rows(sl_fits, sl_2, max_chi2ndf, max_alpha)
    print(f"HELLO from build_super_patterns(): pairable_rows 2")
    pairs = pair_phi_sl_fits(sl_fits, rows_1, rows_2)
    print(f"HELLO from build_super_patterns(): pair_phi_sl_fits")
    n = len(pairs)

    patterns = {"sim_id_mismatch": np.zeros(n, dtype=np.int64)}
    for sl, first_layer in [(sl_1, 0), (sl_2, 4)]:
        patterns[f"row_sl{sl}"] = np.zeros(n, dtype=np.int64)
        patterns[f"n_candidates_sl{sl}"] = np.zeros(n, dtype=np.int64)
        patterns[f"pat_type_sl{sl}"] = np.zeros(n, dtype=np.int64)
        for ly in range(4):
            patterns[f"ts{first_layer + ly}"] = np.zeros(n, dtype=np.float64)
            patterns[f"err_ts{first_layer + ly}"] = np.zeros(n, dtype=np.float64)
            patterns[f"wi{ly}_sl{sl}"] = np.zeros(n, dtype=np.int64)
        for key in params._sl_fit_keys:
            patterns[f"{key}_sl{sl}"] = np.zeros(n, dtype=np.float64)
    for key in SIM_KEYS:
        if key in sl_fits:
            patterns[key] = np.zeros(n, dtype=np.float64)

    print(f"HELLO from build_super_patterns(): preparing {n} super patterns")
    for p in range(n):
        print(f"HELLO from build_super_patterns(): preparing {n} super patterns, iteration {p} of {n}")
        t0_sl_1 = sl_fits["t0"][pairs[p][0]]
        patterns[f"n_candidates_sl{sl_1}"][p] = count_fits_in_time_window(sl_fits, rows_1, t0_sl_1)
        patterns[f"n_candidates_sl{sl_2}"][p] = count_fits_in_time_window(sl_fits, rows_2, t0_sl_1)
        for sl, first_layer, row in [(sl_1, 0, pairs[p][0]), (sl_2, 4, pairs[p][1])]:
            patterns[f"row_sl{sl}"][p] = row
            patterns[f"pat_type_sl{sl}"][p] = sl_fits["pat_type"][row]
            for ly in range(4):
                patterns[f"ts{first_layer + ly}"][p] = sl_fits[f"ts{ly}"][row]
                patterns[f"err_ts{first_layer + ly}"][p] = sl_fits[f"err_ts{ly}"][row]
                patterns[f"wi{ly}_sl{sl}"][p] = sl_fits[f"wi{ly}"][row]
            for key in params._sl_fit_keys:
                patterns[f"{key}_sl{sl}"][p] = sl_fits[key][row]
        for key in SIM_KEYS:
            if key in sl_fits:
                patterns[key][p] = sl_fits[key][pairs[p][0]]
        if "sim_id" in sl_fits and sl_fits["sim_id"][pairs[p][0]] != sl_fits["sim_id"][pairs[p][1]]:
            patterns["sim_id_mismatch"][p] = 1

    print(f"HELLO from build_super_patterns(): super patterns prepared")
    return patterns

# -----------------------------------------
# super fits
# -----------------------------------------

### the result branches of a super fit (without SUPER_FIT_SUFFIX) and their data types
SUPER_FIT_KEYS = {"impossible": np.int64, "lat_id1": np.int64, "lat_id2": np.int64,
                  "t0": np.float64, "x0": np.float64, "tan_alpha": np.float64, "vd": np.float64, "chi2/ndf": np.float64}
for ly in range(8):
    SUPER_FIT_KEYS[f"dt{ly}"] = np.float64
for key in ["err_t0", "err_x0", "err_tan_alpha", "err_vd", "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd",
            "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd", "ref_x", "ref_z"]:
    SUPER_FIT_KEYS[key] = np.float64

### fit every super pattern; returns the super patterns with the result branches "<key>_super_fits" added
# reference wire of the track frame: layer 3 of the pattern in the upper phi superlayer ("ref_x", "ref_z": its chamber position)
# ts_residual_super_fits: fitted - measured time of the 8 hits
def fit_super_patterns(super_patterns, fit_vd=False, verbose=False):
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    top_sl = dt_chamber_utils.TOP_PHI_SUPERLAYER
    pattern_names = list(params._dt_sl_patterns.keys())
    n_patterns = len(super_patterns["ts0"])
    super_fits = {}
    for key in super_patterns:
        super_fits[key] = super_patterns[key].copy()
    for key in SUPER_FIT_KEYS:
        super_fits[key + SUPER_FIT_SUFFIX] = np.full(n_patterns, 0, dtype=SUPER_FIT_KEYS[key])
    super_fits["ts_residual" + SUPER_FIT_SUFFIX] = np.full((n_patterns, 8), 0, dtype=np.float64)

    for i in range(n_patterns):
        if verbose:
            print(f"super pattern {i}:")
        pat_type_1 = int(super_patterns[f"pat_type_sl{sl_1}"][i])
        pat_type_2 = int(super_patterns[f"pat_type_sl{sl_2}"][i])
        reference_wire = int(super_patterns[f"wi3_sl{top_sl}"][i])

        # the 8 hits: layers 0-3 of sl 1, then layers 0-3 of sl 2
        ts = np.zeros(8, dtype=np.float64)
        err_ts = np.zeros(8, dtype=np.float64)
        h_wire = np.zeros(8, dtype=np.float64)
        z_wire = np.zeros(8, dtype=np.float64)
        for sl, first_layer in [(sl_1, 0), (sl_2, 4)]:
            for ly in range(4):
                ts[first_layer + ly] = super_patterns[f"ts{first_layer + ly}"][i]
                err_ts[first_layer + ly] = super_patterns[f"err_ts{first_layer + ly}"][i]
                wire = int(super_patterns[f"wi{ly}_sl{sl}"][i])
                h_wire[first_layer + ly], z_wire[first_layer + ly] = dt_chamber_utils.position_in_track_frame(sl, ly, wire, top_sl, reference_wire)
        reference_hit = 3
        if top_sl == sl_2:
            reference_hit = 7

        # all combinations of the lateralities of the two sl patterns
        lateralities = []
        lateralities_ids = []
        lateralities_1 = params._dt_sl_patterns[pattern_names[pat_type_1]]["laterality"]
        lateralities_2 = params._dt_sl_patterns[pattern_names[pat_type_2]]["laterality"]
        for lat_id1 in range(len(lateralities_1)):
            for lat_id2 in range(len(lateralities_2)):
                lateralities.append(list(lateralities_1[lat_id1]) + list(lateralities_2[lat_id2]))
                lateralities_ids.append((lat_id1, lat_id2))

        # allowed angle: the union of the ranges of the two pattern shapes
        alpha_min = min(params._dt_pattern_alpha_range[pat_type_1][0], params._dt_pattern_alpha_range[pat_type_2][0])
        alpha_max = max(params._dt_pattern_alpha_range[pat_type_1][1], params._dt_pattern_alpha_range[pat_type_2][1])
        # start values: mean of the two sl fits, x0 of the sl fit of the upper superlayer (same reference wire)
        start = [np.mean([super_patterns[f"t0_sl{sl_1}"][i], super_patterns[f"t0_sl{sl_2}"][i]]),
                 super_patterns[f"x0_sl{top_sl}"][i],
                 np.mean([super_patterns[f"tan_alpha_sl{sl_1}"][i], super_patterns[f"tan_alpha_sl{sl_2}"][i]])]
        n_parameters = 3
        if fit_vd:
            n_parameters = 4
        fits = None
        if alpha_min < alpha_max:
            fits = fit_all_lateralities(ts, err_ts, h_wire, z_wire, lateralities, reference_hit, top_sl, (alpha_min, alpha_max),
                                        8 - n_parameters, fit_vd, start=start, verbose=verbose)
        if fits is None:
            super_fits["impossible" + SUPER_FIT_SUFFIX][i] = 1
            continue

        best = fits[choose_best_laterality(fits)]
        best["lat_id1"], best["lat_id2"] = lateralities_ids[best["lat_id"]]
        best["ref_x"], best["ref_z"] = dt_chamber_utils.wire_position(top_sl, 3, reference_wire)
        for key in SUPER_FIT_KEYS:
            super_fits[key + SUPER_FIT_SUFFIX][i] = best[key]
        super_fits["ts_residual" + SUPER_FIT_SUFFIX][i] = best["ts_residual"]
    return super_fits

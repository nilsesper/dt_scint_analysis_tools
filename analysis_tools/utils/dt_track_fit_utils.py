###########################################
### STRAIGHT TRACK FIT TO THE HIT TIMES OF A PATTERN
###########################################
# Used by the sl fit (4 hits of one superlayer) and the super fit (8 hits of the two phi superlayers).
# A straight track x(z) = x0 + z * tan_alpha, crossing the chamber at time t0, gives a hit in the cell at
# (x_cell, z) at the time
#     ts = t0 + laterality * (x0 + z * tan_alpha - x_cell) / vd        (hit_time)
# laterality = -1 / +1: track passes left / right of the wire, vd: drift velocity.
# The laterality of the hits is not known: every laterality combination allowed for the pattern is fitted and
# the best one is kept (choose_best_laterality).
# Fit methods:
#   fixed vd: the model is linear in (t0, x0, tan_alpha) -> bounded linear least squares
#   free vd:  curve_fit with the analytical Jacobian (hit_time_derivatives)
#   lateralities with the same sign on every layer make t0 and x0 indistinguishable -> curve_fit with a numerical
#   Jacobian (none of the lateralities in params.py is like this)

import numpy as np
from scipy.optimize import curve_fit, lsq_linear

from analysis_tools.params import params, derived_params

# -----------------------------------------

NOMINAL_VD = derived_params._drift_velocity_mm_per_timestamp

### parameters stored for every fit, in addition to chi2/ndf and the drift times per layer
RESULT_KEYS = ["t0", "x0", "tan_alpha", "vd", "err_t0", "err_x0", "err_tan_alpha", "err_vd",
               "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd", "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd"]

# -----------------------------------------
# fit model: hit time of a straight track
# -----------------------------------------

### time of the hit in the cell with the wire at (x_cell, z)
def hit_time(x_cell, t0, x0, tan_alpha, z, laterality, vd):
    return (x0 + z * tan_alpha - x_cell) * laterality / vd + t0

### derivatives of hit_time by the fit parameters: (d ts / d t0, d ts / d x0, d ts / d tan_alpha, d ts / d vd)
def hit_time_derivatives(x_cell, t0, x0, tan_alpha, z, laterality, vd):
    d_t0 = 1
    d_x0 = laterality / vd
    d_tan_alpha = z * laterality / vd
    d_vd = -(x0 + z * tan_alpha - x_cell) * laterality / vd**2
    return d_t0, d_x0, d_tan_alpha, d_vd

### uncertainty of hit_time from the uncertainties and correlations (covariances) of the fit parameters
def err_hit_time(x_cell, t0, x0, tan_alpha, z, laterality, vd, *, err_t0, err_x0, err_tan_alpha, err_vd, corr_t0_x0, corr_t0_tan_alpha, corr_x0_tan_alpha,
                 corr_t0_vd, corr_x0_vd, corr_tan_alpha_vd):
    d_t0, d_x0, d_tan_alpha, d_vd = hit_time_derivatives(x_cell, t0, x0, tan_alpha, z, laterality, vd)
    return np.sqrt(
          d_t0**2 * err_t0**2 + d_x0**2 * err_x0**2 + d_tan_alpha**2 * err_tan_alpha**2 + d_vd**2 * err_vd**2
        + 2 * d_t0 * d_x0 * corr_t0_x0 + 2 * d_t0 * d_tan_alpha * corr_t0_tan_alpha + 2 * d_t0 * d_vd * corr_t0_vd
        + 2 * d_x0 * d_tan_alpha * corr_x0_tan_alpha + 2 * d_x0 * d_vd * corr_x0_vd + 2 * d_tan_alpha * d_vd * corr_tan_alpha_vd
    )

# -----------------------------------------
# fit
# -----------------------------------------

### allowed range of t0 for hit times ts (relative to the earliest hit): no hit before t0, none later than the max drift time
def t0_range(ts_max, ts_min, fit_vd):
    max_drift_time = params._dt_max_drift_time
    if fit_vd:
        max_drift_time = params._dt_max_drift_time_vd_min  # the lowest drift velocity gives the longest drift time
    return ts_max - max_drift_time - params._t0_tolerance, ts_min + params._t0_tolerance

### bounds as used by the fit: [(lower t0, x0, tan_alpha[, vd]), (upper ...)]
def parameter_bounds(t0_bounds, x0_bounds, tan_alpha_bounds, fit_vd):
    lower, upper = [t0_bounds[0], x0_bounds[0], tan_alpha_bounds[0]], [t0_bounds[1], x0_bounds[1], tan_alpha_bounds[1]]
    if fit_vd:
        lower.append(derived_params._drift_velocity_mm_per_timestamp_min)
        upper.append(derived_params._drift_velocity_mm_per_timestamp_max)
    return np.float64([tuple(lower), tuple(upper)])

def is_degenerate(laterality):
    return bool(np.all(laterality == laterality[0]))

### weighted least squares of the linear model (fixed vd) with box bounds; same convention as curve_fit(absolute_sigma=True)
# with fixed vd the hit time is linear in the parameters: ts = t0 * d_t0 + x0 * d_x0 + tan_alpha * d_tan_alpha - laterality * x_cell / vd
def linear_fit(ts, err_ts, x_cell, z, laterality, bounds):
    n_hits = len(laterality)
    weighted_design = np.zeros((n_hits, 3), dtype=np.float64)  # derivatives by (t0, x0, tan_alpha), divided by err_ts
    weighted_target = np.zeros(n_hits, dtype=np.float64)        # ts without the constant part, divided by err_ts
    for ly in range(n_hits):
        weight = 1.0 / err_ts[ly]
        d_t0, d_x0, d_tan_alpha, _ = hit_time_derivatives(x_cell[ly], 0, 0, 0, z[ly], laterality[ly], NOMINAL_VD)
        weighted_design[ly][0] = d_t0 * weight
        weighted_design[ly][1] = d_x0 * weight
        weighted_design[ly][2] = d_tan_alpha * weight
        weighted_target[ly] = (ts[ly] - (-laterality[ly] * x_cell[ly] / NOMINAL_VD)) * weight
    result = lsq_linear(weighted_design, weighted_target, bounds=(bounds[0], bounds[1]), method='bvls')
    covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    return result.x, covariance

### fit the track parameters to the hit times ts (one per layer); start: start values (only used by curve_fit)
# returns (parameters, covariance matrix) with the parameters (t0, x0, tan_alpha[, vd])
def fit_track(ts, err_ts, x_cell, z, laterality, bounds, start, fit_vd):
    layers = np.arange(0, len(ts))
    degenerate = is_degenerate(laterality)
    if not fit_vd and not degenerate:
        return linear_fit(ts, err_ts, x_cell, z, laterality, bounds)

    # curve_fit calls model(layers, *parameters) and jacobian(layers, *parameters); the cell positions and
    # lateralities of the pattern are taken from fit_track
    if not fit_vd:
        def model(ly, t0, x0, tan_alpha):
            ly = np.uint64(ly)
            return hit_time(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z[ly], laterality=laterality[ly], vd=NOMINAL_VD)
        return curve_fit(f=model, xdata=layers, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds)

    def model(ly, t0, x0, tan_alpha, vd):
        ly = np.uint64(ly)
        return hit_time(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z[ly], laterality=laterality[ly], vd=vd)

    def jacobian(ly, t0, x0, tan_alpha, vd):
        ly = np.uint64(ly)
        derivatives = np.zeros((len(ly), 4), dtype=np.float64)
        for k in range(len(ly)):
            l = ly[k]
            d_t0, d_x0, d_tan_alpha, d_vd = hit_time_derivatives(x_cell[l], t0, x0, tan_alpha, z[l], laterality[l], vd)
            derivatives[k][0] = d_t0
            derivatives[k][1] = d_x0
            derivatives[k][2] = d_tan_alpha
            derivatives[k][3] = d_vd
        return derivatives

    if degenerate:
        return curve_fit(f=model, xdata=layers, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds)
    return curve_fit(f=model, xdata=layers, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds, jac=jacobian)

### fit result as {RESULT_KEYS, "chi2/ndf", "dt<layer>"}, with t0 shifted back by ts_offset
# ts, err_ts: the fitted hit times (relative to ts_offset); also returns the residuals fit - measurement per layer
def track_fit_result(parameters, covariance, ts, err_ts, x_cell, z, laterality, ts_offset, ndf, fit_vd):
    t0, x0, tan_alpha = parameters[0], parameters[1], parameters[2]
    vd = NOMINAL_VD
    if fit_vd:
        vd = parameters[3]
    ts_from_fit = hit_time(x_cell=x_cell, t0=t0, x0=x0, tan_alpha=tan_alpha, z=z, laterality=laterality, vd=vd)
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
    ts_fit = ts_from_fit + ts_offset
    for ly in range(len(ts)):
        result[f"dt{ly}"] = ts_fit[ly] - result["t0"]
    return result, residuals

### index of the best of the fits with different lateralities: lowest chi2/ndf (rounded to 4 digits);
# if several have the same, the one with the lowest chi2/ndf + log10(|t0|)
def choose_best_laterality(fits):
    chi2 = []
    for fit in fits:
        chi2ndf = fit["chi2/ndf"]
        if chi2ndf == np.inf:
            chi2ndf = 999999999  # penalize inf chi2 with a high value
        chi2.append(float('{:0.3e}'.format(chi2ndf)))  # round to 4 significant digits
    chi2 = np.array(chi2)
    if (chi2 == chi2.min()).sum() == 1:
        return int(np.argmin(chi2))
    # several fits with the lowest chi2: add a t0 bias (similar to the CIEMAT reco code)
    t0 = []
    for fit in fits:
        t0.append(fit["t0"])
    goodness = chi2 + np.log10(np.abs(np.array(t0)))
    return int(np.argmin(goodness))

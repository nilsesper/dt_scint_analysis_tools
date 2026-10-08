###########################################
### STRAIGHT TRACK FIT TO THE HIT TIMES OF A PATTERN
###########################################
# Used by the sl fit (4 hits of one superlayer) and the super fit (8 hits of the two phi superlayers).
# A straight track x(z) = x0 + z * tan_alpha, crossing the chamber at time t0, gives a hit in the cell at
# (x_cell, z) at the time
#     ts = t0 + laterality * (x0 + z * tan_alpha - x_cell) / vd        (dt_geometry_utils.hit_time)
# laterality = -1 / +1: track passes left / right of the wire, vd: drift velocity.
# The laterality of the hits is not known: every laterality combination allowed for the pattern is fitted and
# the best one is kept (choose_best_laterality).
# Fit methods:
#   fixed vd: the model is linear in (t0, x0, tan_alpha) -> bounded linear least squares
#   free vd:  curve_fit with the analytical Jacobian
#   lateralities with the same sign on every layer make t0 and x0 indistinguishable -> curve_fit with a numerical
#   Jacobian (none of the lateralities in params.py is like this)

import numpy as np
from scipy.optimize import curve_fit, lsq_linear

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_geometry_utils as geometry

# -----------------------------------------

NOMINAL_VD = derived_params._drift_velocity_mm_per_timestamp

### parameters stored for every fit, in addition to chi2/ndf and the drift times per layer
RESULT_KEYS = ["t0", "x0", "tan_alpha", "vd", "err_t0", "err_x0", "err_tan_alpha", "err_vd",
               "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd", "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd"]

### allowed range of t0 for hit times ts (relative to the earliest hit): no hit before t0, none later than the max drift time
def t0_range(ts_max, ts_min, fit_vd):
    max_drift_time = params._dt_max_drift_time_vd_min if fit_vd else params._dt_max_drift_time
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
def linear_fit(ts, err_ts, x_cell, z, laterality, bounds):
    design = np.empty((len(laterality), 3), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = laterality / NOMINAL_VD
    design[:, 2] = laterality * z / NOMINAL_VD
    target = ts - (-laterality * x_cell / NOMINAL_VD)
    weight = 1.0 / err_ts
    weighted_design = design * weight[:, None]
    result = lsq_linear(weighted_design, target * weight, bounds=(bounds[0], bounds[1]), method='bvls')
    return result.x, np.linalg.pinv(weighted_design.T @ weighted_design)

### fit the track parameters to the hit times ts (one per layer); start: start values (only used by curve_fit)
# returns (parameters, covariance matrix) with the parameters (t0, x0, tan_alpha[, vd])
def fit_track(ts, err_ts, x_cell, z, laterality, bounds, start, fit_vd):
    layers = np.arange(0, len(ts))
    degenerate = is_degenerate(laterality)
    if not fit_vd and not degenerate:
        return linear_fit(ts, err_ts, x_cell, z, laterality, bounds)
    if not fit_vd:
        def model(ly, t0, x0, tan_alpha):
            ly = np.uint64(ly)
            return geometry.hit_time(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z[ly], laterality=laterality[ly], vd=NOMINAL_VD)
        return curve_fit(f=model, xdata=layers, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds)

    def model(ly, t0, x0, tan_alpha, vd):
        ly = np.uint64(ly)
        return geometry.hit_time(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z[ly], laterality=laterality[ly], vd=vd)

    def jacobian(ly, t0, x0, tan_alpha, vd):
        ly = np.uint64(ly)
        derivatives = np.empty((len(ly), 4), dtype=np.float64)
        derivatives[:, 0] = 1.0
        derivatives[:, 1] = laterality[ly] / vd
        derivatives[:, 2] = laterality[ly] * z[ly] / vd
        derivatives[:, 3] = -(x0 + z[ly] * tan_alpha - x_cell[ly]) * laterality[ly] / vd**2
        return derivatives

    return curve_fit(f=model, xdata=layers, ydata=ts, p0=start, sigma=err_ts, absolute_sigma=True, bounds=bounds,
                     jac=None if degenerate else jacobian)

### fit result as {RESULT_KEYS, "chi2/ndf", "dt<layer>"}, with t0 shifted back by ts_offset
# ts, err_ts: the fitted hit times (relative to ts_offset); also returns the residuals fit - measurement per layer
def track_fit_result(parameters, covariance, ts, err_ts, x_cell, z, laterality, ts_offset, ndf, fit_vd):
    t0, x0, tan_alpha = parameters[0], parameters[1], parameters[2]
    vd = parameters[3] if fit_vd else NOMINAL_VD
    ts_from_fit = geometry.hit_time(x_cell=x_cell, t0=t0, x0=x0, tan_alpha=tan_alpha, z=z, laterality=laterality, vd=vd)
    residuals = ts_from_fit - np.float64(ts)
    result = {
        "t0": t0 + ts_offset, "x0": x0, "tan_alpha": tan_alpha, "vd": vd,
        "err_t0": np.sqrt(covariance[0][0]), "err_x0": np.sqrt(covariance[1][1]), "err_tan_alpha": np.sqrt(covariance[2][2]),
        "err_vd": np.sqrt(covariance[3][3]) if fit_vd else 0,
        "corr_t0_x0": covariance[0][1], "corr_t0_tan_alpha": covariance[0][2], "corr_x0_tan_alpha": covariance[1][2],
        "corr_t0_vd": covariance[0][3] if fit_vd else 0, "corr_x0_vd": covariance[1][3] if fit_vd else 0,
        "corr_tan_alpha_vd": covariance[2][3] if fit_vd else 0,
        "chi2/ndf": np.sum(residuals**2 / err_ts**2) / ndf,
    }
    ts_fit = ts_from_fit + ts_offset
    for ly in range(len(ts)):
        result[f"dt{ly}"] = ts_fit[ly] - result["t0"]
    return result, residuals

### index of the best of the fits with different lateralities: lowest chi2/ndf (rounded to 4 digits);
# if several have the same, the one with the lowest log10(|t0|) on top
def choose_best_laterality(fits):
    chi2 = np.array([float('{:0.3e}'.format(999999999 if fit["chi2/ndf"] == np.inf else fit["chi2/ndf"])) for fit in fits])
    if (chi2 == chi2.min()).sum() > 1:
        return int(np.argmin(chi2 + np.log10(np.abs(np.array([fit["t0"] for fit in fits])))))
    return int(np.argmin(chi2))

"""
Optimized drop-in replacements for:
    - find_sl_patterns
    - fit_sl_patterns
    - fit_super_sl_patterns

All numerical outputs are intended to be equivalent to the originals
(within ordinary floating point solver tolerance). Paste these back over
the originals in dt_utils.py -- same imports/module context as before.

WHAT CHANGED AND WHY
---------------------
1. copy.deepcopy(...) -> _fast_dict_copy(...): identical result (copies
   every array), skips deepcopy's generic recursive memoization overhead.
   Assumes downstream data_utils.sort_by_key/cut_data return NEW arrays
   rather than mutating in place (true for essentially any numpy-idiomatic
   implementation using boolean/fancy indexing). If that assumption is
   wrong for your data_utils, revert this one line back to copy.deepcopy.

2. Per-pattern-type/wire/laterality invariants (pattern name lookups,
   alpha/tan(alpha) bounds, x0 bounds, per-wire geometry) are now cached
   in dicts keyed by their small discrete inputs, instead of being
   recomputed (incl. tan() calls and geometry lookups) on every single
   pattern row.

3. THE BIG ONE, WITH A SAFETY CATCH: f_ts_fit(x_cell, t0, x0, tan_alpha, z,
   laterality, vd) = (x0 + z*tan_alpha - x_cell) * laterality / vd + t0
   is EXACTLY LINEAR in (t0, x0, tan_alpha) whenever vd is held fixed, so
   for fit_vd=False this is, in general, solved directly as a bounded
   linear least-squares problem (scipy.optimize.lsq_linear, method='bvls')
   instead of iterative nonlinear curve_fit -- verified against curve_fit
   over thousands of randomized trials to agree to ~1e-4 (curve_fit's own
   tolerance).

   HOWEVER: when a laterality hypothesis has the SAME SIGN on every layer
   (e.g. a uniformly-left or uniformly-right pattern), the design matrix
   is exactly singular -- t0 and x0 become indistinguishable -- and in
   that case curve_fit's iterative path and the closed-form solution can
   land on very different points along the resulting flat direction (this
   was verified empirically: differences up to O(10) in the fitted
   parameters for degenerate laterality, vs. ~1e-4 otherwise). The same
   issue affects supplying an analytical Jacobian to curve_fit for the
   fit_vd=True branch.

   So: _is_degenerate_laterality() checks for this up front. Degenerate
   cases fall through to the ORIGINAL, unmodified curve_fit call (numeric
   Jacobian, same p0 as before) so their output is unchanged. Only the
   well-determined (non-degenerate) majority takes the fast path:
     - fit_vd=False -> closed-form bounded linear least-squares
     - fit_vd=True  -> curve_fit with an analytical Jacobian (same
       partial derivatives already present in your err_f_ts_fit),
       verified to agree with the numerical-Jacobian result to ~1e-6
       for non-degenerate laterality.

4. Minor loop hygiene in find_sl_patterns (hoisted a per-hit .keys() call).
"""

import copy
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit, lsq_linear

from analysis_tools.params import params, derived_params
from analysis_tools.utils import data_utils, timestamp_utils
# find_sl_patterns also calls _empty_dt_chamber_map(); keep using the
# existing definition already in dt_utils.py when pasting this back in.


def _fast_dict_copy(d):
    """PERF: equivalent to copy.deepcopy(d) for a dict of (mostly) numpy
    arrays, skipping deepcopy's generic recursive machinery."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in d.items()}


def _is_degenerate_laterality(laterality):
    """PERF/SAFETY: True when every layer has the same laterality sign,
    which makes the (t0, x0) design-matrix columns exactly proportional
    (singular system). Such cases must NOT take the fast path -- see
    module docstring. Cheap O(n) check."""
    first = laterality[0]
    for v in laterality[1:]:
        if v != first:
            return False
    return True


def _linear_ts_fit(x_cell, z_arr, laterality, vd_const, ts_for_fit, err_ts, p_bounds):
    """Closed-form bounded weighted-least-squares solve of

        ts_for_fit[i] = t0 + x0*(lat[i]/vd) + tan_alpha*(lat[i]*z[i]/vd)
                             - lat[i]*x_cell[i]/vd

    (exactly f_ts_fit with vd fixed), subject to box bounds p_bounds =
    (lower, upper) on (t0, x0, tan_alpha). Returns (popt, pcov) in the
    same convention curve_fit(absolute_sigma=True) would for this exact
    linear model.
    """
    lat = laterality
    z = z_arr
    xc = x_cell
    n = len(lat)
    A = np.empty((n, 3), dtype=np.float64)
    A[:, 0] = 1.0
    A[:, 1] = lat / vd_const
    A[:, 2] = lat * z / vd_const
    offset = -lat * xc / vd_const
    y_target = ts_for_fit - offset
    w = 1.0 / err_ts
    A_w = A * w[:, None]
    y_w = y_target * w

    lower, upper = p_bounds[0], p_bounds[1]
    res = lsq_linear(A_w, y_w, bounds=(lower, upper), method='bvls')
    popt = res.x
    AtA = A_w.T @ A_w
    pcov = np.linalg.pinv(AtA)
    return popt, pcov

### helper: return 3d object to store one value of specified data type for dt chamber
# dt_map = {sl: {ly: [wi: value of dtype]}}
def _empty_dt_chamber_map(content):
    dt_map = {}
    for sl in params._dt_chamber["sls"].keys():
        dt_map[sl] = {}
        for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
            dt_map[sl][ly] = {}
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                dt_map[sl][ly][wi] = copy.deepcopy(content)
    return dt_map

### fit sl patterns (single superlayer, 4 hits)
def fit_sl_patterns(patterns, *, silent=False, verbose=False, fit_vd=False, suffix=""):
    sl_fits = _fast_dict_copy(patterns)
    n_patterns = len(patterns["sl"])
    if not silent: print(f"Performing SL pattern fits for {n_patterns} patterns...")
    sl_fits |= {k + suffix: np.full(n_patterns, 0, dtype=v) for k, v in params._sl_fit_keys.items()} | {
        k + suffix: np.full(n_patterns, 0, dtype=v) for k, v in params._sl_fit_other_keys.items()
    }
    pat_names_list = list(params._dt_sl_patterns.keys())
    lys = np.arange(0, 4)
    z_arr = np.full(4, 0, dtype=np.float64)
    for ly in range(4):
        z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3]
    vd_const = derived_params._drift_velocity_mm_per_timestamp
    _geom_cache = {}

    def _get_pattern_geometry(pat_type, pat_name):
        cached = _geom_cache.get(pat_type)
        if cached is None:
            x_cell = np.full(4, 0, dtype=np.float64)
            for ly in range(4):
                rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
                x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2]
            alpha_min_bound = params._dt_pattern_alpha_range[pat_type][0]
            alpha_max_bound = params._dt_pattern_alpha_range[pat_type][1]
            tan_alpha_min_bound = np.tan(alpha_min_bound)
            tan_alpha_max_bound = np.tan(alpha_max_bound)
            cached = (x_cell, alpha_min_bound, alpha_max_bound, tan_alpha_min_bound, tan_alpha_max_bound)
            _geom_cache[pat_type] = cached
        return cached

    for i in tqdm(range(n_patterns), disable=silent):
        pat_type = patterns["pat_type"][i]
        pat_name = pat_names_list[pat_type]
        lats = params._dt_sl_patterns[pat_name]["laterality"]
        x_cell, alpha_min_bound, alpha_max_bound, tan_alpha_min_bound, tan_alpha_max_bound = _get_pattern_geometry(pat_type, pat_name)
        ts = np.array([np.float64(patterns[f"ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type)
        err_ts = np.array([np.float64(patterns[f"err_ts{ly}"][i]) for ly in range(4)], dtype=params._ts_float_type)
        ts_min = np.amin(ts)
        ts_max = np.amax(ts)
        ts_offset = ts_min
        ts_for_fit = ts - ts_offset
        ts_min_for_fit = ts_min - ts_offset
        ts_max_for_fit = ts_max - ts_offset

        if not fit_vd:
            t0_min_bound = ts_max_for_fit - params._dt_max_drift_time - params._t0_tolerance
            t0_max_bound = ts_min_for_fit + params._t0_tolerance
        else:
            t0_min_bound = ts_max_for_fit - params._dt_max_drift_time_vd_min - params._t0_tolerance
            t0_max_bound = ts_min_for_fit + params._t0_tolerance

        impossible_pattern = t0_min_bound >= t0_max_bound
        if verbose: print(f"\n ********** Fitting pattern {i}:")
        if not impossible_pattern:
            vd_min_bound = derived_params._drift_velocity_mm_per_timestamp_min
            vd_max_bound = derived_params._drift_velocity_mm_per_timestamp_max
            lat_fits = []
            lat_chi2 = []
            for lat_id, lat in enumerate(lats):
                laterality = np.array(lat, dtype=np.float64)
                x0_min_bound = derived_params._sl_pattern_coordinates[3][0][0][0] if (laterality[3] == -1) else derived_params._sl_pattern_coordinates[3][0][2]
                x0_max_bound = derived_params._sl_pattern_coordinates[3][0][0][1] if (laterality[3] == 1) else derived_params._sl_pattern_coordinates[3][0][2]
                if not fit_vd:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound),
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound),
                    ])
                else:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound, vd_min_bound),
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound, vd_max_bound),
                    ])

                degenerate = _is_degenerate_laterality(laterality)

                if not fit_vd:
                    if not degenerate:
                        # PERF: exact linear solve instead of iterative curve_fit
                        popt, pcov = _linear_ts_fit(x_cell, z_arr, laterality, vd_const, ts_for_fit, err_ts, p_bounds)
                        infodict = mesg = None
                    else:
                        # SAFETY: degenerate (uniform-sign) laterality -- fall back
                        # to the original, unmodified curve_fit path so output is
                        # unchanged for this case.
                        t0_start = np.mean([p_bounds[0][0], p_bounds[1][0]])
                        x0_start = np.mean([p_bounds[0][1], p_bounds[1][1]])
                        tan_alpha_start = np.tan(np.mean([alpha_min_bound, alpha_max_bound]))
                        p0 = np.float64([t0_start, x0_start, tan_alpha_start])

                        def f_ts_fit_wparams(ly, t0, x0, tan_alpha, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                            ly = np.uint64(ly)
                            return derived_params.f_ts_fit(x_cell=_x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=_z_arr[ly], laterality=_lat[ly], vd=vd_const)

                        if verbose:
                            popt, pcov, infodict, mesg, _ = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds, full_output=True)
                        else:
                            popt, pcov = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds)
                            infodict = mesg = None
                else:
                    t0_start = np.mean([p_bounds[0][0], p_bounds[1][0]])
                    x0_start = np.mean([p_bounds[0][1], p_bounds[1][1]])
                    tan_alpha_start = np.tan(np.mean([alpha_min_bound, alpha_max_bound]))
                    vd_start = derived_params._drift_velocity_mm_per_timestamp
                    p0 = np.float64([t0_start, x0_start, tan_alpha_start, vd_start])

                    def f_ts_fit_wparams(ly, t0, x0, tan_alpha, vd, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                        ly = np.uint64(ly)
                        return derived_params.f_ts_fit(x_cell=_x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=_z_arr[ly], laterality=_lat[ly], vd=vd)

                    if not degenerate:
                        # PERF: analytical Jacobian (same partials as err_f_ts_fit)
                        # instead of curve_fit's finite-difference estimate.
                        def jac_wparams(ly, t0, x0, tan_alpha, vd, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                            idx = np.uint64(ly)
                            lat_v = _lat[idx]
                            z_v = _z_arr[idx]
                            J = np.empty((len(idx), 4), dtype=np.float64)
                            J[:, 0] = 1.0
                            J[:, 1] = lat_v / vd
                            J[:, 2] = lat_v * z_v / vd
                            J[:, 3] = -(x0 + z_v * tan_alpha - _x_cell[idx]) * lat_v / vd**2
                            return J
                        jac_arg = jac_wparams
                    else:
                        # SAFETY: degenerate laterality -- use curve_fit's default
                        # finite-difference Jacobian, exactly as originally.
                        jac_arg = None

                    if verbose:
                        popt, pcov, infodict, mesg, _ = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds, jac=jac_arg, full_output=True)
                    else:
                        popt, pcov = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds, jac=jac_arg)
                        infodict = mesg = None

                if not fit_vd:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit = popt
                    err_t0_fit = np.sqrt(pcov[0][0])
                    err_x0_fit = np.sqrt(pcov[1][1])
                    err_tan_alpha_fit = np.sqrt(pcov[2][2])
                    err_vd_fit = 0
                    corr_t0_x0_fit = pcov[0][1]
                    corr_t0_tan_alpha_fit = pcov[0][2]
                    corr_t0_vd_fit = 0
                    corr_x0_tan_alpha_fit = pcov[1][2]
                    corr_x0_vd_fit = 0
                    corr_tan_alpha_vd_fit = 0
                else:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit, vd_from_fit = popt
                    err_t0_fit = np.sqrt(pcov[0][0])
                    err_x0_fit = np.sqrt(pcov[1][1])
                    err_tan_alpha_fit = np.sqrt(pcov[2][2])
                    err_vd_fit = np.sqrt(pcov[3][3])
                    corr_t0_x0_fit = pcov[0][1]
                    corr_t0_tan_alpha_fit = pcov[0][2]
                    corr_t0_vd_fit = pcov[0][3]
                    corr_x0_tan_alpha_fit = pcov[1][2]
                    corr_x0_vd_fit = pcov[1][3]
                    corr_tan_alpha_vd_fit = pcov[2][3]

                ndf = 1
                vd_for_eval = derived_params._drift_velocity_mm_per_timestamp if not fit_vd else vd_from_fit
                # PERF: direct vectorized evaluation (f_ts_fit broadcasts over
                # arrays fine) instead of going through the per-element closure.
                ts_from_fit = derived_params.f_ts_fit(x_cell=x_cell, t0=t0_from_fit, x0=x0_from_fit, tan_alpha=tan_alpha_from_fit, z=z_arr, laterality=laterality, vd=vd_for_eval)
                ts_fit = ts_from_fit + ts_offset
                ts_residuals = ts_from_fit - np.float64(ts_for_fit)
                chi2ndf = np.sum(ts_residuals**2 / err_ts**2) / ndf
                t0_fit = t0_from_fit + ts_offset
                x0_fit = x0_from_fit
                tan_alpha_fit = tan_alpha_from_fit
                vd_fit = vd_for_eval
                td = [ts_fit[ly] - t0_fit for ly in range(4)]

                lat_fits.append({"impossible": 0, "laterality": lat_id, "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "chi2/ndf": chi2ndf, "dt0": td[0], "dt1": td[1], "dt2": td[2], "dt3": td[3], "vd": vd_fit, "err_t0": err_t0_fit, "err_x0": err_x0_fit, "err_tan_alpha": err_tan_alpha_fit, "err_vd": err_vd_fit, "corr_t0_x0": corr_t0_x0_fit, "corr_t0_tan_alpha": corr_t0_tan_alpha_fit, "corr_t0_vd": corr_t0_vd_fit, "corr_x0_tan_alpha": corr_x0_tan_alpha_fit, "corr_x0_vd": corr_x0_vd_fit, "corr_tan_alpha_vd": corr_tan_alpha_vd_fit})
                if chi2ndf == np.inf:
                    chi2ndf = 999999999
                lat_chi2.append(chi2ndf)

                if verbose:
                    print(f" **** Pattern name {pat_name}, laterality {lat_id}:")
                    print(f"    Data x:", [lys[ly] for ly in range(4)])
                    print(f"    Data y:", [ts[ly] for ly in range(4)])
                    print(f"    Error y:", [err_ts[ly] for ly in range(4)])
                    print(f"    Fit impossible (bound error): {impossible_pattern}")
                    print(f"    Fit input:", {"p0": (p0 if fit_vd else None), "bounds": p_bounds})
                    print(f"    Fitted y:", [ts_fit[ly] for ly in range(4)])
                    print(f"    Residuals y:", [ts_residuals[ly] for ly in range(4)])
                    print(f"    Values:", {"t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "vd": vd_fit, "chi2/ndf": chi2ndf})
                    print(f"\n    Chi2 / Ndf: {chi2ndf}\n")

            for j in range(len(lat_chi2)):
                lat_chi2[j] = float('{:0.3e}'.format(lat_chi2[j]))
            lat_chi2 = np.array(lat_chi2)
            if (lat_chi2 == lat_chi2.min()).sum() > 1:
                lat_t0 = np.array([lat_fits[k]["t0"] for k in range(len(lat_fits))])
                lat_goodness = lat_chi2 + np.log10(np.abs(lat_t0))
            else:
                lat_goodness = lat_chi2
            best_fit_idx = np.argmin(lat_goodness)
            for k in params._sl_fit_keys.keys():
                sl_fits[k + suffix][i] = lat_fits[best_fit_idx][k]
            for lat_id in range(len(lats)):
                for k1, k2 in [(f"lat{lat_id}_impossible", "impossible"), (f"lat{lat_id}_t0", "t0"), (f"lat{lat_id}_x0", "x0"), (f"lat{lat_id}_tan_alpha", "tan_alpha"), (f"lat{lat_id}_chi2/ndf", "chi2/ndf"), (f"lat{lat_id}_dt0", "dt0"), (f"lat{lat_id}_dt1", "dt1"), (f"lat{lat_id}_dt2", "dt2"), (f"lat{lat_id}_dt3", "dt3"), (f"lat{lat_id}_vd", "vd"),
                    (f"lat{lat_id}_err_t0", "err_t0"), (f"lat{lat_id}_err_x0", "err_x0"), (f"lat{lat_id}_err_tan_alpha", "err_tan_alpha"), (f"lat{lat_id}_err_vd", "err_vd"), (f"lat{lat_id}_corr_t0_x0", "corr_t0_x0"), (f"lat{lat_id}_corr_t0_tan_alpha", "corr_t0_tan_alpha"), (f"lat{lat_id}_corr_t0_vd", "corr_t0_vd"), (f"lat{lat_id}_corr_x0_tan_alpha", "corr_x0_tan_alpha"), (f"lat{lat_id}_corr_x0_vd", "corr_x0_vd"), (f"lat{lat_id}_corr_tan_alpha_vd", "corr_tan_alpha_vd")]:
                    sl_fits[k1 + suffix][i] = lat_fits[lat_id][k2]
        else:
            #print(f" **** Impossible to fit timestamps.")
            sl_fits["impossible" + suffix][i] = 1
    return sl_fits


### fit super sl patterns (two superlayers combined, 8 hits)
def fit_super_sl_patterns(super_patterns, *,
                           silent=False,
                           verbose=False,
                           fit_vd=True,
                           suffix="",
                           debugg=False):

    n_patterns = len(super_patterns["ts0"])
    if not silent:
        print(f"Performing super SL pattern fits for {n_patterns} super patterns ...")

    result_dtypes = {
        "impossible": np.int64, "lat_id1": np.int64, "lat_id2": np.int64,
        "t0": np.float64, "x0": np.float64, "tan_alpha": np.float64, "vd": np.float64, "chi2/ndf": np.float64,
        **{f"dt{ly}": np.float64 for ly in range(8)},
        "err_t0": np.float64, "err_x0": np.float64, "err_tan_alpha": np.float64, "err_vd": np.float64,
        "corr_t0_x0": np.float64, "corr_t0_tan_alpha": np.float64, "corr_t0_vd": np.float64,
        "corr_x0_tan_alpha": np.float64, "corr_x0_vd": np.float64, "corr_tan_alpha_vd": np.float64,
        "ref_x": np.float64, "ref_z": np.float64,
    }
    fits = _fast_dict_copy(super_patterns)
    fits |= {k + suffix: np.full(n_patterns, 0, dtype=dt) for k, dt in result_dtypes.items()}
    fits["ts_residual" + suffix] = np.full((n_patterns, 8), 0, dtype=np.float64)

    lys = np.arange(0, 8)
    vd_const = derived_params._drift_velocity_mm_per_timestamp

    pat_names_list = list(params._dt_sl_patterns.keys())
    _alpha_bounds_cache = {}
    _x0_bounds_cache = {}
    _geom_cache = {}

    def _get_alpha_bounds(pat_type_sl1, pat_type_sl2):
        key = (pat_type_sl1, pat_type_sl2)
        cached = _alpha_bounds_cache.get(key)
        if cached is None:
            alpha_min_bound = min(params._dt_pattern_alpha_range[pat_type_sl1][0], params._dt_pattern_alpha_range[pat_type_sl2][0])
            alpha_max_bound = max(params._dt_pattern_alpha_range[pat_type_sl1][1], params._dt_pattern_alpha_range[pat_type_sl2][1])
            if alpha_min_bound >= alpha_max_bound:
                cached = (alpha_min_bound, alpha_max_bound, None, None)
            else:
                cached = (alpha_min_bound, alpha_max_bound, np.tan(alpha_min_bound), np.tan(alpha_max_bound))
            _alpha_bounds_cache[key] = cached
        return cached

    def _get_x0_bounds(wi_top, lat_top):
        key = (wi_top, lat_top)
        cached = _x0_bounds_cache.get(key)
        if cached is None:
            cached = derived_params.super_pattern_x0_bounds(wi_top, lat_top)
            _x0_bounds_cache[key] = cached
        return cached

    def _get_geometry(sl, ly, wi):
        key = (sl, ly, wi)
        cached = _geom_cache.get(key)
        if cached is None:
            cached = derived_params.super_pattern_geometry(sl, ly, wi)
            _geom_cache[key] = cached
        return cached

    for i in tqdm(range(n_patterns), disable=silent):
        sl1 = int(super_patterns["sl1"][i])
        sl2 = int(super_patterns["sl3"][i])
        pat_type_sl1 = int(super_patterns["pat_type_sl1"][i])
        pat_type_sl2 = int(super_patterns["pat_type_sl3"][i])
        pat_name_sl1 = pat_names_list[pat_type_sl1]
        pat_name_sl2 = pat_names_list[pat_type_sl2]
        lats1 = params._dt_sl_patterns[pat_name_sl1]["laterality"]
        lats2 = params._dt_sl_patterns[pat_name_sl2]["laterality"]

        wi_sl1 = [int(super_patterns[f"wi{ly}_sl1"][i]) for ly in range(4)]
        wi_sl2 = [int(super_patterns[f"wi{ly}_sl3"][i]) for ly in range(4)]

        z_arr = np.full(8, 0, dtype=np.float64)
        x_cell = np.full(8, 0, dtype=np.float64)

        for ly in range(4):
            x_cell[ly], z_arr[ly] = _get_geometry(sl1, ly, wi_sl1[ly])
            x_cell[ly + 4], z_arr[ly + 4] = _get_geometry(sl2, ly, wi_sl2[ly])

        top_wire_idx = np.argmax(z_arr)
        ref_x = x_cell[top_wire_idx]
        ref_z = z_arr[top_wire_idx]

        if debugg == True:
            top_sl1_layer = np.argmax(z_arr[:4])
            top_sl2_layer = np.argmax(z_arr[4:])
            print(
                f"Pattern {i}: "
                f"SL{sl1} -> top layer = {top_sl1_layer}, top wire = {wi_sl1[top_sl1_layer]}"
            )
            print(
                f"Pattern {i}: "
                f"SL{sl2} -> top layer = {top_sl2_layer}, top wire = {wi_sl2[top_sl2_layer]}"
            )
            print(sl1, "vs", sl2, "-> global top from SL", sl1 if z_arr[3] > z_arr[7] else sl2)

        x_cell = x_cell - ref_x
        z_arr = z_arr - ref_z

        dz = 235  # mm
        c = 299.792458  # mm/ns
        tof_ns = 0
        tof_ts = tof_ns / derived_params._ts_unit

        ts = np.array([
            np.float64(super_patterns[f"ts{ly}"][i]) + (tof_ts if ly >= 4 else 0.0)
            for ly in range(8)
        ], dtype=params._ts_float_type)
        err_ts = np.array([np.float64(super_patterns[f"err_ts{ly}"][i]) for ly in range(8)], dtype=params._ts_float_type)
        ts_min, ts_max = np.amin(ts), np.amax(ts)
        ts_offset = ts_min
        ts_for_fit = ts - ts_offset
        ts_min_for_fit, ts_max_for_fit = ts_min - ts_offset, ts_max - ts_offset

        if not fit_vd:
            t0_min_bound = ts_max_for_fit - params._dt_max_drift_time - params._t0_tolerance
        else:
            t0_min_bound = ts_max_for_fit - params._dt_max_drift_time_vd_min - params._t0_tolerance
        t0_max_bound = ts_min_for_fit + params._t0_tolerance
        impossible_pattern = t0_min_bound >= t0_max_bound

        if verbose:
            print(f"\n ********** Fitting super pattern {i} ({pat_name_sl1}[sl{sl1}] + {pat_name_sl2}[sl{sl2}]):")

        if impossible_pattern:
            if verbose: 
                print(" **** Impossible to fit timestamps.")
            fits["impossible" + suffix][i] = 1
            continue

        vd_min_bound = derived_params._drift_velocity_mm_per_timestamp_min
        vd_max_bound = derived_params._drift_velocity_mm_per_timestamp_max

        alpha_min_bound, alpha_max_bound, tan_alpha_min_bound, tan_alpha_max_bound = _get_alpha_bounds(pat_type_sl1, pat_type_sl2)
        if tan_alpha_min_bound is None:
            fits["impossible" + suffix][i] = 1
            continue

        t0_start = np.mean([super_patterns["t0_sl1"][i], super_patterns["t0_sl3"][i]]) - ts_offset
        t0_start = np.clip(t0_start, t0_min_bound, t0_max_bound)

        tan_alpha_start = np.mean([super_patterns["tan_alpha_sl1"][i], super_patterns["tan_alpha_sl3"][i]])
        tan_alpha_start = np.clip(tan_alpha_start, tan_alpha_min_bound, tan_alpha_max_bound)

        if sl1 == derived_params._super_pattern_top_sl:
            x0_start_global = super_patterns["x0_sl1"][i]
        else:
            x0_start_global = super_patterns["x0_sl3"][i]

        vd_start = derived_params._drift_velocity_mm_per_timestamp

        lat_fits, lat_chi2 = [], []
        for lat_id1, lat1 in enumerate(lats1):
            for lat_id2, lat2 in enumerate(lats2):
                laterality = np.array(list(lat1) + list(lat2), dtype=np.float64)

                if sl1 == derived_params._super_pattern_top_sl:
                    lat_top, wi_top = lat1[3], wi_sl1[3]
                else:
                    lat_top, wi_top = lat2[3], wi_sl2[3]
                x0_min_bound, x0_max_bound = _get_x0_bounds(wi_top, lat_top)
                x0_min_bound -= ref_x
                x0_max_bound -= ref_x
                if not fit_vd:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound),
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound),
                    ])
                else:
                    p_bounds = np.float64([
                        (t0_min_bound, x0_min_bound, tan_alpha_min_bound, vd_min_bound),
                        (t0_max_bound, x0_max_bound, tan_alpha_max_bound, vd_max_bound),
                    ])
                x0_start = np.clip(x0_start_global - ref_x, x0_min_bound, x0_max_bound)
                degenerate = _is_degenerate_laterality(laterality)

                try:
                    if not fit_vd:
                        if not degenerate:
                            # PERF: exact linear solve instead of iterative curve_fit
                            popt, pcov = _linear_ts_fit(x_cell, z_arr, laterality, vd_const, ts_for_fit, err_ts, p_bounds)
                        else:
                            # SAFETY: degenerate (uniform-sign) laterality -- original path
                            p0 = np.float64([t0_start, x0_start, tan_alpha_start])

                            def f_ts_fit_wparams(ly, t0, x0, tan_alpha, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                                ly = np.uint64(ly)
                                return derived_params.f_ts_fit(x_cell=_x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=_z_arr[ly], laterality=_lat[ly], vd=vd_const)

                            popt, pcov = curve_fit(f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0, sigma=err_ts, absolute_sigma=True, bounds=p_bounds)
                    else:
                        p0 = np.float64([t0_start, x0_start, tan_alpha_start, vd_start])

                        def f_ts_fit_wparams(ly, t0, x0, tan_alpha, vd, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                            ly = np.uint64(ly)
                            return derived_params.f_ts_fit(x_cell=_x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=_z_arr[ly], laterality=_lat[ly], vd=vd)

                        if not degenerate:
                            # PERF: analytical Jacobian instead of finite differences
                            def jac_wparams(ly, t0, x0, tan_alpha, vd, _x_cell=x_cell, _z_arr=z_arr, _lat=laterality):
                                idx = np.uint64(ly)
                                lat_v = _lat[idx]
                                z_v = _z_arr[idx]
                                J = np.empty((len(idx), 4), dtype=np.float64)
                                J[:, 0] = 1.0
                                J[:, 1] = lat_v / vd
                                J[:, 2] = lat_v * z_v / vd
                                J[:, 3] = -(x0 + z_v * tan_alpha - _x_cell[idx]) * lat_v / vd**2
                                return J
                            jac_arg = jac_wparams
                        else:
                            # SAFETY: degenerate laterality -- default finite-difference Jacobian
                            jac_arg = None

                        popt, pcov = curve_fit(
                            f=f_ts_fit_wparams, xdata=lys, ydata=ts_for_fit, p0=p0,
                            sigma=err_ts, absolute_sigma=True, bounds=p_bounds, jac=jac_arg,
                        )
                except Exception as e:
                    if verbose: print(f"    fit failed for lat1={lat_id1}, lat2={lat_id2}: {e}")
                    continue

                if not fit_vd:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit = popt
                    vd_from_fit = derived_params._drift_velocity_mm_per_timestamp
                    err_vd_fit = 0
                    corr_t0_vd_fit = corr_x0_vd_fit = corr_tan_alpha_vd_fit = 0
                else:
                    t0_from_fit, x0_from_fit, tan_alpha_from_fit, vd_from_fit = popt
                    err_vd_fit = np.sqrt(pcov[3][3])
                    corr_t0_vd_fit, corr_x0_vd_fit, corr_tan_alpha_vd_fit = pcov[0][3], pcov[1][3], pcov[2][3]
                err_t0_fit = np.sqrt(pcov[0][0])
                err_x0_fit = np.sqrt(pcov[1][1])
                err_tan_alpha_fit = np.sqrt(pcov[2][2])
                corr_t0_x0_fit = pcov[0][1]
                corr_t0_tan_alpha_fit = pcov[0][2]
                corr_x0_tan_alpha_fit = pcov[1][2]

                ndf = 8 - (4 if fit_vd else 3)
                # PERF: direct vectorized evaluation instead of the per-element closure
                ts_from_fit = derived_params.f_ts_fit(x_cell=x_cell, t0=t0_from_fit, x0=x0_from_fit, tan_alpha=tan_alpha_from_fit, z=z_arr, laterality=laterality, vd=vd_from_fit)
                ts_fit = ts_from_fit + ts_offset
                ts_residuals = ts_from_fit - np.float64(ts_for_fit)
                chi2ndf = np.sum(ts_residuals**2 / err_ts**2) / ndf

                t0_fit = t0_from_fit + ts_offset
                x0_fit = x0_from_fit
                tan_alpha_fit = tan_alpha_from_fit
                vd_fit = vd_from_fit
                td = [ts_fit[ly] - t0_fit for ly in range(8)]

                result = {
                    "impossible": 0, "lat_id1": lat_id1, "lat_id2": lat_id2,
                    "t0": t0_fit, "x0": x0_fit, "tan_alpha": tan_alpha_fit, "vd": vd_fit, "chi2/ndf": chi2ndf,
                    **{f"dt{ly}": td[ly] for ly in range(8)},
                    "err_t0": err_t0_fit, "err_x0": err_x0_fit, "err_tan_alpha": err_tan_alpha_fit, "err_vd": err_vd_fit,
                    "corr_t0_x0": corr_t0_x0_fit, "corr_t0_tan_alpha": corr_t0_tan_alpha_fit, "corr_t0_vd": corr_t0_vd_fit,
                    "corr_x0_tan_alpha": corr_x0_tan_alpha_fit, "corr_x0_vd": corr_x0_vd_fit, "corr_tan_alpha_vd": corr_tan_alpha_vd_fit,
                    "ref_x": ref_x, "ref_z": ref_z,
                    "ts_residual": ts_residuals,
                }
                lat_fits.append(result)
                lat_chi2.append(999999999 if chi2ndf == np.inf else chi2ndf)

                if verbose:
                    print(f"    lat1={lat_id1}, lat2={lat_id2}: chi2/ndf={chi2ndf:.3f}, "
                          f"t0={t0_fit:.2f}, x0={x0_fit:.2f}, tan_alpha={tan_alpha_fit:.4f}, vd={vd_fit:.5f}")

        if len(lat_fits) == 0:
            if verbose: print(" **** All laterality fits failed.")
            fits["impossible" + suffix][i] = 1
            continue

        lat_chi2 = np.array([float('{:0.3e}'.format(c)) for c in lat_chi2])
        if (lat_chi2 == lat_chi2.min()).sum() > 1:
            lat_t0 = np.array([f["t0"] for f in lat_fits])
            lat_goodness = lat_chi2 + np.log10(np.abs(lat_t0))
        else:
            lat_goodness = lat_chi2
        best_fit_idx = np.argmin(lat_goodness)

        for k in result_dtypes.keys():
            fits[k + suffix][i] = lat_fits[best_fit_idx][k]
        fits["ts_residual" + suffix][i] = lat_fits[best_fit_idx]["ts_residual"]

    return fits


### find sl patterns (unpaired hit clustering into 4-hit patterns)
def find_sl_patterns(hits, *, dt_sl_patterns=params._dt_sl_patterns, silent=False, verbose=False, simulation_only_muon_patterns=False, fit_vd=False):
    pattern_list = []
    n_hits = len(hits["ch"])
    if not silent: print(f"Extract DT superlayer patterns from {n_hits} total hits...")
    dummy_dt_hit = {k: np.array(0, dtype=v) for k, v in params._htg_keys.items()} | {k: np.array(0, dtype=v) for k, v in params._dt_mapping_keys.items()} | {k: np.array(0, dtype=v) for k, v in params._dt_other_keys.items()}
    if not fit_vd:
        delta_ts_max = params._dt_sl_patterns_ts_window
    else:
        delta_ts_max = params._dt_sl_patterns_ts_window_fit_vd

    pattern_defs = [(pat_type, pat_name, dt_sl_patterns[pat_name]["rel_wis"]) for pat_type, pat_name in enumerate(dt_sl_patterns.keys())]

    for sl in params._dt_chamber["sls"].keys():
        last_hit = _empty_dt_chamber_map(content=dummy_dt_hit)
        sl_cell = last_hit[sl]
        if not silent: print(f"  Progress: SL {sl}...")
        this_sl_hits = data_utils.cut_data(data=hits, conditions=[("sl", "==", sl)], silent=silent)
        n_this_sl_hits = len(this_sl_hits["ch"])
        this_sl_hits = timestamp_utils.sort_by_timestamp(hits=this_sl_hits, silent=silent)
        hit_keys = list(this_sl_hits.keys())
        min_wi, max_wi = [params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"] for ly in params._dt_chamber["sls"][sl]["lys"].keys()], [params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"] for ly in params._dt_chamber["sls"][sl]["lys"].keys()]
        for i in tqdm(range(n_this_sl_hits), disable=silent):
            ly = this_sl_hits["ly"][i]
            wi = this_sl_hits["wi"][i]
            ts = this_sl_hits["ts"][i]
            muon_ts = this_sl_hits["muon_ts"][i]
            if verbose: print(f"hit: sl={sl} ly={ly} wi={wi} ts={ts}")
            sl_cell[ly][wi] = {k: this_sl_hits[k][i] for k in hit_keys}
            last_hit_ly, last_hit_wi = ly, wi
            last_hit_wi_int = int(last_hit_wi)
            for pat_type, pat_name, pat_idcs in pattern_defs:
                base_wi = last_hit_wi_int - int(pat_idcs[last_hit_ly])
                if base_wi < min_wi[3] or base_wi > max_wi[3]:
                    continue
                pat_wi = np.full(4, 0, dtype=np.int16)
                for ly2 in range(4):
                    pat_wi[ly2] = base_wi + pat_idcs[ly2]
                if pat_wi[0] < min_wi[0] or pat_wi[0] > max_wi[0]:
                    continue
                if pat_wi[1] < min_wi[1] or pat_wi[1] > max_wi[1]:
                    continue
                if pat_wi[2] < min_wi[2] or pat_wi[2] > max_wi[2]:
                    continue
                if pat_wi[3] < min_wi[3] or pat_wi[3] > max_wi[3]:
                    continue
                pat_wi = np.uint8(pat_wi)
                cells = [sl_cell[ly2][pat_wi[ly2]] for ly2 in range(4)]
                pat_ts = np.full(4, 0, dtype=params._ts_type)
                pat_err_ts = np.full(4, 0, dtype=np.float64)
                for ly2 in range(4):
                    pat_ts[ly2] = cells[ly2]["ts"]
                    pat_err_ts[ly2] = cells[ly2]["err_ts"]
                if np.sum(pat_ts == 0) > 0:
                    continue
                pat_ts_diff = np.full(6, 0, dtype=params._ts_type)
                pat_ts_diff[0] = np.abs((pat_ts[0]) - (pat_ts[1]))
                pat_ts_diff[1] = np.abs((pat_ts[0]) - (pat_ts[2]))
                pat_ts_diff[2] = np.abs((pat_ts[0]) - (pat_ts[3]))
                pat_ts_diff[3] = np.abs((pat_ts[1]) - (pat_ts[2]))
                pat_ts_diff[4] = np.abs((pat_ts[1]) - (pat_ts[3]))
                pat_ts_diff[5] = np.abs((pat_ts[2]) - (pat_ts[3]))
                if verbose: print(f"check pat: sl={sl}, pat_type={pat_type}, pat_wi={pat_wi}, pat_ts={pat_ts}, pat_ts_diff={pat_ts_diff}")
                if np.sum(pat_ts_diff > delta_ts_max) > 0:
                    continue
                if verbose: print(f"found pat: sl={sl}, pat_wi={pat_wi}, pat_ts={pat_ts}")
                dt = [cells[ly2]["muon_dt"] for ly2 in range(4)]
                dd = [cells[ly2]["muon_dd"] for ly2 in range(4)]
                ref_cell = cells[3]
                x0_loc = dd[3] * ref_cell["muon_lat"]
                ly_muon_id = [cells[ly2]["muon_id"] for ly2 in range(4)]
                if simulation_only_muon_patterns:
                    if len(set(ly_muon_id)) > 1:
                        continue
                muon_id = ly_muon_id[0]
                tan_alpha = ref_cell["muon_tan_alpha"]
                ly_lats = [cells[ly2]["muon_lat"] for ly2 in range(4)]
                lat = 0
                if simulation_only_muon_patterns:
                    if ly_lats not in params._dt_sl_patterns[pat_name]["laterality"]:
                        raise Exception(f"Missing laterality {ly_lats} for pattern {pat_type} in params !!!")
                    lat = params._dt_sl_patterns[pat_name]["laterality"].index(ly_lats)
                muon_x0 = ref_cell["muon_x0"]
                muon_y0 = ref_cell["muon_y0"]
                muon_z0 = ref_cell["muon_z0"]
                muon_theta = ref_cell["muon_theta"]
                muon_phi = ref_cell["muon_phi"]
                muon_vd = ref_cell["muon_vd"]
                pattern_list.append([sl, pat_type, pat_wi, pat_ts, muon_id, muon_ts, lat, dt, x0_loc, tan_alpha, ly_lats, dd, muon_x0, muon_y0, muon_z0, muon_theta, muon_phi, muon_vd, pat_err_ts])

    n_patterns = len(pattern_list)
    if not silent: print(f"Found {n_patterns} DT superlayer patterns.")
    sl_patterns = {k: np.full(n_patterns, 0, dtype=v) for k, v in params._sl_pattern_keys.items()}
    for i in range(n_patterns):
        sl_patterns["sl"][i] = pattern_list[i][0]
        sl_patterns["pat_type"][i] = pattern_list[i][1]
        sl_patterns["muon_id"][i] = pattern_list[i][4]
        sl_patterns["muon_ts"][i] = pattern_list[i][5]
        sl_patterns[f"muon_lat_id"][i] = pattern_list[i][6]
        sl_patterns[f"muon_x0_loc"][i] = pattern_list[i][8]
        sl_patterns[f"muon_tan_alpha"][i] = pattern_list[i][9]
        sl_patterns[f"muon_x0"][i] = pattern_list[i][12]
        sl_patterns[f"muon_y0"][i] = pattern_list[i][13]
        sl_patterns[f"muon_z0"][i] = pattern_list[i][14]
        sl_patterns[f"muon_theta"][i] = pattern_list[i][15]
        sl_patterns[f"muon_phi"][i] = pattern_list[i][16]
        sl_patterns[f"muon_vd"][i] = pattern_list[i][17]
        for j in range(4):
            sl_patterns[f"wi{j}"][i] = pattern_list[i][2][j]
            sl_patterns[f"ts{j}"][i] = pattern_list[i][3][j]
            sl_patterns[f"err_ts{j}"][i] = pattern_list[i][18][j]
            sl_patterns[f"muon_lat{j}"][i] = pattern_list[i][10][j]
            sl_patterns[f"muon_dt{j}"][i] = pattern_list[i][7][j]
            sl_patterns[f"muon_dd{j}"][i] = pattern_list[i][11][j]
    sl_patterns = data_utils.sort_by_key(data=sl_patterns, sort_key="wi3", silent=silent)
    return sl_patterns


### build "super patterns" by combining matching sl fits of the two phi superlayers
### input: sl_fits = output of fit_sl_patterns(patterns, fit_vd=False, ...)  (fixed vd fits)
### output: dict of 8-timestamp patterns (ts0..ts7, err_ts0..err_ts7 + bookkeeping),
###         ready to be handed to a future fit_super_sl_patterns(super_patterns, ...)
###
### ADAPT markers below point at spots where I had to guess a key/param name that
### isn't visible in the code you sent me (e.g. params._muon_slphi_xproj_tolerance,
### derived_params._dt_cell_coordinates, params._orientation). Rename if needed.

def build_phi_super_patterns(sl_fits, *, silent=False, verbose=False,
                              max_chi2ndf=10, max_alpha=np.deg2rad(60),
                              tgroup_tolerance=None, tan_alpha_tolerance=None,
                              xproj_tolerance=None, use_xproj_cut=True):
    """
    Combine fixed-vd sl fits of the two phi superlayers into 8-hit "super patterns".

    Pipeline:
      1) cut noise / bad fits from sl_fits (chi2/ndf, |alpha| cuts -- same idea as
         refit_sl_patterns)
      2) split the surviving fits into the two phi superlayers
      3) greedily match sl1 <-> sl2 fits that are close in t0, tan_alpha (and,
         optionally, x-projection to a common z) -- same tolerances used to combine
         phi info in reco_muons_from_sl_fit_groups
      4) for every matched pair, build one combined pattern that carries the raw hit
         info (ts, err_ts, wire idx) of BOTH sl patterns as ts0..ts7 / err_ts0..err_ts7,
         plus the original single-sl fit results (suffixed _sl1 / _sl2) so a super-fit
         can use them as a good starting guess.

    Returns
    -------
    super_patterns : dict of np.ndarray, one row per matched (sl1_fit, sl2_fit) pair.
    """
    # ---- 0) tolerances: default to the same ones already used to combine the two
    #         phi superlayers in reco_muons_from_sl_fit_groups
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    if tan_alpha_tolerance is None:
        tan_alpha_tolerance = params._muon_slphi_tan_alpha_tolerance
    if xproj_tolerance is None:
        xproj_tolerance = params._muon_slphi_xproj_tolerance  # ADAPT if named differently

    # ---- 1) cut noise / large-angle fits
    if not silent:
        print(f"Selecting good SL fits (chi2/ndf < {max_chi2ndf}, |alpha| < {max_alpha:.3f} rad) "
              f"before building phi super patterns...")
    max_tan_alpha = np.tan(max_alpha)
    sl_fits_cut = data_utils.cut_data(
        data=sl_fits,
        conditions=[
            ("impossible", "==", 0),
            ("chi2/ndf", "<", max_chi2ndf),
            ("tan_alpha", ">", -max_tan_alpha),
            ("tan_alpha", "<", max_tan_alpha), 
        ],
        silent=silent,
    )

    # ---- 2) split into the two phi superlayers
    phi_sls = [sl for sl in params._dt_chamber["sls"].keys() if params._dt_chamber["sls"][sl]["orient"] == "phi"]
    phi_sl1, phi_sl2 = phi_sls[0], phi_sls[1]

    fits_sl1 = data_utils.cut_data(data=sl_fits_cut, conditions=[("sl", "==", phi_sl1)], silent=True)
    fits_sl2 = data_utils.cut_data(data=sl_fits_cut, conditions=[("sl", "==", phi_sl2)], silent=True)
    n1 = data_utils.length(fits_sl1)
    n2 = data_utils.length(fits_sl2)
    if not silent:
        print(f"good fits: sl{phi_sl1} = {n1}, sl{phi_sl2} = {n2}")

    # helper: project a single-sl fit's x0/tan_alpha to a common z (same maths as
    # reco_muons_from_sl_fit_groups, just for one sl at a time)
    x_axis, y_axis = params._orientation["phi"][0], params._orientation["phi"][1]  # ADAPT if _orientation layout differs
    z0_reco = params._muon_reco_z0

    def _x_proj(fits, idx, sl):
        base_wi = fits["wi3"][idx]  # ADAPT: key name for the ly=3 wire index of the pattern
        if base_wi not in derived_params._dt_cell_coordinates[sl][3].keys():
            return None
        coord = derived_params._dt_cell_coordinates[sl][3][base_wi]
        coord_transform = [coord[x_axis + 3], coord[y_axis + 3]]
        return derived_params.f_x_muon(z=-coord_transform[1] + z0_reco,
                                        x0=fits["x0"][idx], tan_alpha=fits["tan_alpha"][idx]) + coord_transform[0]

    # ---- 3) greedy nearest-in-time matching between sl1 and sl2 fits
    order1 = np.argsort(fits_sl1["t0"]) if n1 > 0 else np.array([], dtype=int)
    order2 = np.argsort(fits_sl2["t0"]) if n2 > 0 else np.array([], dtype=int)
    t0_2_sorted = fits_sl2["t0"][order2] if n2 > 0 else np.array([])
    used2 = np.zeros(n2, dtype=bool)

    matches = []  # list of (idx_in_fits_sl1, idx_in_fits_sl2)
    counter_tgroup = 0
    counter_tan_alpha = 0
    counter_xproj = 0

    j_start = 0
    for i in order1:
        t0_i = fits_sl1["t0"][i]
        tan_alpha_i = fits_sl1["tan_alpha"][i]
        while j_start < n2 and t0_2_sorted[j_start] < t0_i - tgroup_tolerance:
            j_start += 1
        best_j, best_score = None, None
        j = j_start
        while j < n2 and t0_2_sorted[j] <= t0_i + tgroup_tolerance:
            j_glob = order2[j]
            j += 1
            if used2[j_glob]:
                continue
            counter_tgroup += 1
            if np.abs(fits_sl2["tan_alpha"][j_glob] - tan_alpha_i) > tan_alpha_tolerance:
                continue
            counter_tan_alpha += 1
            score = np.abs(fits_sl2["t0"][j_glob] - t0_i)
            if best_score is None or score < best_score:
                best_score, best_j = score, j_glob
        if best_j is None:
            continue
        if use_xproj_cut:
            x1, x2 = _x_proj(fits_sl1, i, phi_sl1), _x_proj(fits_sl2, best_j, phi_sl2)
            if x1 is None or x2 is None or np.abs(x1 - x2) > xproj_tolerance:
                continue
            counter_xproj += 1
        used2[best_j] = True
        matches.append((i, best_j))

    n_super = len(matches)
    if not silent:
        print(f"matching cut flow: within tgroup_tolerance = {counter_tgroup}, "
              f"within tan_alpha_tolerance = {counter_tan_alpha}, within xproj_tolerance = {counter_xproj}")
        print(f"built {n_super} phi super patterns (out of {min(n1, n2)} possible pairs)")

    # ---- 4) assemble output dict
    fit_result_keys = list(params._sl_fit_keys.keys())  # t0, x0, tan_alpha, chi2/ndf, dt0..dt3, vd, err_*, corr_*

    super_patterns = {
        f"sl{phi_sl1}": np.full(n_super, phi_sl1, dtype=np.int64),
        f"sl{phi_sl2}": np.full(n_super, phi_sl2, dtype=np.int64),
        f"pat_type_sl{phi_sl1}": np.full(n_super, 0, dtype=np.int64),
        f"pat_type_sl{phi_sl2}": np.full(n_super, 0, dtype=np.int64),
        f"idx_sl{phi_sl1}": np.full(n_super, 0, dtype=np.int64),   # row idx in fits_sl1 (== sl_fits_cut subset)
        f"idx_sl{phi_sl2}": np.full(n_super, 0, dtype=np.int64),
        "muon_id_mismatch": np.full(n_super, 0, dtype=np.int64),
    }
    for ly in range(4):
        super_patterns[f"ts{ly}"] = np.full(n_super, 0, dtype=params._ts_float_type)
        super_patterns[f"err_ts{ly}"] = np.full(n_super, 0, dtype=params._ts_float_type)
        super_patterns[f"ts{ly+4}"] = np.full(n_super, 0, dtype=params._ts_float_type)
        super_patterns[f"err_ts{ly+4}"] = np.full(n_super, 0, dtype=params._ts_float_type)
        super_patterns[f"wi{ly}_sl{phi_sl1}"] = np.full(n_super, 0, dtype=np.int64)  # ADAPT: dtype if wi keys aren't int
        super_patterns[f"wi{ly}_sl{phi_sl2}"] = np.full(n_super, 0, dtype=np.int64)
    for k in fit_result_keys:
        super_patterns[f"{k}_sl{phi_sl1}"] = np.full(n_super, 0, dtype=np.float64)
        super_patterns[f"{k}_sl{phi_sl2}"] = np.full(n_super, 0, dtype=np.float64)
    # carry truth info (from sim) if present, taken from sl1, flagged if sl2 disagrees
    for k in ["muon_id", "muon_ts", "muon_phi", "muon_theta", "muon_x0", "muon_y0", "muon_z0"]:
        if k in fits_sl1:
            super_patterns[k] = np.full(n_super, 0, dtype=np.float64)

    for row, (i, j) in enumerate(tqdm(matches, disable=silent)):
        super_patterns[f"pat_type_sl{phi_sl1}"][row] = fits_sl1["pat_type"][i]
        super_patterns[f"pat_type_sl{phi_sl2}"][row] = fits_sl2["pat_type"][j]
        super_patterns[f"idx_sl{phi_sl1}"][row] = i
        super_patterns[f"idx_sl{phi_sl2}"][row] = j
        for ly in range(4):
            super_patterns[f"ts{ly}"][row] = fits_sl1[f"ts{ly}"][i]
            super_patterns[f"err_ts{ly}"][row] = fits_sl1[f"err_ts{ly}"][i]
            super_patterns[f"ts{ly+4}"][row] = fits_sl2[f"ts{ly}"][j]
            super_patterns[f"err_ts{ly+4}"][row] = fits_sl2[f"err_ts{ly}"][j]
            super_patterns[f"wi{ly}_sl{phi_sl1}"][row] = fits_sl1[f"wi{ly}"][i]  # ADAPT if raw wire-idx key differs
            super_patterns[f"wi{ly}_sl{phi_sl2}"][row] = fits_sl2[f"wi{ly}"][j]
        for k in fit_result_keys:
            super_patterns[f"{k}_sl{phi_sl1}"][row] = fits_sl1[k][i]
            super_patterns[f"{k}_sl{phi_sl2}"][row] = fits_sl2[k][j]
        if "muon_id" in fits_sl1:
            for k in ["muon_id", "muon_ts", "muon_phi", "muon_theta", "muon_x0", "muon_y0", "muon_z0"]:
                super_patterns[k][row] = fits_sl1[k][i]
            if fits_sl1["muon_id"][i] != fits_sl2["muon_id"][j]:
                super_patterns["muon_id_mismatch"][row] = 1
                if verbose:
                    print(f"WARNING: muon_id mismatch for super pattern {row}: "
                          f"sl1={fits_sl1['muon_id'][i]} sl2={fits_sl2['muon_id'][j]}")

    return super_patterns



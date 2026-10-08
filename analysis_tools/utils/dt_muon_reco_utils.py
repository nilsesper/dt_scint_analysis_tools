###########################################
### DT MUONS: super fit (phi view, x-z) + sl fit of the theta superlayer (y-z)
###########################################
# The phi and the theta view share no position information, so a super fit and a theta sl fit are paired in time
# only: the closest unused theta fit with |t0 difference| <= tgroup_tolerance. "n_theta_candidates" > 1 marks muons
# where more than one theta fit was in the window.
# The muon is given at z = params._muon_reco_z0: position (x0, y0), direction (theta, phi), time ts.

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry

# -----------------------------------------

SIM_KEYS = ["sim_id", "sim_ts", "sim_phi", "sim_theta", "sim_x0", "sim_y0", "sim_z0"]

### branches of the dt muons table and their data types
OUTPUT_DTYPES = {}
for key in params._muon_obj_keys:
    OUTPUT_DTYPES[key] = params._muon_obj_keys[key]
OUTPUT_DTYPES["super_fit_idx"] = np.int64
OUTPUT_DTYPES["theta_fit_idx"] = np.int64
OUTPUT_DTYPES["n_theta_candidates"] = np.int64
OUTPUT_DTYPES["delta_t0"] = np.float64
OUTPUT_DTYPES["sim_id_mismatch"] = np.int64

### track of a super fit at z: (position, its error, tan_alpha, its error); the super fit frame is the shared super pattern
# frame shifted by (ref_x, ref_z) of the fit
def super_fit_track(super_fits, i, suffix, z):
    ref_x = geometry.SUPER_FRAME_ORIGIN[0] + super_fits["ref_x" + suffix][i]
    ref_z = geometry.SUPER_FRAME_ORIGIN[1] + super_fits["ref_z" + suffix][i]
    x0, tan_alpha = super_fits["x0" + suffix][i], super_fits["tan_alpha" + suffix][i]
    position = geometry.track_position(z=z - ref_z, x0=x0, tan_alpha=tan_alpha) + ref_x
    err_position = geometry.err_track_position(z=z - ref_z, x0=x0, tan_alpha=tan_alpha, err_x0=super_fits["err_x0" + suffix][i],
                                               err_tan_alpha=super_fits["err_tan_alpha" + suffix][i], corr_x0_tan_alpha=super_fits["corr_x0_tan_alpha" + suffix][i])
    return position, err_position, tan_alpha, super_fits["err_tan_alpha" + suffix][i]

### track of an sl fit at z: (position, its error, tan_alpha, its error)
def sl_fit_track(sl, fits, j, z):
    position = geometry.sl_track_position(sl, fits["wi3"][j], fits["x0"][j], fits["tan_alpha"][j], z)
    err_position = geometry.err_sl_track_position(sl, fits["wi3"][j], fits["x0"][j], fits["tan_alpha"][j], z, err_x0=fits["err_x0"][j],
                                                  err_tan_alpha=fits["err_tan_alpha"][j], corr_x0_tan_alpha=fits["corr_x0_tan_alpha"][j])
    return position, err_position, fits["tan_alpha"][j], fits["err_tan_alpha"][j]

### direction from the slopes in the x-z and y-z planes: tan_alpha_x = tan(theta) cos(phi), tan_alpha_y = tan(theta) sin(phi)
# returns (theta, err_theta, phi, err_phi), phi in [0, 2 pi)
def direction_from_slopes(tan_alpha_x, err_tan_alpha_x, tan_alpha_y, err_tan_alpha_y):
    phi = np.atan2(tan_alpha_y, tan_alpha_x)
    phi = phi - 2 * np.pi * (phi // (2 * np.pi))
    err_phi = np.sqrt((-tan_alpha_x / (tan_alpha_y**2 + tan_alpha_x**2))**2 * err_tan_alpha_x**2
                      + (tan_alpha_y / (tan_alpha_y**2 + tan_alpha_x**2))**2 * err_tan_alpha_y**2)
    theta = np.arctan(tan_alpha_x / np.cos(phi))
    err_theta = np.sqrt(((2 * np.cos(phi)) / (2 * tan_alpha_x**2 + np.cos(2 * phi) + 1))**2 * err_tan_alpha_x**2
                        + ((tan_alpha_x * np.sin(phi)) / (tan_alpha_x**2 + np.cos(phi)**2))**2 * err_phi**2)
    return theta, err_theta, phi, err_phi

### arrival time: mean of the two t0; its error is the error of the mean if they agree within their errors, else their difference
def arrival_time(t0_1, err_t0_1, t0_2, err_t0_2):
    ts = np.mean([t0_1, t0_2])
    difference = np.abs(t0_1 - t0_2)
    if difference <= np.amin([err_t0_1, err_t0_2]):
        return ts, np.sqrt((err_t0_1 / 2)**2 + (err_t0_2 / 2)**2)
    return ts, difference

### pairs of a super fit and a theta sl fit: [(i, j, number of candidates)] (row i of the super fits, row j of the theta fits)
# The super fits are taken in the order of their t0. Each is paired with the closest unused theta fit in time
# (|t0 difference| <= tgroup_tolerance); "number of candidates" counts the unused theta fits in this window.
def pair_super_and_theta_fits(t0_super, t0_theta, tgroup_tolerance):
    n_theta = len(t0_theta)
    order_super = np.argsort(t0_super, kind="stable")
    order_theta = np.argsort(t0_theta, kind="stable")
    used_theta = np.zeros(n_theta, dtype=bool)
    pairs = []
    first_candidate = 0  # position in order_theta of the first theta fit which is not too early
    for i in order_super:
        while first_candidate < n_theta and t0_theta[order_theta[first_candidate]] < t0_super[i] - tgroup_tolerance:
            first_candidate += 1
        closest, closest_distance, n_candidates = None, None, 0
        k = first_candidate
        while k < n_theta and t0_theta[order_theta[k]] <= t0_super[i] + tgroup_tolerance:
            j = order_theta[k]
            k += 1
            if used_theta[j]:
                continue
            n_candidates += 1
            distance = np.abs(t0_theta[j] - t0_super[i])
            if closest is None or distance < closest_distance:
                closest, closest_distance = j, distance
        if closest is None:
            continue
        used_theta[closest] = True
        pairs.append((i, closest, n_candidates))
    return pairs

### muons of one chunk from super fits (after cuts, result branches with suffix) and the sl fits of the theta superlayer
# "super_fit_idx" / "theta_fit_idx": rows of the inputs a muon was built from; "sim_id_mismatch" (simulation): 1 if the
# fits come from different simulated muons
def reco_muons_from_super_fits(super_fits, theta_fits, *, suffix="", tgroup_tolerance=None, verbose=False):
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    theta_sl = dt_chamber_utils.theta_superlayer()
    phi_measures_x = (geometry.measured_axis(dt_chamber_utils.phi_superlayers()[0]) == geometry.X)
    z0 = params._muon_reco_z0
    t0_super = np.asarray(super_fits["t0" + suffix], dtype=np.float64)
    t0_theta = np.asarray(theta_fits["t0"], dtype=np.float64)
    muons = []
    for i, j, n_candidates in pair_super_and_theta_fits(t0_super, t0_theta, tgroup_tolerance):
        if not geometry.is_cell(theta_sl, 3, theta_fits["wi3"][j]):
            continue
        # tracks of both views at z0: (position, its error, tan_alpha, its error)
        phi_view = super_fit_track(super_fits, i, suffix, z0)
        theta_view = sl_fit_track(theta_sl, theta_fits, j, z0)
        if phi_measures_x:
            x_view, y_view = phi_view, theta_view
        else:
            x_view, y_view = theta_view, phi_view
        theta, err_theta, phi, err_phi = direction_from_slopes(x_view[2], x_view[3], y_view[2], y_view[3])
        ts, err_ts = arrival_time(t0_super[i], super_fits["err_t0" + suffix][i], t0_theta[j], theta_fits["err_t0"][j])

        muon = {
            "x0": x_view[0], "y0": y_view[0], "z0": z0, "theta": theta, "phi": phi, "ts": ts,
            "err_x0": x_view[1], "err_y0": y_view[1], "err_z0": 0, "err_theta": err_theta, "err_phi": err_phi, "err_ts": err_ts,
            "super_fit_idx": i, "theta_fit_idx": j, "n_theta_candidates": n_candidates, "delta_t0": t0_theta[j] - t0_super[i],
            "sim_id_mismatch": 0,
        }
        # simulation truth (0 in data)
        for key in SIM_KEYS:
            muon[key] = 0
            if key in super_fits:
                muon[key] = super_fits[key][i]
        if "sim_id" in theta_fits and muon["sim_id"] != theta_fits["sim_id"][j]:
            muon["sim_id_mismatch"] = 1
        if "sim_id_mismatch" in super_fits and super_fits["sim_id_mismatch"][i] != 0:
            muon["sim_id_mismatch"] = 1
        muons.append(muon)
        if verbose:
            print(f"muon: super fit {i} + theta fit {j}: ts = {ts} +- {err_ts}, theta = {theta} +- {err_theta}, phi = {phi} +- {err_phi}")

    reco_muons = {}
    for key in OUTPUT_DTYPES:
        reco_muons[key] = np.full(len(muons), 0, dtype=OUTPUT_DTYPES[key])
    for row in range(len(muons)):
        for key in OUTPUT_DTYPES:
            reco_muons[key][row] = muons[row][key]
    return reco_muons

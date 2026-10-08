###########################################
### DT MUONS: super fit (phi view, x-z) + sl fit of the theta superlayer (y-z)
###########################################
# The phi and the theta view share no position information, so a super fit and a theta sl fit are paired in time
# only: the closest unused theta fit with |t0 difference| <= tgroup_tolerance. "n_theta_candidates" > 1 marks muons
# where more than one theta fit was in the window.
# The muon is given at z = params._muon_reco_z0: position (x0, y0), direction (theta, phi), time ts.

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry, dt_matching_utils

# -----------------------------------------

SIM_KEYS = ["sim_id", "sim_ts", "sim_phi", "sim_theta", "sim_x0", "sim_y0", "sim_z0"]

OUTPUT_DTYPES = params._muon_obj_keys | {
    "super_fit_idx": np.int64, "theta_fit_idx": np.int64, "n_theta_candidates": np.int64, "delta_t0": np.float64, "sim_id_mismatch": np.int64,
}

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
    args = (sl, fits["wi3"][j], fits["x0"][j], fits["tan_alpha"][j], z)
    position = geometry.sl_track_position(*args)
    err_position = geometry.err_sl_track_position(*args, err_x0=fits["err_x0"][j], err_tan_alpha=fits["err_tan_alpha"][j],
                                                           corr_x0_tan_alpha=fits["corr_x0_tan_alpha"][j])
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

### muons of one chunk from super fits (after cuts, result branches with suffix) and the sl fits of the theta superlayer
# "super_fit_idx" / "theta_fit_idx": rows of the inputs a muon was built from; "sim_id_mismatch" (simulation): 1 if the
# fits come from different simulated muons
def reco_muons_from_super_fits(super_fits, theta_fits, *, suffix="", tgroup_tolerance=None, verbose=False):
    if tgroup_tolerance is None:
        tgroup_tolerance = params._muon_tgroup_tolerance
    theta_sl = dt_chamber_utils.theta_superlayer()
    z0 = params._muon_reco_z0
    t0_super = np.asarray(super_fits["t0" + suffix], dtype=np.float64)
    t0_theta = np.asarray(theta_fits["t0"], dtype=np.float64)
    muons = []
    for i, j, n_candidates in dt_matching_utils.match_nearest_in_time(t0_super, t0_theta, tgroup_tolerance):
        if not geometry.is_cell(theta_sl, 3, theta_fits["wi3"][j]):
            continue
        phi_view = super_fit_track(super_fits, i, suffix, z0)
        theta_view = sl_fit_track(theta_sl, theta_fits, j, z0)
        x_view, y_view = (phi_view, theta_view) if geometry.measured_axis(dt_chamber_utils.phi_superlayers()[0]) == 0 else (theta_view, phi_view)
        theta, err_theta, phi, err_phi = direction_from_slopes(x_view[2], x_view[3], y_view[2], y_view[3])
        ts, err_ts = arrival_time(t0_super[i], super_fits["err_t0" + suffix][i], t0_theta[j], theta_fits["err_t0"][j])
        sim = {k: (super_fits[k][i] if k in super_fits else 0) for k in SIM_KEYS}
        theta_from_other_muon = "sim_id" in theta_fits and sim["sim_id"] != theta_fits["sim_id"][j]
        phi_from_two_muons = "sim_id_mismatch" in super_fits and super_fits["sim_id_mismatch"][i] != 0
        mismatch = int(theta_from_other_muon or phi_from_two_muons)
        muons.append({
            "x0": x_view[0], "y0": y_view[0], "z0": z0, "theta": theta, "phi": phi, "ts": ts,
            "err_x0": x_view[1], "err_y0": y_view[1], "err_z0": 0, "err_theta": err_theta, "err_phi": err_phi, "err_ts": err_ts,
            "super_fit_idx": i, "theta_fit_idx": j, "n_theta_candidates": n_candidates, "delta_t0": t0_theta[j] - t0_super[i], "sim_id_mismatch": mismatch,
        } | sim)
        if verbose:
            print(f"muon: super fit {i} + theta fit {j}: ts = {ts} +- {err_ts}, theta = {theta} +- {err_theta}, phi = {phi} +- {err_phi}")
    reco_muons = {k: np.full(len(muons), 0, dtype=dtype) for k, dtype in OUTPUT_DTYPES.items()}
    for row, muon in enumerate(muons):
        for k in OUTPUT_DTYPES.keys():
            reco_muons[k][row] = muon[k]
    return reco_muons

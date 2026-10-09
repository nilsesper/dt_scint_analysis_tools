###########################################
### DT MUONS: super fit (phi view, x-z) + sl fit of the theta superlayer (y-z)
###########################################
# The super fit gives the track in the x-z plane, the sl fit of the theta superlayer the track in the y-z plane.
# The two views share no position information, so they are paired in time only: every super fit gets the closest
# unused theta sl fit with |t0 difference| <= tgroup_tolerance. "n_theta_candidates" > 1 marks muons where more than
# one theta fit was in the time window; "n_candidates_sl1" / "n_candidates_sl3" (from the super fit) the same for the
# sl fits of the two phi superlayers. Unambiguous muons: all three are 1.
# The muon is given at z = params._muon_reco_z0: position (x0, y0), direction (theta, phi), time ts
# (see "3. MUONS" in dt_chamber_utils.py).

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils.dt_fit_utils import SUPER_FIT_SUFFIX, SIM_KEYS

# -----------------------------------------

### branches of the dt muons table and their data types
MUON_KEYS = {}
for key in params._muon_obj_keys:
    MUON_KEYS[key] = params._muon_obj_keys[key]
MUON_KEYS["n_theta_candidates"] = np.int64
for sl in dt_chamber_utils.phi_superlayers():
    MUON_KEYS[f"n_candidates_sl{sl}"] = np.int64
MUON_KEYS["delta_t0"] = np.float64
MUON_KEYS["sim_id_mismatch"] = np.int64
MUON_KEYS["super_fit_row"] = np.int64
for sl in dt_chamber_utils.superlayers():
    MUON_KEYS[f"sl{sl}_fit_row"] = np.int64

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

### pairs (i, j, number of candidates) of super fit i and theta sl fit j
# The super fits are taken in the order of their t0. Each is paired with the closest unused theta fit in time
# (|t0 difference| <= tgroup_tolerance); "number of candidates" counts all theta fits in this window.
def pair_super_and_theta_fits(t0_super, t0_theta, tgroup_tolerance):
    n_theta = len(t0_theta)
    order_super = np.argsort(t0_super)
    order_theta = np.argsort(t0_theta)
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
            n_candidates += 1  # every theta fit in the window counts, also one already taken by another super fit
            if used_theta[j]:
                continue
            distance = np.abs(t0_theta[j] - t0_super[i])
            if closest is None or distance < closest_distance:
                closest, closest_distance = j, distance
        if closest is None:
            continue
        used_theta[closest] = True
        pairs.append((i, closest, n_candidates))
    return pairs

### muons from the super fits and the sl fits of one chunk
# first_super_fit_row, first_sl_fit_row: row numbers (in their files) of the first super fit / sl fit of this chunk;
# the muons get the rows of the fits they were made of: "super_fit_row", "sl<n>_fit_row"
# "sim_id_mismatch" (simulation): 1 if the fits come from different simulated muons
def reco_muons(super_fits, sl_fits, first_super_fit_row, first_sl_fit_row, tgroup_tolerance, verbose=False):
    sfx = SUPER_FIT_SUFFIX
    theta_sl = dt_chamber_utils.theta_superlayer()
    z0 = params._muon_reco_z0

    theta_rows = []
    for row in range(len(sl_fits["sl"])):
        if sl_fits["sl"][row] == theta_sl:
            theta_rows.append(row)
    t0_super = np.asarray(super_fits["t0" + sfx], dtype=np.float64)
    t0_theta = np.asarray(sl_fits["t0"][theta_rows], dtype=np.float64)

    muons = []
    for i, j, n_candidates in pair_super_and_theta_fits(t0_super, t0_theta, tgroup_tolerance):
        theta_row = theta_rows[j]

        # phi view (super fit) at z0: position, its error, slope, its error
        ref_x, ref_z = super_fits["ref_x" + sfx][i], super_fits["ref_z" + sfx][i]
        x = dt_chamber_utils.track_position_in_chamber(ref_x, ref_z, super_fits["x0" + sfx][i], super_fits["tan_alpha" + sfx][i], z0)
        err_x = dt_chamber_utils.err_track_position_in_chamber(ref_z, z0, super_fits["err_x0" + sfx][i], super_fits["err_tan_alpha" + sfx][i],
                                                      super_fits["corr_x0_tan_alpha" + sfx][i])
        tan_alpha_x, err_tan_alpha_x = super_fits["tan_alpha" + sfx][i], super_fits["err_tan_alpha" + sfx][i]

        # theta view (theta sl fit) at z0
        ref_y, ref_z = dt_chamber_utils.wire_position(theta_sl, 3, int(sl_fits["wi3"][theta_row]))
        y = dt_chamber_utils.track_position_in_chamber(ref_y, ref_z, sl_fits["x0"][theta_row], sl_fits["tan_alpha"][theta_row], z0)
        err_y = dt_chamber_utils.err_track_position_in_chamber(ref_z, z0, sl_fits["err_x0"][theta_row], sl_fits["err_tan_alpha"][theta_row],
                                                      sl_fits["corr_x0_tan_alpha"][theta_row])
        tan_alpha_y, err_tan_alpha_y = sl_fits["tan_alpha"][theta_row], sl_fits["err_tan_alpha"][theta_row]

        theta, err_theta, phi, err_phi = direction_from_slopes(tan_alpha_x, err_tan_alpha_x, tan_alpha_y, err_tan_alpha_y)
        ts, err_ts = arrival_time(t0_super[i], super_fits["err_t0" + sfx][i], t0_theta[j], sl_fits["err_t0"][theta_row])

        muon = {
            "x0": x, "y0": y, "z0": z0, "theta": theta, "phi": phi, "ts": ts,
            "err_x0": err_x, "err_y0": err_y, "err_z0": 0, "err_theta": err_theta, "err_phi": err_phi, "err_ts": err_ts,
            "n_theta_candidates": n_candidates, "delta_t0": t0_theta[j] - t0_super[i], "sim_id_mismatch": 0,
            "super_fit_row": first_super_fit_row + i, f"sl{theta_sl}_fit_row": first_sl_fit_row + theta_row,
        }
        for sl in dt_chamber_utils.phi_superlayers():
            muon[f"sl{sl}_fit_row"] = super_fits[f"row_sl{sl}"][i]  # already the row in the sl fits file
            muon[f"n_candidates_sl{sl}"] = super_fits[f"n_candidates_sl{sl}"][i]
        # simulation truth (0 in data)
        for key in SIM_KEYS:
            muon[key] = 0
            if key in super_fits:
                muon[key] = super_fits[key][i]
        if "sim_id" in sl_fits and muon["sim_id"] != sl_fits["sim_id"][theta_row]:
            muon["sim_id_mismatch"] = 1
        if "sim_id_mismatch" in super_fits and super_fits["sim_id_mismatch"][i] != 0:
            muon["sim_id_mismatch"] = 1
        muons.append(muon)
        if verbose:
            print(f"muon: super fit {i} + theta fit {theta_row}: ts = {ts} +- {err_ts}, theta = {theta} +- {err_theta}, phi = {phi} +- {err_phi}")

    dt_muons = {}
    for key in MUON_KEYS:
        dt_muons[key] = np.full(len(muons), 0, dtype=MUON_KEYS[key])
    for row in range(len(muons)):
        for key in MUON_KEYS:
            dt_muons[key][row] = muons[row][key]
    return dt_muons

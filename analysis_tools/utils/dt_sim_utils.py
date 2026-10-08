###########################################
### DT SIMULATION: hits of cosmic muons, noise hits, secondary hits
###########################################
# Simulated hits have the same branches as data hits; the simulation truth is stored in the "sim_..." branches.
# All random numbers come from np.random (seed it for reproducible output).

import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils, dt_dumpfile_utils, dt_geometry_utils as geometry, muon_utils

# -----------------------------------------

### dt hits table with n rows, all branches 0
def empty_dt_hits(n):
    hits = {}
    for key in params._htg_keys:
        hits[key] = np.full(n, 0, dtype=params._htg_keys[key])
    for key in params._dt_mapping_keys:
        hits[key] = np.full(n, 0, dtype=params._dt_mapping_keys[key])
    for key in params._dt_other_keys:
        hits[key] = np.full(n, 0, dtype=params._dt_other_keys[key])
    return hits

### fill the branches which follow from the cell (readout channel) and from the timestamp (oc, bx, tdc, err_ts)
def set_readout_and_time_keys(hits):
    hits["err_ts"][:] = dt_dumpfile_utils.DT_HIT_TS_UNCERTAINTY
    ts_int = np.uint64(np.round(hits["ts"], 0))
    hits["oc"][:] = (ts_int % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp
    hits["bx"][:] = (ts_int % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp
    hits["tdc"][:] = (ts_int % derived_params._bx_to_timestamp) // derived_params._tdc_to_timestamp
    for i in range(len(hits["ts"])):
        readout_keys = dt_chamber_utils.readout_keys_of_cell(hits["sl"][i], hits["ly"][i], hits["wi"][i])
        for key in readout_keys:
            hits[key][i] = readout_keys[key]

### dt hits table from a list of hits, each a dict {"sl", "ly", "wi", "ts", and optionally simulation truth keys}
def dt_hits_from_list(hit_list):
    hits = empty_dt_hits(len(hit_list))
    for i in range(len(hit_list)):
        for key in hit_list[i]:
            hits[key][i] = hit_list[i][key]
    set_readout_and_time_keys(hits)
    return hits

def sort_by_time(hits):
    order = np.argsort(hits["ts"])
    sorted_hits = {}
    for key in hits:
        sorted_hits[key] = hits[key][order]
    return sorted_hits

### several dt hits tables in one, sorted by time
def merge_and_sort_by_time(parts):
    merged = {}
    for key in parts[0]:
        arrays = []
        for part in parts:
            arrays.append(part[key])
        merged[key] = np.concatenate(arrays).astype(parts[0][key].dtype, copy=False)
    return sort_by_time(merged)

### the wire of layer (sl, ly) whose cell contains the point (x, y), or None
def cell_at(sl, ly, x, y):
    for wi in dt_chamber_utils.wires(sl, ly):
        box = geometry.cell(sl, ly, wi)
        inside_x = box["low"][geometry.X] <= x < box["high"][geometry.X]
        inside_y = box["low"][geometry.Y] <= y < box["high"][geometry.Y]
        if inside_x and inside_y:
            return wi
    return None

### hits of straight muon tracks: every layer the track crosses gives a hit (with probability params._dt_cell_efficiency)
# at the time ts(muon) + drift distance / drift velocity
# noise_ampl > 0: gaussian time jitter (sigma in ts units) of every hit
# sys_miscalib_ampl > 0: constant gaussian time offset (sigma in ts units) of every cell
def hits_from_muons(muons, *, noise_ampl=0, sys_miscalib_ampl=0):
    miscalibration = dt_chamber_utils.chamber_map(0)
    if sys_miscalib_ampl > 0:
        for sl in dt_chamber_utils.superlayers():
            for ly in dt_chamber_utils.layers(sl):
                for wi in dt_chamber_utils.wires(sl, ly):
                    miscalibration[sl][ly][wi] = np.random.normal(loc=0, scale=1) * sys_miscalib_ampl

    hit_list = []
    for sl in dt_chamber_utils.superlayers():
        axis = geometry.measured_axis(sl)
        for ly in dt_chamber_utils.layers(sl):
            x, y, _ = muon_utils.propagate_muons(muons=muons, z=geometry.layer_z(sl, ly))
            for i in range(len(muons["x0"])):
                wi = cell_at(sl, ly, x[i], y[i])
                if wi is None:
                    continue
                if np.random.uniform(low=0, high=1) > params._dt_cell_efficiency:
                    continue
                if axis == geometry.X:
                    track_position = x[i]
                    tan_alpha = np.tan(muons["theta"][i]) * np.cos(muons["phi"][i])
                else:
                    track_position = y[i]
                    tan_alpha = np.tan(muons["theta"][i]) * np.sin(muons["phi"][i])
                wire_position = geometry.cell(sl, ly, wi)["center"][axis]
                jitter = 0
                if noise_ampl > 0:
                    jitter = np.random.normal(loc=0, scale=1) * noise_ampl
                drift_distance = np.float64(np.clip(np.abs(track_position - wire_position), a_min=0, a_max=params._dt_cell_width / 2))
                drift_time = np.float64(drift_distance / derived_params._drift_velocity_mm_per_timestamp) + jitter + miscalibration[sl][ly][wi]
                laterality = 1 if track_position >= wire_position else -1
                hit_list.append({
                    "sl": sl, "ly": ly, "wi": wi, "ts": muons["ts"][i] + drift_time,
                    "sim_ts": muons["ts"][i], "sim_dt": drift_time, "sim_dd": drift_distance, "sim_id": muons["sim_id"][i],
                    "sim_lat": laterality, "sim_tan_alpha": tan_alpha, "sim_vd": derived_params._drift_velocity_mm_per_timestamp,
                    "sim_x0": muons["x0"][i], "sim_y0": muons["y0"][i], "sim_z0": muons["z0"][i], "sim_theta": muons["theta"][i], "sim_phi": muons["phi"][i],
                })
    return sort_by_time(dt_hits_from_list(hit_list))

### random noise hits in every cell (rate ref_cell_noise_rate in Hz) during ts_range = [ts_min, ts_max], added to the hits
def add_noise(hits, *, ts_range, ref_cell_noise_rate):
    noise_rate = ref_cell_noise_rate * 0.78e-9  # per ts unit
    parts = []
    for sl in derived_params._dt_inverted_remap_table:
        for ly in derived_params._dt_inverted_remap_table[sl]:
            for wi in dt_chamber_utils.wires(sl, ly):
                n_noise = np.random.poisson(lam=noise_rate * (ts_range[1] - ts_range[0]))
                times_between_hits = np.random.exponential(1.0 / noise_rate, n_noise)
                noise_ts = ts_range[0] + np.cumsum(times_between_hits)
                noise_list = []
                for ts in noise_ts:
                    noise_list.append({"sl": sl, "ly": ly, "wi": wi, "ts": ts})
                parts.append(dt_hits_from_list(noise_list))
    parts.append(hits)
    return merge_and_sort_by_time(parts)

### secondary hits (delta rays, photo ionisation): with probability secondary_hit_probability a hit is followed by a
# second hit in the same cell, uniformly distributed in the time window secondary_hit_window after it
def add_secondary_hits(hits, *, secondary_hit_window, secondary_hit_probability):
    secondary_list = []
    for i in range(len(hits["ts"])):
        if np.random.uniform(low=0, high=1) < secondary_hit_probability:
            ts = hits["ts"][i] + np.random.uniform(low=secondary_hit_window[0], high=secondary_hit_window[1])
            secondary_list.append({"sl": hits["sl"][i], "ly": hits["ly"][i], "wi": hits["wi"][i], "ts": ts})
    return merge_and_sort_by_time([hits, dt_hits_from_list(secondary_list)])

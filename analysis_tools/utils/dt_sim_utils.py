###########################################
### DT SIMULATION: hits of cosmic muons, noise hits, secondary hits
###########################################
# Simulated hits have the same branches as data hits; the simulation truth is stored in the "sim_..." branches.
# All random numbers come from np.random (seed it for reproducible output).

import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils, dt_dumpfile_utils, dt_geometry_utils as geometry, muon_utils

# -----------------------------------------

### empty dt hits table with n rows (all branches 0)
def empty_dt_hits(n):
    return {k: np.full(n, 0, dtype=v) for k, v in (params._htg_keys | params._dt_mapping_keys | params._dt_other_keys).items()}

### dt hits table from the cell and time of every hit: adds err_ts, the readout keys and oc / bx / tdc of the timestamp
def dt_hits_from_cells(sl, ly, wi, ts, truth=None):
    hits = empty_dt_hits(len(ts))
    hits["sl"][:], hits["ly"][:], hits["wi"][:], hits["ts"][:] = sl, ly, wi, ts
    hits["err_ts"][:] = dt_dumpfile_utils.DT_HIT_TS_UNCERTAINTY
    ts_int = np.uint64(np.round(hits["ts"], 0))
    hits["oc"][:] = (ts_int % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp
    hits["bx"][:] = (ts_int % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp
    hits["tdc"][:] = (ts_int % derived_params._bx_to_timestamp) // derived_params._tdc_to_timestamp
    for i in range(len(ts)):
        for k, value in dt_chamber_utils.readout_keys_of_cell(hits["sl"][i], hits["ly"][i], hits["wi"][i]).items():
            hits[k][i] = value
    for k, values in (truth or {}).items():
        hits[k][:] = values
    return hits

def merge_and_sort_by_time(parts):
    merged = {k: np.concatenate([part[k] for part in parts]).astype(parts[0][k].dtype, copy=False) for k in parts[0].keys()}
    order = np.argsort(merged["ts"])
    return {k: v[order] for k, v in merged.items()}

### the wire of layer (sl, ly) whose cell contains the point (x, y), or None
def cell_at(sl, ly, x, y):
    for wi in dt_chamber_utils.wires(sl, ly):
        cell = geometry.cell(sl, ly, wi)
        if cell.low[geometry.X] <= x < cell.high[geometry.X] and cell.low[geometry.Y] <= y < cell.high[geometry.Y]:
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
        measures_x = geometry.measured_axis(sl) == geometry.X
        for ly in dt_chamber_utils.layers(sl):
            x, y, _ = muon_utils.propagate_muons(muons=muons, z=geometry.layer_z(sl, ly))
            for i in range(len(muons["x0"])):
                wi = cell_at(sl, ly, x[i], y[i])
                if wi is None or np.random.uniform(low=0, high=1) > params._dt_cell_efficiency:
                    continue
                track_position = x[i] if measures_x else y[i]
                wire_position = geometry.cell(sl, ly, wi).center[geometry.measured_axis(sl)]
                jitter = np.random.normal(loc=0, scale=1) * noise_ampl if noise_ampl > 0 else 0
                drift_distance = np.float64(np.clip(np.abs(track_position - wire_position), a_min=0, a_max=params._dt_cell_width / 2))
                drift_time = np.float64(drift_distance / derived_params._drift_velocity_mm_per_timestamp) + jitter + miscalibration[sl][ly][wi]
                theta, phi = muons["theta"][i], muons["phi"][i]
                hit_list.append({
                    "sl": sl, "ly": ly, "wi": wi, "ts": muons["ts"][i] + drift_time,
                    "sim_ts": muons["ts"][i], "sim_dt": drift_time, "sim_dd": drift_distance, "sim_id": muons["sim_id"][i],
                    "sim_lat": 1 if track_position >= wire_position else -1,
                    "sim_tan_alpha": np.tan(theta) * np.cos(phi) if measures_x else np.tan(theta) * np.sin(phi),
                    "sim_vd": derived_params._drift_velocity_mm_per_timestamp,
                    "sim_x0": muons["x0"][i], "sim_y0": muons["y0"][i], "sim_z0": muons["z0"][i], "sim_theta": theta, "sim_phi": phi,
                })
    columns = {k: [hit[k] for hit in hit_list] for k in (hit_list[0].keys() if hit_list else ["sl", "ly", "wi", "ts"])}
    hits = dt_hits_from_cells(columns.pop("sl"), columns.pop("ly"), columns.pop("wi"), columns.pop("ts"), truth=columns)
    order = np.argsort(hits["ts"])
    return {k: v[order] for k, v in hits.items()}

### random noise hits in every cell (rate ref_cell_noise_rate in Hz) during ts_range = [ts_min, ts_max], added to the hits
def add_noise(hits, *, ts_range, ref_cell_noise_rate):
    noise_rate = ref_cell_noise_rate * 0.78e-9  # per ts unit
    parts = []
    for sl in derived_params._dt_inverted_remap_table.keys():
        for ly in derived_params._dt_inverted_remap_table[sl].keys():
            for wi in dt_chamber_utils.wires(sl, ly):
                n_noise = np.random.poisson(lam=noise_rate * (ts_range[1] - ts_range[0]))
                noise_ts = ts_range[0] + np.cumsum(np.random.exponential(1.0 / noise_rate, n_noise))
                parts.append(dt_hits_from_cells(sl, ly, wi, noise_ts))
    return merge_and_sort_by_time(parts + [hits])

### secondary hits (delta rays, photo ionisation): with probability secondary_hit_probability a hit is followed by a
# second hit in the same cell, uniformly distributed in the time window secondary_hit_window after it
def add_secondary_hits(hits, *, secondary_hit_window, secondary_hit_probability):
    cells, times = [], []
    for i in range(len(hits["ts"])):
        if np.random.uniform(low=0, high=1) < secondary_hit_probability:
            cells.append((hits["sl"][i], hits["ly"][i], hits["wi"][i]))
            times.append(hits["ts"][i] + np.random.uniform(low=secondary_hit_window[0], high=secondary_hit_window[1]))
    secondary = dt_hits_from_cells([c[0] for c in cells], [c[1] for c in cells], [c[2] for c in cells], np.array(times, dtype=np.float64))
    return merge_and_sort_by_time([hits, secondary])

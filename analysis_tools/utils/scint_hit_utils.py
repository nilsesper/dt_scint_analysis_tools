###########################################
### SCINTILLATOR: strips, raw scint hits (single SiPMs), scint hits (strip hits), correlation with dt muons
###########################################
# The scintillator (hodoscope below the chamber) has 2 layers of 16 strips; every strip has a SiPM at each end.
#
#   raw scint hit = hit of one SiPM                     (ly, st, sipm, ts)
#   scint hit     = both SiPMs of a strip hit together  (ly, st, ts = mean of the two SiPM times)
#
# Positions are in the chamber frame of the dt chamber (see dt_chamber_utils.py); the scintillator is about 1 m
# below SL 1. Layer 0 measures y (strips along x, "theta" orientation), layer 1 measures x (strips along y, "phi").
#
#        y ^   layer 1 (phi): strips along y            layer 0 (theta): strips along x
#          |   | | | | | | |   st 15 ... st 0           ====== st 0
#          |   | | | | | | |   (x decreases with st)    ====== st 1   (y decreases with st)
#          |   | | | | | | |                            ====== ...
#          +---------------> x
#
# A strip is a box {"low": [x, y, z], "high": [x, y, z], "center": [x, y, z]} like the dt cells.

import copy
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils, timestamp_utils

# -----------------------------------------
# strips
# -----------------------------------------

def layers():
    return sorted(params._scintillator["lys"].keys())

def strips(ly):
    return list(range(params._scintillator["lys"][ly]["n_sts"]))

def orientation(ly):
    return params._scintillator["lys"][ly]["orient"]

### axis measured by a layer: X for phi, Y for theta
def measured_axis(ly):
    return dt_chamber_utils.MEASURED_AXIS[orientation(ly)]

### box of strip st of layer ly: strip 0 at "strip_0" (lower corner), strip st shifted by st * "pitch" along the measured axis
def strip_box(ly, st):
    layer = params._scintillator["lys"][ly]
    low = list(layer["strip_0"])
    axis = measured_axis(ly)
    low[axis] = low[axis] + st * layer["pitch"]
    return dt_chamber_utils.corner_box(low, layer["strip_size"])

### z of the middle of the strips of a layer
def layer_z(ly):
    return strip_box(ly, 0)["center"][dt_chamber_utils.Z]

### box around all strips of the scintillator
def scintillator_box():
    low = [np.inf, np.inf, np.inf]
    high = [-np.inf, -np.inf, -np.inf]
    for ly in layers():
        for st in strips(ly):
            box = strip_box(ly, st)
            for axis in range(3):
                low[axis] = min(low[axis], box["low"][axis])
                high[axis] = max(high[axis], box["high"][axis])
    center = []
    for axis in range(3):
        center.append((low[axis] + high[axis]) / 2)
    return {"low": low, "high": high, "center": center}

### signed distance of a position (along the measured axis of the layer) to strip st: 0 inside the strip,
# < 0 below its lower edge, > 0 above its upper edge
def distance_to_strip(ly, st, position):
    box = strip_box(ly, st)
    axis = measured_axis(ly)
    if position < box["low"][axis]:
        return position - box["low"][axis]
    if position > box["high"][axis]:
        return position - box["high"][axis]
    return 0.0

# -----------------------------------------
# raw scint hits
# -----------------------------------------

### extract raw scint hits (single SiPM hits) from the hit data of one block of the dumpfile
# keeps the words of the SiPM channels (params._raw_scint_mapping), adds ly, st, sipm, ch_id and the timestamp
# overflow_state: orbit counter overflows of the earlier blocks, see timestamp_utils.add_timestamp
# returns the raw scint hits sorted by timestamp
def extract_raw_scint_hits(hits, *, overflow_state=None, silent=True):
    tmp_hits = copy.deepcopy(hits)
    n_hits = len(tmp_hits["ch"])
    scint_mask = np.full(n_hits, False, dtype=bool)
    for ro_ch in derived_params._raw_scint_ro_chs:
        tmp_mask = np.isin(tmp_hits["ro_ch"], [ro_ch])
        tmp_mask &= np.isin(tmp_hits["ch"], list(derived_params._raw_scint_remap_table[ro_ch].keys()))
        scint_mask |= tmp_mask
    for k in tmp_hits.keys():
        tmp_hits[k] = tmp_hits[k][scint_mask]
    n_scint_hits = len(tmp_hits["ch"])
    if not silent: print(f"Found {n_scint_hits} raw scint hits in {n_hits} hits.")
    for k in ["ly", "st", "sipm", "ch_id"]:
        tmp_hits[k] = np.zeros(n_scint_hits, dtype=params._raw_scint_keys[k])
    for i in range(n_scint_hits):
        sipm_info = derived_params._raw_scint_remap_table[tmp_hits["ro_ch"][i]][tmp_hits["ch"][i]]
        for k in ["ly", "st", "sipm", "ch_id"]:
            tmp_hits[k][i] = sipm_info[k]
    tmp_hits = timestamp_utils.add_timestamp(hits=tmp_hits, silent=silent, overflow_state=overflow_state)
    return tmp_hits

### dead time of every SiPM channel: a hit is dropped if it comes less than dead_time after the previous hit of the
# same channel (the previous hit counts also if it was dropped itself); the first hit of a channel is dropped as well.
# raw_hits: sorted by timestamp. last_ts_of_channel: {(ly, st, sipm): ts of the last hit}, filled here, so that it can
# be passed on to the next chunk of hits. Returns a list of the indices of the hits which are kept.
def dead_time_kept_indices(raw_hits, dead_time, last_ts_of_channel):
    kept = []
    for i in range(len(raw_hits["ts"])):
        channel = (int(raw_hits["ly"][i]), int(raw_hits["st"][i]), int(raw_hits["sipm"][i]))
        ts = raw_hits["ts"][i]
        if channel in last_ts_of_channel and ts - last_ts_of_channel[channel] >= dead_time:
            kept.append(i)
        last_ts_of_channel[channel] = ts
    return kept

# -----------------------------------------
# scint hits: coincidence of the two SiPMs of a strip
# -----------------------------------------

### groups of raw scint hits (sorted by timestamp): a hit belongs to the group of the hit before it if their time
# difference is below ts_tolerance. Returns a list of groups, each group a list of hit indices.
def group_raw_scint_hits(raw_hits, ts_tolerance):
    groups = []
    current_group = []
    for i in range(len(raw_hits["ts"])):
        if len(current_group) > 0 and raw_hits["ts"][i] - raw_hits["ts"][i - 1] >= ts_tolerance:
            groups.append(current_group)
            current_group = []
        current_group.append(i)
    if len(current_group) > 0:
        groups.append(current_group)
    return groups

### strip hits of one group of raw scint hits; returns a list of rows (dicts with the keys of params._scint_hit_keys)
# Only the first hit of every SiPM channel in the group is used. In every layer a strip is hit if both of its SiPMs
# have a hit within params._scintillator_ts_acceptance_interval (or the unmasked SiPM, for a strip in
# params._scint_masked_sipms).
# isolation_criterion: the layer must have exactly one hit strip and no other SiPM hit, else no strip hit is taken from it.
# raw_row: row number (in the raw scint hits file) of every raw hit.
def strip_hits_of_group(group, raw_hits, raw_row, isolation_criterion):
    # first hit of every SiPM channel: {(ly, st, sipm): hit index}
    first_hit_of_channel = {}
    for i in group:
        channel = (int(raw_hits["ly"][i]), int(raw_hits["st"][i]), int(raw_hits["sipm"][i]))
        if channel not in first_hit_of_channel:
            first_hit_of_channel[channel] = i
    n_raw_hits_group = len(first_hit_of_channel)

    rows = []
    for ly in layers():
        # hits of every strip of this layer: {st: {sipm: hit index}}
        hits_of_strip = {}
        n_hits_layer = 0
        for (hit_ly, st, sipm), i in first_hit_of_channel.items():
            if hit_ly != ly:
                continue
            n_hits_layer += 1
            if st not in hits_of_strip:
                hits_of_strip[st] = {}
            hits_of_strip[st][sipm] = i

        # strips with a hit: both SiPMs, or the unmasked SiPM of a strip with a masked SiPM
        hit_strips = []
        for st in sorted(hits_of_strip.keys()):
            masked_sipm = params._scint_masked_sipms.get(ly, {}).get(st, None)
            if masked_sipm is None and len(hits_of_strip[st]) == 2:
                hit_strips.append(st)
            if masked_sipm is not None and (1 - masked_sipm) in hits_of_strip[st]:
                hit_strips.append(st)
        if len(hit_strips) == 0:
            continue
        if isolation_criterion and (n_hits_layer > 2 or len(hit_strips) > 1):
            continue

        for st in hit_strips:
            masked_sipm = params._scint_masked_sipms.get(ly, {}).get(st, None)
            if masked_sipm is None:
                i0, i1 = hits_of_strip[st][0], hits_of_strip[st][1]
                ts0, ts1 = raw_hits["ts"][i0], raw_hits["ts"][i1]
                if np.abs(ts0 - ts1) > params._scintillator_ts_acceptance_interval:
                    continue
                row = {"ly": ly, "st": st, "ts": np.mean([ts0, ts1]),
                       "err_ts": np.sqrt(raw_hits["err_ts"][i0]**2 + raw_hits["err_ts"][i1]**2),
                       "sipm_delta_ts": np.abs(ts0 - ts1), "sipm_delta_ts_signed": ts0 - ts1,
                       "raw_row_sipm0": raw_row[i0], "raw_row_sipm1": raw_row[i1]}
            else:
                i = hits_of_strip[st][1 - masked_sipm]
                row = {"ly": ly, "st": st, "ts": raw_hits["ts"][i], "err_ts": raw_hits["err_ts"][i],
                       "sipm_delta_ts": 0, "sipm_delta_ts_signed": 0, "raw_row_sipm0": -1, "raw_row_sipm1": -1}
                row[f"raw_row_sipm{1 - masked_sipm}"] = raw_row[i]
            row["n_raw_hits_group"] = n_raw_hits_group
            rows.append(row)
    return rows

### list of strip hit rows (from strip_hits_of_group) -> scint hits table, sorted by timestamp
def scint_hits_table(rows):
    scint_hits = {}
    for key in params._scint_hit_keys:
        scint_hits[key] = np.zeros(len(rows), dtype=params._scint_hit_keys[key])
    for i in range(len(rows)):
        for key in params._scint_hit_keys:
            scint_hits[key][i] = rows[i][key]
    order = np.argsort(scint_hits["ts"], kind="stable")
    for key in scint_hits:
        scint_hits[key] = scint_hits[key][order]
    return scint_hits

# -----------------------------------------
# correlation of scint hits and dt muons
# -----------------------------------------

### pairs (scint hit index, dt muon index) of scint hits and dt muons at the same time
# Every scint hit (in time order) looks for dt muons with |ts_dt - ts_scint| <= window. It is paired only if there is
# exactly one dt muon in the window and this muon is not paired yet with an earlier scint hit.
def pair_scint_hits_and_dt_muons(ts_scint, ts_dt, window):
    order_dt = np.argsort(ts_dt, kind="stable")
    sorted_ts_dt = ts_dt[order_dt]
    used_dt = np.zeros(len(ts_dt), dtype=bool)
    pairs = []
    for i in np.argsort(ts_scint, kind="stable"):
        first = np.searchsorted(sorted_ts_dt, ts_scint[i] - window, side="left")
        after_last = np.searchsorted(sorted_ts_dt, ts_scint[i] + window, side="right")
        if after_last - first != 1:
            continue
        j = order_dt[first]
        if used_dt[j]:
            continue
        used_dt[j] = True
        pairs.append((i, j))
    return pairs

### correlation table: one row per pair of scint hit and dt muon
# delta_ts = ts scint - ts dt; x_at_strip, y_at_strip: the dt muon track at the z of the hit strip;
# dist_to_strip: distance of the track to the hit strip along the axis the layer measures (0: track crosses the strip)
def correlation_rows(scint_hits, dt_muons, pairs, first_scint_row=0, first_dt_row=0):
    n = len(pairs)
    keys = {"scint_hit_row": np.int64, "dt_muon_row": np.int64, "ly": np.uint8, "st": np.uint8,
            "ts_scint": np.float64, "ts_dt": np.float64, "delta_ts": np.float64, "err_delta_ts": np.float64,
            "x_at_strip": np.float64, "y_at_strip": np.float64, "dist_to_strip": np.float64}
    corr = {}
    for key in keys:
        corr[key] = np.zeros(n, dtype=keys[key])
    for p in range(n):
        i, j = pairs[p]
        ly, st = int(scint_hits["ly"][i]), int(scint_hits["st"][i])
        z = layer_z(ly)
        muon = (dt_muons["x0"][j], dt_muons["y0"][j], dt_muons["z0"][j], dt_muons["theta"][j], dt_muons["phi"][j])
        x = dt_chamber_utils.muon_track_position("phi", z, *muon)
        y = dt_chamber_utils.muon_track_position("theta", z, *muon)
        corr["scint_hit_row"][p] = first_scint_row + i
        corr["dt_muon_row"][p] = first_dt_row + j
        corr["ly"][p], corr["st"][p] = ly, st
        corr["ts_scint"][p] = scint_hits["ts"][i]
        corr["ts_dt"][p] = dt_muons["ts"][j]
        corr["delta_ts"][p] = np.float64(scint_hits["ts"][i]) - np.float64(dt_muons["ts"][j])
        corr["err_delta_ts"][p] = np.sqrt(scint_hits["err_ts"][i]**2 + dt_muons["err_ts"][j]**2)
        corr["x_at_strip"][p], corr["y_at_strip"][p] = x, y
        corr["dist_to_strip"][p] = distance_to_strip(ly, st, [x, y][measured_axis(ly)])
    return corr

###########################################
### DT PATTERN SEARCH: 4 hits of one superlayer from one muon
###########################################
# The hits of a superlayer are processed in time order. Every cell remembers its last hit. With each new hit,
# every pattern shape (params._dt_sl_patterns: wire offsets per layer) is placed so that it contains the new hit;
# a pattern is found if all 4 cells of the shape have a hit and these hits are close enough in time.
#
# Parallel search: the hits of a superlayer are split into pieces in time. Every piece also gets the hits of
# the time window before it ("lookback"), which only fill the cells, so it finds exactly the patterns which
# the search over all hits finds for the hits of the piece.

import multiprocessing
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils

# -----------------------------------------

def ts_window(wide):
    return params._dt_sl_patterns_ts_window_fit_vd if wide else params._dt_sl_patterns_ts_window

def pattern_shapes():
    return [(pat_type, pat_name, np.array(shape["rel_wis"])) for pat_type, (pat_name, shape) in enumerate(params._dt_sl_patterns.items())]

### one found pattern as a row of the sl patterns table (params._sl_pattern_keys)
# the simulation truth ("sim_...") is taken from the hits, it is 0 in data; new_hit: the hit which completed the pattern
def pattern_row(sl, pat_type, pat_name, wires, hits, hit_idx, new_hit, *, only_single_muon_patterns):
    def per_layer(key):
        return [hits[key][i] for i in hit_idx]
    top = hit_idx[3]  # layer 3: reference layer of the local frame of the superlayer
    row = {"sl": sl, "pat_type": pat_type, "sim_id": hits["sim_id"][hit_idx[0]], "sim_ts": hits["sim_ts"][new_hit], "sim_lat_id": 0,
           "sim_x0_loc": hits["sim_dd"][top] * hits["sim_lat"][top], "sim_tan_alpha": hits["sim_tan_alpha"][top]}
    for key in ["sim_x0", "sim_y0", "sim_z0", "sim_theta", "sim_phi", "sim_vd"]:
        row[key] = hits[key][top]
    for ly, (wi, ts, err_ts, lat, dt, dd) in enumerate(zip(wires, per_layer("ts"), per_layer("err_ts"), per_layer("sim_lat"), per_layer("sim_dt"), per_layer("sim_dd"))):
        row |= {f"wi{ly}": wi, f"ts{ly}": ts, f"err_ts{ly}": err_ts, f"sim_lat{ly}": lat, f"sim_dt{ly}": dt, f"sim_dd{ly}": dd}
    if only_single_muon_patterns:
        lateralities = params._dt_sl_patterns[pat_name]["laterality"]
        layer_lats = per_layer("sim_lat")
        if layer_lats not in lateralities:
            raise Exception(f"Missing laterality {layer_lats} for pattern {pat_type} in params !!!")
        row["sim_lat_id"] = lateralities.index(layer_lats)
    return row

### pattern search in the hits of one superlayer (sorted by time)
# n_lookback: the first n_lookback hits only fill the cells, no patterns are searched for them
# only_single_muon_patterns (simulation): drop patterns made of hits of different simulated muons
def find_patterns_in_superlayer(sl, hits, *, wide_ts_window=False, n_lookback=0, only_single_muon_patterns=False, verbose=False):
    max_ts_difference = ts_window(wide_ts_window)
    shapes = pattern_shapes()
    wire_ranges = [(min(dt_chamber_utils.wires(sl, ly)), max(dt_chamber_utils.wires(sl, ly))) for ly in range(4)]
    last_hit_of_cell = np.full((4, 256), -1, dtype=np.int64)
    rows = []
    for i in range(len(hits["ts"])):
        ly, wi = int(hits["ly"][i]), int(hits["wi"][i])
        last_hit_of_cell[ly, wi] = i
        if i < n_lookback:
            continue
        for pat_type, pat_name, rel_wis in shapes:
            wires = wi - rel_wis[ly] + rel_wis
            if any(wires[l] < lo or wires[l] > hi for l, (lo, hi) in enumerate(wire_ranges)):
                continue
            hit_idx = last_hit_of_cell[np.arange(4), wires]
            if np.any(hit_idx < 0):
                continue
            ts = hits["ts"][hit_idx]
            if np.amax(ts) - np.amin(ts) > max_ts_difference:
                continue
            if only_single_muon_patterns and len(set(hits["sim_id"][hit_idx])) > 1:
                continue
            if verbose:
                print(f"found pattern: sl={sl}, pattern {pat_name}, wires={wires}, ts={ts}")
            rows.append(pattern_row(sl, pat_type, pat_name, np.uint8(wires), hits, hit_idx, i, only_single_muon_patterns=only_single_muon_patterns))
    return rows

def _find_patterns_in_piece(job):
    sl, hits, kwargs = job
    return find_patterns_in_superlayer(sl, hits, **kwargs)

### pieces of the hits of one superlayer for the parallel search: [(hits of the piece incl. lookback, n_lookback)]
def split_in_time(hits, n_pieces, max_ts_difference):
    n_hits = len(hits["ts"])
    ts = np.asarray(hits["ts"], dtype=np.float64)
    bounds = np.linspace(0, n_hits, n_pieces + 1).astype(int)
    pieces = []
    for first, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= first:
            continue
        lookback_start = int(np.searchsorted(ts, ts[first] - max_ts_difference, side="left")) if first > 0 else 0
        pieces.append(({k: v[lookback_start:stop] for k, v in hits.items()}, first - lookback_start))
    return pieces

### find the patterns of all superlayers; returns the sl patterns table, sorted by the wire of layer 3
# pool: an open multiprocessing pool for the parallel search (or n_proc > 1 to open one for this call)
def find_sl_patterns(hits, *, wide_ts_window=False, only_single_muon_patterns=False, n_proc=1, pool=None, min_hits_per_piece=2000, verbose=False):
    kwargs = {"wide_ts_window": wide_ts_window, "only_single_muon_patterns": only_single_muon_patterns, "verbose": verbose}
    parallel = (n_proc > 1 or pool is not None) and not verbose
    n_workers = n_proc if pool is None else pool._processes
    jobs = []
    for sl in dt_chamber_utils.superlayers():
        in_sl = hits["sl"] == sl
        sl_hits = {k: v[in_sl] for k, v in hits.items()}
        time_order = np.argsort(sl_hits["ts"])
        sl_hits = {k: v[time_order] for k, v in sl_hits.items()}
        n_pieces = max(1, min(n_workers, len(sl_hits["ts"]) // max(1, min_hits_per_piece))) if parallel else 1
        if n_pieces == 1:
            jobs.append((sl, sl_hits, kwargs))
        else:
            jobs += [(sl, piece, kwargs | {"n_lookback": n_lookback}) for piece, n_lookback in split_in_time(sl_hits, n_pieces, ts_window(wide_ts_window))]
    if parallel and len(jobs) > 1:
        if pool is not None:
            results = pool.map(_find_patterns_in_piece, jobs)
        else:
            with multiprocessing.Pool(n_proc) as new_pool:
                results = new_pool.map(_find_patterns_in_piece, jobs)
    else:
        results = [_find_patterns_in_piece(job) for job in jobs]
    rows = [row for result in results for row in result]

    patterns = {k: np.full(len(rows), 0, dtype=v) for k, v in params._sl_pattern_keys.items()}
    for i, row in enumerate(rows):
        for k, value in row.items():
            patterns[k][i] = value
    order = np.argsort(patterns["wi3"])
    return {k: v[order] for k, v in patterns.items()}

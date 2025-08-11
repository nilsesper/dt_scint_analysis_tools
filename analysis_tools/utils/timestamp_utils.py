###########################################
### TIMESTAMP UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### add timestamp (integer value concatenating all existing timestamp keys oc,bx,tdc into one value with key ts) to hits object
# timestamp formula: ts = (tdc) + n_tdc*(bx) + n_tdc*n_bunches*(orbit) + n_tdc*n_bunches*n_orbits*(orbit_overflow)
# timestamp unit: 0.78 ns
# timestamp data type: uint64 i.e. max. value ~1.844e19 timestamp units (0.78 ns) = ~1.438e10 seconds = ~456 days
def add_timestamp(hits, *, silent=False):
    ts_hits = copy.deepcopy(hits)
    n_hits = len(ts_hits["ch"])
    if not silent: print(f"Add converted timestamp to {n_hits} hits...")
    ts_hits |= {"ts": np.full(n_hits, 0, dtype=params._ts_type)}
    oc_overflow = 0 # count how many times the orbit counter overflowed -> to have non-jumping but continous timestamp
    last_oc = 0
    for i in tqdm(range(n_hits)):
        tdc = ts_hits["tdc"][i]
        bx = ts_hits["bx"][i]
        oc = ts_hits["oc"][i]
        if last_oc > oc: # if last oc > current oc i.e. overflow detected -> increment oc_overflow counter to "smooth out" timestamp and not have jumps in it
            oc_overflow += 1
            if not silent: print(f"  Orbit counter overflow detected for hit #{i}. Incrementing overflow counter to {oc_overflow}.")
        ts_hits["ts"][i] =  tdc * derived_params._tdc_to_timestamp + bx * derived_params._bx_to_timestamp + oc * derived_params._orbit_to_timestamp + oc_overflow * derived_params._orbit_overflow_to_timestamp
        last_oc = oc
    return ts_hits

### sort hits by timestamp
# sort hints in ascending order depending on timestamp value ("ts" key)
def sort_by_timestamp(hits, *, silent=False):
    sorted_hits = copy.deepcopy(hits)
    n_hits = len(sorted_hits["ch"])
    if not silent: print(f"Sorting {n_hits} hits by timestamp...")
    new_idx_order = np.argsort(sorted_hits["ts"])
    for k in hits.keys(): # sort all keys of hit dict depending on order in timestamp key
        sorted_hits[k] = sorted_hits[k][new_idx_order]
    return sorted_hits

### calculate back ox,bx,tdc from timestamp value
def remap_htg_timestamp(ts):
    oc = (ts % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp
    bx = (ts % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp
    tdc = (ts % derived_params._bx_to_timestamp) // derived_params._tdc_to_timestamp
    return (oc, bx, tdc)


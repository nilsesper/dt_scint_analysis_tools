###########################################
### TIMESTAMP UTILS
###########################################

import numpy as np
import copy
from tqdm import tqdm

from analysis_tools.utils import data_utils

from analysis_tools.params import params
from analysis_tools.params import derived_params

# -----------------------------------------


### add timestamp (integer value concatenating all existing timestamp keys oc,bx,tdc into one value with key ts) to hits object
# timestamp formula: ts = (tdc) + n_tdc*(bx) + n_tdc*n_bunches*(orbit) + n_tdc*n_bunches*n_orbits*(orbit_overflow)
# timestamp unit: 0.78 ns
# timestamp data type: uint64 i.e. max. value ~1.844e19 timestamp units (0.78 ns) = ~1.438e10 seconds = ~456 days
# overflow_state (optional): {"oc_overflow": overflows found so far, "last_oc": last orbit counter}, to continue counting
#   the orbit counter overflows when a file is processed block by block; it is updated here
def add_timestamp(hits, *, silent=False, overflow_state=None):
    ts_hits = copy.deepcopy(hits)
    n_hits = data_utils.length(hits)
    if not silent: print(f"Add converted timestamp to {n_hits} entries...")
    ts_hits |= {"ts": np.full(n_hits, 0, dtype=params._ts_type), "err_ts": np.full(n_hits, 0, dtype=np.float64)}
    # assign timestamp for full dataset (together)
    oc_overflow = 0 # count how many times the orbit counter overflowed -> to have non-jumping but continous timestamp
    last_oc = None
    if overflow_state is not None: # continue from the previous block
        oc_overflow = overflow_state["oc_overflow"]
        last_oc = overflow_state["last_oc"]
    for i in tqdm(range(n_hits), disable=silent):
        tdc = ts_hits["tdc"][i]
        bx = ts_hits["bx"][i]
        oc = ts_hits["oc"][i]
        ro_ch = ts_hits["ro_ch"][i]
        if last_oc == None: # set last_oc at first hit
            last_oc = oc
        if int(last_oc) - int(oc) > params._oc_difference_for_overflow: # if last oc > current oc i.e. overflow detected -> increment oc_overflow counter to "smooth out" timestamp and not have jumps in it
            oc_overflow += 1
            print(f"Detected OC overflow -- i={i}, ro_ch={ro_ch} -- last_oc={last_oc} -- oc={oc}")
            #if not silent: print(f"  Orbit counter overflow detected for hit #{i}. Incrementing overflow counter to {oc_overflow}.")
        ts_hits["ts"][i] =  tdc * derived_params._tdc_to_timestamp + bx * derived_params._bx_to_timestamp + oc * derived_params._orbit_to_timestamp + oc_overflow * derived_params._orbit_overflow_to_timestamp
        ts_hits["err_ts"][i] = 1/np.sqrt(12) # digitization error
        last_oc = oc
    if overflow_state is not None:
        overflow_state["oc_overflow"] = oc_overflow
        overflow_state["last_oc"] = last_oc
    # sort hits by timestamp
    ts_hits = sort_by_timestamp(hits=ts_hits)
    return ts_hits

### sort hits by timestamp
# sort hints in ascending order depending on timestamp value ("ts" key)
def sort_by_timestamp(hits, *, silent=False):
    sorted_hits = copy.deepcopy(hits)
    n_hits = data_utils.length(sorted_hits)
    if not silent: print(f"Sorting {n_hits} entries by timestamp...")
    sorted_hits = data_utils.sort_by_key(data=hits, sort_key="ts", silent=silent)
    return sorted_hits

### calculate back ox,bx,tdc from timestamp value
def remap_htg_timestamp(ts):
    ts = np.uint64(np.round(ts,0))
    oc = (ts % derived_params._orbit_overflow_to_timestamp) // derived_params._orbit_to_timestamp
    bx = (ts % derived_params._orbit_to_timestamp) // derived_params._bx_to_timestamp
    tdc = (ts % derived_params._bx_to_timestamp) // derived_params._tdc_to_timestamp
    return (oc, bx, tdc)

### calculate time distance (in ts units) relative the start of this orbit (oc=this_orbit, bx=0, tdc=0)
# store in key "ts_orbit"
def add_timestamp_this_orbit(hits, *, silent=False):
    n_hits = data_utils.length(hits)
    ts_hits = copy.deepcopy(hits)
    if not silent: print(f"Add timestamp relative to orbit to {n_hits} entries...")
    ts_hits |= {"ts_orbit": np.full(n_hits, 0, dtype=params._ts_type), "err_ts_orbit": np.full(n_hits, 0, dtype=params._ts_type)}
    for i in tqdm(range(n_hits), disable=silent):
        tdc = ts_hits["tdc"][i]
        bx = ts_hits["bx"][i]
        #this_oc = ts_hits["oc"][i]
        # calculate delta_ts to beginning of orbit
        ts_hits["ts_orbit"][i] =  tdc * derived_params._tdc_to_timestamp + bx * derived_params._bx_to_timestamp
        ts_hits["err_ts_orbit"][i] = 1/np.sqrt(12)
    return ts_hits


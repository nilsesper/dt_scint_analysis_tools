###########################################
### SCINTILLATOR-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path

import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### add timestamp (integer value concatenating all existing timestamp keys oc,bx,tdc into one value with key ts) to hits object
# timestamp formula: ts = (tdc) + n_tdc*(bx) + n_tdc*n_bunches*(orbit)
# timestamp unit = 0.78 ns
def add_timestamp(hits, *, silent=False):
    ts_hits = copy.deepcopy(hits)
    n_hits = len(ts_hits["ch"])
    if not silent: print(f"Add converted timestamp to {n_hits} hits...")
    ts_hits |= {"ts": np.full(n_hits, 0, dtype=params._ts_type)}
    oc_overflow = 0 # count how many times the orbit counter overflowed -> to have non-jumping but continous timestamp
    last_oc = 0
    for i in range(n_hits):
        tdc = ts_hits["tdc"][i]
        bx = ts_hits["bx"][i]
        oc = ts_hits["oc"][i]
        if last_oc > oc: # if last oc > current oc i.e. overflow detected -> increment oc_overflow counter to "smooth out" timestamp and not have jumps in it
            oc_overflow += 1
            if not silent: print(f"Orbit counter overflow detected for hit #{i}. Incrementing overflow counter to {oc_overflow}.")
        ts_hits["ts"][i] = tdc * derived_params._tdc_to_timestamp + bx * derived_params._bx_to_timestamp + oc * derived_params._orbit_to_timestamp + oc_overflow * derived_params._orbit_overflow_to_timestamp
        last_oc = oc
    return ts_hits





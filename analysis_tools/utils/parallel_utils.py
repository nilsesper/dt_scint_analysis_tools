###########################################
### PARALLEL PROCESSING
###########################################
# All functions give the same result as running on one process.

import multiprocessing
import numpy as np

from analysis_tools.utils import root_utils

# -----------------------------------------

### a pool of n_proc worker processes, or None for n_proc <= 1; close it with close_pool
def open_pool(n_proc):
    if n_proc <= 1:
        return None
    return multiprocessing.Pool(n_proc)

def close_pool(pool):
    if pool is not None:
        pool.close()
        pool.join()

### split a data dict into n_parts dicts with (nearly) the same number of rows
def split_rows(data, n_parts):
    n_rows = root_utils.length(data)
    parts = []
    for i_part in range(n_parts):
        start = (i_part * n_rows) // n_parts
        stop = ((i_part + 1) * n_rows) // n_parts
        part = {}
        for key in data:
            part[key] = data[key][start:stop]
        parts.append(part)
    return parts

def merge_rows(parts):
    merged = {}
    for key in parts[0]:
        arrays = []
        for part in parts:
            arrays.append(part[key])
        merged[key] = np.concatenate(arrays)
    return merged

### fit_function(rows, fit_vd, suffix, verbose) for a fit function which treats every row on its own (one output row per
# input row, e.g. dt_sl_fit_utils.fit_sl_patterns): the rows are split into n_proc parts, which are fitted in parallel,
# and the results are merged again
def run_fits_in_parallel(fit_function, rows, fit_vd, suffix, verbose, n_proc):
    n_rows = root_utils.length(rows)
    if n_proc <= 1 or n_rows < 4 * n_proc:
        return fit_function(rows, fit_vd, suffix, verbose)
    jobs = []
    for part in split_rows(rows, n_proc):
        jobs.append((part, fit_vd, suffix, verbose))
    pool = open_pool(n_proc)
    results = pool.starmap(fit_function, jobs)
    close_pool(pool)
    return merge_rows(results)

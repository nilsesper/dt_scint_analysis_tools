###########################################
### PARALLEL PROCESSING
###########################################
# All functions give the same result as running on one process.

import collections
import contextlib
import multiprocessing
import numpy as np

from analysis_tools.utils import root_utils

# -----------------------------------------

### a pool of n_proc worker processes, or None for n_proc <= 1 (use in a with statement)
@contextlib.contextmanager
def optional_pool(n_proc):
    if n_proc <= 1:
        yield None
        return
    pool = multiprocessing.Pool(n_proc)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

### function(item) for every item, in the order of the items; on the pool if one is given
# at most 2 * (pool size) items are in work at a time, so a generator of items is never read far ahead
def map_in_order(function, items, pool=None):
    if pool is None:
        for item in items:
            yield function(item)
        return
    in_work = collections.deque()
    for item in items:
        in_work.append(pool.apply_async(function, (item,)))
        if len(in_work) >= 2 * pool._processes:
            yield in_work.popleft().get()
    while in_work:
        yield in_work.popleft().get()

def _call_with_rows(job):
    function, rows, rows_argument, kwargs = job
    return function(**{rows_argument: rows}, **kwargs)

### function(rows, **kwargs) for a function which treats every row on its own (one output row per input row),
# with the rows split over n_proc processes
def run_rowwise(function, rows, rows_argument, kwargs, n_proc):
    n_rows = root_utils.length(rows)
    if n_proc <= 1 or n_rows < 4 * n_proc:
        return function(**{rows_argument: rows}, **kwargs)
    parts = [{k: v[part] for k, v in rows.items()} for part in np.array_split(np.arange(n_rows), n_proc)]
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(_call_with_rows, [(function, part, rows_argument, kwargs) for part in parts])
    return {k: np.concatenate([result[k] for result in results]) for k in results[0].keys()}

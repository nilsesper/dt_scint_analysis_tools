###########################################
### PROCESSING UTILS (for parallel execution)
###########################################

import numpy as np
import copy
import os.path
import multiprocessing
from tqdm import tqdm
import sys

import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### execute function on dataset with multiprocessing
# split data into batches and run on n_processes multiprocessing pool
# approximate batch size of data processed with each subprocess should be specified with n_batches
# give function to be executed and (constant) keyword arguments
# pass data (to be split) and data key explicitly
# mute=True allows suppression of function output of multiprocesses
def multiprocess_data(n_processes, n_batches, function, data_key, data, *, kwargs={}, silent=False, mute=False, give_idx_offset=False, return_unmerged=False): # kwargs are all other keys passed to function, data_key is implicitly added
    # calculate no of parts depending on data length and batch size
    any_key = list(data.keys())[0]
    n_data = len(data[any_key])
    n_parts = max([1, n_data//n_batches])
    print(f"Executing function \"{function.__name__}\" for dataset with {n_data} entries in multiprocessing environment with {n_processes} processes and batches of ~{n_batches} data entries (i.e. {n_parts} sub-datasets)...")
    # split data
    split_data = data_utils.split_dataset(data=data, n_parts=n_parts, silent=silent)
    # prepare function to be f(data), all kwargs implicit
    #def parsed_function(data_in):
    #    parsed_kwargs = kwargs
    #    parsed_kwargs[data_key] = data_in
    #    function(**parsed_kwargs)
    # prepare kwargs for all parts
    kwarg_list = []
    idx_offset = 0
    for part in range(n_parts):
        cur_kwargs = kwargs | {data_key: split_data[part]} | {"silent": mute}
        if give_idx_offset:
             cur_kwargs |= {"idx_offset": idx_offset}
        kwarg_list.append(cur_kwargs)
        idx_offset += data_utils.length(split_data[part])
    # execute function in multiprocessing pool
    results = [None for i in range(n_parts)]
    with multiprocessing.Pool(processes=n_processes) as pool, tqdm(total=n_parts) as pbar:
            future_parameters = [(pool.apply_async(function, kwds=kwargs), kwargs) for kwargs in kwarg_list]
            for part, (future, parameters) in enumerate(future_parameters):
                result = future.get()
                results[part] = result
                pbar.update()
                pbar.refresh()
    if not return_unmerged:
        # merge results if desired
        merged_data = data_utils.merge_dataset(split_data=results, silent=silent)
        return merged_data
    else:
        return results





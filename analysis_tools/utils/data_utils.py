###########################################
### DATA IMPORT, CUTTING & CONVERSION UTILS
###########################################

import numpy as np
import copy
import pickle


# -----------------------------------------


### return data array with applied conditions (cuts)
# arguments: data dict
# conditions: list of conditions [(name, operator, value)]
#       name: name of data key to compare with
#       operator: =,>,<,>=,<=,in as string
#       value: value to compare with
#       all conditions are "AND-ed" together
def cut_data(data, conditions=[], *, silent=False):
    if not silent: print(f"Cutting data according to conditions {conditions}...")
    ## calculate masks for data
    #mask = np.full(len(data["ch"]), True)
    last_data = copy.deepcopy(data)
    any_key = list(last_data.keys())[0]
    mask = np.full(len(last_data[any_key]), True)
    for c in conditions: # calculate mask for all conditions and AND them together
        if c[1] == "==": mask &= (data[c[0]] == c[2])
        elif c[1] == "!=": mask &= (data[c[0]] != c[2])
        elif c[1] == ">": mask &= (data[c[0]] > c[2])
        elif c[1] == "<": mask &= (data[c[0]] < c[2])
        elif c[1] == ">=": mask &= (data[c[0]] >= c[2])
        elif c[1] == "<=": mask &= (data[c[0]] <= c[2])
        elif c[1] == "in": mask &= np.ma.isin(data[c[0]], c[2])
        else: raise Exception(f"Invalid operator \"{c[1]}\".")
    ## apply mask to data
    masked_data = {}
    for name in data.keys():
        # if python list at this key
        masked_data[name] = []
        if isinstance(last_data[name], list):
            for i in range(len(last_data[name])):
                if mask[i]:
                    masked_data[name].append(last_data[name][i])
        # if numpy array at this key
        elif isinstance(last_data[name], np.ndarray):
            masked_data[name] = copy.deepcopy(last_data[name][mask])
        else:
            raise Exception(f"CUT_DATA ERROR: data[{name}] is of unsupported type {type(last_data[name])}. can only cut lists or numpy arrays")
    last_data = copy.deepcopy(masked_data)
    one_key = list(masked_data.keys())[0]
    if not silent:
        if len(data[one_key]) > 0: print(f"Cut flow: {len(masked_data[one_key])} / {len(data[one_key])} = {len(masked_data[one_key])/len(data[one_key])}")
        else: print(f"Cut flow: {len(masked_data[one_key])} / {len(data[one_key])}")
    return masked_data

### sort hits by any key
# sort hints in ascending order depending on key value
def sort_by_key(data, sort_key, *, silent=False):
    sorted_data = copy.deepcopy(data)
    any_key = list(sorted_data.keys())[0]
    n_data = len(sorted_data[any_key])
    if not silent: print(f"Sorting {n_data} hits by key \"{sort_key}\"...")
    new_idx_order = np.argsort(data[sort_key])
    for name in data.keys(): # sort all keys of hit dict depending on order
        # if python list at this key
        if isinstance(data[name], list):
            for i in range(len(data[name])):
                j = new_idx_order[i]
                sorted_data[name][i] = data[name][j]
        # if numpy array at this key
        elif isinstance(data[name], np.ndarray):
            sorted_data[name] = data[name][new_idx_order]
        else:
            raise Exception(f"SORT_BY_KEY ERROR: data[{name}] is of unsupported type {type(data[name])}. can only sort lists or numpy arrays")
    return sorted_data

### store arbitrary object as pickle file
def store_pickle(data, file, *, silent=False):
    if not silent: print(f"Storing object to pickle file \"{file}\"...")
    with open(file, 'wb') as file_obj:
        pickle.dump(obj=data, file=file_obj)
    return

### load arbitrary object from pickle file
def load_pickle(file, *, silent=False):
    if not silent: print(f"Loading object from pickle file \"{file}\"...")
    with open(file, 'rb') as file_obj:
        data = pickle.load(file=file_obj)
    return data





### get length of data object
def length(data):
    any_key = list(data.keys())[0]
    return len(data[any_key])



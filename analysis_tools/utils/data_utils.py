###########################################
### DATA IMPORT, CUTTING & CONVERSION UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm
import pickle

import analysis_tools.params.params as params

# -----------------------------------------

### import raw file (.txt recorded with htg box)
# extract data from dumpfile and convert to numbers
# taken from private exchange with A. Bergnoli (INFN Padova/Legnaro)
# return dict of np arrays
def import_raw(file_name, *, silent=False):
    if not silent: print(f"Importing raw file \"{file_name}\"...")
    with open(file_name) as ascii_bin_file:
        lines = ascii_bin_file.readlines()
    if not silent: print(f"Converting raw file to dictionary of np arrays...")
    n_hits = len(lines)
    hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()}
    for i in tqdm(range(n_hits), disable=silent):
        d = lines[i]
        hits["ch"][i] = (int(d) & params._htg_shifted_mask["ch"]) >> params._htg_bitshift["ch"]
        hits["bx"][i] = (int(d) & params._htg_shifted_mask["bx"]) >> params._htg_bitshift["bx"]
        hits["tdc"][i] = (int(d) & params._htg_shifted_mask["tdc"]) >> params._htg_bitshift["tdc"]
        hits["oc"][i] = (int(d) & params._htg_shifted_mask["oc"]) >> params._htg_bitshift["oc"]
        hits["ro_ch"][i] = (int(d) & params._htg_shifted_mask["ro_ch"]) >> params._htg_bitshift["ro_ch"]
    return hits

### return data array with applied conditions (cuts)
# arguments: data dict
# conditions: list of conditions [(name, operator, value)]
#       name: name of data key to compare with
#       operator: =,>,<,>=,<=,in as string
#       value: value to compare with
#       all conditions are "AND-ed" together
def cut_data(data, conditions=[], *, silent=False):
    if not silent: print(f"Cutting data according to conditions {conditions}...")
    # calculate masks for data
    #mask = np.full(len(data["ch"]), True)
    last_data = copy.deepcopy(data)
    any_key = list(last_data.keys())[0]
    mask = np.full(len(last_data[any_key]), True)
    for c in conditions: # calculate mask for all conditions and AND them together
        if c[1] == "==": mask &= (data[c[0]] == c[2])
        elif c[1] == ">": mask &= (data[c[0]] > c[2])
        elif c[1] == "<": mask &= (data[c[0]] < c[2])
        elif c[1] == ">=": mask &= (data[c[0]] >= c[2])
        elif c[1] == "<=": mask &= (data[c[0]] <= c[2])
        elif c[1] == "in": mask &= np.ma.isin(data[c[0]], c[2])
        else: raise Exception(f"Invalid operator \"{c[1]}\".")
    # apply mask to data
    masked_data = {}
    for name in data.keys():
        masked_data[name] = copy.deepcopy(last_data[name][mask])
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
    for k in data.keys(): # sort all keys of hit dict depending on order
        sorted_data[k] = data[k][new_idx_order]
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

### split given data into n_parts
# to be calculated in parallel
def split_dataset(data, n_parts, *, silent=False):
    split_data = [{} for i in range(n_parts)]
    for k in data.keys():
        split_array = np.array_split(data[k], n_parts) # near-equal array division
        for i in range(n_parts):
            split_data[i][k] = split_array[i]
    return split_data # [data_part[i] for i in range(n_parts)]

### merge split dataset into one
# after parallel calculation
# assume all data has same keys
def merge_dataset(split_data, *, silent=False):
    n_parts = len(split_data)
    any_key = list(split_data[0].keys())[0]
    n_data_parts = [len(split_data[i][any_key]) for i in range(n_parts)] # data entries of each part
    n_data = np.sum(n_data_parts) # total no of data entries
    merged_data = {k: np.full(n_data, 0, dtype=v.dtype) for k,v in split_data[0].items()}
    offset = 0
    for part in range(n_parts):
        for k in split_data[part].keys():
            for i in range(n_data_parts[part]):
                merged_data[k][i+offset] = copy.deepcopy(split_data[part][k][i])
        if n_data_parts[part] > 0:
            offset += i+1
    return merged_data

### get length of data object
def length(data):
    any_key = list(data.keys())[0]
    return len(data[any_key])

###########################################
### DATA IMPORT, CUTTING & CONVERSION UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

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
    for i in tqdm(range(n_hits)):
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
    if not silent: print(f"Cut flow: {len(masked_data[one_key])} / {len(data[one_key])} = {len(masked_data[one_key])/len(data[one_key])}")
    return masked_data

### sort hits by any key
# sort hints in ascending order depending on key value
def sort_by_key(hits, sort_key, *, silent=False):
    sorted_hits = copy.deepcopy(hits)
    any_key = list(sorted_hits.keys())[0]
    n_hits = len(sorted_hits[any_key])
    if not silent: print(f"Sorting {n_hits} hits by key \"{sort_key}\"...")
    new_idx_order = np.argsort(sorted_hits[sort_key])
    for k in hits.keys(): # sort all keys of hit dict depending on order
        sorted_hits[k] = sorted_hits[k][new_idx_order]
    return sorted_hits





###########################################
### GENERATE DUMMY DUMPFILE DATA
###########################################

import numpy as np
import copy
import os.path

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### write hits to dumpfile
# use hit keys {oc, bx, tdc, ch, ro_ch} for writing to dumpfile
def write_to_dumpfile(file_name, hits, *, silent=False):
    # convert to integer
    n_hits = len(hits["ch"])
    if not silent: print(f"Write {n_hits} hits to dummy dumpfile \"{file_name}\"...")
    num_content = np.full(n_hits, 0, dtype=np.uint64)
    for i in range(n_hits):
        num_content[i] = 0
        for k in derived_params._dumpfile_keys:
            num_content[i] |= ((int(hits[k][i]) << (params._htg_bitshift[k])) & (params._htg_shifted_mask[k]))
    # convert to ascii content
    content = ""
    for i in range(n_hits):
        content += str(num_content[i])+"\n"
    # ascii content file dump
    with open(file_name, "w") as f:
        f.write(content)
    return

### convert hit list (list of single hit dicts) to hits in normal format (dict of np arrays)
def hit_list_to_hits(hit_list, *, silent=False):
    n_hits = len(hit_list)
    if not silent: print(f"Convert list of {n_hits} single hits to normal hits format...")
    hits = {k: np.full(n_hits, 0, dtype=v) for k,v in params._htg_keys.items()}
    for i in range(n_hits):
        for k in derived_params._dumpfile_keys:
            hits[k][i] = hit_list[i][k]
    return hits




###########################################
### SCINTILLATOR-SPECIFIC UTILS
###########################################

import numpy as np
import copy
import os.path



# -----------------------------------------

### add scintillator specific keys/mapping to hits
# take information about this mapping from params.py
def add_dt_keys(hits):
    hits = copy.deepcopy(hits)
    return hits


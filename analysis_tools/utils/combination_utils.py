###########################################
### DT-SCINT-COMBINATION/CORRELATION UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm
from scipy.optimize import curve_fit

import analysis_tools.utils.data_utils as data_utils
import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.muon_utils as muon_utils
import analysis_tools.utils.hist_utils as hist_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### group together hits of two different datasets (which have synchronized timestamps "ts")
# for each entry/index in data1, return list of data2 entries/indices which lie in relative time window [ts_data1 - data2_ts_tolerance, ts_data1 + data2_ts_tolerance]
# give name of timestamp key of data sets (default "ts")
# returns:
#   time_grouping_list [[indices of data2 within ts tolerance window] index of data1 element]
def time_grouping_indices(data1, data2, data2_ts_tolerance, data1_ts_key="ts", data2_ts_key="ts"):
    time_grouping_list = []
    data1 = copy.deepcopy(data1)
    data2 = copy.deepcopy(data2)
    n_data_1, n_data_2 = data_utils.length(data1), data_utils.length(data2)
    # sort data by timestamp
    data1 = data_utils.sort_by_key(data=data1, sort_key=data1_ts_key)
    data2 = data_utils.sort_by_key(data=data2, sort_key=data2_ts_key)
    # go event by event through data1 and look for data2 elements in time correlation window
    for i in range(n_data_1):
        # find all indices in data2 which are in time window ts_data2 = [ts_data1 - data2_ts_tolerance, ts_data1 + data2_ts_tolerance]
        this_data1_ts = data1[data1_ts_key][i]
        this_mask = (data2[data2_ts_key] >= this_data1_ts - data2_ts_tolerance) & (data2[data2_ts_key] <= this_data1_ts + data2_ts_tolerance)
        this_mask_indices = np.where(this_mask)[0]
        # append to list
        time_grouping_list.append(this_mask_indices)
    return time_grouping_list










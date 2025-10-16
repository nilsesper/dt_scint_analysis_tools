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

### group hits of 3 different datasets (with sync timestamps)
# for each entry/index in data1, return list of data2 entries/indices which lie in relative time window [ts_data1 - data2_ts_tolerance, ts_data1 + data2_ts_tolerance]
# give name of timestamp key of data sets (default "ts")
# then group together ts_data1 hits and merge the respective sets of data2 indices
# returns:
#   data_idx_grouped = [ [index of data for groups within ts_window] for all found groups in dataset ]
#   mean_ts_grouped = [ mean_ts_of_group for all found groups in dataset ]
def time_grouping_indices_2(data, ts_tolerance, data_ts_key="ts"):
    data = copy.deepcopy(data)
    # sort data by timestamp
    data = data_utils.sort_by_key(data=data, sort_key=data_ts_key, silent=True)
    n_data = data_utils.length(data)
    # grouping by closely lying timestamps
    ts_diff = data[data_ts_key][1:] - data[data_ts_key][:-1]
    gps = np.concatenate([[0],np.cumsum(ts_diff >= ts_tolerance)])
    data_idx_grouped = [ np.arange(0,n_data)[gps == i] for i in range(gps[-1]+1) ] # [ [index of data for groups within ts_window] for all found groups in dataset ]
    mean_ts_grouped = [ np.mean(data[data_ts_key][ data_idx_grouped[i] ]) for i in range(len(data_idx_grouped)) ]
    return data_idx_grouped, mean_ts_grouped

### group together hits of two different datasets (which have synchronized timestamps "ts")
# for each entry/index in data1, return list of data2 entries/indices which lie in relative time window [ts_data1 - data2_ts_tolerance, ts_data1 + data2_ts_tolerance]
# give name of timestamp key of data sets (default "ts")
# returns:
#   time_grouping_list [[indices of data2 within ts tolerance window] index of data1 element]
def time_grouping_indices_3(data1, data2, data3, ts_tolerance):
    time_grouping_list = []
    data1, data2, data3 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data3)
    n_data_1, n_data_2, n_data_3 = len(data1), len(data2), len(data3)
    # sort data
    data1 = data1[np.argsort(data1)]
    data2 = data2[np.argsort(data2)]
    data3 = data3[np.argsort(data3)]
    # go event by event through data1 and look for data2 elements in time correlation window
    for i in range(n_data_1):
        # find all indices in data2 which are in time window ts_data2 = [ts_data1 - data2_ts_tolerance, ts_data1 + data2_ts_tolerance]
        this_data1_ts = data1[i]
        this_data2_mask = (data2 >= this_data1_ts - ts_tolerance) & (data2 <= this_data1_ts + ts_tolerance)
        this_data2_mask_indices = np.where(this_data2_mask)[0]
        this_data3_mask = (data3 >= this_data1_ts - ts_tolerance) & (data3 <= this_data1_ts + ts_tolerance)
        this_data3_mask_indices = np.where(this_data3_mask)[0]
        # append to list
        time_grouping_list.append((this_data2_mask_indices, this_data3_mask_indices))
    return time_grouping_list






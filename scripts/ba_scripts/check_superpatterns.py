import os
import argparse
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params
import numpy as np
import matplotlib.pyplot as plt

base_path = "data_ba/"
dataset_name = "cosmic_82-18_3600-1800-1200_test1_th20"
sl_patterns_file = base_path + "pcls/" + dataset_name + "_sl_patterns.pcl"  
sl_fits_file = base_path + "pcls/" + dataset_name + "_sl_fits.pcl"
sl_refits_file = base_path + "pcls/" + dataset_name + "_sl_refits.pcl"

sl_fits = data_utils.load_pickle(file=sl_fits_file)

super_patterns = dt_utils.build_phi_super_patterns(sl_fits)
print(super_patterns.keys())

# prepare angle diff hist for both sl 
tanalpha_diff = np.arctan(super_patterns["tan_alpha_sl1"]) - np.arctan(super_patterns["tan_alpha_sl3"])
print(np.mean(tanalpha_diff))
print(np.std(tanalpha_diff))
plt.figure()
plt.hist(tanalpha_diff, bins=100)
plt.savefig("tan_alpha_hist.png")


delta_t0 = (super_patterns["t0_sl1"] -super_patterns["t0_sl3"]) * 0.78 

plt.figure()
plt.hist(delta_t0, bins=100)
plt.savefig("delta_t0.png")
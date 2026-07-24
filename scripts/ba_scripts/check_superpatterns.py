import os
import argparse
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

base_path = "data_ba/"
dataset_name = "cosmic_82-18_3550-1800-1200_run1_th20_cut_50"
sl_patterns_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_patterns.pcl"  
sl_fits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_fits.pcl"
sl_refits_file = base_path + f"pcls/{dataset_name}/" + dataset_name + "_sl_refits.pcl"
super_fits_path = base_path + f"pcls/{dataset_name}/" + dataset_name + "_super_fits.pcl"
plots_path = base_path + "plots/sl_fits/"
plot_type = ".png"

sl_fits = data_utils.load_pickle(file=sl_fits_file)

super_patterns = dt_utils.build_phi_super_patterns(sl_fits)


super_fits = dt_utils.fit_super_sl_patterns(super_patterns, fit_vd=True, suffix = "_free_vd_super_fit")
print(f"Saving Superfits to {super_fits_path}...")
data_utils.store_pickle(data = super_fits, file = super_fits_path)
print(f"\nDone saving data under {super_fits_path}")
"""
print(super_fits.keys())

plt.figure()
plt.hist(super_fits["vd_free_vd_super_fit"])
plt.savefig(plots_path + "vd_free_vd_super_fit_hist" + plot_type)

plt.figure()
plt.hist(super_fits["tan_alpha_free_vd_super_fit"])
plt.savefig(plots_path + "tan_alpha_free_vd_super_fit_hist" + plot_type)


plt.figure()
plt.hist(super_fits["t0_free_vd_super_fit"])
plt.savefig(plots_path + "t0_free_vd_super_fit_hist" + plot_type)
"""
"""
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
"""
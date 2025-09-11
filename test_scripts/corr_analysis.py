#################################################################
### analysis plots
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# get REPO_PATH from bash env
if "REPO_PATH" not in os.environ:
    raise Exception(f"REPO_PATH is not in bash environment. Please source env.sh before executing this script!")
REPO_PATH = os.environ["REPO_PATH"]
pcl_path = REPO_PATH+"/data_files"
plot_path = REPO_PATH+"/plots"
# data input files:
cosmic_muon_file = pcl_path+"/dummy_cosmic_muons.pcl"
# dt
cosmic_muon_dt_hits_file = pcl_path+"/dummy_cosmic_muon_dt_hits.pcl"
cosmic_muon_sl_dt_patterns_file = pcl_path+"/dummy_cosmic_muon_sl_dt_patterns.pcl"
cosmic_muon_sl_dt_fits_file = pcl_path+"/dummy_cosmic_muon_dt_sl_fits.pcl"
cosmic_muon_dt_muons_file = pcl_path+"/dummy_cosmic_muon_dt_muons.pcl"
# scint
cosmic_muon_scint_hits_file = pcl_path+"/dummy_cosmic_muon_scint_hits.pcl"
cosmic_muon_scint_muon_areas_file = pcl_path+"/dummy_cosmic_muon_scint_muon_areas.pcl"
# data output files:
# -

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### data import
    print(f"###### Importing all data...")
    cosmic_muons = data_utils.load_pickle(file=cosmic_muon_file)
    # dt
    cosmic_muon_dt_hits = data_utils.load_pickle(file=cosmic_muon_dt_hits_file)
    cosmic_muon_sl_dt_patterns = data_utils.load_pickle(file=cosmic_muon_sl_dt_patterns_file)
    cosmic_muon_sl_dt_fits = data_utils.load_pickle(file=cosmic_muon_sl_dt_fits_file)
    cosmic_muon_dt_muons = data_utils.load_pickle(file=cosmic_muon_dt_muons_file)
    # scint
    cosmic_muon_scint_hits = data_utils.load_pickle(file=cosmic_muon_scint_hits_file)
    cosmic_muon_scint_muon_areas = data_utils.load_pickle(file=cosmic_muon_scint_muon_areas_file)


    ### correlate scintillator & dt reco
    cosmic_muon_corr_muons = muon_utils.correlate_muons_and_muon_areas(muons=cosmic_muon_dt_muons, muon_areas=cosmic_muon_scint_muon_areas)
    print("cosmic_muon_corr_muons =",cosmic_muon_corr_muons)

    ### corr muon plots
    print(f"### correlated muons plots")
    n_hist_bins = 100
    hist_bins = {
        "delta_xcenter": 30 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "delta_ycenter": 30 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "delta_ts": np.arange(-30,30+1),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muon_corr_muons, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muon_corr_muons)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/corr_muons_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)



    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

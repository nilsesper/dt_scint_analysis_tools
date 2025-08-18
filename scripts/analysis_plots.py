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

    ### true cosmic muons
    print(f"### true cosmic muons")
    n_hist_bins = 100
    hist_bins = {
        "x0": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "y0": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "z0": np.linspace(params._scintillator["pos"][2]-10, params._scintillator["pos"][2]+10, n_hist_bins),
        "theta": np.linspace(0, np.pi, n_hist_bins),
        "phi": np.linspace(0, 2*np.pi, n_hist_bins),
        "ts": np.linspace(0, int(1.05e8), n_hist_bins),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muons, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muons)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{true})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/true_muons_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### dt hits
    print(f"### dt hits")
    n_hist_bins = 100
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
        "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
        "wi": np.arange(0, 70),
        "ly": np.arange(0, 3+1),
        "sl": np.arange(1, 3+1),
        "ts": np.linspace(0, int(1.05e8), n_hist_bins),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muon_dt_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muon_dt_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/dt_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### dt reco muons
    print(f"### dt reco muons")
    n_hist_bins = 100
    hist_bins = {
        "x0": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "y0": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "z0": np.linspace(params._scintillator["pos"][2]-10, params._scintillator["pos"][2]+10, n_hist_bins),
        "theta": np.linspace(0, np.pi, n_hist_bins),
        "phi": np.linspace(0, 2*np.pi, n_hist_bins),
        "ts": np.linspace(0, int(1.05e8), n_hist_bins),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muon_dt_muons, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muon_dt_muons)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/dt_muons_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### dt reco vs. true cosmic muons
    print(f"### dt reco muons vs. cosmic muons")
    n_hist_bins = 100
    hist_bins = {
        "x0": 3 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "y0": 3 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "z0": 3 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "theta": 0.01 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "phi": 0.1 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "ts": np.arange(-10,10+1),
    }
    reco_muon_delta = copy.deepcopy(cosmic_muon_dt_muons) # = reco - simulated_cosmic
    reco_muon_delta["ts"] = np.full(data_utils.length(cosmic_muon_dt_muons), 0, dtype=np.int64)
    for i in range(data_utils.length(cosmic_muon_dt_muons)):
        muon_id = cosmic_muon_dt_muons["muon_id"][i]
        for k in hist_bins.keys():
            if k in ["ts"]: 
                reco_muon_delta[k][i] = int(cosmic_muon_dt_muons[k][i]) - int(cosmic_muons[k][muon_id])
            else:
                reco_muon_delta[k][i] = cosmic_muon_dt_muons[k][i] - cosmic_muons[k][muon_id]
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=reco_muon_delta, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(reco_muon_delta)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{DT})-$"+params._key_symbols[k]+"$(\\text{true})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/true_vs_dt_muons_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### scintillator hits
    print(f"### scintillator hits")
    n_hist_bins = 100
    hist_bins = {
        "ro_ch": np.arange(0, 32),
        "ch": np.arange(0, 255),
        "tdc": np.arange(0, params._lhc_tdc_count+1),
        "bx": np.linspace(0, params._lhc_bunch_count, n_hist_bins),
        "oc": np.linspace(0, params._lhc_orbit_count, n_hist_bins),
        "ly": np.arange(0, 3+1),
        "st": np.arange(1, 16+1),
        "ts": np.linspace(0, int(1.05e8), n_hist_bins),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muon_scint_hits, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muon_scint_hits)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/scint_hits_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### scint reco muon areas
    print(f"### scint reco muon areas")
    n_hist_bins = 100
    hist_bins = {
        "xmin": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "xmax": np.linspace(params._scintillator["pos"][0]-10, params._scintillator["pos"][0]+params._scintillator["size"][0]+10, n_hist_bins),
        "ymin": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "ymax": np.linspace(params._scintillator["pos"][1]-10, params._scintillator["pos"][1]+params._scintillator["size"][1]+10, n_hist_bins),
        "z0": np.linspace(params._scintillator["pos"][2]-10, params._scintillator["pos"][2]+params._scintillator["size"][2]+10, n_hist_bins),
    }
    for k in hist_bins.keys():
        hists, edges, centers, underflow, overflow = hist_utils.calculate_hist(data=cosmic_muon_scint_muon_areas, key=k, bin_centers=hist_bins[k], silent=True)
        print(f"key \"{k}\": entries={data_utils.length(cosmic_muon_scint_muon_areas)} underflow={underflow}, overflow={overflow}")
        round_digits = 0 if k in ["ts"] else 2
        xlabel = params._key_symbols[k]+"$(\\text{scint})$"
        xlabel += " ["+params._key_units[k]+"]" if (params._key_units[k] != "") else ""
        plotname =  plot_path+f"/scint_reco_{k}.png"
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=xlabel, round_digits=round_digits, bin_labels=False, silent=True, store=plotname)

    ### correlate scintillator & dt reco


    input("Press enter to exit.")
    exit()






if __name__ == "__main__":
    main()
    print(f"###### Done.")

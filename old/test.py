#########################
# generate dummy data dumpfile
#########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils
from analysis_tools.params import params, derived_params

dummy_filename = "dumpfiles/dummy_data.txt"
data_filename = "dumpfiles/test_theta_v50.txt"

hit_list = [
    # wrong hits
    { "ro_ch": 7, "ch": 55, "oc": 0, "bx": 0, "tdc": 0, },
    { "ro_ch": 8, "ch": 254, "oc": 0, "bx": 20, "tdc": 0, },
    { "ro_ch": 24, "ch": 42, "oc": 0, "bx": 30, "tdc": 20, },
    # dt hits
    { "ro_ch": 8, "ch": 186, "oc": 1, "bx": 50, "tdc": 0, },
    { "ro_ch": 8, "ch": 187, "oc": 1, "bx": 50, "tdc": 3, },
    { "ro_ch": 8, "ch": 189, "oc": 1, "bx": 50, "tdc": 10, },
    { "ro_ch": 8, "ch": 188, "oc": 1, "bx": 50, "tdc": 25, },
    # scint hits
    { "ro_ch": 24, "ch": 1, "oc": 2, "bx": 5, "tdc": 3, },
    { "ro_ch": 24, "ch": 9, "oc": 2, "bx": 19, "tdc": 7, },
    { "ro_ch": 25, "ch": 0, "oc": 2, "bx": 23, "tdc": 9, },
    { "ro_ch": 25, "ch": 3, "oc": 2, "bx": 55, "tdc": 15, },
    { "ro_ch": 25, "ch": 3, "oc": 1, "bx": 55, "tdc": 15, }, # overflow
]

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    dummy_hits = dummy_gen.hit_list_to_hits(hit_list)
    dummy_gen.write_to_dumpfile(file_name=dummy_filename, hits=dummy_hits)

    dumpfile_hits = data_utils.import_raw(file_name=dummy_filename) # dummy_filename, data_filename
    print("dumpfile_hits =",dumpfile_hits)

    dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits)
    print("dt_hits =",dt_hits)

    scint_hits = scint_utils.extract_scint_hits(hits=dumpfile_hits)
    print("scint_hits =",scint_hits)

    dt_sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits)
    print("dt_sl_patterns =",dt_sl_patterns)

    # generate cosmic muons
    #dummy_muon = {"x0": 1000, "y0": 1000, "z0": 100, "theta": 10*np.pi/180, "phi": 20*np.pi/180, "ts": 1000}
    n_muons = 1
    t_start = 10000
    t_step = 1000
    cosmic_muons = muon_utils.generate_cosmic_muons(
        n = n_muons,
        ts = t_start+t_step*np.arange(0,n_muons),
        xrange = [ params._scintillator["pos"][0] , params._scintillator["pos"][0]+params._scintillator["size"][0] ],
        yrange = [ params._scintillator["pos"][1] , params._scintillator["pos"][1]+params._scintillator["size"][1] ],
        z0 = params._scintillator["pos"][2],
        phirange = [ 0 , 2*np.pi ],
        thetarange = [ 0 , np.pi/4 ]
    )
    
    # propagate cosmic muons through dt chamber
    dt_muon_hits = dt_utils.hits_from_muons(muons=cosmic_muons, noise_ampl=0)
    print("dt_muon_hits =",dt_muon_hits)

    # treat these dt hits as dummy data -> apply clustering algorithm
    sl_dt_patterns = dt_utils.find_sl_patterns(hits=dt_muon_hits)
    print("sl_muon_patterns =",sl_dt_patterns)

    # fit patterns
    sl_dt_fits = dt_utils.fit_sl_patterns(patterns=sl_dt_patterns, verbose=False)
    n_patterns = len(sl_dt_fits["sl"])
    print("sl_dt_fits =",sl_dt_fits)

    # plot all fitted patterns
    for pattern_id in range(n_patterns):
        ### plot sl pattern
        show_wires = True
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot sl pattern
        ax = geoplot_utils.sl_fit_ax(ax, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id, wire=show_wires)
        # plot originally simulated muon
        ax = geoplot_utils.sl_muon_proj_ax(ax, muons=cosmic_muons, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id, color="tab:green")
        # plot dt hits of simulated muon
        ax = geoplot_utils.sl_dt_hits_proj_ax(ax, dt_hits=dt_muon_hits, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id, color="tab:green", other_lat=True)
        # plot fitted muon
        ax = geoplot_utils.sl_muon_fit_ax(ax, sl_dt_fits=sl_dt_fits, pattern_id=pattern_id)
        # axis limits
        ax.margins(x=0.05, y=0.05)
        # text labels
        axbox = ax.get_position()
        x_topleft = axbox.p0[0]
        x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
        ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
        ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
        description = f"Pattern #{pattern_id}, SL {sl_dt_fits["sl"][pattern_id]}"
        ax.set_xlabel("$x_\\text{rel}$ [mm]")
        ax.set_ylabel("$z_\\text{rel}$ [mm]")
        ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
        # show/store figure
        fig.show()
        
    # reco muons from fitted patterns
    reco_muons = dt_utils.reco_muons_from_sl_fits(fits=sl_dt_fits, verbose=False)
    n_reco_muons = len(reco_muons["ts"])
    print("reco_muons =",reco_muons)
    print("cosmic_muons =",cosmic_muons)

    """
    # plot differences between sim + reco
    n_hist_bins = 50
    hist_bins = {
        "x0": 2 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "y0": 2 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "z0": 2 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "theta": 0.1 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "phi": 0.1 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
        "ts": 10 * np.arange(-n_hist_bins//2,n_hist_bins//2+1,1)/n_hist_bins*2,
    }
    reco_muon_delta = copy.deepcopy(reco_muons) # = reco - simulated_cosmic
    reco_muon_delta["ts"] = np.full(n_reco_muons, 0, dtype=np.int64)
    for i in range(n_reco_muons):
        muon_id = reco_muons["muon_id"][i]
        for k in hist_bins.keys():
            if k in ["ts"]: 
                reco_muon_delta[k][i] = int(reco_muons[k][i]) - int(cosmic_muons[k][muon_id])
            else:
                reco_muon_delta[k][i] = reco_muons[k][i] - cosmic_muons[k][muon_id]
    for k in hist_bins.keys():
        hists, edges, centers = hist_utils.calculate_hist(data=reco_muon_delta, key=k, bin_centers=hist_bins[k])
        round_digits = 0 if k in ["ts"] else 2
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=k, round_digits=round_digits)
    """

    # cut reco muons to only the ones that hit the scintillator
    reco_muons = muon_utils.cut_muons_by_area(muons=reco_muons, xmin=derived_params._scintillator_sensitive_coordinates[0][0], xmax=derived_params._scintillator_sensitive_coordinates[0][1], ymin=derived_params._scintillator_sensitive_coordinates[1][0], ymax=derived_params._scintillator_sensitive_coordinates[1][1], z0=derived_params._scintillator_sensitive_coordinates[2])

    # calculate expected scintillator hits from reco muons
    scint_reco_muon_hits = scint_utils.hits_from_muons(muons=reco_muons)
    print("scint_reco_muon_hits =",scint_reco_muon_hits)
    
    # reco muon areas from scintillator hits
    reco_muon_areas = scint_utils.reco_muon_area_from_hits(hits=scint_reco_muon_hits)
    print("reco_muon_areas =",reco_muon_areas)

    #pickle_file = "test.pcl"
    #data_utils.store_pickle(data=reco_muon_areas, file=pickle_file)
    #data_utils.load_pickle(file=pickle_file)
    #print("reco_muon_areas =",reco_muon_areas)

    #input("Press enter to exit.")
    #exit()

    """
    # create hists for dt hits
    hist_bins = {
        "dt": np.arange(-75, 575+1, 5),
        "sl": np.arange(1, 3+1, 1),
        "bx": "auto",
        "tdc": np.arange(0, 32, 1),
        "muon_id": np.arange(0, n_muons, 1),
    }
    for k in hist_bins.keys():
        hists, edges, centers = hist_utils.calculate_hist(data=dt_muon_hits, key=k, bin_centers=hist_bins[k])
        round_digits = 1 if k in ["tdc"] else 0
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=k, round_digits=round_digits)
    """

    ### plot full geometry

    # generate dt cell data
    dt_cell_data = dt_utils._chamber_data()
    ## illustrate patterns that should be recognized
    #for i, (pat_name, pat_rel_wi) in enumerate(params._dt_sl_patterns.items()):
    #    start_wi = 6*i+3
    #    for ly in range(4):
    #        cell_data[1][ly][start_wi+pat_rel_wi[ly]]["color"] = "tab:red"
    # mark dt hits in chamber
    for i in range(len(dt_muon_hits["ch"])):
        sl, ly, wi = dt_muon_hits["sl"][i], dt_muon_hits["ly"][i], dt_muon_hits["wi"][i]
        dt_cell_data[sl][ly][wi]["color"] = "aqua"
        
    # generate scintillator cell data
    scint_cell_data = scint_utils._scint_data()
    # mark scint hits in chamber
    for i in range(len(scint_reco_muon_hits["ch"])):
        ly, st = scint_reco_muon_hits["ly"][i], scint_reco_muon_hits["st"][i]
        scint_cell_data[ly][st]["color"] = "aqua"
            
    # actual plotting
    show_wires = True
    for orient in ["phi", "theta"]:
        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=(12,4))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
        # plot chamber geometry
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=show_wires)
        # plot scintillator geometry
        ax = geoplot_utils.scintillator_ax(ax=ax, orient=orient, cell_data=scint_cell_data)
        # plot muon simulated track
        for i in range(n_muons):
            ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=cosmic_muons, muon_id=i, color="tab:green")
        # plot individual simulated muon dt hits
        for i in range(n_muons):
            ax = geoplot_utils.cell_hits_ax(ax=ax, orient=orient, dt_hits=dt_muon_hits, muon_id=i, color="tab:green")
        # plot individual simulated muon dt sl fits
        for i in range(n_patterns):
            ax = geoplot_utils.chamber_muon_fit_ax(ax=ax, orient=orient, sl_dt_fits=sl_dt_fits, pattern_id=i, color="red")
        # plot reconstructed muon
        for i in range(n_reco_muons):
            ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muons=reco_muons, muon_id=i, color="tab:blue")
        # plot reconstructed muon scint hits
        for i in range(n_reco_muons):
            ax = geoplot_utils.scint_hits_ax(ax=ax, orient=orient, scint_hits=scint_reco_muon_hits, muon_id=i, color="tab:blue")
        # plot reconstructed reconstructed muon area in scintillator
        ax = geoplot_utils.scint_muon_area_ax(ax=ax, orient=orient, scint_muon_areas=reco_muon_areas, muon_id=i, color="red")
        # axis limits
        ax.margins(x=0.05, y=0.05)
        # text labels
        axbox = ax.get_position()
        x_topleft = axbox.p0[0]
        x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
        ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
        ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
        #description = params._dt_chamber["name"]
        description = ""
        if orient == "theta":
            description += "$y$-$z$-plane (SL-$\\theta$ view)"
            ax.set_xlabel("$y$ [mm]")
            ax.set_ylabel("$z$ [mm]")
        elif orient == "phi":
            description += "$x$-$z$-plane (SL-$\\phi$ view)"
            ax.set_xlabel("$x$ [mm]")
            ax.set_ylabel("$z$ [mm]")
        ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
        # show/store figure
        fig.show()

    input("Press enter to exit.")


if __name__ == "__main__":
    main()










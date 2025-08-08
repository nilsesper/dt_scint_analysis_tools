#########################
# generate dummy data dumpfile
#########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    dt_hits = timestamp_utils.add_timestamp(hits=dt_hits)
    dt_hits = timestamp_utils.sort_by_timestamp(hits=dt_hits)
    print("dt_hits =",dt_hits)

    scint_hits = scint_utils.extract_scint_hits(hits=dumpfile_hits)
    scint_hits = timestamp_utils.add_timestamp(hits=scint_hits)
    scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
    print("scint_hits =",scint_hits)

    #cut_dt_hits = data_utils.cut_data(data=dt_hits, conditions=[("sl", "==", 1), ("wi", ">=", 20), ("wi", "<=", 40)])
    dt_sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits)
    print("dt_sl_patterns =",dt_sl_patterns)

    # generate cosmic muons
    #dummy_muon = {"x0": 1000, "y0": 1000, "z0": 100, "theta": 10*np.pi/180, "phi": 20*np.pi/180, "ts": 1000}
    n_muons = 10
    t_step = 1000
    cosmic_muons = []
    for i in range(n_muons):
        cosmic_muons.append( muon_utils.generate_cosmic_muon(xrange=[-100, 2300], yrange=[-100, 2600], z0=100, ts=t_step*i) )
    
    # propagate cosmic muons through dt chamber
    dt_muon_hits = muon_utils.dt_hits_from_muons(muons=cosmic_muons)
    print("dt_muon_hits =",dt_muon_hits)

    # create hists for dt hits
    hist_bins = {
        "dt": np.arange(-75, 575+1, 5),
        "sl": np.arange(1, 3+1, 1),
        "bx": "auto",
        "tdc": np.arange(-0.5, 32.5+1, 1),
        "muon_id": "auto"
    }
    for k in hist_bins.keys():
        hists, edges, centers = hist_utils.calculate_hist(data=dt_muon_hits, key=k, bin_centers=hist_bins[k])
        round_digits = 1 if k == "tdc" else 0
        hist_utils.plot_1hist(hist=hists, centers=centers, xlabel=k, round_digits=round_digits)


    #"""
    ### plot chamber

    # generate cell data to plot chamber
    cell_data = dt_utils._chamber_data()
    ## illustrate patterns that should be recognized
    #for i, (pat_name, pat_rel_wi) in enumerate(params._dt_sl_patterns.items()):
    #    start_wi = 6*i+3
    #    for ly in range(4):
    #        cell_data[1][ly][start_wi+pat_rel_wi[ly]]["color"] = "tab:red"
    # mark muon hits in chamber
    for i in range(len(dt_muon_hits["ch"])):
        sl, ly, wi = dt_muon_hits["sl"][i], dt_muon_hits["ly"][i], dt_muon_hits["wi"][i]
        cell_data[sl][ly][wi]["color"] = "aqua"
            
    ### plot chamber (phi orientation)
    orient = "phi"
    show_wires = False
    # generate plot
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
    # plot chamber geometry
    ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=cell_data, wire=show_wires)
    # plot muon track
    for i in range(n_muons):
        ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muon=cosmic_muons[i], color="red")
    # axis limits
    ax.margins(x=0.05, y=0.05)
    # text labels
    axbox = ax.get_position()
    x_topleft = axbox.p0[0]
    x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
    ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
    ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
    description = params._dt_chamber["name"]
    if orient == "theta":
        description += ", SL-$\\theta$ view"
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    elif orient == "phi":
        description += ", SL-$\\phi$ view"
        ax.set_xlabel("$y$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
    # show/store figure
    fig.show()

    ### plot chamber (theta orientation)
    orient = "theta"
    # generate plot
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
    # plot chamber geometry
    ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=cell_data, wire=show_wires)
    # plot muon track
    for i in range(n_muons):
        ax = geoplot_utils.muon_ax(ax=ax, orient=orient, muon=cosmic_muons[i], color="red")
    # axis limits
    ax.margins(x=0.05, y=0.05)
    # text labels
    axbox = ax.get_position()
    x_topleft = axbox.p0[0]
    x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
    ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
    ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
    description = params._dt_chamber["name"]
    if orient == "theta":
        description += ", SL-$\\theta$ view"
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    elif orient == "phi":
        description += ", SL-$\\phi$ view"
        ax.set_xlabel("$y$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
    # show/store figure
    fig.show()
    #"""

    input("Press enter to exit.")


if __name__ == "__main__":
    main()










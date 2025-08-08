###########################################
### HISTROGRAM UTILS
###########################################

import numpy as np
import copy
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params
import analysis_tools.utils.muon_utils as muon_utils

# -----------------------------------------

### generate one histogram from given data
# binning and given conditions for selection of hits from data for one specific key
# arguments:
#   data: data dict
#   key: key of data dict to create histogram from
#   bin_centers: centers of bins as list/array
#       special keywords:
#           auto: leave binning up to np.histogram
#           step1: bin width is fixed to 1, automatically choose binning from min to max value
def calculate_hist(data, key, bin_centers):
    print(f"Calculating histogram for data key \"{key}\"...")
    hists, edges, centers = [], [], []
    ### in case data is empty, return empty arrays
    if len(data[key]) == 0:
        return hists, edges, centers
    ### set bins
    # check for special keywords
    if type(bin_centers) in [type("")]: # use string keyword for bins option in np.histogram if given
        if bin_centers == "auto":
            edges = "auto" # automatic binning
        elif bin_centers == "step1": # automatic binning with bin width of 1
            dmin = int(np.amin(data[key]))
            dmax = int(np.amax(data[key]))
            centers = np.linspace(dmin-1, dmax+1, dmax-dmin+3)
    # else use given bin centers
    else:
        centers = bin_centers
    # calculate edges from centers
    distance = np.mean(np.diff(centers))/2
    if edges != "auto":
        edges = np.zeros(len(centers)+1)
        for i in range(len(centers)):
            edges[i] = centers[i]-distance
        edges[len(centers)] = centers[-1]+distance
    ### calculate actual histograms
    hists, edges = np.histogram(data[key], bins=edges)
    centers = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    return hists, edges, centers

### plot one histogram
# arguments:
# hist: histogram entries (bin heights)
# centers: centers of histograms
@mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12})
def plot_1hist(hist, centers, *, vmin=None, vmax=None, scale="norm", bin_labels=True, show=True, store=False, xlabel="", rel_spacing=0, round_digits=0):
    print(f"Plotting one histogram...")
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    # plot hist
    barwidth = np.mean(np.diff(centers))*(1-rel_spacing) # relative spacing between bins
    ax.bar(centers, hist, width=barwidth, align="center")
    # bin labels
    if bin_labels == True:
        for i in range(len(centers)):
            if hist[i] == 0: continue
            if round_digits == 0:
                text_str = str(int(np.round(centers[i],round_digits)))
            elif round_digits == None:
                text_str = str(centers[i])
            else:
                text_str = str(np.round(centers[i],round_digits))
            ax.text(centers[i], hist[i], text_str,
                    horizontalalignment="center", verticalalignment="bottom", fontsize=6)
    # axis scale
    if scale == "norm": pass
    elif scale == "log": ax.set_yscale("log")
    # axis limits
    if len(hist) > 0:
        if vmin == None:
            if scale == "norm": vmin = 0
            if scale == "log": vmin = 0.5
        if vmax == None:
            if scale == "norm": vmax = np.amax(hist)*1.1
            if scale == "log": vmax = np.amax(hist)*np.exp(1.1)
        ax.set_ylim(bottom=vmin, top=vmax)
    if xlabel != "": ax.set_xlabel(xlabel)
    if show == True: fig.show()
    if store != False: fig.savefig(store)
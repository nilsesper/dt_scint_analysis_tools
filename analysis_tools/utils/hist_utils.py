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
def calculate_hist(data, key, *, bin_centers=None, bin_edges=None, silent=False):
    if not silent: print(f"Calculating histogram for data key \"{key}\"...")
    hists, edges, centers, underflow, overflow = [], [], [], 0, 0
    ### in case data is empty, return empty arrays
    if len(data[key]) == 0:
        print("EMPTY HIST DATA !!!")
        return hists, edges, centers, underflow, overflow
    ### set bins
    # BIN CENTERS
    if type(bin_centers) != type(None):
        # check for special keywords
        if type(bin_centers) in [type("")]: # use string keyword for bins option in np.histogram if given
            if bin_centers == "auto":
                #edges = "auto" # automatic binning
                n_auto_bins = 20
                dmin = np.int64(np.amin(data[key]))
                dmax = np.int64(np.amax(data[key]))
                centers = np.linspace(dmin-1, dmax+1, n_auto_bins)
            elif "auto" in bin_centers: 
                # "autoXX" = auto binning with XX bins
                n_auto_bins = int(bin_centers[4:])
                dmin = np.int64(np.amin(data[key]))
                dmax = np.int64(np.amax(data[key]))
                centers = np.linspace(dmin-1, dmax+1, n_auto_bins)
            elif bin_centers == "step1": # automatic binning with bin width of 1
                dmin = np.int64(np.amin(data[key]))
                dmax = np.int64(np.amax(data[key]))
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
    # BIN EDGES
    elif type(bin_edges) != type(None):
        edges = bin_edges
    # ---
    ### calculate actual histograms
    # create edges w/ over/underflow
    ou_step = 1
    ou_clip = [np.amin(edges)-ou_step, np.amax(edges)+ou_step]
    edges_with_ou = np.array(copy.deepcopy(edges))
    edges_with_ou = np.insert(edges_with_ou, 0, ou_clip[0])
    edges_with_ou = np.append(edges_with_ou, ou_clip[1])
    data_ou_clip = np.clip(data[key], a_min=ou_clip[0], a_max=ou_clip[1])
    hists_with_ou, edges_with_ou = np.histogram(data_ou_clip, bins=edges_with_ou)
    underflow = hists_with_ou[0]
    overflow = hists_with_ou[-1]
    # calculate hists, bin centers & edges w/o over/underflow
    hists = hists_with_ou[1:-1]
    edges = edges_with_ou[1:-1]
    centers = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    return hists, edges, centers, underflow, overflow

### plot one histogram
# arguments:
# hist: histogram entries (bin heights)
# centers: centers of histograms
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) # 'font.sans-serif': 'Arial'
def plot_1hist(hist, centers, *, vmin=None, vmax=None, scale="norm", bin_labels=True, show=True, store=False, xlabel="", ylabel="", rel_spacing=0, round_digits=0, silent=False, title=None):
    if not silent: print(f"Plotting one histogram...")
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
    if ylabel != "": ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)
    # tight layout
    fig.tight_layout()
    # show / save figure
    if show == True: fig.show()
    if store != False: fig.savefig(store)

### determine interval/range of histogram peak
# give rel_thres to determine where the peak starts & stops relative to the max value of the (whole) histogram
# returns list of lists indices of hist values / bin centers of all bins belonging to this peak - outer list is for all peaks
# [[peak indixes] for peaks]
def find_peak_indices(hist, rel_thres=0.01,*, silent=False):
    peak_indices = []
    peak_no = -1
    current_peak = False
    n_hist = len(hist)
    if n_hist == 0:
        return []
    thres = np.amax(hist)*rel_thres
    for i in range(n_hist):
        if hist[i] < thres:
            current_peak = False
        else:
            if not current_peak:
                current_peak = True
                peak_indices.append([])
                peak_no += 1
            peak_indices[peak_no].append(i)
    peak_indices = [np.array(idx_list) for idx_list in list(peak_indices)]
    return peak_indices

### calculate histogram peak position with weighted mean (bin centers = x, hist values = weights)
def weighted_mean_peak_position(hist, centers, err_hist, err_centers, *, silent=False):
    if len(hist) != len(centers) or len(hist) != len(err_centers) or len(hist) != len(err_hist):
        raise Exception("All lists must have the same length.")
    mean = np.sum(centers*hist)/np.sum(hist)
    err_mean = ( np.sum( ( hist/np.sum(hist) )**2 * err_centers**2 ) + np.sum( ( centers/np.sum(hist) - np.sum(centers*hist)/np.sum(hist)**2 )**2 * err_hist**2 ) )**(1/2)
    return mean, err_mean



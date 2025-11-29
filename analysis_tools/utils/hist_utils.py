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
                dmin = np.amin(data[key])
                dmax = np.amax(data[key])
                drange = dmax - dmin
                if drange > 0:
                    centers = np.linspace(dmin-drange*0.1, dmax+drange*0.1, n_auto_bins)
                else:
                    centers = np.linspace(dmin-1, dmax+1, n_auto_bins)
            elif "auto" in bin_centers: 
                # "autoXX" = auto binning with XX bins
                n_auto_bins = int(bin_centers[4:])
                dmin = np.amin(data[key])
                dmax = np.amax(data[key])
                drange = dmax - dmin
                #print(key, dmin, dmax, drange)
                if drange > 0:
                    centers = np.linspace(dmin-drange*0.1, dmax+drange*0.1, n_auto_bins)
                else:
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
        #if edges != "auto":
        #    edges = np.zeros(len(centers)+1)
        #    for i in range(len(centers)):
        #        edges[i] = centers[i]-distance
        #    edges[len(centers)] = centers[-1]+distance
    # BIN EDGES
        edges = np.zeros(len(centers)+1)
        for i in range(len(centers)):
            edges[i] = centers[i]-distance
        edges[len(centers)] = centers[-1]+distance
    # if edges given
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
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) # 'font.sans-serif': 'Arial'
def plot_1hist(hist, centers, *, vmin=None, vmax=None, scale="norm", bin_labels=True, show=True, store=False, xlabel="", ylabel="", rel_spacing=0, round_digits=0, silent=False, title=None, figsize=(12,8)):
    if not silent: print(f"Plotting one histogram...")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
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


###################################
###################################

### calculate bin centers from bin edges
def centers_from_edges(edges):
    centers = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    return centers

### calculate bin edges from bin centers
def edges_from_centers(centers):
    edges = np.zeros(len(centers)+1)
    distance = np.mean(np.diff(centers))/2
    for i in range(len(centers)):
        edges[i] = centers[i]-distance
    edges[len(centers)] = centers[-1]+distance
    return edges

### generate histogram edges
# manually or automatically (requires min max value of data)
# arg options:
#   "linear,start,stop,n_bins+1" ,
#   "range,start,stop+1" ,
#   "auto,n_bins" ,
#   "step1" ,
def generate_histogram_edges(arg, *, data_min_val=None, data_max_val=None):
    arg_split = arg.split(",")
    edges = None
    ## manual binning
    # "linear" = linear bin edges
    if arg_split[0] == "linear":
        if len(arg_split) != 3+1:
            raise Exception(f"arg: Need linear,start,stop,n_bins+1.")
        edges = np.linspace(float(arg_split[1]), float(arg_split[2]), int(arg_split[3]))
    # "range" = integer range bin edges
    elif arg_split[0] == "range":
        if len(arg_split) != 2+1:
            raise Exception(f"arg: Need range,start,stop+1.")
        edges = np.arange(int(arg_split[1]), int(arg_split[2]))
    ## automatic binning
    # "auto" = automatic bin edges for fixed no of bins
    elif arg_split[0] == "auto":
        if len(arg_split) != 1+1:
            raise Exception(f"arg: Need auto,n_bins.")
        n_auto_bins = int(arg_split[1])
        edges = np.linspace(data_min_val, data_max_val, n_auto_bins+1)
        #edges = hist_utils.edges_from_centers(centers)
    # "step1" =  automatic bin edges for bin width = 1
    elif arg_split[0] == "step1":
        if len(arg_split) != 1:
            raise Exception(f"arg: Need step1.")
        edges = np.linspace(int(data_min_val)-1, int(data_max_val)+1, int(data_max_val)-int(data_min_val)+3)
    else:
        raise Exception(f"arg: Need linear / range / auto / step1.")
    n_bins = len(edges)-1
    centers = centers_from_edges(edges)
    return edges, n_bins, centers

### prepare empty hist
def create_empty_histogram(edges):
    n_bins = len(edges)-1
    centers = centers_from_edges(edges)
    hist = np.zeros(n_bins) # data hist
    entries, underflow, overflow = 0, 0, 0
    hist_err_right = np.zeros(n_bins) # data + err_data hist 
    hist_err_left = np.zeros(n_bins) # data - err_data hist
    return centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left

### generate one histogram from given data
# for given bin edges
def calculate_histogram(data, edges):
    # create edges w/ over/underflow
    ou_step = 1
    ou_clip = [np.amin(edges)-ou_step, np.amax(edges)+ou_step]
    edges_with_ou = np.array(copy.deepcopy(edges))
    edges_with_ou = np.insert(edges_with_ou, 0, ou_clip[0])
    edges_with_ou = np.append(edges_with_ou, ou_clip[1])
    data_ou_clip = np.clip(data, a_min=ou_clip[0], a_max=ou_clip[1])
    hist_with_ou, edges_with_ou = np.histogram(data_ou_clip, bins=edges_with_ou)
    underflow = hist_with_ou[0] # entries in underflow
    overflow = hist_with_ou[-1] # entries in overflow
    entries = np.sum(hist_with_ou[1:-1]) # entries in hist, so that: (entries + overflow+ + underflow) = len(data)
    # calculate hists, bin centers & edges w/o over/underflow
    hist = hist_with_ou[1:-1]
    edges = edges_with_ou[1:-1]
    centers = centers_from_edges(edges)
    return hist, edges, centers, entries, underflow, overflow
    
### generate one histogram and two histograms with data shifted by +- err_data
# data passed as np array / list
def calculate_histogram_and_shifted_histograms(data, edges, err_data=None):
    # prepare data frame
    n_bins = len(edges)-1
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = create_empty_histogram(edges)
    # create histogram of specified key
    hist, edges, centers, entries, underflow, overflow = calculate_histogram(data=data, edges=edges)
    # propagate error of data into histogram
    if type(err_data) != type(None):
        hist_err_right = np.zeros(n_bins)
        hist_err_left = np.zeros(n_bins)
        #  shift right (data+err_data)
        hist_err_right, _, _, _, _, _ = calculate_histogram(data=data+err_data, edges=edges)
        #  shift left (data-err_data)
        hist_err_left, _, _, _, _, _ = calculate_histogram(data=data-err_data, edges=edges)
    else:
        hist_err_right = None
        hist_err_left = None
    return hist, edges, centers, entries, underflow, overflow, hist_err_right, hist_err_left

### combine hist and shifted hists to calculate uncertainty
# also calculate poisson uncertainty per bin
def calculate_hist_uncertainty(hist, *, hist_err_right=None, hist_err_left=None, do_stat_err=True):
    n_bins = len(hist)
    ## stat err
    if do_stat_err:
        # statistical error, slip to 1 entry if no entries
        err_hist_stat = np.clip(a=np.sqrt(hist), a_min=1, a_max=None)
    else:
        err_hist_stat = np.zeros(n_bins)
    ## data err
    if type(hist_err_right) != type(None) and type(hist_err_left) != type(None):
        # error of data entries (mean shift from left & right)
        err_hist_data = (np.abs(hist_err_right-hist)+np.abs(hist_err_left-hist))/2
    else:
        err_hist_data = np.zeros(n_bins)
    ## combine errors (assume uncorrelated)
    err_hist = np.sqrt(err_hist_stat**2 + err_hist_data**2)
    return err_hist

### plot histogram with uncertainty bar into given ax, return ax
def plot_histogram(ax, hist, centers, *, err_hist=None):
    barwidth = np.mean(np.diff(centers))
    ax.bar(centers, hist, width=barwidth, align="center", facecolor="tab:blue")
    ax.bar(centers, bottom=hist-err_hist, height=2*err_hist, width=barwidth, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
    ax.set_ylim(bottom=0, top=np.amax(hist+err_hist)*1.1)
    return ax





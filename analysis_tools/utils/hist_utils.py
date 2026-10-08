###########################################
### HISTROGRAM UTILS
###########################################

import numpy as np
import copy
from matplotlib.ticker import ScalarFormatter

import analysis_tools.params.params as params

# -----------------------------------------




### calculate histogram peak position with weighted mean (bin centers = x, hist values = weights)
def weighted_mean_peak_position(hist, centers, err_hist, err_centers, *, silent=False):
    if len(hist) != len(centers) or len(hist) != len(err_centers) or len(hist) != len(err_hist):
        raise Exception("All lists must have the same length.")
    mean = np.sum(centers*hist)/np.sum(hist)
    err_mean = ( np.sum( ( hist/np.sum(hist) )**2 * err_centers**2 ) + np.sum( ( centers/np.sum(hist) - np.sum(centers*hist)/np.sum(hist)**2 )**2 * err_hist**2 ) )**(1/2)
    return mean, err_mean


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

### calculate bin centers from bin edges
def centers_from_edges(edges):
    centers = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    return centers



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
# calculates symm and asymm errors
def calculate_hist_uncertainty(hist, *, hist_err_right=None, hist_err_left=None, do_stat_err=True, min_stat_err=1):
    n_bins = len(hist)
    ## stat err (clip to min_stat_err)
    if do_stat_err:
        err_hist_stat = np.clip(a=np.sqrt(hist), a_min=min_stat_err, a_max=None)
    else:
        err_hist_stat = np.zeros(n_bins)
    ## final err
    err_hist_up = err_hist_stat
    # asymm err: for now use same as symm err
    err_hist_down = err_hist_stat
    err_hist = err_hist_stat
    return err_hist, err_hist_down, err_hist_up

### plot histogram with uncertainty bar into given ax, return ax
# give hist to plot
# optionally give err_hist (symm errors)
# or err_hist_down and err_hist_up (asymm errors)
def plot_histogram(ax, hist, centers, *, err_hist=None, err_hist_down=None, err_hist_up=None, log_scale=False, power_limits=[-2, 2], add_info=False, overflow=None, underflow=None, entries=None, bin_unit=None, bin_width_digits=3, set_y_label=True, info_font_size=params._info_font_size, info_loc="top right", bool_plus_label = False, pluslabel = ""):
    barwidth = np.mean(np.diff(centers))
    ax.bar(centers, hist, width=barwidth, align="center", facecolor="tab:blue")
    # if up down errors given
    if type(err_hist_down) != type(None) and type(err_hist_up) != type(None):
        ax.bar(centers, bottom=hist-err_hist_down, height=err_hist_down+err_hist_up, width=barwidth, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
        max_hist_val = np.amax(hist+err_hist_up)
    # if symm errors given
    elif type(err_hist) != type(None):
        ax.bar(centers, bottom=hist-err_hist, height=2*err_hist, width=barwidth, align="center", hatch="xxx", fill=False, edgecolor="0.2", linestyle="")
        max_hist_val = np.amax(hist+err_hist)
    if not log_scale:
        ax.set_ylim(bottom=0, top=max_hist_val*1.1)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits(power_limits) # 10^X power limits for prescale
    else:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=max_hist_val*np.exp(1.1))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 
    ax.xaxis.get_major_formatter().set_powerlimits(power_limits) # 10^X power limits for prescale
    # add text in histogram with hist info / stats
    if add_info:
        #barwidth_str = math_utils.latex_float(barwidth)
        barwidth_str = f"{barwidth:.{bin_width_digits}g}"
        if bin_unit != None:
            barwidth_str += f" {bin_unit}"
        if bool_plus_label == False:
            info_str = f"entries = {entries:,}\nunderflow = {underflow:,}\noverflow = {overflow:,}\ntotal = {entries+overflow+underflow:,}\nbin count = {len(centers):,}\nbin width = {barwidth_str}"
        elif bool_plus_label == True:
            info_str = f"entries = {entries:,}\nunderflow = {underflow:,}\noverflow = {overflow:,}\ntotal = {entries+overflow+underflow:,}\nbin count = {len(centers):,}\nbin width = {barwidth_str}\n{pluslabel}"
        #ax.text(0.99, 0.99, info_str, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
        ax = add_infobox(ax, info_str=info_str, info_font_size=info_font_size, info_loc=info_loc)
    # set y label
    if set_y_label:
        ax.set_ylabel(f"Counts")
    return ax

### add infobox to ax subplot
def add_infobox(ax, info_str, info_font_size=params._info_font_size, info_loc="top right"):
    if info_loc == "top right":
        ax.annotate(info_str, xy=(1, 1), xytext=(-info_font_size-0.5, -info_font_size-0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "bottom right":
        ax.annotate(info_str, xy=(1, 0), xytext=(-info_font_size-0.5, info_font_size+0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "bottom left":
        ax.annotate(info_str, xy=(0, 0), xytext=(info_font_size+0.5, info_font_size+0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='left', verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "top left":
        ax.annotate(info_str, xy=(0, 1), xytext=(info_font_size+0.5, -info_font_size-0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "top center":
        ax.annotate(info_str, xy=(0.5, 1), xytext=(0, -info_font_size-0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='center', verticalalignment='top', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "bottom center":
        ax.annotate(info_str, xy=(0.5, 0), xytext=(0, info_font_size+0.5), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='center', verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "center left":
        ax.annotate(info_str, xy=(0, 0.5), xytext=(info_font_size+0.5, 0), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    elif info_loc == "center right":
        ax.annotate(info_str, xy=(1, 0.5), xytext=(-info_font_size-0.5, 0), xycoords='axes fraction', textcoords='offset points', fontsize=info_font_size, horizontalalignment='right', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='lightgray', alpha=params._hist_info_alpha))
    return ax


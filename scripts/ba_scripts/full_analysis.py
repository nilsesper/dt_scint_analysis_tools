#################################################################
### import dumpfile and extract dt hits (and optionally raw scint hits)
# store dt hits (and optionally raw scint hits) as pkl file
#################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params, params_justus

import subprocess
import atexit
import sys
import time
from tqdm import tqdm
from scipy.optimize import curve_fit
# ---------------------------------------------------------------
# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    ###################################################
    # IMPORTANT
    # When not using example data (dt_cosmics.txt) use params_justus
    main_path = "example_data/"
    dumpfile_name = "cosmic_85-15_3600-1800-1200_run2_th20.txt"
    dataset_name =  "cosmic_85-15_3600-1800-1200_run2_th20"
    dt_hits_name = dataset_name + "_hits.pcl"
    raw_scint_hits_name = dataset_name + "_raw_scint_hits.pcl"
    ts_range_name = dataset_name + "_ts_range.txt"
    ### --- manuell gesetzte Parameter (ersetzt argparse) ---
    input_dumpfile      = main_path + dumpfile_name
    dt_hit_diff_name = dataset_name + "_hit_diff.pcl"
    
    nodeadtime          = True  # True setzen, um dead time zu ignorieren
    deadtime_preffix = "nodeadtime" if nodeadtime else "deadtime"
    dt_hits_file        = main_path + dt_hits_name
    dt_hit_diff_hist_file = main_path + dt_hit_diff_name
    # optionale Schritte:
    use_timestamp_sync   = True   # add_timestamp + sort_by_timestamp anwenden
    extract_scint_hits    = True   # raw scint hits extrahieren und speichern
    raw_scint_hits_file   = main_path + raw_scint_hits_name  # nur relevant falls extract_scint_hits=True

    create_ts_file        = True   # ts_range Datei erzeugen
    ts_range_file          = main_path + ts_range_name  # nur relevant falls create_ts_file=True
    plot_save_path         = main_path + "plots/photo_peak/"
    plot_type = ".png"
    # ---------------------------------------------------------

    #################
    ### data import
    print(f"###### Importing dumpfile \"{input_dumpfile}\"...")
    dumpfile_hits = data_utils.import_raw(file_name=input_dumpfile) # dummy_filename, data_filename
    # cut first entries of data (which might be old data from htg buffer)
    dumpfile_hits = data_utils.cut_first_entries(data=dumpfile_hits, n_cut=params._dumpfile_hits_to_skip)

    ### optionally sync timestamps
    if use_timestamp_sync:
        dumpfile_hits = timestamp_utils.add_timestamp(hits=dumpfile_hits)
        dumpfile_hits = timestamp_utils.sort_by_timestamp(hits=dumpfile_hits)

    print("dumpfile_hits =",dumpfile_hits)

    ### extract dt hits
    print(f"###### Extracting dt hits...")
    dt_hits = dt_utils.extract_dt_hits(
        hits=dumpfile_hits,
        has_timestamp=use_timestamp_sync,
        ignore_deadtime=nodeadtime,
    )
    print("dt_hits =",dt_hits)


    ### store dt hits to pcl file
    print(f"###### Storing dt hit data to file \"{dt_hits_file}\"...")
    data_utils.store_pickle(data=dt_hits, file=dt_hits_file)


    ####################

    ### fixed bins
    n_bins = 2500
    edges = np.linspace(0, 5000, n_bins+1) # in tu

    ### prepare hists
    centers, hist, entries, underflow, overflow, hist_err_right, hist_err_left = hist_utils.create_empty_histogram(edges=edges)

    ### calculate histograms for sub datasets, merge hists consecutively
    ## import all data and apply the respective timing offset
    ## extract the data of the specified hist key and calculate hist
    print(f"CALCULATING DT HIT TIME DIFFERENCE HISTOGRAM...")

    sub_data = dt_hits
    ## apply ts shift
    #for ts_key in ts_keys:
    #    if ts_key in sub_data.keys():
    #        sub_data[ts_key] = sub_data[ts_key] + ts_offset[data_idx]
    ### do something with data
    ## calculate time difference between hits
    ch_list = []
    err_ch_list = []
    cut_layers = True # cut layers to calculate time difference only for hits in the same layer

    for sl in range(1,4):
        for ly in range(0,4):
            for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
                sub_data_cut = data_utils.cut_data(data=sub_data, conditions=[("sl","==",sl), ("ly","==",ly), ("wi","==",wi)], silent=True)
                sub_data_cut = timestamp_utils.sort_by_timestamp(hits=sub_data_cut, silent=True)
                n_sub_data_cut = data_utils.length(sub_data_cut)
                ts_diff_list = []
                err_ts_diff_list = []
                for i in range(1,n_sub_data_cut):
                    ts_diff_list.append(sub_data_cut["ts"][i] - sub_data_cut["ts"][i-1])
                    err_ts_diff_list.append( np.sqrt(sub_data_cut["err_ts"][i]**2 + sub_data_cut["err_ts"][i]**2) )
                ts_diff_list = np.array(ts_diff_list)
                err_ts_diff_list = np.array(err_ts_diff_list)
                ch_list.append({"key": ts_diff_list})
                err_ch_list.append({"key": err_ts_diff_list})
    merged_ts_diff = data_utils.merge_dataset(split_data=ch_list, silent=True)["key"]
    merged_err_ts_diff = data_utils.merge_dataset(split_data=err_ch_list, silent=True)["key"]

    # create histogram of specified key and shifted hists to respect data error
    hist_, _, _, entries_, underflow_, overflow_, hist_err_right_, hist_err_left_ = hist_utils.calculate_histogram_and_shifted_histograms(data=merged_ts_diff, edges=edges, err_data=merged_err_ts_diff)
    # add to combined histogram
    hist += hist_
    entries += entries_
    underflow += underflow_
    overflow += overflow_
    hist_err_right += hist_err_right_
    hist_err_left += hist_err_left_


    ### error calculation for full hist
    err_hist, err_hist_down, err_hist_up = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    ### calculate once only stat unc
    err_hist_stat = np.sqrt(hist)

    print(f"created histogram:")
    print(f"  entries   =  {entries}  ,  underflow =  {underflow}  ,  overflow  =  {overflow}")

    ### store histogram into file
    specific_data= {
        "edges": edges,
        "centers": centers,
        "hist": hist,
        "err_hist": err_hist,
        "err_hist_stat": err_hist_stat,
        "err_hist_down": err_hist_down,
        "err_hist_up": err_hist_up,
        "entries": entries,
        "underflow": underflow,
        "overflow": overflow,
    }
    specific_data_file = dt_hit_diff_hist_file 
    print(f"storing specific data as {specific_data_file}...")
    data_utils.store_pickle(data=specific_data, file=specific_data_file)


    ####################

    cell_half_width = 21000 # um
    err_cell_half_width = 100 # um

    legend_font_size = 13

    ### hist to plot
    # read data
    start_idx = 0
    hist = np.array(specific_data["hist"])[start_idx:]
    err_hist = np.array(specific_data["err_hist"])[start_idx:]
    err_hist_down = np.array(specific_data["err_hist_down"])[start_idx:]
    err_hist_up = np.array(specific_data["err_hist_up"])[start_idx:]
    edges = np.array(specific_data["edges"])[start_idx:]*0.78 # convert from tu to ns
    centers = hist_utils.centers_from_edges(edges)
    bins = centers
    overflow = specific_data["overflow"]
    underflow = specific_data["underflow"]

    ######################
    ##### poisson bg subtraction

    ### plot dt hit differences
    # plot hist, 
    print("Plotting full t_diff hist...")
    fig_size = (8, 6)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = hist_utils.plot_histogram(ax, hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
    ax.set_xlim(0,np.amax(bins))
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    hist_plot_file = plot_save_path + "_DIFF_SPECIFIC_ALL" + plot_type
    print(f"store histogram plot as {hist_plot_file}.")
    fig.savefig(hist_plot_file)
    print(f"Done saving hist as {hist_plot_file}\n")

    ### remove exponential "poisson" background
    print("\nFitting exp. backgrund...")
    boarder = 2000 # ns
    fit_index_range = (bins > boarder) # > 1000 ns
    extrapol_index_range = (bins <= boarder)
    fit_bins = bins[fit_index_range]
    fit_hist = hist[fit_index_range]
    err_fit_hist = err_hist[fit_index_range]
    def f_bg_fit(x, a, b):
        return a*np.exp(-x/b)
    def err_f_bg_fit(x, a, b, err_a, err_b):
        return np.sqrt( (err_a*np.exp(-x/b))**2 + (-1/b*a*np.exp(-x/b)*err_b)**2 )
    p0 = (1000, 100)
    popt, pcov, infodict, mesg, _ = curve_fit(f=f_bg_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True)
    a_fit, b_fit = popt
    err_a_fit = np.sqrt(pcov[0][0])
    err_b_fit = np.sqrt(pcov[1][1])
    chi2 = np.sum((fit_hist - f_bg_fit(x=fit_bins, a=a_fit, b=b_fit))**2/err_fit_hist**2)
    ndf = len(fit_hist)-2
    chi2ndf = chi2/ndf
    print(f"exp fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) ns")
    print(f"  a = {a_fit} +- {err_a_fit}")
    print(f"  b = {b_fit} +- {err_b_fit}")
    print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

    ## plot fit, with residual plot
    print("\nFit successfull \n beginn plotting of dt hit diff with bg fit...")
    fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5,1))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins))*(1-rel_spacing)
    ax[0] = hist_utils.plot_histogram(ax[0], hist=hist, centers=bins, err_hist_down=err_hist_down, err_hist_up=err_hist_up, log_scale=True, power_limits=[-4,4], add_info=True, entries=int(np.sum(hist)), overflow=overflow, underflow=underflow, bin_unit="ns")
    fit_label = f"""Exponential fit:
    $f(\\Delta T) = a \\cdot e^{{-x/b}}$
    $a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$
    $b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
    $\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
    ax[0].plot(fit_bins, f_bg_fit(fit_bins, a=a_fit, b=b_fit), color="tab:red", label=fit_label)
    ax[0].fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
    ax[0].plot(bins[extrapol_index_range], f_bg_fit(bins[extrapol_index_range], a=a_fit, b=b_fit), color="tab:red", linestyle="--", label="Extrapolated fit")
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
    ax[0].legend(loc="lower right", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
    residuals = fit_hist - f_bg_fit(fit_bins, a=a_fit, b=b_fit)
    err_residuals = err_fit_hist
    ax[1].axhline(y=0, color="gray", linewidth=1)
    ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1, linestyle="")
    ax[1].set_xlim(0,np.amax(bins))
    ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    hist_plot_file = plot_save_path + "t_diff_bgfit" + plot_type
    print(f"store histogram plot as {hist_plot_file}.")
    fig.savefig(hist_plot_file)
    print(f"plot saved to {hist_plot_file}")

    ### subtract exp bg
    print("\nSubtracting background from hist...")
    hist_nobg = hist - f_bg_fit(bins, a=a_fit, b=b_fit)
    err_hist_nobg = np.sqrt( err_hist**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    err_hist_nobg_down = np.sqrt( err_hist_down**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    err_hist_nobg_up = np.sqrt( err_hist_up**2 + err_f_bg_fit(bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit)**2 )
    bins_nobg = bins
    print("\nDone subtracting bg from hist.")

    # plot wo bg
    print("\nPlotting hist without bg...")
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing)
    ax = hist_utils.plot_histogram(ax, hist=hist_nobg, centers=bins_nobg, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
    info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")
    ax.set_xlim(0,600)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
    fig.tight_layout()
    fig.show()
    ## store plot
    hist_plot_file = plot_save_path + "t_diff_bg_sub" + plot_type
    print(f"store histogram plot as {hist_plot_file}.")
    fig.savefig(hist_plot_file)
    print(f"\nSaved plot to {hist_plot_file}")

    """
    # plot wo bg -- IN TDC UNITS
    fig, ax = plt.subplots(1, 1, figsize=(7,6.5))
    rel_spacing = 0
    barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing)
    ax = hist_utils.plot_histogram(ax, hist=hist_nobg, centers=bins_nobg/0.78, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
    info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg/0.78)):.3g} TU"
    ax = hist_utils.add_infobox(ax=ax, info_str=info_str, info_loc="top right")
    ax.set_xlim(0,600/0.78)
    ax.set_xlabel("$\\Delta T_\\text{cell}$ [TU]")
    fig.tight_layout()
    fig.show()
    ## store plot
    hist_plot_file = plot_save_path + "_DIFF_SPECIFIC_NOBG_tdc" + plot_type
    print(f"store histogram plot as {hist_plot_file}.")
    fig.savefig(hist_plot_file)
    """


    ######################
    ##### fit peak position

    ### fit parabola photopeak to determine position
    fit_index_range = (bins_nobg >= 400) & (bins_nobg <= 440)
    fit_bins = bins_nobg[fit_index_range]
    fit_hist = hist_nobg[fit_index_range]
    err_fit_hist = err_hist_nobg[fit_index_range]
    def f_peak_fit(x, a, b, c):
        return a*(x-b)**2+c
    def err_f_peak_fit(x, a, b, c, err_a, err_b, err_c):
        return np.sqrt( ( err_a*(x-b)**2 )**2 + ( -2*a*(x-b)*err_b )**2 + ( err_c )**2 )
    p0 = (-1, 415, 1000)
    try: 
        popt, pcov, infodict, mesg, _ = curve_fit(f=f_peak_fit, xdata=fit_bins, ydata=fit_hist, p0=p0, sigma=err_fit_hist, absolute_sigma=True, full_output=True, )
        a_fit, b_fit, c_fit = popt
        err_a_fit = np.sqrt(pcov[0][0])
        err_b_fit = np.sqrt(pcov[1][1])
        err_c_fit = np.sqrt(pcov[2][2])
        chi2 = np.sum((fit_hist - f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit))**2/err_fit_hist**2)
        ndf = len(fit_hist)-2
        chi2ndf = chi2/ndf
        print(f"parabola fit to interval delta_t = ({np.amin(fit_bins)}, {np.amax(fit_bins)}) TU")
        print(f"  a = {a_fit} +- {err_a_fit}")
        print(f"  b = {b_fit} +- {err_b_fit}")
        print(f"  c = {c_fit} +- {err_c_fit}")
        print(f"  chi2/ndf = {chi2} / {ndf} = {chi2ndf}")

        # plot fit
        fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5,1))
        rel_spacing = 0
        barwidth = np.mean(np.diff(bins_nobg))*(1-rel_spacing)
        ax[0] = hist_utils.plot_histogram(ax[0], hist=hist_nobg, centers=bins_nobg, err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up, log_scale=False, power_limits=[-3,3])
        info_str = f"entries = {int(np.sum(hist_nobg))}\nbin count = {len(centers)}\nbin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
        ax[0] = hist_utils.add_infobox(ax=ax[0], info_str=info_str, info_loc="top left")

        fit_label = f"""Parabolic fit:
        $f(\\Delta T) = a \\cdot (\\Delta T-b)^2+c$
        $a=({np.round(a_fit,2):.2f}\\pm{np.round(err_a_fit,2):.2f})$ 1/ns${{}}^2$
        $b=({np.round(b_fit,2):.2f}\\pm{np.round(err_b_fit,2):.2f})$ ns
        $c=({np.round(c_fit,0):.0f}\\pm{np.round(err_c_fit,0):.0f})$
        $\\chi^2 / N_{{df}} = {chi2:.1f}\\; / \\;{ndf:.0f} ={np.round(chi2ndf,1):.1f}$"""
        ax[0].plot(fit_bins, f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit), color="tab:red", label=fit_label)
        ax[0].fill_between(fit_bins, y1=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)-err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), y2=f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit)+err_f_peak_fit(x=fit_bins, a=a_fit, b=b_fit, c=c_fit, err_a=err_a_fit, err_b=err_b_fit, err_c=err_c_fit), color="tab:red", alpha=0.1)
        ax[0].axvline(x=b_fit, color="tab:red", linestyle="--", label="Peak position $b$")
        ax[0].axvspan(xmin=b_fit-err_b_fit, xmax=b_fit+err_b_fit, color="tab:red", alpha=0.1)
        ax[0].set_ylim(bottom=0, top=np.amax(hist_nobg)*1.1)
        ax[0].legend(loc="lower left", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
        residuals = fit_hist - f_peak_fit(fit_bins, a=a_fit, b=b_fit, c=c_fit)
        err_residuals = err_fit_hist
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(x=fit_bins, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=2, linewidth=1, linestyle="")
        ax[1].set_xlim(0,600)
        ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        fig.show()
        ## store plot

        hist_plot_file = plot_save_path + "t_diff_peak_fit" + plot_type
        print(f"store histogram plot as {hist_plot_file}.")
        fig.savefig(hist_plot_file)

        ### estimate drift velocity
        v_drift = cell_half_width / b_fit # um/ns
        err_v_drift = np.sqrt(
            (-cell_half_width/b_fit**2)**2 * err_b_fit**2
            + (1/b_fit)**2 * err_cell_half_width**2
        )
        print(f"v_drift = {v_drift} +- {err_v_drift} um/ns")
    except:
        print("###############################\n Fit of Parabola to peak failed...\n###############################")
if __name__ == "__main__":
    main()
    print(f"###### Done.")
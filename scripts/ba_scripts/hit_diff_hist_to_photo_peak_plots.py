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
from analysis_tools.params import params, derived_params   #, params_justus

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

    data_set_name_list = ["cosmic_82-18_3600-1800-1200_test1_th20", "cosmic_82-18_3600-1800-1200_test2_th20",
                          "cosmic_85-15_3550-1800-1200_test1_cut", "cosmic_85-15_3550-1800-1200_test1",
                          "cosmic_85-15_3575-1800-1200_run1_th20_cut", "cosmic_85-15_3600-1800-1200_run2_th20_cut", 
                          "cosmic_85-15_3600-1800-1200_run2_th20"]
    
    def data_analysis(*, dataset_name):
            # IMPORTANT
        # When not using example data (dt_cosmics.txt) use params_justus
        main_path = "data_ba/"
        pcls_path = "pcls/"
        dataset_name = dataset_name
        dt_hits_name = dataset_name + "_hits.pcl"
        raw_scint_hits_name = dataset_name + "_raw_scint_hits.pcl"
        ts_range_name = dataset_name + "_ts_range.txt"
        ### --- manuell gesetzte Parameter (ersetzt argparse) ---
        dt_hit_diff_name = dataset_name + "_hit_diff.pcl"
        plot_title = r"$\Delta_t$ Photopeak method " + dataset_name
        
        nodeadtime          = True  # True setzen, um dead time zu ignorieren
        deadtime_preffix = "nodeadtime" if nodeadtime else "deadtime"
        dt_hits_file        = main_path + pcls_path + dt_hits_name
        dt_hit_diff_hist_file = main_path + pcls_path +dt_hit_diff_name
        # optionale Schritte:
        use_timestamp_sync   = True   # add_timestamp + sort_by_timestamp anwenden
        extract_scint_hits    = True   # raw scint hits extrahieren und speichern
        raw_scint_hits_file   = main_path + pcls_path + raw_scint_hits_name  # nur relevant falls extract_scint_hits=True

        create_ts_file        = True   # ts_range Datei erzeugen
        plot_save_path = main_path + "plots/photo_peak/" + dataset_name + "/"
        os.makedirs(plot_save_path, exist_ok=True)  
        plot_type = ".png"
        # ---------------------------------------------------------

        ####################
        specific_data = data_utils.load_pickle(dt_hit_diff_hist_file)
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
        path = f"{plot_save_path}{dataset_name}_DIFF_SPECIFIC_ALL{plot_type}"
        print(f"store histogram plot as {path}.")
        fig.savefig(path)
        print(f"Done saving hist as {path}\n")

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
        path = f"{plot_save_path}{dataset_name}_t_diff_bgfit{plot_type}"
        print(f"store histogram plot as {path}.")
        fig.savefig(path)
        print(f"plot saved to {path}")

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
        path = f"{plot_save_path}{dataset_name}_t_diff_nobg{plot_type}"
        print(f"store histogram plot as {path}.")
        fig.savefig(path)
        print(f"\nSaved plot to {path}")

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

    
        fit_index_range = (bins_nobg >= 390) & (bins_nobg <= 420)
        fit_bins = bins_nobg[fit_index_range]
        fit_hist = hist_nobg[fit_index_range]
        err_fit_hist = err_hist_nobg[fit_index_range]
        #p0 = (2000, -0.005, -10, 1000, 417, 60) #cosmic_85-15_3600-1800-1200_run2_th20
                #a     b    c    d    mu   sig
        # p0 = (1500, -0.01, 1000, 400, 20)
        """
        def f_peak_fit(x, a, b, c):
            return a*(x-b)**2+c
        def err_f_peak_fit(x, a, b, c, err_a, err_b, err_c):
            return np.sqrt( ( err_a*(x-b)**2 )**2 + ( -2*a*(x-b)*err_b )**2 + ( err_c )**2 )
            p0 = (-1, 415, 1000)
        
        """
        p0 = (500, 410, 10)
        #       A     mu   sigma

        def f_peak_fit(x, A, mu, sigma):
            peak = A*np.exp(-0.5*((x-mu)/sigma)**2)
            return peak


        def err_f_peak_fit(
            x, A, mu, sigma,
            err_A, err_mu, err_sigma
        ):
            gauss = np.exp(-0.5*((x-mu)/sigma)**2)

            df_dA = gauss
            df_dmu = A*gauss*(x-mu)/sigma**2
            df_dsigma = A*gauss*(x-mu)**2/sigma**3

            return np.sqrt(
                (df_dA*err_A)**2 +
                (df_dmu*err_mu)**2 +
                (df_dsigma*err_sigma)**2
            )


        """
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

        fit_label = (
            f"Parabolic fit:\n"
            f"$f(\\Delta T) = a\\cdot(\\Delta T-b)^2+c$\n"
            f"$a=({a_fit:.2f}\\pm{err_a_fit:.2f})$ 1/ns$^2$\n"
            f"$b=({b_fit:.2f}\\pm{err_b_fit:.2f})$ ns\n"
            f"$c=({c_fit:.0f}\\pm{err_c_fit:.0f})$\n"
            f"$\\chi^2/N_{{df}}={chi2:.1f}/{ndf}={chi2ndf:.1f}$"
    )
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
        """
        popt, pcov, infodict, mesg, _ = curve_fit(
            f=f_peak_fit,
            xdata=fit_bins,
            ydata=fit_hist,
            p0=p0,
            sigma=err_fit_hist,
            absolute_sigma=True,
            full_output=True,
        )

        perr = np.sqrt(np.diag(pcov))

        param_names = ["A", "mu", "sig"]
        fit_params = dict(zip(param_names, popt))
        errors = dict(zip(param_names, perr))

        fit_values = f_peak_fit(fit_bins, *popt)
        chi2 = np.sum((fit_hist - fit_values)**2 / err_fit_hist**2)
        ndf = len(fit_hist) - len(popt)
        chi2ndf = chi2 / ndf

        print(f"Fit interval ΔT = ({fit_bins.min()}, {fit_bins.max()}) ns")
        for name in param_names:
            print(f"  {name:>3} = {fit_params[name]:.6g} ± {errors[name]:.2g}")
        print(f"  chi²/ndf = {chi2:.2f} / {ndf} = {chi2ndf:.2f}")

            # --- estimate drift velocity ---
        mu_val = fit_params["mu"]
        err_mu = errors["mu"]
        v_drift = cell_half_width / mu_val
        err_v_drift = np.sqrt(
            (err_cell_half_width / mu_val)**2 +
            (cell_half_width * err_mu / mu_val**2)**2
        )

        print(f"v_drift = {v_drift:.4g} ± {err_v_drift:.2g} um/ns")

        fit_label = (
        "Gaussian fit\n"
        r"$f(\Delta T)=A\,e^{-\frac{1}{2}((\Delta T-\mu)/\sigma)^2}$"
        )
        for name in param_names:
            fit_label += f"\n${name}=({fit_params[name]:.3g}\\pm {errors[name]:.2g})$"
        fit_label += f"\n$v_{{\\mathrm{{drift}}}}=({v_drift:.3g}\\pm {err_v_drift:.2g})$"
        fit_err = err_f_peak_fit(fit_bins, *popt, *perr)

        # --- build the actual figure/axes for this plot ---
        fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True, height_ratios=(5, 1))

        ax[0] = hist_utils.plot_histogram(
            ax[0], hist=hist_nobg, centers=bins_nobg,
            err_hist_down=err_hist_nobg_down, err_hist_up=err_hist_nobg_up,
            log_scale=False, power_limits=[-3, 3],
        )
        info_str = (
            f"entries = {int(np.sum(hist_nobg))}\n"
            f"bin count = {len(centers)}\n"
            f"bin width = {np.mean(np.diff(bins_nobg)):.3g} ns"
        )
        ax[0] = hist_utils.add_infobox(ax=ax[0], info_str=info_str, info_loc="top left")

        ax[0].plot(fit_bins, fit_values, color="tab:red", label=fit_label)
        ax[0].fill_between(
            fit_bins,
            fit_values - fit_err,
            fit_values + fit_err,
            color="tab:red",
            alpha=0.1,
        )

        mu = fit_params["mu"]
        err_mu = errors["mu"]
        lims = [0, 500]
        ax[0].axvline(x=mu, color="tab:red", linestyle="--", label="Peak position $\\mu$")
        ax[0].axvspan(xmin=mu - err_mu, xmax=mu + err_mu, color="tab:red", alpha=0.1)
        ax[0].set_ylim(bottom=0, top=np.amax(hist_nobg) * 1.1)
        ax[0].legend(loc="lower left", prop={'size': legend_font_size}, fancybox=False, framealpha=params._legend_alpha)
        ax[0].set_xlim(left=lims[0], right=lims[1])
        ax[0].set_title(plot_title)
        # --- residuals panel ---
        residuals = fit_hist - fit_values
        err_residuals = err_fit_hist
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(
            x=fit_bins, y=residuals, yerr=err_residuals,
            color="black", marker="o", markersize=2, linewidth=1, linestyle="",
        )
        ax[1].set_xlim(left=lims[0], right=lims[1])
        ax[1].set_ylim(
            -np.amax(residuals + err_residuals) * 1.1,
            np.amax(residuals + err_residuals) * 1.1,
        )
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("$\\Delta T_\\text{cell}$ [ns]")

        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        fig.show()
        path = f"{plot_save_path}{dataset_name}_t_diff_peak_fit{plot_type}"
        print(f"store histogram plot as {path}.")
        fig.savefig(path)

        return
    
    for dataset in data_set_name_list:
        data_analysis(dataset_name = dataset)
    return
       
if __name__ == "__main__":
    main()
    print(f"###### Done.")
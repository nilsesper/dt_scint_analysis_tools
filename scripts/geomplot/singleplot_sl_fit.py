###########################################
### plot single sl pattern fits
###########################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
from scipy.optimize import curve_fit

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# -----------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sl_fits_file",
        type     = str,
        help     = "input file path: sl fits (pcl file)",
    )
    # plotting / store plot
    parser.add_argument(
        "--indices",
        help     = "indices to plot, separated by \",\"",
    )
    # ---
    args = parser.parse_args()
    sl_fits_file = args.sl_fits_file
    plot_idcs = []
    if args.indices:
        for s in args.indices.split(","):
            plot_idcs.append(int(s))

    #################

    ### data import
    print(f"###### Importing all data...")
    sl_fits = data_utils.load_pickle(file=sl_fits_file)
    n_sl_fits = data_utils.length(sl_fits)


    for idx in plot_idcs:

        fit = {k: sl_fits[k][idx] for k in sl_fits.keys()}
        # data
        lys = np.arange(0, 4)
        ts = np.array([fit[f"ts{ly}"] for ly in range(4)])
        err_ts = np.array([fit[f"err_ts{ly}"] for ly in range(4)])
        sl = fit["sl"]
        # pattern
        pat_type = fit["pat_type"]
        pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
        lats = params._dt_sl_patterns[pat_name]["laterality"] # list of [lat for ly0,1,2,3] laterality lists
        lat_idx = fit["laterality"]
        laterality = np.array(lats[lat_idx])
        # fit results
        t0 = fit["t0"]
        x0 = fit["x0"]
        tan_alpha = fit["tan_alpha"]
        vd = fit["vd"]
        err_t0 = fit["err_t0"]
        err_x0 = fit["err_x0"]
        err_tan_alpha = fit["err_tan_alpha"]
        err_vd = fit["err_vd"]
        corr_t0_x0 = fit["corr_t0_x0"]
        corr_t0_tan_alpha = fit["corr_t0_tan_alpha"]
        corr_t0_vd = fit["corr_t0_vd"]
        corr_x0_tan_alpha = fit["corr_x0_tan_alpha"]
        corr_x0_vd = fit["corr_x0_vd"]
        corr_tan_alpha_vd = fit["corr_tan_alpha_vd"]
        chi2ndf = fit["chi2/ndf"]
        # other params
        z_arr, x_cell = np.full(4, 0, dtype=np.float64), np.full(4, 0, dtype=np.float64)
        lys = np.arange(0, 4)
        for ly in lys:
            z_arr[ly] = derived_params._sl_pattern_coordinates[ly][0][3] #-1*(3-ly)*params._cell_height # z coord for ly0,1,2,3. note coordinate system with ly3 = (z=0)
            rel_wi = params._dt_sl_patterns[pat_name]["rel_wis"][ly]
            x_cell[ly] = derived_params._sl_pattern_coordinates[ly][rel_wi][2] # x values for fit => x positions of wires / cell centers for each layer, depends on pattern layout
        # fit function
        fit_ts, err_fit_ts = np.zeros(4), np.zeros(4)
        for ly in lys:
            fit_ts[ly] = derived_params.f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd)
            err_fit_ts[ly] = derived_params.err_f_ts_fit(x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly], laterality=laterality[ly], vd=vd, err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd, corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=corr_t0_vd, corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=corr_x0_vd, corr_tan_alpha_vd=corr_tan_alpha_vd)

        ## print
        print(f"pattern fit #{idx}:")
        print(f"  sl = {sl}")
        print(f"  pat_type = {pat_type}")
        print(f"  lat_idx = {lat_idx}")
        print(f"  laterality = {laterality}")
        print(f"  t0 = {t0}")
        print(f"  x0 = {x0}")
        print(f"  tan_alpha = {tan_alpha}")
        print(f"  vd = {vd}")
        print(f"  err_t0 = {err_t0}")
        print(f"  err_x0 = {err_x0}")
        print(f"  err_tan_alpha = {err_tan_alpha}")
        print(f"  err_vd = {err_vd}")
        print(f"  corr_t0_x0 = {corr_t0_x0}")
        print(f"  corr_t0_tan_alpha = {corr_t0_tan_alpha}")
        print(f"  corr_t0_vd = {corr_t0_vd}")
        print(f"  corr_x0_tan_alpha = {corr_x0_tan_alpha}")
        print(f"  corr_x0_vd = {corr_x0_vd}")
        print(f"  corr_tan_alpha_vd = {corr_tan_alpha_vd}")
        print(f"  chi2/ndf = {chi2ndf}")

        ################################
        ###### plot fit

        ## plot fit, with residual plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=(5,1))
        # main plot
        ts_label = "Hit timestamps"
        ax[0].errorbar(x=lys-0.02, y=ts, yerr=err_ts, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        fit_label = f"""Track fit:
$T_0=({np.round(t0,0):.0f}\\pm{np.round(err_t0,0):.0f})$ {params._key_units['t0']}
$x_0=({np.round(x0,1):.1f}\\pm{np.round(err_x0,1):.1f})$ {params._key_units['x0']}
$\\tan\\alpha=({np.round(tan_alpha,2):.2f}\\pm{np.round(err_tan_alpha,2):.2f})$ {params._key_units['tan_alpha']}
$\\chi^2/N_{{df}}={np.round(chi2ndf,2):.2f}$"""
        ax[0].errorbar(x=lys+0.02, y=fit_ts, yerr=err_fit_ts, color="tab:red", marker="v", markersize=7, linestyle="", label=fit_label)
        #ax[0].axhline(y=t0, color="tab:green", linestyle="--", linewidth=2, label="Arrival time $T_0$")
        #ax[0].axhspan(ymin=t0-err_t0, ymax=t0+err_t0, color="tab:green", alpha=0.2)
        
        #ax[0].plot(fit_bins, f_bg_fit(fit_bins, a=a_fit, b=b_fit), color="tab:red", label=fit_label)
        #ax[0].plot(bins[0:fit_index_range[0]], f_bg_fit(bins[0:fit_index_range[0]], a=a_fit, b=b_fit), color="tab:red", linestyle="--", label="Extrapolated fit")
        #ax[0].fill_between(bins, y1=f_bg_fit(x=bins, a=a_fit, b=b_fit)-err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), y2=f_bg_fit(x=bins, a=a_fit, b=b_fit)+err_f_bg_fit(x=bins, a=a_fit, b=b_fit, err_a=err_a_fit, err_b=err_b_fit), color="tab:red", alpha=0.1)
        #ax[0].set_yscale("log")
        #ax[0].set_ylim(bottom=0.5, top=np.amax(hist)*np.exp(1.1))
        ax[0].set_ylabel("Timestamp $T_{ly}$ [TU]")
        ax[0].legend(prop = { "size": 18 })
        ax[0].set_title(f"SL {sl} ({params._dt_chamber["sls"][sl]["orient"]}), Pattern {pat_type}, Laterality {[int(l) for l in laterality]}")
        
        # residual plot
        residuals = ts - fit_ts
        err_residuals = err_ts
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].errorbar(x=lys, y=residuals, yerr=err_residuals , color="black", marker="o", markersize=7, linestyle="")
        ax[1].set_ylim(-np.amax(residuals+err_residuals)*1.1, np.amax(residuals+err_residuals)*1.1)
        ax[1].set_ylabel("Residuals")
        ax[1].set_xlabel("Layer $ly$")
        ax[1].set_xticks([i for i in range(4)])
        ax[1].set_xticklabels([f"{i}" for i in range(4)])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        fig.show()

        ################################
        ###### plot projected local sl track

        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        # draw dt sl pattern
        ax = geoplot_utils.empty_sl_pattern_ax(ax, pat_name, wire=True)
        # calculate hit positions
        x_hits = x_cell + laterality*(ts-t0)*vd
        err_x_hits = np.sqrt(
              (laterality*vd)**2 * err_ts**2
            + (-laterality*vd)**2 * err_t0**2
            + (-laterality*(ts-t0)*vd**2)**2 * err_vd**2
            + 2*(laterality*vd)*(-laterality*(ts-t0)*vd**2) * corr_t0_vd
        )
        ax.errorbar(x=x_hits, y=z_arr, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        # plot local sl fit
        _z0 = derived_params._sl_pattern_coordinates[3][0][3] # z_cell (wire position) of ly=3
        _z1 = derived_params._sl_pattern_coordinates[2][0][3] # z_cell (wire position) of ly=2
        #ax.axline((derived_params.f_x_muon(z=_z0, x0=x0, tan_alpha=tan_alpha), _z0), (derived_params.f_x_muon(z=_z1, x0=x0, tan_alpha=tan_alpha), _z1), c="tab:red", linewidth=params._color_info["muon"]["linewidth"], label=fit_label)
        z_range = np.linspace(np.amin(z_arr)-params._cell_height, np.amax(z_arr)+params._cell_height, 1000)
        track = derived_params.f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha)
        err_track = derived_params.err_f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        ax.plot(track, z_range, linewidth=2, color="tab:red", label=fit_label)
        ax.fill_betweenx(x1=track-err_track, x2=track+err_track, y=z_range, color="tab:red", alpha=0.2)
        ax.legend(prop = { "size": 18 }, loc="center left")
        if sl != 2:
            ax.set_xlabel("$x-x_\\text{wire,ly=3}$ [mm]")
        else:
            ax.set_xlabel("$y-y_\\text{wire,ly=3}$ [mm]")
        ax.set_ylabel("$z-z_\\text{wire,ly=3}$ [mm]")
        ax.set_ylim(np.amin(z_range),np.amax(z_range))
        ax.set_title(f"SL {sl} ({params._dt_chamber["sls"][sl]["orient"]}), Pattern {pat_type}, Laterality {[int(l) for l in laterality]}")
        fig.tight_layout()
        fig.show()



    input("Press enter to exit.")
    exit()




if __name__ == "__main__":
    main()
    print(f"###### Done.")









###########################################
### plot super sl pattern fits (8-layer, dual-SL)
### mirrors plot_sl_pattern_fits.py, adapted for the output of
### fit_super_sl_patterns (8 layers spanning sl1 + sl2, lat_id1/lat_id2,
### local frame relative to the topmost wire via ref_x/ref_z)
###########################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

# -----------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main():

    ### ------------------------------------------------------------
    ### manual dataset / path configuration (edit as needed, no argparse)
    ### ------------------------------------------------------------
    base_path = "data_ba/"
    data_sets = []

    #begin for loop her for plots of fits in all datasets in data_sets
    dataset_name = "cosmic_82-18_3550-1800-1200_run1_th20_cut_50"

    super_fits_path = base_path + f"pcls/{dataset_name}/" + dataset_name + "_super_fits.pcl"
    plot_save_path = base_path + f"plots/sl_fits/{dataset_name}/"
    os.makedirs(plot_save_path, exist_ok=True)
    plot_type = ".png"

    suffix = "_free_vd_super_fit"            # decide which fit should be plotted, e.g. "" or "_refit"
    plot_idcs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]         # indices to plot

    super_fits_file = super_fits_path
    save_path = plot_save_path

    #################

    ### data import
    print(f"###### Importing super fits...")
    super_fits = data_utils.load_pickle(file=super_fits_file)
    n_super_fits = data_utils.length(super_fits)
    print("### imported super fits data from file: " + super_fits_file)


    for idx in plot_idcs:

        fit = {k: super_fits[k][idx] for k in super_fits.keys()}

        # -------------------------------------------------------------
        # pattern / laterality bookkeeping (mirrors fit_super_sl_patterns)
        # -------------------------------------------------------------
        sl1 = int(fit["sl1"])
        sl2 = int(fit["sl3"])
        pat_type_sl1 = int(fit["pat_type_sl1"])
        pat_type_sl2 = int(fit["pat_type_sl3"])
        pat_name_sl1 = list(params._dt_sl_patterns.keys())[pat_type_sl1]
        pat_name_sl2 = list(params._dt_sl_patterns.keys())[pat_type_sl2]
        lats1 = params._dt_sl_patterns[pat_name_sl1]["laterality"]
        lats2 = params._dt_sl_patterns[pat_name_sl2]["laterality"]

        lat_id1 = int(fit["lat_id1" + suffix])
        lat_id2 = int(fit["lat_id2" + suffix])
        laterality = np.array(list(lats1[lat_id1]) + list(lats2[lat_id2]))  # length 8

        wi_sl1 = [int(fit[f"wi{ly}_sl1"]) for ly in range(4)]
        wi_sl2 = [int(fit[f"wi{ly}_sl3"]) for ly in range(4)]

        # data (8 layers)
        lys = np.arange(0, 8)
        ts = np.array([fit[f"ts{ly}"] for ly in range(8)])
        err_ts = np.array([fit[f"err_ts{ly}"] for ly in range(8)])

        # -------------------------------------------------------------
        # geometry: rebuilt exactly as in fit_super_sl_patterns
        # (global frame -> shifted into local frame via stored ref_x/ref_z)
        # -------------------------------------------------------------
        z_arr_glob, x_cell_glob = np.full(8, 0, dtype=np.float64), np.full(8, 0, dtype=np.float64)
        for ly in range(4):
            x_cell_glob[ly],     z_arr_glob[ly]     = derived_params.super_pattern_geometry(sl1, ly, wi_sl1[ly])
            x_cell_glob[ly + 4], z_arr_glob[ly + 4] = derived_params.super_pattern_geometry(sl2, ly, wi_sl2[ly])

        ref_x = fit["ref_x" + suffix]
        ref_z = fit["ref_z" + suffix]
        # local frame (relative to the topmost wire), used for the fit itself
        x_cell = x_cell_glob - ref_x
        z_arr = z_arr_glob - ref_z

        # -------------------------------------------------------------
        # fit results
        # -------------------------------------------------------------
        t0 = fit["t0" + suffix]
        x0 = fit["x0" + suffix]
        tan_alpha = fit["tan_alpha" + suffix]
        vd = fit["vd" + suffix]
        err_t0 = fit["err_t0" + suffix]
        err_x0 = fit["err_x0" + suffix]
        err_tan_alpha = fit["err_tan_alpha" + suffix]
        err_vd = fit["err_vd" + suffix]
        corr_t0_x0 = fit["corr_t0_x0" + suffix]
        corr_t0_tan_alpha = fit["corr_t0_tan_alpha" + suffix]
        corr_t0_vd = fit["corr_t0_vd" + suffix]
        corr_x0_tan_alpha = fit["corr_x0_tan_alpha" + suffix]
        corr_x0_vd = fit["corr_x0_vd" + suffix]
        corr_tan_alpha_vd = fit["corr_tan_alpha_vd" + suffix]
        chi2ndf = fit["chi2/ndf" + suffix]

        # fit function evaluated at all 8 layers
        fit_ts, err_fit_ts = np.zeros(8), np.zeros(8)
        for ly in range(8):
            fit_ts[ly] = derived_params.f_ts_fit(
                x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly],
                laterality=laterality[ly], vd=vd,
            )
            err_fit_ts[ly] = derived_params.err_f_ts_fit(
                x_cell=x_cell[ly], t0=t0, x0=x0, tan_alpha=tan_alpha, z=z_arr[ly],
                laterality=laterality[ly], vd=vd,
                err_t0=err_t0, err_x0=err_x0, err_tan_alpha=err_tan_alpha, err_vd=err_vd,
                corr_t0_x0=corr_t0_x0, corr_t0_tan_alpha=corr_t0_tan_alpha, corr_t0_vd=corr_t0_vd,
                corr_x0_tan_alpha=corr_x0_tan_alpha, corr_x0_vd=corr_x0_vd, corr_tan_alpha_vd=corr_tan_alpha_vd,
            )

        ## print
        print(f"super pattern fit #{idx}:")
        print(f"  sl1 = {sl1}, sl2 = {sl2}")
        print(f"  pat_type_sl1 = {pat_type_sl1} ({pat_name_sl1}), pat_type_sl2 = {pat_type_sl2} ({pat_name_sl2})")
        print(f"  lat_id1 = {lat_id1}, lat_id2 = {lat_id2}")
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
        ###### plot fit: timestamps vs fit, 8 layers, with residuals

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(5, 1))
        ts_label = "Hit timestamps"
        ax[0].errorbar(x=lys - 0.04, y=ts, yerr=err_ts, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label)
        fit_label = f"""Track fit:
$T_0=({np.round(t0,0):.0f}\\pm{np.round(err_t0,0):.0f})$ {params._key_units['t0']}
$x_0=({np.round(x0,1):.1f}\\pm{np.round(err_x0,1):.1f})$ {params._key_units['x0']}
$\\tan\\alpha=({np.round(tan_alpha,2):.2f}\\pm{np.round(err_tan_alpha,2):.2f})$ {params._key_units['tan_alpha']}
$V_d=({np.round(vd,4):.4f}\\pm{np.round(err_vd,4):.4f})$ mm/TU
$\\chi^2/N_{{df}}={np.round(chi2ndf,2):.2f}$"""
        ax[0].errorbar(x=lys + 0.04, y=fit_ts, yerr=err_fit_ts, color="tab:red", marker="v", markersize=7, linestyle="", label=fit_label)
        # visual separator between the two superlayers (ly 0-3 = sl1, ly 4-7 = sl2)
        ax[0].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[0].set_ylabel("Timestamp $T_{ly}$ [TU]")
        ax[0].legend(prop={"size": 16}, fancybox=False, framealpha=params._legend_alpha)
        ax[0].set_title(f"SL {sl1} + SL {sl2}, Patterns {pat_name_sl1}/{pat_name_sl2}, Laterality {[int(l) for l in laterality]}")
        ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # residual plot
        residuals = ts - fit_ts
        err_residuals = err_ts
        ax[1].axhline(y=0, color="gray", linewidth=1)
        ax[1].axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
        ax[1].errorbar(x=lys, y=residuals, yerr=err_residuals, color="black", marker="o", markersize=7, linestyle="")
        y_span = np.amax(np.abs(residuals) + err_residuals) * 1.1
        ax[1].set_ylim(-y_span, y_span)
        ax[1].set_ylabel("Residuals [TU]")
        ax[1].set_xlabel(f"Layer $ly$ (0-3: SL{sl1}, 4-7: SL{sl2})")
        ax[1].set_xticks([i for i in range(8)])
        ax[1].set_xticklabels([f"{i}" for i in range(8)])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.1)
        fig.savefig(fname=(f"{save_path}super_muon_fit{suffix}_{idx}{plot_type}"))

        ################################
        ###### plot projected local track (8 hits, spanning both SLs)

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        # NOTE: geoplot_utils.empty_sl_pattern_ax() draws a single 4-layer SL
        # pattern background and doesn't directly apply to an 8-layer super
        # pattern spanning two SLs. If you have (or add) an equivalent helper
        # for super patterns, swap it in here, e.g.:
        #   ax = geoplot_utils.empty_super_pattern_ax(ax, pat_name_sl1, pat_name_sl2, wire=True)
        # In the meantime, cells are drawn as simple open markers colored by
        # which SL they belong to.
        for ly in range(8):
            color = "tab:gray" if ly < 4 else "tab:olive"
            ax.scatter(x_cell[ly], z_arr[ly], marker="s", s=60, facecolors="none", edgecolors=color, zorder=1)

        # calculate hit positions (drift distance resolved via laterality)
        x_hits = x_cell + laterality * (ts - t0) * vd
        err_x_hits = np.sqrt(
              (laterality * vd) ** 2 * err_ts ** 2
            + (-laterality * vd) ** 2 * err_t0 ** 2
            + (-laterality * (ts - t0) * vd ** 2) ** 2 * err_vd ** 2
            + 2 * (laterality * vd) * (-laterality * (ts - t0) * vd ** 2) * corr_t0_vd
        )
        ax.errorbar(x=x_hits, y=z_arr, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label, zorder=2)

        # plot local track fit across the full z range of both SLs
        z_range = np.linspace(np.amin(z_arr) - params._cell_height, np.amax(z_arr) + params._cell_height, 1000)
        track = derived_params.f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha)
        err_track = derived_params.err_f_x_muon(z=z_range, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        ax.plot(track, z_range, linewidth=2, color="tab:red", label=fit_label, zorder=3)
        ax.fill_betweenx(x1=track - err_track, x2=track + err_track, y=z_range, color="tab:red", alpha=0.2, zorder=0)

        ax.legend(prop={"size": 16}, loc="center left", fancybox=False, framealpha=params._legend_alpha)
        ax.set_xlabel("$x-x_\\text{wire,top}$ [mm]")
        ax.set_ylabel("$z-z_\\text{wire,top}$ [mm]")
        ax.set_ylim(np.amin(z_range), np.amax(z_range))
        ax.set_title(f"SL {sl1} + SL {sl2}, Patterns {pat_name_sl1}/{pat_name_sl2}, Laterality {[int(l) for l in laterality]}")
        fig.tight_layout()
        fig.show()

        ################################
        ###### plot track inside the full detector geometry (global chamber view)
        # super patterns span sl1 + sl3, i.e. the "phi" superlayers

        orient = "phi"

        dt_cell_data = dt_utils._chamber_data()
        # highlight the cells that were actually used in the fit
        for ly in range(4):
            dt_cell_data[sl1][ly][wi_sl1[ly]]["color"] = "aqua"
            dt_cell_data[sl2][ly][wi_sl2[ly]]["color"] = "aqua"

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax = geoplot_utils.chamber_ax(ax=ax, orient=orient, cell_data=dt_cell_data, wire=True)

        # -----------------------------------------------------------------
        # IMPORTANT: super_pattern_geometry() (used inside fit_super_sl_patterns,
        # and thus the fit's ref_x/ref_z) lives in a DIFFERENT coordinate frame
        # than _dt_cell_coordinates (the source chamber_ax/dt_cell_data draw from).
        # Mixing the two caused the track/hits to be offset from the cells.
        # So here we independently rebuild the 8-layer geometry from
        # _dt_cell_coordinates -- the same source the chamber background uses --
        # and re-derive our own "top wire" reference in THAT frame.
        # -----------------------------------------------------------------
        ref_axis = 0  # phi view -> x-axis (would be 1 for theta/y)

        z_arr_ch, x_cell_ch = np.full(8, 0, dtype=np.float64), np.full(8, 0, dtype=np.float64)
        for ly in range(4):
            x_cell_ch[ly]     = derived_params._dt_cell_coordinates[sl1][ly][wi_sl1[ly]][ref_axis + 3]
            z_arr_ch[ly]      = derived_params._dt_cell_coordinates[sl1][ly][wi_sl1[ly]][5]
            x_cell_ch[ly + 4] = derived_params._dt_cell_coordinates[sl2][ly][wi_sl2[ly]][ref_axis + 3]
            z_arr_ch[ly + 4]  = derived_params._dt_cell_coordinates[sl2][ly][wi_sl2[ly]][5]

        # same "top wire" (largest z) that the fit used as its local-frame origin,
        # just expressed here in the chamber-drawing coordinate frame
        top_wire_idx = int(np.argmax(z_arr_ch))
        x_ref_ch = x_cell_ch[top_wire_idx]
        z_ref_ch = z_arr_ch[top_wire_idx]

        # hit positions directly in chamber coordinates -- the drift-distance term
        # (laterality * (ts - t0) * vd) is a pure physical offset in mm and is
        # translation-invariant, so it's valid to add it to the chamber-frame cell centers
        x_hits_ch = x_cell_ch + laterality * (ts - t0) * vd

        # extrapolate the track across the FULL chamber z-range, not just the
        # local 8-layer hit height, so it renders as an actual track through
        # the detector rather than a short segment
        z_range_glob = np.linspace(
            derived_params._dt_cell_coordinates[1][0][1][5] - params._cell_height * 2,
            derived_params._dt_cell_coordinates[3][3][1][5] + params._cell_height * 2,
            1000,
        )
        z_range_local = z_range_glob - z_ref_ch
        track_local = derived_params.f_x_muon(z=z_range_local, x0=x0, tan_alpha=tan_alpha)
        err_track_local = derived_params.err_f_x_muon(z=z_range_local, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)
        track_glob = track_local + x_ref_ch

        ax.errorbar(x=x_hits_ch, y=z_arr_ch, xerr=err_x_hits, color="tab:blue", marker="o", markersize=7, linestyle="", label=ts_label, zorder=2)
        ax.plot(track_glob, z_range_glob, linewidth=2, color="tab:red", label=fit_label, zorder=3)
        ax.fill_betweenx(x1=track_glob - err_track_local, x2=track_glob + err_track_local, y=z_range_glob, color="tab:red", alpha=0.2, zorder=0)

        ax.legend(prop={"size": 14}, fancybox=False, framealpha=params._legend_alpha, loc="center right")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
        ax.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))
        ax.set_title(f"DT chamber ($\\phi$ view) — SL {sl1} + SL {sl2} track fit")
        fig.tight_layout()
        fig.savefig(fname=(f"{save_path}super_muon_fit_chamber{suffix}_{idx}{plot_type}"))
        fig.show()



        ################################
        ###### zoomed chamber view (x-axis only)

        fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(8, 10))
        ax_zoom = geoplot_utils.chamber_ax(
            ax=ax_zoom,
            orient=orient,
            cell_data=dt_cell_data,
            wire=True,
        )

        # Draw the same hits and fitted track
        ax_zoom.errorbar(
            x=x_hits_ch,
            y=z_arr_ch,
            xerr=err_x_hits,
            color="tab:blue",
            marker="o",
            markersize=7,
            linestyle="",
            label=ts_label,
            zorder=2,
        )

        ax_zoom.plot(
            track_glob,
            z_range_glob,
            linewidth=2,
            color="tab:red",
            label=fit_label,
            zorder=3,
        )

        ax_zoom.fill_betweenx(
            y=z_range_glob,
            x1=track_glob - err_track_local,
            x2=track_glob + err_track_local,
            color="tab:red",
            alpha=0.2,
            zorder=0,
        )

        # ---------------------------------------------------------
        # Zoom only in x
        # ---------------------------------------------------------
        margin = 20.0  # mm; adjust as desired

        xmin = min(
            np.min(track_glob - err_track_local),
            np.min(x_hits_ch - err_x_hits),
        )
        xmax = max(
            np.max(track_glob + err_track_local),
            np.max(x_hits_ch + err_x_hits),
        )

        ax_zoom.set_xlim(xmin - margin, xmax + margin)

        # Keep the full chamber height
        ax_zoom.set_ylim(np.amin(z_range_glob), np.amax(z_range_glob))

        ax_zoom.set_xlabel("$x$ [mm]")
        ax_zoom.set_ylabel("$z$ [mm]")
        ax_zoom.set_title(f"DT chamber ($\\phi$ view) — Zoomed track fit")
        ax_zoom.legend(
            prop={"size": 14},
            fancybox=False,
            framealpha=params._legend_alpha,
            loc="center right",
        )

        fig_zoom.tight_layout()

        fig_zoom.savefig(
            fname=f"{save_path}super_muon_fit_chamber_zoom{suffix}_{idx}{plot_type}"
        )

        fig_zoom.show()

    return




if __name__ == "__main__":
    main()
    print(f"###### Done.")
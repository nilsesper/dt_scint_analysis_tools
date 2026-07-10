
# give  as first argument

args=("$@") # obtain args from script call
# give as arguments:
#   basedir in this machine
#   dumpfile name on this machine (as rel path)
#   internal name of dataset

source env.sh

BASEDIR="${args[0]}"
PLOTDIR="plots/"
DUMPFILE="${args[1]}"
DATASETNAME="${args[2]}"
CONFIGFILE="config_${DATASETNAME}.txt"
CONFIGFILE_DIR="${BASEDIR}/${CONFIGFILE}"

echo "BASEDIR = ${BASEDIR}"
echo "PLOTDIR = ${PLOTDIR}"
echo "DUMPFILE = ${DUMPFILE}"
echo "DATASETNAME = ${DATASETNAME}"
echo "CONFIGFILE = ${CONFIGFILE}"
echo "CONFIGFILE_DIR = ${CONFIGFILE_DIR}"

touch ${CONFIGFILE_DIR}
echo "${DUMPFILE},${DATASETNAME}" > ${CONFIGFILE_DIR}
sleep 1

###### import hit data
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dump_import" --task_for_each_set --n_proc 12 
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dump_dt_nodeadtime" --task_for_each_set --n_proc 12
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_corr" --task_for_each_set --n_proc 12

### hits occupancy & rate w/o dead time
yes "" | python split_scripts/split_utils_dt/dt_hits_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME
yes "" | python split_scripts/split_utils_dt/plot_dt_hits_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/${PLOTDIR} --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME
# hist: "ts" +- "err_ts"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME --key "ts" --edges "auto,50"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME --key "ts" --err_key "err_ts" --fig_size "7,6" --use_asymm_err --info_box_loc "bottom center" --new_unit_name "s" --new_unit_conversion "0.78e-9" --new_unit_name "s" --new_unit_conversion "0.78e-9"

### hits occupancy & rate w/ dead time
yes "" | python split_scripts/split_utils_dt/dt_hits_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS
yes "" | python split_scripts/split_utils_dt/plot_dt_hits_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS

### photo peak hits
yes "" | python split_scripts/split_utils_dt/dt_hit_diff_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME
yes "" | python split_scripts/split_utils_dt/plot_dt_hit_diff_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/${PLOTDIR} --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_HITS_NODEADTIME --fig_size "7,6"

###### calculate sl patterns
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_patterns" --task_for_each_set --n_proc 12

### sl pattern occupancy
yes "" | python split_scripts/split_utils_dt/sl_patterns_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_PATTERNS
yes "" | python split_scripts/split_utils_dt/plot_sl_patterns_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_PATTERNS
# hist: "pat_type"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_PATTERNS --key "pat_type" --edges "step1"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_PATTERNS --key "pat_type" --fig_size "7,6" --use_asymm_err --info_box_only_entries

###### calculate sl fake patterns
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_fake_patterns" --task_for_each_set --n_proc 12

### sl fake pattern occupancy
yes "" | python split_scripts/split_utils_dt/sl_patterns_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FAKE_PATTERNS
yes "" | python split_scripts/split_utils_dt/plot_sl_patterns_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FAKE_PATTERNS
# hist: "pat_type"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FAKE_PATTERNS --key "pat_type" --edges "step1"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FAKE_PATTERNS --key "pat_type" --fig_size "7,6" --use_asymm_err --info_box_only_entries --new_x_label "SL fake pattern type"

###### calculate sl fits
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_fits" --task_for_each_set --n_proc 12

### pattern occupancy after sl fit cuts & time box histogram
yes "" | python split_scripts/split_utils_dt/sl_fits_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS
yes "" | python split_scripts/split_utils_dt/plot_sl_fits_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --fig_size "7,6"

###### calculate sl fit cuts
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_fit_cuts" --task_for_each_set --n_proc 12

### sl fit histograms after cuts
# hist: "chi2/ndf"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "chi2/ndf" --edges "auto,50"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "chi2/ndf" --fig_size "7,6" --use_asymm_err --info_box_loc "top right"
# hist: "t0" +- "err_t0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "t0" --err_key "err_t0" --edges "auto,50"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "t0" --fig_size "7,6" --use_asymm_err --info_box_loc "bottom center" --new_unit_name "s" --new_unit_conversion "0.78e-9" --n_x_ticks 6 --x_tick_minmax 0,20000
# hist: "x0" +- "err_x0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "x0" --err_key "err_x0" --edges "linear,-21,21,101"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "x0" --fig_size "7,6" --use_asymm_err --info_box_loc "top left"
# hist: "tan_alpha" +- "err_tan_alpha"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "tan_alpha" --err_key "err_tan_alpha" --edges "linear,-1.62,1.62,101"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "tan_alpha" --fig_size "7,6" --use_asymm_err --info_box_loc "top left"
# hist: "err_t0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_t0" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_t0" --fig_size "7,6" --use_asymm_err
# hist: "err_x0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_x0" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_x0" --fig_size "7,6" --use_asymm_err
# hist: "err_tan_alpha"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_tan_alpha" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FITS_AFTERCUTS --key "err_tan_alpha" --fig_size "7,6" --use_asymm_err

### single fit event display
yes "" | python scripts/geomplot/singleplot_sl_fit.py --sl_fits_file ${BASEDIR}/${DATASETNAME}_SL_FITS_AFTERCUTS.pcl --indices 8   
# manually store plots

###### calculate sl fit groups
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_fit_groups" --task_for_each_set --n_proc 12
# hist: "n_fits"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FIT_GROUPS --key "n_fits" --edges "step1"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FIT_GROUPS --key "n_fits" --fig_size "7,6" --log_y_scale --use_asymm_err

###### calculate sl fit groups after cuts
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_fit_group_cuts" --task_for_each_set --n_proc 12

# hist: "n_fits"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FIT_GROUPS_AFTERCUTS --key "n_fits" --edges "step1"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset SL_FIT_GROUPS_AFTERCUTS --key "n_fits" --fig_size "7,6" --log_y_scale --use_asymm_err

###### calculate dt muons
yes "" | python split_scripts/split_utils/analysis_run_control.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --task_list "dt_muons" --task_for_each_set --n_proc 12

### general histograms
## hist: "ts" +- "err_ts"
# calculate & plot
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "ts" --err_key "err_ts" --edges "auto,50"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "ts" --fig_size "7,6" --use_asymm_err --info_box_loc "bottom center" --new_unit_name "s" --new_unit_conversion "0.78e-9"
# hist: "theta" +- "err_theta"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "theta" --err_key "err_theta" --edges "linear,0,1.58,100"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "theta" --fig_size "7,6" --use_asymm_err --info_box_loc "top right"
# hist: "phi" +- "err_phi"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "phi" --err_key "err_phi" --edges "linear,0,6.29,200"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "phi" --fig_size "7,6" --use_asymm_err --info_box_loc "bottom center"
# hist: "x0" +- "err_x0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "x0" --err_key "err_x0" --edges "auto,200"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "x0" --fig_size "7,6" --use_asymm_err --info_box_loc "top right"
# hist: "y0" +- "err_y0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "y0" --err_key "err_y0" --edges "auto,200"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "y0" --fig_size "7,6" --use_asymm_err --info_box_loc "top right"
# hist: "err_ts"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_ts" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_ts" --fig_size "7,6" --use_asymm_err
# hist: "err_theta"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_theta" --edges "linear,0,0.5,20"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_theta" --fig_size "7,6" --use_asymm_err
# hist: "err_phi"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_phi" --edges "linear,0,0.5,20"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_phi" --fig_size "7,6" --use_asymm_err
# hist: "err_x0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_x0" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_x0" --fig_size "7,6" --use_asymm_err
# hist: "err_y0"
yes "" | python split_scripts/split_utils/hist_from_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_y0" --edges "auto,10"
yes "" | python split_scripts/split_utils/plot_hist_from_split_data.py --base_path ${BASEDIR}/ --store_path ${BASEDIR}/plots/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS --key "err_y0" --fig_size "7,6" --use_asymm_err

###### merge split dt muon files into one, because some plotting scripts only work with merged data
yes "" | python split_scripts/split_utils/merge_split_data.py --base_path ${BASEDIR}/ --data_config_file ${BASEDIR}/${CONFIGFILE} --dataset DT_MUONS

### event display of one dt muon
yes "" | python scripts/geomplot/singleplot_dt_muon.py --dt_muons_file ${BASEDIR}/${DATASETNAME}_DT_MUONS.pcl --sl_fits_file ${BASEDIR}/${DATASETNAME}_SL_FITS_AFTERCUTS.pcl --sl_fit_groups_file ${BASEDIR}/${DATASETNAME}_SL_FIT_GROUPS.pcl --dt_muon_idcs 5
# manually store plots

### dt track geometry plotting & other analysis
yes "" | python scripts/dt/plot_dt_muons.py --show_plots --dt_muons_file ${BASEDIR}/${DATASETNAME}_MERGED_DT_MUONS.pcl --fig_size "7,6" --store_path ${BASEDIR}/plots/









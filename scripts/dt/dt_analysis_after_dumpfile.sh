### RUN DT ANALYSIS CHAIN

args=("$@") # obtain args from script call
# give as arguments:
#   filename of dumpfile (must be stored in ~/masterarbeit/zynq_read-out_software/output/) without txt
NO_ARGUMENTS=1

# check no of arguments is correct
if [[ $# != $NO_ARGUMENTS ]] ; then
  echo "*** ERROR: wrong number of arguments. the script expects ${NO_ARGUMENTS} argument(s)."
  exit 0
fi

############################

echo "****** dumpfile_to_dt_hits ******"
python scripts/dt/dumpfile_to_dt_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/${args[0]}.txt --dt_hits_file data_files/${args[0]}_dt_hits.pcl

echo "****** dt_hits_to_sl_patterns ******"
python scripts/dt/dt_hits_to_sl_patterns.py --dt_hits_file data_files/${args[0]}_dt_hits.pcl --sl_patterns_file data_files/${args[0]}_sl_patterns.pcl --dt_tp_corrections_file data_files/dt_tp_corrections_6.pcl --n_proc 16

echo "****** sl_patterns_to_sl_fits ******"
python scripts/dt/sl_patterns_to_sl_fits.py --sl_patterns_file data_files/${args[0]}_sl_patterns.pcl --sl_fits_file data_files/${args[0]}_sl_fits.pcl --n_proc 16

echo "****** apply_cuts ******"
python scripts/general/apply_cuts.py --input_data_file data_files/${args[0]}_sl_fits.pcl --cut_data_file data_files/${args[0]}_sl_fits_aftercuts.pcl --cuts "chi2/ndf,<,10;x0,<=,21;x0,>=,-21;dt0,>=,0;dt0,<=,params._dt_max_drift_time;dt1,>=,0;dt1,<=,params._dt_max_drift_time;dt2,>=,0;dt2,<=,params._dt_max_drift_time;dt3,>=,0;dt3,<=,params._dt_max_drift_time"

echo "****** sl_fits_to_sl_fit_groups ******"
python scripts/dt/sl_fits_to_sl_fit_groups.py --sl_fits_file data_files/${args[0]}_sl_fits_aftercuts.pcl --sl_fit_groups_file data_files/${args[0]}_sl_fit_groups.pcl

echo "****** sl_fit_groups_to_dt_muons ******"
python scripts/dt/sl_fit_groups_to_dt_muons.py --sl_fits_file data_files/${args[0]}_sl_fits_aftercuts.pcl --sl_fit_groups_file data_files/${args[0]}_sl_fit_groups.pcl --dt_muons_file data_files/${args[0]}_dt_muons.pcl


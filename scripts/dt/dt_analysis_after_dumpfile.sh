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

python scripts/dt/dt_hits_to_sl_patterns.py --dt_hits_file data_files/${args[0]}_hits.pcl --sl_patterns_file data_files/${args[0]}_patterns.pcl
python scripts/dt/sl_patterns_to_sl_fits.py --sl_patterns_file data_files/${args[0]}_patterns.pcl --sl_fits_file data_files/${args[0]}_fits.pcl


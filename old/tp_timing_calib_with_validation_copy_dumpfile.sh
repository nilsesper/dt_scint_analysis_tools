### RUN SCINT ANALYSIS CHAIN

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

# copy dumpfile from rack
~/utils/scp_dumpfile_from_rack.sh ${args[0]}.txt ~/masterarbeit/zynq_read-out_software/output/

# do analysis
python test_scripts/tp_timing_calib.py --inputfile ~/masterarbeit/zynq_read-out_software/output/${args[0]}.txt --validationfile ~/masterarbeit/zynq_read-out_software/output/${args[0]}_validation.txt




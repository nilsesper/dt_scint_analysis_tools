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

python scripts/scint/raw_scint_hits_to_scint_hits.py --raw_scint_hits_file ~/masterarbeit/dt_scint_analysis_tools/data_files/${args[0]}_raw_hits.pcl --scint_hits_file ~/masterarbeit/dt_scint_analysis_tools/data_files/${args[0]}_hits.pcl 
python scripts/scint/scint_hits_to_areas.py --scint_hits_file /home/nils/masterarbeit/dt_scint_analysis_tools/data_files/${args[0]}_hits.pcl --scint_areas_file /home/nils/masterarbeit/dt_scint_analysis_tools/data_files/${args[0]}_areas.pcl




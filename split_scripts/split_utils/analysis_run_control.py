########################################
### analysis scripts run controller
# to run several scripts (tasks) on several split data files sequentially
########################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse
import subprocess
import atexit
import sys
import time

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils
from analysis_tools.params import params, derived_params

### kill subprocesses on exit of program
all_processes = []
def cleanup():
    timeout_sec = 5
    for p in all_processes: # list of your processes
        p_sec = 0
        for second in range(timeout_sec):
            if p.poll() == None:
                time.sleep(1)
                p_sec += 1
        if p_sec >= timeout_sec:
            p.kill() # supported from python 2.6
    print('cleaned up all running processes!')
atexit.register(cleanup)

# ---------------------------------------------------------------

allowed_tasks = {
    ### dumpfile workflow
    "dump_import": # dumpfile to dt hits & scint hits
        "python scripts/combined/dumpfile_to_dt_and_raw_scint_hits.py --input_dumpfile [DUMPFILE] --dt_hits_file [DT_HITS] --raw_scint_hits_file [RAW_SCINT_HITS] --ts_range_file [TS_RANGE]",
    "dump_dt_nodeadtime": # only dt hits w/o dead time
        "python scripts/dt/dumpfile_to_dt_hits.py --input_dumpfile [DUMPFILE] --dt_hits_file [DT_HITS_NODEADTIME] --nodeadtime",
    ### dt workflow
    "dt_corr": # apply timing correction (from testpulses) to dt hit timestamps
        "python scripts/dt/dt_hits_timing_correction.py --dt_hits_file [DT_HITS] --dt_tp_corrections_file [DT_CORRECTIONS] --corr_dt_hits_file [DT_CORR_HITS]",
    "dt_patterns": # construct track-like sl patterns from cell hits
        "python scripts/dt/dt_hits_to_sl_patterns.py --dt_hits_file [DT_CORR_HITS] --sl_patterns_file [SL_PATTERNS] --n_proc [N_PROC]",
    "dt_fake_patterns": # construct fake (not track-like) patterns from cell hits
        "python scripts/dt/dt_hits_to_sl_fake_patterns.py --dt_hits_file [DT_CORR_HITS] --sl_patterns_file [SL_FAKE_PATTERNS] --n_proc [N_PROC]",
    "dt_fits": # fit track-like sl patterns to straight tracks
        "python scripts/dt/sl_patterns_to_sl_fits.py --sl_patterns_file [SL_PATTERNS] --sl_fits_file [SL_FITS] --n_proc [N_PROC]",
    "dt_fit_cuts": # apply cuts to resulting sl pattern fits
        "python scripts/dt/sl_fits_apply_cuts.py --input_data_file [SL_FITS] --cut_data_file [SL_FITS_AFTERCUTS]",
    "dt_fit_groups": # group fits together within same sl, which lie close in time
        "python scripts/dt/sl_fits_to_sl_fit_groups.py --sl_fits_file [SL_FITS_AFTERCUTS] --sl_fit_groups_file [SL_FIT_GROUPS] --n_proc [N_PROC]",
    "dt_fit_group_cuts": # apply cuts to fit groups
        "python scripts/dt/sl_fit_groups_apply_cuts.py --input_data_file [SL_FIT_GROUPS] --cut_data_file [SL_FIT_GROUPS_AFTERCUTS]",
    "dt_muons": # construct global muon tracks by combining pattern fits from all three sls
        "python scripts/dt/sl_fit_groups_to_dt_muons.py --sl_fits_file [SL_FITS_AFTERCUTS] --sl_fit_groups_file [SL_FIT_GROUPS_AFTERCUTS] --dt_muons_file [DT_MUONS]",
    ### scint workflow
    "scint_dead_time": # apply offline dead time
        "python scripts/scint/raw_scint_hits_dead_time.py --raw_scint_hits_file [RAW_SCINT_HITS] --raw_scint_hits_deadtime_file [RAW_SCINT_HITS_DEADTIME]",
    "scint_raw_groups": # construct groups of sipm hits, which lie close in time
        "python scripts/scint/raw_scint_hits_to_raw_groups.py --raw_scint_hits_file [RAW_SCINT_HITS_DEADTIME] --raw_scint_groups_file [RAW_SCINT_GROUPS] --n_proc [N_PROC]",
    "scint_hits": # construct scintillator strip hits from groups
        "python scripts/scint/raw_groups_to_scint_hits.py --raw_scint_hits_file [RAW_SCINT_HITS_DEADTIME] --raw_scint_groups_file [RAW_SCINT_GROUPS] --scint_hits_file [SCINT_HITS]",
    "scint_pixels": # construct scintillator pixel hits from groups
        "python scripts/scint/raw_groups_to_scint_areas.py --raw_scint_hits_file [RAW_SCINT_HITS_DEADTIME] --raw_scint_groups_file [RAW_SCINT_GROUPS] --scint_areas_file [SCINT_AREAS]",
    ### simulation workflow
    "gen_muons_sim": # generate cosmic muon tracks with monte carlo method
        "python scripts/sim/gen_cosmic_tracks.py --cosmic_muons_file [SIM_MUONS]",
    "dt_hits_sim": # propagate cosmic muon tracks through dt chamber geometry and generate simulated dt hits
        "python scripts/sim/cosmic_tracks_to_dt_hits.py --cosmic_muons_file [SIM_MUONS] --dt_hits_file [DT_HITS_SIM] --ts_range_file [TS_RANGE]",
    "dt_cut_dead_cells": # cut away simulated cell hits which are dead in the real detector
        "python scripts/sim/sim_dt_hits_apply_cuts.py --input_data_file [DT_HITS_SIM] --cut_data_file [DT_HITS]",
    "dt_skip_corr": # skip correction (as in simulation it is not necessary)
        "cp [DT_HITS] [DT_CORR_HITS]",
}
### filepath wildcards:
prefix_wildcard_list = [
    "DT_HITS", "DT_HITS_NODEADTIME", "DT_CORR_HITS", "SL_PATTERNS", "SL_FAKE_PATTERNS", "SL_FITS", "SL_FITS_AFTERCUTS", "SL_FIT_GROUPS", "SL_FIT_GROUPS_AFTERCUTS", "DT_MUONS",
    "RAW_SCINT_HITS", "RAW_SCINT_HITS_DEADTIME", "RAW_SCINT_GROUPS", "SCINT_HITS", "SCINT_AREAS",
    "DT_HIT_DIFFERENCES",
    "SIM_MUONS", "DT_HITS_SIM",
]
def replace_wildcards(command, base_path, dump_file, file_prefix, n_proc):
    wildcard_dict = {
        "["+n+"]": base_path+"/"+file_prefix+"_"+n+".pcl" for n in prefix_wildcard_list
    } | {
        "[DUMPFILE]": base_path+"/"+dump_file,
        "[TS_RANGE]": base_path+"/"+file_prefix+"_TS_RANGE.txt",
        "[DT_CORRECTIONS]": base_path+"/"+"DT_CORRECTIONS.pcl",
    } | {
        "[N_PROC]": n_proc,
    }
    for k,v in wildcard_dict.items():
        command = command.replace(k,v)
    return command

### data config file is .txt file with no empty lines or spaces and the following content
# [dumpfilename.txt],[datafilesprefix]
# the data file prefix is used to store all file wildcards for this dumpfile

# ---------------------------------------------------------------

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():

    ### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type     = str,
        help     = "base / working directory",
        required=True,
    )
    parser.add_argument(
        "--data_config_file",
        type     = str,
        help     = "path to data config file which stores the data file names to be considered for the analysis",
        required=True,
    )
    parser.add_argument(
        "--task_list",
        type     = str,
        help     = "list of tasks to be executed",
        required=True,
    )
    # optional
    parser.add_argument( # specify --n_proc for multiprocessing tasks
        "--n_proc",
        type     = str,
        help     = "n_proc argument for tasks (if available)",
        default  = "16",
    )
    parser.add_argument( # if this flag is given: invert data & task loop order: i.e. run all tasks for one ds before going to the next ds
        "--task_for_each_set",
        help     = "n_proc argument for tasks (if available)",
        action = "store_true",
    )
    # ---
    args = parser.parse_args()
    # base file path
    base_path = args.base_path
    # tasks to be executed
    task_list = []
    for task_string in args.task_list.split(","):
        if task_string not in allowed_tasks.keys():
            raise Exception(f"Illegal task {task_string}")
        task_list.append(task_string)
    n_tasks = len(task_list)
    # list of data files to be used
    dump_files = [] # list of dumpfile names
    file_prefixes = [] # list of data file prefixes to be used
    data_config_file = args.data_config_file
    with open(data_config_file) as f:
        lines = f.readlines()
        for line in lines:
            dump_file, file_prefix = line.split(",")
            file_prefixes.append(file_prefix.replace("\n","").replace("\r","").replace("\t",""))
            dump_files.append(dump_file.replace("\n","").replace("\r","").replace("\t",""))
    n_data = len(dump_files)
    # other
    n_proc = args.n_proc
    run = True
    task_for_each_set = False
    if args.task_for_each_set:
        task_for_each_set = True

    ####################

    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+++++++++++ TODO:")
    print(f"+++++++++++  task_list ({n_tasks:3} total)      =  {task_list}")
    print(f"+++++++++++  dump_files ({n_data:3} total)     =  {dump_files}")
    print(f"+++++++++++  file_prefixes ({n_data:3} total)  =  {file_prefixes}")

    if not task_for_each_set: ## default order: one task for all ds then next task

        for task_step, task in enumerate(task_list):
            print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            for data_step, data_idx in enumerate(range(n_data)):
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++ EXECUTING TASK \"{task}\" ({task_step+1} / {n_tasks}): +++++++++++")
                print(f"+++++++++++ -> FOR DATA \"{file_prefixes[data_idx]}\" ({data_step+1} / {n_data}): +++++++++++")
                raw_command = allowed_tasks[task]
                command = replace_wildcards(raw_command, base_path=base_path, dump_file=dump_files[data_idx], file_prefix=file_prefixes[data_idx], n_proc=n_proc)
                print(f"+++++++++++ bash command = {command}")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                if run:
                    my_env = os.environ.copy()
                    my_env["PATH"] = f"/usr/sbin:/sbin:{my_env['PATH']}"
                    process = subprocess.Popen(command, env=my_env, shell=True) # launch subprocess
                    all_processes.append(process) # log running process
                    process.wait() # wait till completion
                    all_processes.remove(process)

    else: ## inverted order: all tasks for one ds then next ds

        for data_step, data_idx in enumerate(range(n_data)):
            print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            for task_step, task in enumerate(task_list):
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++ FOR DATA \"{file_prefixes[data_idx]}\" ({data_step+1} / {n_data}): +++++++++++")
                print(f"+++++++++++ -> EXECUTING TASK \"{task}\" ({task_step+1} / {n_tasks}): +++++++++++")
                raw_command = allowed_tasks[task]
                command = replace_wildcards(raw_command, base_path=base_path, dump_file=dump_files[data_idx], file_prefix=file_prefixes[data_idx], n_proc=n_proc)
                print(f"+++++++++++ bash command = {command}")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                if run:
                    my_env = os.environ.copy()
                    my_env["PATH"] = f"/usr/sbin:/sbin:{my_env['PATH']}"
                    process = subprocess.Popen(command, env=my_env, shell=True) # launch subprocess
                    all_processes.append(process) # log running process
                    process.wait() # wait till completion
                    all_processes.remove(process)



if __name__ == "__main__":
    main()
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")











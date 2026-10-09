#################################################################
### run the complete dt workflow on one dumpfile: the stage scripts one after the other
# raw dumpfile -> dt hits (timing corrected with --dt_tp_corrections_file) -> hit diff hist, cell counts
#   -> sl patterns -> sl fits -> cut sl fits -> super fits (phi superlayers) -> cut super fits
#   -> dt muons (cut super fits + theta sl fits)
#
# Every stage is one script of this directory, started as
#   python scripts/<stage script>.py --<input file> ... --<output file> ...
# and the command is printed before, so a stage can also be repeated by hand. All files are written to --output_dir
# as <prefix>_<stage>.root (<prefix>: the dumpfile name without ending, unless --prefix is given).
# Use --from_stage / --to_stage to run only a part, e.g. to redo everything after the sl fits.
#################################################################

import argparse
import os
import subprocess
import sys
import time

from analysis_tools.utils.dt_fit_utils import SUPER_FIT_SUFFIX
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

STAGES = ["dt_hits", "hit_diff_hist", "cell_counts", "sl_patterns", "sl_fits", "sl_fits_cut", "super_fits", "super_fits_cut", "dt_muons"]
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

### the command of one stage: [python, script, arguments...]
def stage_command(stage, args, files):
    python = sys.executable
    if stage == "dt_hits":
        command = [python, os.path.join(SCRIPTS_DIR, "dumpfile_to_dt_hits.py"), "--input_dumpfile", args.input_dumpfile,
                   "--dt_hits_file", files["dt_hits"], "--n_lines_to_skip", str(args.n_lines_to_skip), "--block_lines", str(args.block_lines)]
        if args.dt_tp_corrections_file is not None:
            command += ["--dt_tp_corrections_file", args.dt_tp_corrections_file]
    elif stage == "hit_diff_hist":
        command = [python, os.path.join(SCRIPTS_DIR, "dt_hits_to_hit_diff_hist.py"), "--dt_hits_file", files["dt_hits"],
                   "--hit_diff_hist_file", files["hit_diff_hist"], "--chunk_size", str(args.chunk_size)]
    elif stage == "cell_counts":
        command = [python, os.path.join(SCRIPTS_DIR, "dt_hits_to_cell_counts.py"), "--dt_hits_file", files["dt_hits"],
                   "--cell_counts_file", files["cell_counts"], "--chunk_size", str(args.chunk_size)]
    elif stage == "sl_patterns":
        command = [python, os.path.join(SCRIPTS_DIR, "dt_hits_to_sl_patterns.py"), "--dt_hits_file", files["dt_hits"],
                   "--sl_patterns_file", files["sl_patterns"], "--chunk_size", str(args.chunk_size), "--n_proc", str(args.n_proc)]
    elif stage == "sl_fits":
        command = [python, os.path.join(SCRIPTS_DIR, "sl_patterns_to_sl_fits.py"), "--sl_patterns_file", files["sl_patterns"],
                   "--sl_fits_file", files["sl_fits"], "--n_proc", str(args.n_proc)]
    elif stage == "sl_fits_cut":
        command = [python, os.path.join(SCRIPTS_DIR, "apply_cuts.py"), "--input_file", files["sl_fits"],
                   "--output_file", files["sl_fits_cut"], "--cuts", args.sl_fit_cuts]
    elif stage == "super_fits":
        command = [python, os.path.join(SCRIPTS_DIR, "sl_fits_to_super_fits.py"), "--sl_fits_file", files["sl_fits_cut"],
                   "--super_fits_file", files["super_fits"], "--max_alpha_deg", str(args.max_alpha_deg), "--n_proc", str(args.n_proc)]
        if args.super_fit_free_vd:
            command += ["--free_vd"]
    elif stage == "super_fits_cut":
        command = [python, os.path.join(SCRIPTS_DIR, "apply_cuts.py"), "--input_file", files["super_fits"],
                   "--output_file", files["super_fits_cut"], "--cuts", args.super_fit_cuts]
    elif stage == "dt_muons":
        command = [python, os.path.join(SCRIPTS_DIR, "super_fits_to_dt_muons.py"), "--super_fits_file", files["super_fits_cut"],
                   "--sl_fits_file", files["sl_fits_cut"], "--dt_muons_file", files["dt_muons"]]
        if args.tgroup_tolerance is not None:
            command += ["--tgroup_tolerance", str(args.tgroup_tolerance)]
    if args.params_file is not None:
        command += ["--params_file", args.params_file]
    return command

def main():
    parser = argparse.ArgumentParser(description="Run the complete dt workflow on one dumpfile.")
    parser.add_argument("--input_dumpfile", type=str, default=None, help="input file path: raw dumpfile (.txt); not needed if --from_stage is after dt_hits")
    parser.add_argument("--output_dir", type=str, required=True, help="directory for all output files")
    parser.add_argument("--prefix", type=str, default=None, help="prefix of the output file names (default: name of the dumpfile without ending)")
    parser.add_argument("--from_stage", type=str, choices=STAGES, default=STAGES[0], help="first stage to run (earlier outputs have to exist in --output_dir)")
    parser.add_argument("--to_stage", type=str, choices=STAGES, default=STAGES[-1], help="last stage to run")
    parser.add_argument("--skip_stages", type=str, default="", help="comma separated list of stages to leave out, e.g. \"hit_diff_hist,cell_counts\"")
    parser.add_argument("--dt_tp_corrections_file", type=str, default=None,
                        help="optional: testpulse timing calibration (.root from dumpfile_to_dt_tp_corrections.py, or old .pcl). "
                             "If given, the timestamps of the dt hits are corrected in stage dt_hits")
    # settings of the stages (same defaults as the stage scripts)
    parser.add_argument("--n_lines_to_skip", type=int, default=999, help="dumpfile lines to ignore at the start")
    parser.add_argument("--block_lines", type=int, default=500_000, help="dumpfile lines processed at once")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of dt hits per chunk (patterns are only searched inside a chunk)")
    parser.add_argument("--sl_fit_cuts", type=str, default="impossible,==,0;chi2/ndf,<,20",
                        help="cuts on the sl fits (stage sl_fits_cut), format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--max_alpha_deg", type=float, default=60, help="max |track angle| in degrees of phi sl fits which are paired to super fits")
    parser.add_argument("--super_fit_free_vd", action="store_true", help="fit the drift velocity as free parameter in the super fit (default: fixed)")
    parser.add_argument("--super_fit_cuts", type=str, default=f"impossible{SUPER_FIT_SUFFIX},==,0;chi2/ndf{SUPER_FIT_SUFFIX},<,20",
                        help="cuts on the super fits (stage super_fits_cut)")
    parser.add_argument("--tgroup_tolerance", type=float, default=None,
                        help="max |t0 difference| between super fit and theta sl fit of a muon in timestamp units (default: params._muon_tgroup_tolerance)")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes for the pattern search and the fits (does not change the results)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    if args.prefix is None:
        if args.input_dumpfile is None:
            parser.error("give --prefix when running without --input_dumpfile")
        args.prefix = os.path.splitext(os.path.basename(args.input_dumpfile))[0]
    skip = []
    for stage in args.skip_stages.split(","):
        if stage.strip() != "":
            skip.append(stage.strip())
    for stage in skip:
        if stage not in STAGES:
            parser.error(f"unknown stage \"{stage}\" in --skip_stages, allowed: {STAGES}")
    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if first > last:
        parser.error("--from_stage is after --to_stage")
    stages_to_run = []
    for stage in STAGES[first:last + 1]:
        if stage not in skip:
            stages_to_run.append(stage)
    if "dt_hits" in stages_to_run and args.input_dumpfile is None:
        parser.error("--input_dumpfile is needed for stage dt_hits")

    files = {}
    for stage in STAGES:
        files[stage] = os.path.join(args.output_dir, f"{args.prefix}_{stage}.root")
    os.makedirs(args.output_dir, exist_ok=True)
    log(f"###### stages to run: {stages_to_run}")
    t_start = time.perf_counter()

    for stage in stages_to_run:
        command = stage_command(stage, args, files)
        log(f"###### stage {stage}: {' '.join(command)}")
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"stage {stage} failed (exit code {result.returncode})")

    log(f"###### all stages done in {(time.perf_counter() - t_start) / 60:.1f} minutes, output files:")
    for stage in stages_to_run:
        log(f"    {stage:14s} -> {files[stage]}")

if __name__ == "__main__":
    main()
    log("###### Done.")

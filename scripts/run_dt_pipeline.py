#################################################################
### run the complete dt workflow on one dumpfile, stage by stage
# raw dumpfile -> dt hits (timing corrected with --dt_tp_corrections_file) -> sl patterns -> sl fits -> cut sl fits
#   -> super fits (phi superlayers) -> cut super fits -> dt muons (cut super fits + theta sl fits)
#
# This calls the same functions as the individual stage scripts in this directory, with the same
# default settings. All files are written to --output_dir as <prefix>_<stage>.root, where <prefix>
# is the dumpfile name without ending unless --prefix is given.
# Use --from_stage / --to_stage to run only a part, e.g. to redo everything after the sl fits.
#################################################################

import argparse
import os
import time
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import cut_utils, dt_pipeline_utils, root_utils

# ---------------------------------------------------------------

STAGES = ["dt_hits", "hit_diff_hist", "cell_counts", "sl_patterns", "sl_fits", "sl_fits_cut", "super_fits", "super_fits_cut", "dt_muons"]

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the complete dt workflow on one dumpfile.")
    parser.add_argument("--input_dumpfile", type=str, default=None, help="input file path: raw dumpfile (.txt); not needed if --from_stage is after dt_hits")
    parser.add_argument("--output_dir", type=str, required=True, help="directory for all output files")
    parser.add_argument("--prefix", type=str, default=None, help="prefix of the output file names (default: name of the dumpfile without ending)")
    parser.add_argument("--from_stage", type=str, choices=STAGES, default=STAGES[0], help="first stage to run (earlier outputs have to exist in --output_dir)")
    parser.add_argument("--to_stage", type=str, choices=STAGES, default=STAGES[-1], help="last stage to run")
    parser.add_argument("--dt_tp_corrections_file", type=str, default=None,
                        help="optional: testpulse timing calibration (.root from dumpfile_to_dt_tp_corrections.py, or old .pcl). "
                             "If given, the timestamps of the dt hits are corrected in stage dt_hits")
    parser.add_argument("--skip_stages", type=str, default="", help="comma separated list of stages to leave out, e.g. \"hit_diff_hist,cell_counts\"")
    # settings of the individual stages (same defaults as the stage scripts)
    parser.add_argument("--n_lines_to_skip", type=int, default=50_000, help="dumpfile lines to ignore at the start")
    parser.add_argument("--block_lines", type=int, default=2_000_000, help="dumpfile lines processed at once")
    parser.add_argument("--step_size", type=str, default=root_utils.DEFAULT_STEP_SIZE, help="how much of a ROOT file is read at once: memory size like \"200 MB\" or a number of rows")
    parser.add_argument("--sl_fit_cuts", type=str, default="impossible,==,0;chi2/ndf,<,20",
                        help="cuts on the sl fits (stage sl_fits_cut), format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--max_alpha_deg", type=float, default=60, help="max |track angle| in degrees of phi sl fits which are combined to super fits")
    parser.add_argument("--super_fit_free_vd", action="store_true", help="fit the drift velocity as free parameter in the super fit (default: fixed)")
    parser.add_argument("--super_fit_suffix", type=str, default=dt_pipeline_utils.DEFAULT_SUPER_FIT_SUFFIX, help="suffix of the super fit result branches")
    parser.add_argument("--super_fit_cuts", type=str, default=None,
                        help="cuts on the super fits (stage super_fits_cut), default: \"impossible<suffix>,==,0;chi2/ndf<suffix>,<,20\"")
    parser.add_argument("--tgroup_tolerance", type=float, default=None,
                        help="max |t0 difference| between super fit and theta sl fit of a muon in timestamp units (default: params._muon_tgroup_tolerance)")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes for the dumpfile conversion, the pattern search and the fit stages (does not change the results)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    if args.prefix is None:
        if args.input_dumpfile is None:
            parser.error("give --prefix when running without --input_dumpfile")
        args.prefix = os.path.splitext(os.path.basename(args.input_dumpfile))[0]
    skip = []
    for s in args.skip_stages.split(","):
        if s.strip() != "":
            skip.append(s.strip())
    for s in skip:
        if s not in STAGES:
            parser.error(f"unknown stage \"{s}\" in --skip_stages, allowed: {STAGES}")
    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if first > last:
        parser.error("--from_stage is after --to_stage")
    todo = []  # stages to run
    for s in STAGES[first:last + 1]:
        if s not in skip:
            todo.append(s)
    if "dt_hits" in todo and args.input_dumpfile is None:
        parser.error("--input_dumpfile is needed for stage dt_hits")

    super_fit_cuts = args.super_fit_cuts
    if super_fit_cuts is None:
        super_fit_cuts = f"impossible{args.super_fit_suffix},==,0;chi2/ndf{args.super_fit_suffix},<,20"
    step_size = root_utils.parse_step_size(args.step_size)
    f = {s: os.path.join(args.output_dir, f"{args.prefix}_{s}.root") for s in STAGES}
    hits = f["dt_hits"]  # dt hits used by the following stages
    if args.dt_tp_corrections_file is not None and "dt_hits" not in todo:
        log(f"###### WARNING: --dt_tp_corrections_file is only used in stage dt_hits, which is not run; {hits} is used as it is")
    os.makedirs(args.output_dir, exist_ok=True)
    root_utils.log(f"###### stages to run: {todo}")
    t_start = time.perf_counter()

    for stage in todo:
        if stage == "dt_hits":
            dt_pipeline_utils.dumpfile_to_dt_hits(
                args.input_dumpfile, f["dt_hits"], n_lines_to_skip=args.n_lines_to_skip, block_n_lines=args.block_lines, n_proc=args.n_proc,
                dt_tp_corrections_file=args.dt_tp_corrections_file,
            )
        elif stage == "hit_diff_hist":
            dt_pipeline_utils.dt_hits_to_hit_diff_hist(
                hits, f["hit_diff_hist"], step_size=step_size
            )
        elif stage == "cell_counts":
            dt_pipeline_utils.dt_hits_to_cell_counts(
                hits, f["cell_counts"], step_size=step_size
            )
        elif stage == "sl_patterns":
            dt_pipeline_utils.dt_hits_to_sl_patterns(
                hits, f["sl_patterns"], step_size=step_size, n_proc=args.n_proc
            )
        elif stage == "sl_fits":
            dt_pipeline_utils.sl_patterns_to_sl_fits(
                f["sl_patterns"], f["sl_fits"], fit_vd=False, step_size=step_size, n_proc=args.n_proc
            )
        elif stage == "sl_fits_cut":
            dt_pipeline_utils.apply_cuts(
                f["sl_fits"], f["sl_fits_cut"], cut_utils.parse_cuts(args.sl_fit_cuts), step_size=step_size
            )
        elif stage == "super_fits":
            dt_pipeline_utils.sl_fits_to_super_fits(
                f["sl_fits_cut"], f["super_fits"], max_alpha=np.deg2rad(args.max_alpha_deg), fit_vd=args.super_fit_free_vd, suffix=args.super_fit_suffix,
                n_proc=args.n_proc,
            )
        elif stage == "super_fits_cut":
            dt_pipeline_utils.apply_cuts(
                f["super_fits"], f["super_fits_cut"], cut_utils.parse_cuts(super_fit_cuts), step_size=step_size
            )
        elif stage == "dt_muons":
            dt_pipeline_utils.super_fits_to_dt_muons(
                f["super_fits_cut"], f["sl_fits_cut"], f["dt_muons"], suffix=args.super_fit_suffix, tgroup_tolerance=args.tgroup_tolerance,
            )

    root_utils.log(f"###### all stages done in {(time.perf_counter() - t_start) / 60:.1f} minutes, output files:")
    for stage in todo:
        if os.path.isfile(f[stage]):
            root_utils.log(f"    {stage:14s} -> {f[stage]}")
        else:
            root_utils.log(f"    {stage:14s} -> {f[stage]}   (not written)")

if __name__ == "__main__":
    main()
    log("###### Done.")

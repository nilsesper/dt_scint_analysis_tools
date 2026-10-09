#################################################################
### cut super fits (phi superlayers) + cut sl fits (theta superlayer) -> dt muons
# every super fit is combined with the theta sl fit closest in time (within --tgroup_tolerance), see dt_muon_reco_utils.py
#   phi view of the muon:   position and slope of the super fit
#   theta view of the muon: position and slope of the theta sl fit
# inputs:
#   --super_fits_file: super fits after quality cuts (apply_cuts.py), at least "impossible_super_fits,==,0"
#   --sl_fits_file:    the cut sl fits file the super fits were made from
# Only fits with the same "chunk_id" are combined. Output file: see OUTPUT_FILES.md.
#################################################################

import argparse
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import data_utils, dt_muon_reco_utils, root_utils
from analysis_tools.utils.dt_fit_utils import SUPER_FIT_SUFFIX
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combine phi super fits and theta sl fits to dt muons.")
    parser.add_argument("--super_fits_file", type=str, required=True, help="input file path: super fits after cuts (.root)")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: cut sl fits the super fits were made from (.root)")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="output file path: dt muons (.root)")
    parser.add_argument("--tgroup_tolerance", type=float, default=params._muon_tgroup_tolerance,
                        help="max |t0 difference| between super fit and theta sl fit in timestamp units (default: params._muon_tgroup_tolerance)")
    parser.add_argument("--verbose", action="store_true", help="print every muon")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "super fits + theta sl fits -> dt muons"
    root_utils.check_input_file(args.super_fits_file)
    root_utils.check_input_file(args.sl_fits_file)
    log(f"[{label}] START \"{args.super_fits_file}\" + \"{args.sl_fits_file}\" -> \"{args.dt_muons_file}\" "
        f"(tgroup_tolerance={args.tgroup_tolerance:.2f} ts units)")

    ### rows of every chunk_id in the two input files
    super_fit_chunks = root_utils.rows_of_each_chunk_id(args.super_fits_file)
    sl_fit_rows_of_chunk_id = {}
    for chunk_id, start, stop in root_utils.rows_of_each_chunk_id(args.sl_fits_file):
        sl_fit_rows_of_chunk_id[chunk_id] = (start, stop)

    output_file = root_utils.create_file(args.dt_muons_file)
    n_super_fits_total, n_muons_total, n_ambiguous = 0, 0, 0
    ts_min, ts_max = None, None
    for i_chunk in range(len(super_fit_chunks)):
        chunk_id, super_start, super_stop = super_fit_chunks[i_chunk]
        if chunk_id not in sl_fit_rows_of_chunk_id:
            raise RuntimeError(f"No sl fits with chunk_id {chunk_id} in {args.sl_fits_file}. Is this the file the super fits were made from?")
        sl_start, sl_stop = sl_fit_rows_of_chunk_id[chunk_id]
        super_fits = root_utils.read_tree(args.super_fits_file, root_utils.DEFAULT_TREE, super_start, super_stop)
        sl_fits = root_utils.read_tree(args.sl_fits_file, root_utils.DEFAULT_TREE, sl_start, sl_stop)
        if np.any(super_fits["impossible" + SUPER_FIT_SUFFIX] != 0):
            raise RuntimeError(f"{args.super_fits_file} contains super fits flagged \"impossible\". Apply cuts first "
                               f"(at least \"impossible{SUPER_FIT_SUFFIX},==,0\"), see scripts/apply_cuts.py.")

        dt_muons = dt_muon_reco_utils.reco_muons(super_fits, sl_fits, super_start, sl_start, args.tgroup_tolerance, args.verbose)
        dt_muons[root_utils.CHUNK_ID_KEY] = np.full(root_utils.length(dt_muons), chunk_id, dtype=np.int64)
        dt_muons = data_utils.sort_by_key(data=dt_muons, sort_key="ts", silent=True)
        root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, dt_muons)

        n_muons = root_utils.length(dt_muons)
        n_super_fits_total += root_utils.length(super_fits)
        n_muons_total += n_muons
        n_ambiguous += int(np.sum(dt_muons["n_theta_candidates"] > 1))
        if n_muons > 0:
            if ts_min is None or np.amin(dt_muons["ts"]) < ts_min:
                ts_min = np.amin(dt_muons["ts"])
            if ts_max is None or np.amax(dt_muons["ts"]) > ts_max:
                ts_max = np.amax(dt_muons["ts"])
        log(f"    chunk {i_chunk + 1:,} / {len(super_fit_chunks):,}: {root_utils.length(super_fits):,} super fits -> {n_muons:,} dt muons")
    output_file.close()

    rate_text = ""
    if ts_min is not None and ts_max > ts_min:
        duration = (ts_max - ts_min) * 0.78e-9
        rate_text = f", rate {n_muons_total / duration:.2f} Hz over {duration:.2f} s"
    log(f"[{label}] DONE. super_fits={n_super_fits_total:,}, dt_muons={n_muons_total:,} "
        f"({n_ambiguous:,} with more than one theta candidate){rate_text}")

if __name__ == "__main__":
    main()
    log("###### Done.")

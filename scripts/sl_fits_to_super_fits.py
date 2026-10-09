#################################################################
### cut sl fits -> super fits
# pairs the sl fits of the two phi superlayers to 8-hit "super patterns" and fits one track through the 8 hits
# (default: fixed drift velocity, --free_vd makes it a fit parameter), see dt_fit_utils.py
# input: sl fits after quality cuts (apply_cuts.py), at least "impossible,==,0"
# Only sl fits with the same "chunk_id" (= from the same chunk of dt hits) are paired. With --n_proc N, the rows of
# N chunk_ids are processed at the same time, each in its own process (same result as with 1).
# Output files: see OUTPUT_FILES.md.
#################################################################

import argparse
import multiprocessing
import time
import numpy as np

from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils import dt_fit_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

### the sl fits of one chunk_id -> super patterns and super fits; runs in its own process if --n_proc > 1
def super_fits_of_chunk(job):
    sl_fits_file, chunk_id, start, stop, max_chi2ndf, max_alpha, fit_vd, verbose = job
    t_start = time.perf_counter()
    sl_fits = root_utils.read_tree(sl_fits_file, root_utils.DEFAULT_TREE, start, stop)
    super_patterns = dt_fit_utils.build_super_patterns(sl_fits, max_chi2ndf, max_alpha)
    n_super_patterns = root_utils.length(super_patterns)
    # rows of the paired sl fits in the whole file instead of in this chunk
    for sl in dt_chamber_utils.phi_superlayers():
        super_patterns[f"row_sl{sl}"] = super_patterns[f"row_sl{sl}"] + start
    super_patterns[root_utils.CHUNK_ID_KEY] = np.full(n_super_patterns, chunk_id, dtype=np.int64)
    super_fits = dt_fit_utils.fit_super_patterns(super_patterns, fit_vd, verbose)
    return super_patterns, super_fits, root_utils.length(sl_fits), time.perf_counter() - t_start

def main():
    parser = argparse.ArgumentParser(description="Combine the fits of the two phi superlayers and fit the 8 hits together.")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="input file path: sl fits after cuts (.root)")
    parser.add_argument("--super_fits_file", type=str, required=True, help="output file path: super fits (.root)")
    parser.add_argument("--super_patterns_file", type=str, default=None,
                        help="optional output file path: the super patterns before the fit (.root)")
    parser.add_argument("--max_chi2", type=float, default=None,
                        help="additional max chi2/ndf of sl fits which may be paired (default: no additional cut)")
    parser.add_argument("--max_alpha_deg", type=float, default=60, help="max |track angle| in degrees of sl fits which may be paired")
    parser.add_argument("--free_vd", action="store_true", help="fit the drift velocity as free parameter in the super fit (default: fixed)")
    parser.add_argument("--n_proc", type=int, default=1, help="number of chunk_ids processed at the same time")
    parser.add_argument("--verbose", action="store_true", help="print every fit (switches off --n_proc)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "sl fits -> super fits"
    root_utils.check_input_file(args.sl_fits_file)
    max_chi2ndf = np.inf
    if args.max_chi2 is not None:
        max_chi2ndf = args.max_chi2
    max_alpha = np.deg2rad(args.max_alpha_deg)
    log(f"[{label}] START \"{args.sl_fits_file}\" -> \"{args.super_fits_file}\" "
        f"(max_chi2={args.max_chi2}, max_alpha={max_alpha:.4f} rad, fit_vd={args.free_vd}, n_proc={args.n_proc})")
    n_proc = args.n_proc
    if args.verbose:
        n_proc = 1

    ### the chunks: rows of every chunk_id
    jobs = []
    for chunk_id, start, stop in root_utils.rows_of_each_chunk_id(args.sl_fits_file):
        jobs.append((args.sl_fits_file, chunk_id, start, stop, max_chi2ndf, max_alpha, args.free_vd, args.verbose))

    ### process the chunks: with n_proc > 1, n_proc chunks at the same time; the results come in the order of the chunks
    if n_proc > 1:
        pool = multiprocessing.Pool(n_proc)
        results = pool.imap(super_fits_of_chunk, jobs)
    else:
        results = map(super_fits_of_chunk, jobs)

    output_file = root_utils.create_file(args.super_fits_file)
    patterns_file = None
    if args.super_patterns_file is not None:
        patterns_file = root_utils.create_file(args.super_patterns_file)
    n_sl_fits_total, n_super_fits_total = 0, 0
    i_chunk = 0
    for super_patterns, super_fits, n_sl_fits, seconds in results:
        i_chunk += 1
        root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, super_fits)
        if patterns_file is not None:
            root_utils.write_rows(patterns_file, root_utils.DEFAULT_TREE, super_patterns)
        n_sl_fits_total += n_sl_fits
        n_super_fits_total += root_utils.length(super_fits)
        log(f"    chunk {i_chunk:,} / {len(jobs):,}: {n_sl_fits:,} sl fits -> {root_utils.length(super_fits):,} super fits ({seconds:.2f}s)")
    output_file.close()
    if patterns_file is not None:
        patterns_file.close()
    if n_proc > 1:
        pool.close()
        pool.join()
    log(f"[{label}] DONE. sl_fits={n_sl_fits_total:,}, super_fits={n_super_fits_total:,}")

if __name__ == "__main__":
    main()
    log("###### Done.")

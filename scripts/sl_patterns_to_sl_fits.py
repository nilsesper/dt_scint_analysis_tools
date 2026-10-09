#################################################################
### sl patterns -> sl fits (straight track through the 4 hits of every pattern, the best laterality is kept)
# default: fixed drift velocity. The output keeps all pattern branches and adds the fit results (see OUTPUT_FILES.md).
# The patterns are fitted in chunks of --chunk_size rows; with --n_proc N, N chunks are fitted at the same time,
# each one in its own process (same result as with 1).
#################################################################

import argparse
import multiprocessing
import time

from analysis_tools.utils import dt_fit_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

### one chunk of patterns -> sl fits; runs in its own process if --n_proc > 1
def fit_chunk(job):
    sl_patterns_file, start, stop, fit_vd, verbose = job
    t_start = time.perf_counter()
    sl_patterns = root_utils.read_tree(sl_patterns_file, root_utils.DEFAULT_TREE, start, stop)
    sl_fits = dt_fit_utils.fit_sl_patterns(sl_patterns, fit_vd, verbose)
    return sl_fits, time.perf_counter() - t_start

def main():
    parser = argparse.ArgumentParser(description="Fit a track segment to every superlayer pattern.")
    parser.add_argument("--sl_patterns_file", type=str, required=True, help="input file path: sl patterns (.root)")
    parser.add_argument("--sl_fits_file", type=str, required=True, help="output file path: sl fits (.root)")
    parser.add_argument("--fit_vd", action="store_true", help="fit the drift velocity as free parameter (default: fixed)")
    parser.add_argument("--chunk_size", type=int, default=2_000, help="number of patterns per chunk")
    parser.add_argument("--n_proc", type=int, default=1, help="number of chunks fitted at the same time")
    parser.add_argument("--verbose", action="store_true", help="print every fit (switches off --n_proc)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "sl patterns -> sl fits"
    root_utils.check_input_file(args.sl_patterns_file)
    log(f"[{label}] START \"{args.sl_patterns_file}\" -> \"{args.sl_fits_file}\" (fit_vd={args.fit_vd}, n_proc={args.n_proc})")
    n_proc = args.n_proc
    if args.verbose:
        n_proc = 1

    ### the chunks: (input file, first row, row after the last row, settings)
    n_patterns = root_utils.number_of_rows(args.sl_patterns_file, root_utils.DEFAULT_TREE)
    jobs = []
    for start in range(0, n_patterns, args.chunk_size):
        stop = min(start + args.chunk_size, n_patterns)
        jobs.append((args.sl_patterns_file, start, stop, args.fit_vd, args.verbose))

    ### fit the chunks: with n_proc > 1, n_proc chunks at the same time; the results come in the order of the chunks
    if n_proc > 1:
        pool = multiprocessing.Pool(n_proc)
        results = pool.imap(fit_chunk, jobs)
    else:
        results = map(fit_chunk, jobs)

    output_file = root_utils.create_file(args.sl_fits_file)
    i_chunk = 0
    for sl_fits, seconds in results:
        i_chunk += 1
        root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, sl_fits)
        log(f"    chunk {i_chunk:,} / {len(jobs):,}: {root_utils.length(sl_fits):,} patterns fitted ({seconds:.2f}s)")
    output_file.close()
    if n_proc > 1:
        pool.close()
        pool.join()
    log(f"[{label}] DONE. {n_patterns:,} patterns fitted")

if __name__ == "__main__":
    main()
    log("###### Done.")

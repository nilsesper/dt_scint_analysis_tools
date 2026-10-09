#################################################################
### dt hits -> sl patterns (4 hits in the 4 layers of one superlayer, close in time)
# The hits are read in chunks of --chunk_size hits. In every chunk the dead time cut is applied
# (params._dt_ts_individual_dead_time) and the patterns are searched (dt_pattern_utils.py). Patterns made of hits of
# two different chunks are not found. Every pattern gets the number of its chunk in the branch "chunk_id".
# With --n_proc N, N chunks are processed at the same time, each one in its own process (same result as with 1).
#################################################################

import argparse
import multiprocessing
import time
import numpy as np

from analysis_tools.utils import dt_hit_utils, dt_pattern_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

### one chunk of hits -> its patterns; runs in its own process if --n_proc > 1
def find_patterns_in_chunk(job):
    dt_hits_file, start, stop, chunk_id, apply_dead_time, wide_ts_window, only_single_muon_patterns, verbose = job
    t_start = time.perf_counter()
    hits = root_utils.read_tree(dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
    n_hits_in = root_utils.length(hits)
    if apply_dead_time:
        hits = dt_hit_utils.apply_dead_time(hits)
    n_hits = root_utils.length(hits)
    sl_patterns = dt_pattern_utils.find_sl_patterns(hits, wide_ts_window, only_single_muon_patterns, verbose)
    sl_patterns[root_utils.CHUNK_ID_KEY] = np.full(root_utils.length(sl_patterns), chunk_id, dtype=np.int64)
    return sl_patterns, n_hits_in, n_hits, time.perf_counter() - t_start

def main():
    parser = argparse.ArgumentParser(description="Find superlayer hit patterns in dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--sl_patterns_file", type=str, required=True, help="output file path: sl patterns (.root)")
    parser.add_argument("--ts_window", type=str, choices=["free_vd", "fixed_vd"], default="free_vd",
                        help="max time difference between the hits of a pattern: \"free_vd\" = wide window for fits with free "
                             "drift velocity (params._dt_sl_patterns_ts_window_fit_vd), \"fixed_vd\" = params._dt_sl_patterns_ts_window")
    parser.add_argument("--no_dead_time", action="store_true", help="do not apply the dead time cut")
    parser.add_argument("--simulation_only_muon_patterns", action="store_true",
                        help="simulation only: keep only patterns whose four hits come from the same simulated muon")
    parser.add_argument("--chunk_size", type=int, default=100_000, help="number of hits per chunk")
    parser.add_argument("--n_proc", type=int, default=1, help="number of chunks processed at the same time")
    parser.add_argument("--verbose", action="store_true", help="print every pattern found (switches off --n_proc)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dt hits -> sl patterns"
    root_utils.check_input_file(args.dt_hits_file)
    log(f"[{label}] START \"{args.dt_hits_file}\" -> \"{args.sl_patterns_file}\" (chunks of {args.chunk_size:,} hits, n_proc={args.n_proc})")
    n_proc = args.n_proc
    if args.verbose:
        n_proc = 1

    ### the chunks: (input file, first row, row after the last row, chunk_id, settings)
    n_hits = root_utils.number_of_rows(args.dt_hits_file, root_utils.DT_HITS_TREE)
    jobs = []
    for start in range(0, n_hits, args.chunk_size):
        stop = min(start + args.chunk_size, n_hits)
        chunk_id = len(jobs) + 1
        jobs.append((args.dt_hits_file, start, stop, chunk_id, not args.no_dead_time, args.ts_window == "free_vd",
                     args.simulation_only_muon_patterns, args.verbose))

    ### process the chunks: with n_proc > 1, n_proc chunks at the same time; the results come in the order of the chunks
    if n_proc > 1:
        pool = multiprocessing.Pool(n_proc)
        results = pool.imap(find_patterns_in_chunk, jobs)
    else:
        results = map(find_patterns_in_chunk, jobs)

    output_file = root_utils.create_file(args.sl_patterns_file)
    n_hits_in_total, n_hits_after_dead_time_total, n_patterns_total = 0, 0, 0
    i_chunk = 0
    for sl_patterns, n_hits_in, n_hits_after_dead_time, seconds in results:
        i_chunk += 1
        root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, sl_patterns)
        n_patterns = root_utils.length(sl_patterns)
        n_hits_in_total += n_hits_in
        n_hits_after_dead_time_total += n_hits_after_dead_time
        n_patterns_total += n_patterns
        log(f"    chunk {i_chunk:,} / {len(jobs):,}: {n_hits_in:,} hits in, {n_hits_after_dead_time:,} after dead time cut "
            f"({100 * n_hits_after_dead_time / max(1, n_hits_in):.1f}%) -> {n_patterns:,} patterns ({seconds:.2f}s)")
    output_file.close()
    if n_proc > 1:
        pool.close()
        pool.join()
    log(f"[{label}] DONE. {len(jobs):,} chunks, hits_in={n_hits_in_total:,}, "
        f"hits_after_deadtime={n_hits_after_dead_time_total:,}, patterns={n_patterns_total:,}")

if __name__ == "__main__":
    main()
    log("###### Done.")

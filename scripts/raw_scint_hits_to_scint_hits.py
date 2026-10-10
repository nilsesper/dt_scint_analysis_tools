#################################################################
### raw scint hits (single SiPMs) -> scint hits (strip hits: both SiPMs of a strip within a short time)
# 1. optional dead time of every SiPM channel (--dead_time, default params._raw_scintillator_ts_individual_dead_time)
# 2. the raw hits are grouped in time: a hit belongs to the group of the hit before it if they are closer than
#    params._raw_scint_hits_grouping_ts_tolerance
# 3. in every group and layer: a strip hit if both SiPMs of a strip are hit within params._scintillator_ts_acceptance_interval;
#    with the isolation criterion (default on, switch off with --no_isolation) only if this is the only hit strip of
#    the layer in the group and the layer has no other SiPM hit
# The raw hits are read in chunks of --chunk_size rows. The last group of a chunk may continue in the next chunk,
# therefore its hits are processed together with the next chunk (no edge effects).
#
# example:
#   python scripts/raw_scint_hits_to_scint_hits.py --raw_scint_hits_file out/run_raw_scint_hits.root --scint_hits_file out/run_scint_hits.root
#################################################################

import argparse
import time
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import root_utils, scint_hit_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

### rows (list of indices) of a table
def select_rows(table, indices):
    selected = {}
    for key in table:
        selected[key] = table[key][indices]
    return selected

### table_1 followed by table_2
def join_tables(table_1, table_2):
    joined = {}
    for key in table_1:
        joined[key] = np.concatenate([table_1[key], table_2[key]])
    return joined

def main():
    parser = argparse.ArgumentParser(description="Combine the hits of the two SiPMs of a strip to scint hits.")
    parser.add_argument("--raw_scint_hits_file", type=str, required=True, help="input file path: raw scint hits (.root)")
    parser.add_argument("--scint_hits_file", type=str, required=True, help="output file path: scint hits (.root)")
    parser.add_argument("--dead_time", type=float, default=params._raw_scintillator_ts_individual_dead_time,
                        help="dead time of every SiPM channel in timestamp units, 0: none "
                             "(default: params._raw_scintillator_ts_individual_dead_time)")
    parser.add_argument("--no_isolation", action="store_true",
                        help="also take strip hits from layers with more than one hit strip / more SiPM hits in the group")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of raw scint hits read at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "raw scint hits -> scint hits"
    root_utils.check_input_file(args.raw_scint_hits_file)
    isolation_criterion = not args.no_isolation
    log(f"[{label}] START \"{args.raw_scint_hits_file}\" -> \"{args.scint_hits_file}\" (dead time {args.dead_time} TU, "
        f"grouping {params._raw_scint_hits_grouping_ts_tolerance} TU, coincidence window {params._scintillator_ts_acceptance_interval} TU, "
        f"isolation criterion {'on' if isolation_criterion else 'off'})")

    t_start = time.perf_counter()
    n_rows = root_utils.number_of_rows(args.raw_scint_hits_file, root_utils.RAW_SCINT_HITS_TREE)
    n_chunks = (n_rows + args.chunk_size - 1) // args.chunk_size
    output_file = root_utils.create_file(args.scint_hits_file)
    last_ts_of_channel = {}  # dead time: last hit of every SiPM channel, passed on from chunk to chunk
    deferred_hits, deferred_rows = None, None  # hits of the last (maybe incomplete) group of the previous chunk
    n_after_dead_time, n_groups, n_scint_hits_total = 0, 0, 0
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_rows)
        raw_hits = root_utils.read_tree(args.raw_scint_hits_file, root_utils.RAW_SCINT_HITS_TREE, start, stop)
        raw_rows = np.arange(start, stop, dtype=np.int64)  # row numbers in the raw scint hits file

        if args.dead_time > 0:
            kept = scint_hit_utils.dead_time_kept_indices(raw_hits, args.dead_time, last_ts_of_channel)
            raw_hits = select_rows(raw_hits, kept)
            raw_rows = raw_rows[kept]
        n_after_dead_time += len(raw_rows)

        if deferred_hits is not None:
            raw_hits = join_tables(deferred_hits, raw_hits)
            raw_rows = np.concatenate([deferred_rows, raw_rows])
        order = np.argsort(raw_hits["ts"], kind="stable")
        raw_hits = select_rows(raw_hits, order)
        raw_rows = raw_rows[order]

        groups = scint_hit_utils.group_raw_scint_hits(raw_hits, params._raw_scint_hits_grouping_ts_tolerance)
        deferred_hits, deferred_rows = None, None
        if i_chunk < n_chunks - 1 and len(groups) > 0:
            last_group = groups.pop()
            deferred_hits = select_rows(raw_hits, last_group)
            deferred_rows = raw_rows[last_group]

        rows = []
        for group in groups:
            rows.extend(scint_hit_utils.strip_hits_of_group(group, raw_hits, raw_rows, isolation_criterion))
        scint_hits = scint_hit_utils.scint_hits_table(rows)
        root_utils.write_rows(output_file, root_utils.SCINT_HITS_TREE, scint_hits)
        n_groups += len(groups)
        n_scint_hits_total += len(rows)
        log(f"[{label}] chunk {i_chunk + 1:,} / {n_chunks:,}: {stop - start:,} raw scint hits, {len(groups):,} groups "
            f"-> {len(rows):,} scint hits")

    root_utils.write_summary(output_file, {
        "n_raw_scint_hits": n_rows, "n_after_dead_time": n_after_dead_time, "n_groups": n_groups,
        "n_scint_hits": n_scint_hits_total, "dead_time": args.dead_time, "isolation_criterion": int(isolation_criterion),
        "grouping_ts_tolerance": params._raw_scint_hits_grouping_ts_tolerance,
        "coincidence_window": params._scintillator_ts_acceptance_interval})
    output_file.close()
    log(f"[{label}] DONE. {n_rows:,} raw scint hits ({n_after_dead_time:,} after dead time), {n_groups:,} groups "
        f"-> {n_scint_hits_total:,} scint hits, took {time.perf_counter() - t_start:.1f}s")

if __name__ == "__main__":
    main()
    log("###### Done.")

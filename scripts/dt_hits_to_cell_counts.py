#################################################################
### dt hits -> number of hits per cell and duration of the run
# before any dead time cut. Output file: see OUTPUT_FILES.md.
#################################################################

import argparse
import numpy as np

from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils import dt_hit_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Count the hits per cell and determine the duration of the run.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--cell_counts_file", type=str, required=True, help="output file path: cell counts (.root)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of hits read at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dt hits -> cell counts"
    root_utils.check_input_file(args.dt_hits_file)
    log(f"[{label}] START \"{args.dt_hits_file}\" -> \"{args.cell_counts_file}\"")

    ### count the hits of every cell, and find the first and the last hit
    cell_counts = dt_chamber_utils.chamber_map(0)
    ts_min, ts_max = None, None
    n_hits = root_utils.number_of_rows(args.dt_hits_file, root_utils.DT_HITS_TREE)
    if n_hits == 0:
        raise RuntimeError(f"No dt hits in {args.dt_hits_file}.")
    n_chunks = (n_hits + args.chunk_size - 1) // args.chunk_size
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_hits)
        hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        dt_hit_utils.count_hits_per_cell(cell_counts, hits)
        if ts_min is None or np.amin(hits["ts"]) < ts_min:
            ts_min = np.amin(hits["ts"])
        if ts_max is None or np.amax(hits["ts"]) > ts_max:
            ts_max = np.amax(hits["ts"])
        log(f"    chunk {i_chunk + 1:,} / {n_chunks:,}: {root_utils.length(hits):,} hits counted")
    duration_seconds = float(ts_max - ts_min) * 0.78 * 1e-9

    ### output: one row per cell, the numbers of the whole run, and a 2d histogram (x = wire, y = 4 * (sl - 1) + ly)
    cells = dt_chamber_utils.chamber_cells()
    rows = {"sl": np.zeros(len(cells), dtype=np.uint8), "ly": np.zeros(len(cells), dtype=np.uint8),
            "wi": np.zeros(len(cells), dtype=np.uint8), "count": np.zeros(len(cells), dtype=np.int64)}
    for i in range(len(cells)):
        sl, ly, wi = cells[i]
        rows["sl"][i], rows["ly"][i], rows["wi"][i] = sl, ly, wi
        rows["count"][i] = cell_counts[sl][ly][wi]
    n_wires = int(np.amax(rows["wi"])) + 1
    counts_2d = np.zeros((n_wires, 12))
    for i in range(len(cells)):
        sl, ly, wi = cells[i]
        counts_2d[wi][4 * (sl - 1) + ly] = rows["count"][i]

    output_file = root_utils.create_file(args.cell_counts_file)
    root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, rows)
    root_utils.write_summary(output_file, {"duration_seconds": duration_seconds, "ts_min": np.float64(ts_min), "ts_max": np.float64(ts_max), "n_hits": n_hits})
    root_utils.write_histogram(output_file, "cell_counts", (counts_2d, np.arange(n_wires + 1) - 0.5, np.arange(13) - 0.5))
    output_file.close()
    log(f"[{label}] DONE. {n_hits:,} hits, duration {duration_seconds:.3f} s")

if __name__ == "__main__":
    main()
    log("###### Done.")

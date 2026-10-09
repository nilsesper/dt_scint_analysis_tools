#################################################################
### dt hits -> histogram of the time difference between consecutive hits of the same cell
# for every cell separately the hits are ordered in time and the differences between neighbours are filled;
# the histograms of all cells are summed. Uses the hits before any dead time cut.
# Differences across the border of two chunks of hits are not counted.
# default range: 0 .. 5000 timestamp units (3.9 us) in 5000 bins, larger differences are counted as overflow;
# change it with --ts_max and --n_bins. Output file: see OUTPUT_FILES.md.
#################################################################

import argparse
import numpy as np

from analysis_tools.utils import dt_hit_utils, hist_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Histogram the time difference between consecutive hits of the same cell.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--hit_diff_hist_file", type=str, required=True, help="output file path: histogram (.root)")
    parser.add_argument("--n_bins", type=int, default=5000, help="number of bins")
    parser.add_argument("--ts_max", type=float, default=5000, help="upper edge of the histogram in timestamp units (lower edge is 0)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of hits read at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dt hits -> hit diff hist"
    root_utils.check_input_file(args.dt_hits_file)
    log(f"[{label}] START \"{args.dt_hits_file}\" -> \"{args.hit_diff_hist_file}\"")
    edges = np.linspace(0, args.ts_max, args.n_bins + 1)
    counts = np.zeros(args.n_bins, dtype=np.int64)
    entries, underflow, overflow = 0, 0, 0

    n_hits = root_utils.number_of_rows(args.dt_hits_file, root_utils.DT_HITS_TREE)
    n_chunks = (n_hits + args.chunk_size - 1) // args.chunk_size
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_hits)
        hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        differences = np.array(dt_hit_utils.hit_time_differences(hits), dtype=np.float64)
        if len(differences) > 0:
            chunk_counts, _, _, chunk_entries, chunk_underflow, chunk_overflow = hist_utils.calculate_histogram(data=differences, edges=edges)
            counts += chunk_counts
            entries += int(chunk_entries)
            underflow += int(chunk_underflow)
            overflow += int(chunk_overflow)
        log(f"    chunk {i_chunk + 1:,} / {n_chunks:,}: {root_utils.length(hits):,} hits, {entries:,} entries in histogram so far")

    ### output: one row per bin, the numbers of the whole histogram, and the histogram as ROOT object
    bins = {
        "edge_low": edges[:-1], "edge_high": edges[1:], "center": (edges[:-1] + edges[1:]) / 2,
        "hist": counts.astype(np.float64),
        "err_hist": np.clip(np.sqrt(counts), 1, None),  # statistical uncertainty, at least 1
    }
    output_file = root_utils.create_file(args.hit_diff_hist_file)
    root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, bins)
    root_utils.write_summary(output_file, {"entries": entries, "underflow": underflow, "overflow": overflow, "n_hits": n_hits})
    root_utils.write_histogram(output_file, "hit_diff_hist", (counts, edges))
    output_file.close()
    log(f"[{label}] DONE. {n_hits:,} hits, {entries:,} entries in histogram")

if __name__ == "__main__":
    main()
    log("###### Done.")

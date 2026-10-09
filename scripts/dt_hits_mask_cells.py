#################################################################
### remove the hits of given cells from a dt hits file
# e.g. to give simulated hits the same dead cells as the real chamber
# default cells: the ones listed in params._dt_wire_mask and params._dt_dead_wires
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils import plot_utils, root_utils

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Remove the hits of given cells from dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--masked_dt_hits_file", type=str, required=True, help="output file path: dt hits without the given cells (.root)")
    parser.add_argument("--cells", type=str, default=None,
                        help="cells to remove as \"sl:ly:wi,sl:ly:wi,...\" (default: the cells in params._dt_wire_mask and params._dt_dead_wires)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of hits read at once")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    root_utils.check_input_file(args.dt_hits_file)
    if args.cells is not None:
        cells = plot_utils.parse_cells(args.cells)
    else:
        cells = sorted(dt_chamber_utils.excluded_cells())
    if len(cells) == 0:
        raise RuntimeError("No cells to remove: give --cells or fill params._dt_wire_mask / params._dt_dead_wires.")
    cells_to_remove = set()
    for sl, ly, wi in cells:
        cells_to_remove.add((int(sl), int(ly), int(wi)))

    n_in = root_utils.number_of_rows(args.dt_hits_file, root_utils.DT_HITS_TREE)
    n_chunks = (n_in + args.chunk_size - 1) // args.chunk_size
    n_out = 0
    output_file = root_utils.create_file(args.masked_dt_hits_file)
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_in)
        hits = root_utils.read_tree(args.dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        n_hits = root_utils.length(hits)
        keep = np.full(n_hits, True)
        for i in range(n_hits):
            if (int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i])) in cells_to_remove:
                keep[i] = False
        kept_hits = {}
        for key in hits:
            kept_hits[key] = hits[key][keep]
        n_kept = root_utils.length(kept_hits)
        n_out += n_kept
        root_utils.write_rows(output_file, root_utils.DT_HITS_TREE, kept_hits)
        log(f"    chunk {i_chunk + 1:,} / {n_chunks:,}: {n_kept:,} of {n_hits:,} hits kept")
    output_file.close()
    log(f"###### Removed the hits of {len(cells):,} cells: {n_out:,} / {n_in:,} hits kept, stored in {args.masked_dt_hits_file}")

if __name__ == "__main__":
    main()
    log("###### Done.")

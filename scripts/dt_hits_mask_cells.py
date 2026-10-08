#################################################################
### remove the hits of given cells from a dt hits file
# e.g. to give simulated hits the same dead cells as the real chamber
# default cells: the ones listed in params._dt_wire_mask and params._dt_dead_wires
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils, root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Remove the hits of given cells from dt hits.")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="input file path: dt hits (.root)")
    parser.add_argument("--masked_dt_hits_file", type=str, required=True, help="output file path: dt hits without the given cells (.root)")
    parser.add_argument("--cells", type=str, default=None,
                        help="cells to remove as \"sl:ly:wi,sl:ly:wi,...\" (default: the cells in params._dt_wire_mask and params._dt_dead_wires)")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    root_utils.check_input_file(args.dt_hits_file)
    if args.cells is not None:
        cells = plot_utils.parse_cells(args.cells)
    else:
        cells = plot_utils.masked_and_dead_cells()
    if len(cells) == 0:
        raise RuntimeError("No cells to remove: give --cells or fill params._dt_wire_mask / params._dt_dead_wires.")
    cells_to_remove = set()
    for sl, ly, wi in cells:
        cells_to_remove.add((int(sl), int(ly), int(wi)))

    n_in, n_out = 0, 0
    writer = root_utils.TreeWriter(args.masked_dt_hits_file, root_utils.DT_HITS_TREE)
    chunks = root_utils.chunk_ranges(args.dt_hits_file, root_utils.DT_HITS_TREE)
    for i_chunk in range(len(chunks)):
        start, stop = chunks[i_chunk]
        hits = root_utils.read_entries(args.dt_hits_file, root_utils.DT_HITS_TREE, start, stop)
        n_hits = root_utils.length(hits)
        keep = np.full(n_hits, True)
        for i in range(n_hits):
            if (int(hits["sl"][i]), int(hits["ly"][i]), int(hits["wi"][i])) in cells_to_remove:
                keep[i] = False
        kept_hits = {}
        for key in hits:
            kept_hits[key] = hits[key][keep]
        n_kept = root_utils.length(kept_hits)
        n_in += n_hits
        n_out += n_kept
        writer.write(kept_hits)
        log(f"    chunk {i_chunk + 1:,} / {len(chunks):,}: {n_kept:,} of {n_hits:,} hits kept")
    writer.close()
    log(f"###### Removed the hits of {len(cells):,} cells: {n_out:,} / {n_in:,} hits kept, stored in {args.masked_dt_hits_file}")

if __name__ == "__main__":
    main()
    log("###### Done.")

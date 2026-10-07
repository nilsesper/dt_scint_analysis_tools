#################################################################
### remove the hits of given cells from a dt hits file
# e.g. to give simulated hits the same dead cells as the real chamber
# default cells: the ones listed in params._dt_wire_mask and params._dt_dead_wires
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, dt_utils, muon_utils, plot_utils, root_utils
from analysis_tools.params import params, derived_params

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
    cells = plot_utils.parse_cells(args.cells) if args.cells is not None else plot_utils.masked_and_dead_cells()
    if len(cells) == 0:
        raise RuntimeError("No cells to remove: give --cells or fill params._dt_wire_mask / params._dt_dead_wires.")
    n_in, n_out = 0, 0
    with root_utils.TreeWriter(args.masked_dt_hits_file, root_utils.DT_HITS_TREE) as writer:
        n_chunks_total = root_utils.n_steps(args.dt_hits_file, root_utils.DT_HITS_TREE)
        for i_chunk, (_, chunk) in enumerate(root_utils.iterate_tree(args.dt_hits_file, root_utils.DT_HITS_TREE), start=1):
            mask = np.full(root_utils.length(chunk), True)
            for sl, ly, wi in cells:
                mask &= ~((chunk["sl"] == sl) & (chunk["ly"] == ly) & (chunk["wi"] == wi))
            n_in += len(mask)
            n_out += int(mask.sum())
            writer.write({k: v[mask] for k, v in chunk.items()})
            log(f"    chunk {i_chunk} / {n_chunks_total}: {int(mask.sum())} of {len(mask)} hits kept")
    log(f"###### Removed the hits of {len(cells)} cells: {n_out} / {n_in} hits kept, stored in {args.masked_dt_hits_file}")

if __name__ == "__main__":
    main()
    log("###### Done.")

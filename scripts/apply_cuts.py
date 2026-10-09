#################################################################
### apply cuts to any ROOT file of this workflow, keep the rows which pass all cuts
# example: --cuts "impossible,==,0;chi2/ndf,<,10;x0,>=,-21;x0,<=,21;dt0,>=,0;dt0,<=,params._dt_max_drift_time"
#################################################################

import argparse

from analysis_tools.utils import data_utils, root_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Keep only the rows of a ROOT file which pass all cuts.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path (.root)")
    parser.add_argument("--output_file", type=str, required=True, help="output file path: rows which pass the cuts (.root)")
    parser.add_argument("--cuts", type=str, required=True,
                        help="cuts in format \"key1,operator1,value1;key2,operator2,value2;...\", operators: == != > < >= <=, "
                             "value: a number or a parameter written as params._name")
    parser.add_argument("--tree", type=str, default=None, help="name of the tree (default: \"tree\", or \"dt_hits\" for dt hit files)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="number of rows read at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    root_utils.check_input_file(args.input_file)
    cuts = data_utils.parse_cuts(args.cuts)
    if len(cuts) == 0:
        raise ValueError("No cuts given.")
    tree = args.tree
    if tree is None:
        tree = root_utils.DEFAULT_TREE
        if tree not in root_utils.tree_names(args.input_file):
            tree = root_utils.DT_HITS_TREE
    branches = root_utils.branch_names(args.input_file, tree)
    for key, operator, value in cuts:
        if key not in branches:
            raise KeyError(f"Cut key \"{key}\" is not a branch of tree \"{tree}\" in {args.input_file}.")
    log(f"[apply cuts] START \"{args.input_file}\" -> \"{args.output_file}\" (tree \"{tree}\")")

    n_in = root_utils.number_of_rows(args.input_file, tree)
    n_after_cut = []  # number of rows which pass the first 1, 2, ... cuts
    for cut in cuts:
        n_after_cut.append(0)
    n_chunks = (n_in + args.chunk_size - 1) // args.chunk_size
    output_file = root_utils.create_file(args.output_file)
    for i_chunk in range(n_chunks):
        start = i_chunk * args.chunk_size
        stop = min(start + args.chunk_size, n_in)
        rows = root_utils.read_tree(args.input_file, tree, start, stop)
        n_rows = root_utils.length(rows)
        for i_cut in range(len(cuts)):
            rows = data_utils.cut_data(data=rows, conditions=[cuts[i_cut]], silent=True)
            n_after_cut[i_cut] += root_utils.length(rows)
        root_utils.write_rows(output_file, tree, rows)
        log(f"    chunk {i_chunk + 1:,} / {n_chunks:,}: {root_utils.length(rows):,} of {n_rows:,} rows pass the cuts")
    output_file.close()

    log(f"[apply cuts] cut flow w.r.t. the {n_in:,} input rows:")
    for i_cut in range(len(cuts)):
        key, operator, value = cuts[i_cut]
        n = n_after_cut[i_cut]
        log(f"    {key} {operator} {value}: {n:,} / {n_in:,} = {n / max(1, n_in):.4f}")
    if n_after_cut[-1] == 0:
        log("[apply cuts] WARNING: no rows pass the cuts")
    log("[apply cuts] DONE.")

if __name__ == "__main__":
    main()
    log("###### Done.")

#################################################################
### print the content of a ROOT file of this workflow: trees, number of rows, branches
# optionally print the first rows of selected branches
#################################################################

import argparse
import uproot

from analysis_tools.utils import root_utils

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Print trees, row counts and branches of a ROOT file.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path (.root)")
    parser.add_argument("--branches", type=str, default=None, help="comma separated branch names to print values of, e.g. \"sl,t0,chi2/ndf\"")
    parser.add_argument("--n_rows", type=int, default=10, help="number of rows to print for --branches")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    root_utils.check_input_file(args.input_file)
    with uproot.open(args.input_file) as f:
        for key in root_utils.tree_names(args.input_file):
            name = key.split(";")[0]
            tree = f[name]
            print(f"tree \"{name}\": {tree.num_entries:,} rows, {len(tree.keys()):,} branches")
            for branch in tree.keys():
                print(f"    {branch:40s} {tree[branch].typename}")
    if args.branches is not None:
        tree = root_utils.DEFAULT_TREE
        if tree not in root_utils.tree_names(args.input_file):
            tree = root_utils.DT_HITS_TREE
        n_rows = min(args.n_rows, root_utils.number_of_rows(args.input_file, tree))
        rows = root_utils.read_tree(args.input_file, tree, 0, n_rows)
        for key in args.branches.split(","):
            key = key.strip()
            if key not in rows:
                raise KeyError(f"Branch \"{key}\" not found.")
            print(f"{key} = {rows[key]}")

if __name__ == "__main__":
    main()

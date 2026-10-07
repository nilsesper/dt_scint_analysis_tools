#################################################################
### print the content of a ROOT file of this workflow: trees, number of rows, branches
# optionally print the first rows of selected branches
#################################################################

import argparse
import uproot

from analysis_tools.utils import root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Print trees, row counts and branches of a ROOT file.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path (.root)")
    parser.add_argument("--branches", type=str, default=None, help="comma separated branch names to print values of, e.g. \"sl,t0,chi2/ndf\"")
    parser.add_argument("--n_rows", type=int, default=10, help="number of rows to print for --branches")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    root_utils.check_input_file(args.input_file)
    with uproot.open(args.input_file) as f:
        tree_names = [k.split(";")[0] for k in f.keys(filter_classname="TTree")]
        for name in tree_names:
            tree = f[name]
            print(f"tree \"{name}\": {tree.num_entries} rows, {len(tree.keys())} branches")
            for key in tree.keys():
                print(f"    {key:40s} {tree[key].typename}")
    if args.branches is not None:
        wanted = [b.strip() for b in args.branches.split(",")]
        for _, chunk in root_utils.iterate_tree(args.input_file, step_size=max(1, args.n_rows)):
            for key in wanted:
                if key not in chunk:
                    raise KeyError(f"Branch \"{key}\" not found.")
                print(f"{key} = {chunk[key][:args.n_rows]}")
            break

if __name__ == "__main__":
    main()

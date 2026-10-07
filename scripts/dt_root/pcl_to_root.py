#################################################################
### convert a .pcl data file ({key: np.ndarray}) into a ROOT file of this workflow
# e.g. dt hits stored as .pcl -> dt hits ROOT file:
#   python scripts/dt_root/pcl_to_root.py --input_pcl_file sim_dt_hits.pcl --output_file sim_dt_hits.root --tree dt_hits
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, root_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a .pcl data file (dict of numpy arrays) into a ROOT file.")
    parser.add_argument("--input_pcl_file", type=str, required=True, help="input file path (.pcl)")
    parser.add_argument("--output_file", type=str, required=True, help="output file path (.root)")
    parser.add_argument("--tree", type=str, default=root_utils.DEFAULT_TREE,
                        help=f"name of the tree: \"{root_utils.DT_HITS_TREE}\" for dt hits, \"{root_utils.DEFAULT_TREE}\" for everything else")
    parser.add_argument("--sort_key", type=str, default=None, help="sort the rows by this key before writing (dt hits have to be sorted by \"ts\")")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    root_utils.check_input_file(args.input_pcl_file)
    data = data_utils.load_pickle(file=args.input_pcl_file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict of numpy arrays in {args.input_pcl_file}, found {type(data)}.")
    jagged = [k for k, v in data.items() if isinstance(v, list)]
    bad = [k for k, v in data.items() if not isinstance(v, (list, np.ndarray))]
    if len(bad) > 0:
        raise TypeError(f"Keys {bad} are neither numpy arrays nor lists, cannot be stored as branches.")
    if args.sort_key is not None:
        data = data_utils.sort_by_key(data=data, sort_key=args.sort_key)
    root_utils.write_tree(args.output_file, data, tree=args.tree, jagged=jagged)
    log(f"Wrote {root_utils.length(data)} rows with {len(data)} branches to tree \"{args.tree}\" in {args.output_file}")

if __name__ == "__main__":
    main()
    log("###### Done.")

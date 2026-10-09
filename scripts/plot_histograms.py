#################################################################
### histograms of the branches of any ROOT file of this workflow
# (dt hits, sl patterns, sl fits, super fits, dt muons)
#
# By default one histogram per "basic" branch is made: all number branches except
#   - the results of the not-selected lateralities ("lat0_...", "lat1_...", ...)
#   - bookkeeping branches (chunk_id, nidcs)
#   - branches which hold the same value in every row (e.g. the simulation truth branches in data)
# Use --branches to select branches by name, or --all_branches to get everything.
#
# examples:
#   python scripts/plot_histograms.py --input_file out/run_sl_fits.root --store_plots plots/sl_fits
#   python scripts/plot_histograms.py --input_file out/run_sl_fits.root --store_plots plots/sl_fits_good \
#          --cuts "impossible,==,0;chi2/ndf,<,10" --branches "t0,x0,tan_alpha,chi2/ndf" --split_by sl
#################################################################

import argparse
import os
import re
import sys
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import plot_utils
plot_utils.setup_backend(show_plots="--show_plots" in sys.argv)
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_tools.utils import data_utils, root_utils

# ---------------------------------------------------------------

BOOKKEEPING_BRANCHES = [root_utils.CHUNK_ID_KEY, "nidcs"]

### branch with the results of a not-selected laterality ("lat0_...", "lat1_...", ...)
def is_other_laterality_branch(key):
    return re.match(r"^lat\d+_", key) is not None

### the tree to read: the given one, else "tree" or "dt_hits", else the only tree besides "summary"
def find_tree(path, tree):
    names = root_utils.tree_names(path)
    if tree is not None:
        if tree not in names:
            raise KeyError(f"Tree \"{tree}\" not found in {path}. Available trees: {names}")
        return tree
    if root_utils.DEFAULT_TREE in names:
        return root_utils.DEFAULT_TREE
    if root_utils.DT_HITS_TREE in names:
        return root_utils.DT_HITS_TREE
    data_trees = []
    for name in names:
        if name != root_utils.SUMMARY_TREE:
            data_trees.append(name)
    if len(data_trees) == 1:
        return data_trees[0]
    raise KeyError(f"Cannot decide which tree to read in {path}. Available trees: {names}. Give it with --tree.")

### fixed-size array branches (e.g. ts_residual with 8 values per row): one histogram per element
# returns [(name, values)], for a normal branch only [(key, column)]
def split_array_branch(key, column):
    sub_columns = []
    if column.ndim == 1:
        sub_columns.append((key, column))
        return sub_columns
    n_elements = int(np.prod(column.shape[1:]))
    columns_2d = column.reshape(len(column), -1)
    for j in range(n_elements):
        sub_columns.append((f"{key}[{j}]", columns_2d[:, j]))
    return sub_columns

### one histogram line per value of the split branch, all in the same plot
def draw_split_histograms(ax, values, finite, edges, split_column, split_values, split_by, xlabel, log_scale):
    top = 0
    for value in split_values:
        selected = values[finite & (split_column == value)]
        hist, _ = np.histogram(selected, bins=edges)
        ax.stairs(hist, edges, linewidth=2, label=f"{plot_utils.key_label(split_by)} = {value:g} ({len(selected):,} rows)")
        if len(hist) > 0:
            top = max(top, hist.max())
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=max(top, 1) * np.exp(1.1))
    else:
        ax.set_ylim(bottom=0, top=max(top, 1) * 1.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.legend(prop={"size": 14}, fancybox=False)

@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 20})
def main():
    parser = argparse.ArgumentParser(description="Histogram the branches of a ROOT file of the dt workflow.")
    parser.add_argument("--input_file", type=str, required=True, help="input file path (.root)")
    parser.add_argument("--tree", type=str, default=None, help="name of the tree (default: found automatically)")
    parser.add_argument("--branches", type=str, default=None, help="comma separated branch names to plot (default: all basic branches)")
    parser.add_argument("--all_branches", action="store_true", help="also plot other-laterality, bookkeeping and constant branches")
    parser.add_argument("--cuts", type=str, default=None,
                        help="cuts applied before histogramming, format \"key1,operator1,value1;key2,operator2,value2;...\"")
    parser.add_argument("--split_by", type=str, default=None,
                        help="branch with few different values (e.g. sl): draw one histogram per value into the same plot")
    parser.add_argument("--n_bins", type=int, default=50, help="number of bins for branches which are not whole numbers")
    parser.add_argument("--full_range", action="store_true",
                        help="bin the full range of the data (default: the central 99%%, the rest is counted as underflow / overflow)")
    parser.add_argument("--log_scale", action="store_true", help="logarithmic y axis")
    parser.add_argument("--prefix", type=str, default=None, help="prefix of the plot file names (default: name of the input file)")
    plot_utils.add_plot_arguments(parser)
    args = parser.parse_args()
    plot_utils.check_plot_arguments(parser, args)

    root_utils.check_input_file(args.input_file)
    tree = find_tree(args.input_file, args.tree)
    prefix = args.prefix
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(args.input_file))[0]
    available = root_utils.branch_names(args.input_file, tree)
    n_rows_file = root_utils.number_of_rows(args.input_file, tree)
    log(f"###### {args.input_file}: tree \"{tree}\", {n_rows_file:,} rows, {len(available):,} branches")
    if n_rows_file == 0:
        raise RuntimeError("The tree is empty, nothing to plot.")

    ### which branches
    if args.branches is not None:
        keys = []
        for k in args.branches.split(","):
            if k.strip() != "":
                keys.append(k.strip())
        missing = []
        for k in keys:
            if k not in available:
                missing.append(k)
        if len(missing) > 0:
            raise KeyError(f"Branches {missing} not found. Available: {available}")
    elif args.all_branches:
        keys = list(available)
    else:
        keys = []
        for k in available:
            if k not in BOOKKEEPING_BRANCHES and not is_other_laterality_branch(k):
                keys.append(k)

    ### selection mask from cuts
    cuts = data_utils.parse_cuts(args.cuts)
    mask = np.full(n_rows_file, True)
    if len(cuts) > 0:
        cut_keys = []
        for c in cuts:
            if c[0] not in cut_keys:
                cut_keys.append(c[0])
        cut_data = root_utils.read_tree(args.input_file, tree, branches=sorted(cut_keys))
        cut_data["__row"] = np.arange(n_rows_file)
        selected_rows = data_utils.cut_data(data=cut_data, conditions=cuts, silent=True)["__row"]
        mask = np.isin(np.arange(n_rows_file), selected_rows)
        log(f"###### cuts {cuts}: {int(mask.sum()):,} / {n_rows_file:,} rows selected")
        if not mask.any():
            raise RuntimeError("No rows pass the cuts, nothing to plot.")

    ### split values
    split_values = [None]
    split_column = None
    if args.split_by is not None:
        split_column = root_utils.read_tree(args.input_file, tree, branches=[args.split_by])[args.split_by][mask]
        split_values = list(np.unique(split_column))
        if len(split_values) > 12:
            raise ValueError(f"--split_by {args.split_by}: {len(split_values)} different values, this is meant for branches with few values (e.g. sl).")

    ### histograms
    skipped = {"constant": [], "not numbers": [], "no finite values": []}
    n_plots = 0
    for key in keys:
        column = root_utils.read_tree(args.input_file, tree, branches=[key])[key]
        if column.dtype == object or column.dtype.kind not in "iufb":
            skipped["not numbers"].append(key)
            continue
        column = column[mask]
        sub_columns = split_array_branch(key, column)
        for name, values in sub_columns:
            values = values.astype(np.float64)
            finite = np.isfinite(values)
            n_not_finite = int(np.sum(~finite))
            if not finite.any():
                skipped["no finite values"].append(name)
                continue
            if np.amin(values[finite]) == np.amax(values[finite]) and not args.all_branches and args.branches is None:
                skipped["constant"].append(f"{name}={values[finite][0]:g}")
                continue
            edges = plot_utils.choose_edges(values[finite], n_bins=args.n_bins, full_range=args.full_range)
            xlabel = name
            if name == key:
                xlabel = plot_utils.key_label(key)
            title = ""
            if len(cuts) > 0:
                cut_texts = []
                for c in cuts:
                    cut_texts.append(f"{c[0]} {c[1]} {c[2]:g}")
                title = "cuts: " + ", ".join(cut_texts)
            if split_column is None:
                plot_utils.plot_histogram(values, f"{prefix}_{plot_utils.safe_name(name)}", args, xlabel=xlabel, title=title, edges=edges)
                log(f"hist: {name}: n_data={len(values):,}, not finite={n_not_finite:,}, n_bins={len(edges) - 1}")
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                draw_split_histograms(ax, values, finite, edges, split_column, split_values, args.split_by, xlabel, args.log_scale)
                log(f"hist: {name} split by {args.split_by}: n_data={len(values):,}, not finite={n_not_finite:,}, n_bins={len(edges) - 1}")
                ax.set_title(title, fontsize=12)
                fig.tight_layout()
                plot_utils.save_figure(fig, f"{prefix}_{plot_utils.safe_name(name)}_by_{plot_utils.safe_name(args.split_by)}", args)
            n_plots += 1

    log(f"###### {n_plots:,} histograms made")
    for reason in skipped:
        if len(skipped[reason]) > 0:
            log(f"skipped ({reason}): {', '.join(skipped[reason])}")
    plot_utils.show_figures(args)

if __name__ == "__main__":
    main()
    log("###### Done.")

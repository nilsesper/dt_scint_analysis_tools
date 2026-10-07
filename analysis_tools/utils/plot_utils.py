###########################################
### PLOTTING UTILS FOR THE ROOT FILE WORKFLOW
###########################################
# Shared by the plotting scripts in scripts/dt_root/: figure output, axis labels, generic histograms.
# The histogram calculation and drawing itself is done with hist_utils (same look as the older plotting scripts).

import os
import re
import numpy as np
import matplotlib as mpl

from analysis_tools.params import params
from analysis_tools.utils import hist_utils, root_utils

# -----------------------------------------

TS_UNIT_NS = 0.78  # ns per timestamp unit (TU)

### choose the matplotlib backend: no window is needed unless plots are shown
# has to be called before matplotlib.pyplot is imported
def setup_backend(show_plots):
    if not show_plots:
        mpl.use("Agg")

### file-name-safe version of a branch name ("chi2/ndf" -> "chi2ndf")
def safe_name(key):
    return re.sub(r"[^A-Za-z0-9_.\-]+", "", key.replace("/", ""))

### store a figure in the output directory (if one is given) and close it unless plots are shown
def finish_figure(fig, name, *, store_plots=None, show_plots=False, file_format="png", dpi=120):
    import matplotlib.pyplot as plt
    if store_plots is not None:
        os.makedirs(store_plots, exist_ok=True)
        path = os.path.join(store_plots, f"{name}.{file_format}")
        fig.savefig(path, dpi=dpi)
        root_utils.log(f"stored {path}")
    if not show_plots:
        plt.close(fig)

### show all open figures at the end of a script (only if plots are shown)
def show_figures(show_plots):
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()

### arguments shared by all plotting scripts
def add_plot_arguments(parser):
    parser.add_argument("--store_plots", type=str, default=None, help="output directory for the plots (created if needed)")
    parser.add_argument("--show_plots", action="store_true", help="show the plots in windows")
    parser.add_argument("--format", type=str, default="png", help="file format of the stored plots, e.g. png or pdf")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")

def check_plot_arguments(parser, args):
    if args.store_plots is None and not args.show_plots:
        parser.error("give --store_plots <directory> and/or --show_plots")

### axis label of a branch: symbol and unit from params._key_symbols / params._key_units if known, else the branch name
# suffixes of refits / super fits ("t0_refit") are recognised and written behind the symbol
def key_label(key, *, known_suffixes=("_super_fits", "_free_vd_super_fit", "_refit")):
    base, suffix_text = key, ""
    if key not in params._key_symbols:
        for suffix in known_suffixes:
            if key.endswith(suffix) and key[:-len(suffix)] in params._key_symbols:
                base, suffix_text = key[:-len(suffix)], " (" + suffix.strip("_").replace("_", " ") + ")"
                break
        else:
            m = re.match(r"^(.*)_sl(\d)$", key)  # super pattern branches like "t0_sl1"
            if m and m.group(1) in params._key_symbols:
                base, suffix_text = m.group(1), f" (SL {m.group(2)})"
    if base not in params._key_symbols:
        return key.replace("_", "\\_") if "$" in key else key
    label = params._key_symbols[base] + suffix_text
    unit = params._key_units.get(base, "")
    return label if unit == "" else f"{label} [{unit}]"

### parse a cell list "sl:ly:wi,sl:ly:wi,..." into [(sl, ly, wi)]
def parse_cells(cells_str):
    cells = []
    if cells_str is None or cells_str.strip() == "":
        return cells
    for item in cells_str.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Cannot read cell \"{item}\". Expected format: sl:ly:wi")
        cells.append(tuple(int(p) for p in parts))
    return cells

### cells listed in params._dt_wire_mask and params._dt_dead_wires as [(sl, ly, wi)]
def masked_and_dead_cells():
    cells = []
    for table in (getattr(params, "_dt_wire_mask", {}), getattr(params, "_dt_dead_wires", {})):
        for sl, lys in table.items():
            for ly, wis in lys.items():
                cells.extend((sl, ly, wi) for wi in wis)
    return sorted(set(cells))

### bin edges for one data array
# - data which only holds whole numbers over a small range gets one bin per value
# - everything else gets n_bins equal bins over the full range, or (full_range=False) over the central part of the
#   distribution given by range_percentiles; values outside are counted as underflow / overflow
def choose_edges(data, *, n_bins=50, full_range=False, range_percentiles=(0.5, 99.5), max_integer_bins=300):
    lo, hi = float(np.amin(data)), float(np.amax(data))
    whole_numbers = data.dtype.kind in "iub" or bool(np.all(data == np.round(data)))
    if whole_numbers and (hi - lo) <= max_integer_bins:
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    if not full_range:
        p_lo, p_hi = np.percentile(data, range_percentiles)
        if p_hi > p_lo:
            lo, hi = float(p_lo), float(p_hi)
    if hi == lo:
        lo, hi = lo - 0.5, hi + 0.5
    # widen by 1% on both sides, so that values sitting exactly on the range limits are inside the histogram
    pad = 0.01 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)

### calculate and draw one histogram with statistical uncertainty and info box into ax
# returns (hist, edges, entries, underflow, overflow)
def draw_histogram(ax, data, edges, *, xlabel="", log_scale=False, info_loc="top right", bin_unit=None, scale=1.0):
    centers = hist_utils.centers_from_edges(edges)
    hist, _, _, entries, underflow, overflow, hist_err_right, hist_err_left = \
        hist_utils.calculate_histogram_and_shifted_histograms(data=data, edges=edges)
    err_hist, _, _ = hist_utils.calculate_hist_uncertainty(hist=hist, hist_err_right=hist_err_right, hist_err_left=hist_err_left, do_stat_err=True)
    err_hist = np.where(hist > 0, err_hist, 0)  # no error bar on empty bins
    hist_utils.plot_histogram(
        ax=ax, hist=hist, centers=centers * scale, err_hist=err_hist, log_scale=log_scale, add_info=True,
        entries=int(entries), overflow=int(overflow), underflow=int(underflow), bin_unit=bin_unit, info_loc=info_loc, power_limits=[-3, 4],
    )
    ax.set_xlabel(xlabel)
    return hist, edges, int(entries), int(underflow), int(overflow)

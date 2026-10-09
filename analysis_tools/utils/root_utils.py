###########################################
### ROOT FILES AND LOG OUTPUT
###########################################
# All scripts exchange their data as ROOT files (read and written with the python package uproot).
# In python, a table of data is a dict {branch name: numpy array}, one array per branch, all of the same length
# ("rows"). In the ROOT file this table is a tree with one branch per key. The names of the trees and branches of
# every output file are listed in OUTPUT_FILES.md.
#
# Reading:  rows = root_utils.read_tree(path, "tree")                  (all rows)
#           rows = root_utils.read_tree(path, "tree", start, stop)     (only rows start ... stop - 1)
# Writing:  output_file = root_utils.create_file(path)
#           root_utils.write_rows(output_file, "tree", rows)           (call again to append more rows)
#           output_file.close()

import os
import re
import sys
import time
import numpy as np
import uproot

# -----------------------------------------

DT_HITS_TREE = "dt_hits"    # tree of the dt hit files
DEFAULT_TREE = "tree"       # tree of all other files
SUMMARY_TREE = "summary"    # one row with numbers of the whole run (some files)
CHUNK_ID_KEY = "chunk_id"   # branch with the number of the dt hit chunk a row comes from (see scripts/README.md)

START_TIME = time.perf_counter()

# -----------------------------------------
# log output
# -----------------------------------------

### print a message with the time since the start of the script; flushed, so that it shows up immediately in batch logs
# colours (only in a terminal): time in cyan, the step name in square brackets at the start of the message in yellow,
# progress counters like "chunk 3 / 10" in magenta. DT_SCINT_COLOR=1 / 0 in the environment forces them on / off.
def log(message):
    message = str(message)
    prefix = f"[{time.perf_counter() - START_TIME:9.1f}s]"
    if use_color():
        prefix = "\033[36m" + prefix + "\033[0m"
        step = re.match(r"^(\s*)(\[[^\]]+\])", message)
        if step is not None:
            message = step.group(1) + "\033[33m" + step.group(2) + "\033[0m" + message[step.end():]
        for counter in set(re.findall(r"\b(?:chunk|block) [\d,]+ ?/ ?[\d,]+", message)):
            message = message.replace(counter, "\033[35m" + counter + "\033[0m")
    print(f"{prefix} {message}", flush=True)

def use_color():
    setting = os.environ.get("DT_SCINT_COLOR", "").strip().lower()
    if setting in ["1", "true", "yes", "on"]:
        return True
    if setting in ["0", "false", "no", "off"]:
        return False
    return sys.stdout.isatty()

# -----------------------------------------
# reading
# -----------------------------------------

def check_input_file(path):
    if path is None:
        raise ValueError("Missing input file argument.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

### number of rows of a table (all arrays have the same length)
def length(rows):
    if rows is None or len(rows) == 0:
        return 0
    first_key = list(rows.keys())[0]
    return len(rows[first_key])

def number_of_rows(path, tree):
    with uproot.open(path) as f:
        return f[tree].num_entries

def tree_names(path):
    names = []
    with uproot.open(path) as f:
        classnames = f.classnames()
        for key in classnames:
            if classnames[key].startswith("TTree"):
                names.append(key.split(";")[0])
    return names

def branch_names(path, tree):
    with uproot.open(path) as f:
        return list(f[tree].keys())

### read the rows start ... stop - 1 of a tree (default: all rows) as {branch name: numpy array}
# branches: read only these branches (default: all)
def read_tree(path, tree, start=None, stop=None, branches=None):
    with uproot.open(path) as f:
        if branches is not None:
            for branch in branches:
                if branch not in f[tree].keys():
                    raise KeyError(f"Branch \"{branch}\" not found in tree \"{tree}\" of {path}.")
        rows = f[tree].arrays(branches, entry_start=start, entry_stop=stop, library="np")
    # uproot can return arrays with the byte order of the file; convert them to normal numpy arrays
    native_rows = {}
    for key in rows:
        array = rows[key]
        if not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
        native_rows[key] = array
    return native_rows

### the rows of every chunk_id: list of (chunk_id, first row, row after the last row)
# the rows of one chunk_id are next to each other in the file
def rows_of_each_chunk_id(path, tree=DEFAULT_TREE):
    chunk_ids = read_tree(path, tree, branches=[CHUNK_ID_KEY])[CHUNK_ID_KEY]
    chunks = []
    start = 0
    for row in range(1, len(chunk_ids) + 1):
        if row == len(chunk_ids) or chunk_ids[row] != chunk_ids[start]:
            chunks.append((int(chunk_ids[start]), start, row))
            start = row
    return chunks

# -----------------------------------------
# writing
# -----------------------------------------

### new (empty) ROOT file, the directory is created if needed
def create_file(path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return uproot.recreate(path)

### write rows into a tree: the first call creates the tree, every further call appends the rows at the end
def write_rows(output_file, tree, rows):
    if tree not in output_file:
        # create the (empty) tree: type of every branch from its array
        branch_types = {}
        for key in rows:
            array = rows[key]
            if not isinstance(array, np.ndarray):
                branch_types[key] = "var * int64"         # variable-length lists (awkward array, see pcl_to_root.py)
            elif array.ndim == 1:
                branch_types[key] = array.dtype
            else:
                branch_types[key] = (array.dtype, array.shape[1:])  # fixed-size array per row, e.g. 8 residuals
        output_file.mktree(tree, branch_types)
    if length(rows) > 0:
        output_file[tree].extend(rows)

### tree "summary" with one row: numbers of the whole run, e.g. {"n_hits": 1234, "duration_seconds": 15.4}
def write_summary(output_file, summary):
    rows = {}
    for key in summary:
        rows[key] = np.array([summary[key]])
    write_rows(output_file, SUMMARY_TREE, rows)

### a histogram object which ROOT can draw directly: TH1D from (contents, edges), TH2D from (contents[x][y], x edges, y edges)
def write_histogram(output_file, name, histogram):
    parts = []
    for part in histogram:
        parts.append(np.asarray(part, dtype=np.float64))
    output_file[name] = tuple(parts)

### one complete table as a new file
def write_file(path, rows, tree=DEFAULT_TREE):
    output_file = create_file(path)
    write_rows(output_file, tree, rows)
    output_file.close()

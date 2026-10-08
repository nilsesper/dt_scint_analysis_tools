###########################################
### ROOT FILES (uproot) AND LOG OUTPUT
###########################################
# All workflow stages exchange data as ROOT files. A tree holds one data dict {key: np.ndarray}, one branch per key.
#   - dt hits are stored in a tree called "dt_hits", everything else in a tree called "tree"
#   - branch names are the dict keys, unchanged (also "chi2/ndf")
#
# Large files are read in chunks of rows: chunk_ranges() gives the (start, stop) rows of every chunk,
# read_entries() reads the rows of one chunk.
# Stages which combine rows (super fits, muons) must only combine rows from the same chunk of dt hits. The pattern
# stage therefore writes the number of its dt hit chunk into the branch "chunk_id", and all later files keep it;
# chunk_id_ranges() gives the rows of every chunk_id.

import os
import re
import sys
import time
import numpy as np
import uproot

# -----------------------------------------

DT_HITS_TREE = "dt_hits"
DEFAULT_TREE = "tree"
SUMMARY_TREE = "summary"
CHUNK_ID_KEY = "chunk_id"
DEFAULT_STEP_SIZE = "500 MB"

START_TIME = time.perf_counter()

# -----------------------------------------
# log output
# -----------------------------------------

# colours of the log lines: time in cyan, step description (text in square brackets at the start of a message,
# e.g. "[dt hits -> cell counts]") in yellow, progress counters ("chunk 3 / 10", "block 3 / 10") in magenta.
# Colours are used when the output goes to a terminal. Set the environment variable DT_SCINT_COLOR=1 to force
# them (e.g. for batch logs which are viewed with a colour-aware tool) or DT_SCINT_COLOR=0 to switch them off.
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def use_color():
    setting = os.environ.get("DT_SCINT_COLOR", "").strip().lower()
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    return sys.stdout.isatty()

def add_colors(msg):
    # step description: text in square brackets at the start
    step = re.match(r"^(\s*)(\[[^\]]+\])", msg)
    if step is not None:
        msg = step.group(1) + YELLOW + step.group(2) + RESET + msg[step.end():]
    # progress counters
    for counter in set(re.findall(r"\b(?:chunk|block) [\d,]+ ?/ ?[\d,]+", msg)):
        msg = msg.replace(counter, MAGENTA + counter + RESET)
    return msg

### print with the time since the start of the script, always flushed (so it shows up immediately in batch logs)
def log(msg):
    msg = str(msg)
    prefix = f"[{time.perf_counter() - START_TIME:9.1f}s]"
    if use_color():
        prefix = CYAN + prefix + RESET
        msg = add_colors(msg)
    print(f"{prefix} {msg}", flush=True)

# -----------------------------------------
# files
# -----------------------------------------

### number of rows of a data dict
def length(data):
    if data is None or len(data) == 0:
        return 0
    first_key = list(data.keys())[0]
    return len(data[first_key])

### make sure the directory of an output file exists
def prepare_output_file(path):
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

### fail early with a clear message if an input file is missing
def check_input_file(path):
    if path is None:
        raise ValueError("Missing input file argument.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

### command line value of --step_size: a number of rows ("50000") or a memory size ("200 MB")
def parse_step_size(value):
    if value.strip().isdigit():
        return int(value)
    return value

### pick the tree to read: the given name, else "tree", else "dt_hits", else the only tree in the file
def resolve_tree_name(path, tree=None):
    with uproot.open(path) as f:
        names = []
        for key in f.keys(filter_classname="TTree"):
            names.append(key.split(";")[0])
    if tree is not None:
        if tree not in names:
            raise KeyError(f"Tree \"{tree}\" not found in {path}. Available trees: {names}")
        return tree
    if DEFAULT_TREE in names:
        return DEFAULT_TREE
    if DT_HITS_TREE in names:
        return DT_HITS_TREE
    data_trees = []
    for name in names:
        if name != SUMMARY_TREE:
            data_trees.append(name)
    if len(data_trees) == 1:
        return data_trees[0]
    raise KeyError(f"Cannot decide which tree to read in {path}. Available trees: {names}. Pass the tree name explicitly.")

### number of rows of a tree
def n_entries(path, tree=None):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        return f[tree].num_entries

### uproot can return arrays with non-native byte order: convert them to plain numpy arrays
def to_native_arrays(data):
    native = {}
    for key in data:
        array = data[key]
        if array.dtype != object and not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
        native[key] = array
    return native

### read a complete tree into memory as {key: np.ndarray}
def read_tree(path, tree=None):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        return to_native_arrays(f[tree].arrays(library="np"))

### read the rows start ... stop - 1 of a tree as {key: np.ndarray}
def read_entries(path, tree, start, stop):
    with uproot.open(path) as f:
        return to_native_arrays(f[tree].arrays(entry_start=start, entry_stop=stop, library="np"))

### split a tree into chunks: list of (first row, row after the last row) of every chunk
# step_size: number of rows (int) or memory size (str, e.g. "200 MB")
def chunk_ranges(path, tree, step_size=DEFAULT_STEP_SIZE):
    with uproot.open(path) as f:
        n_rows = f[tree].num_entries
        if isinstance(step_size, int):
            rows_per_chunk = step_size
        else:
            rows_per_chunk = f[tree].num_entries_for(step_size)
    rows_per_chunk = max(1, rows_per_chunk)
    ranges = []
    start = 0
    while start < n_rows:
        stop = min(start + rows_per_chunk, n_rows)
        ranges.append((start, stop))
        start = stop
    return ranges

### rows of every chunk_id: list of (chunk_id, first row, row after the last row)
# the rows of a chunk_id are next to each other in the file; a tree without chunk_id branch is one block with chunk_id 0
def chunk_id_ranges(path, tree=DEFAULT_TREE):
    with uproot.open(path) as f:
        n_rows = f[tree].num_entries
        if n_rows == 0:
            return []
        if CHUNK_ID_KEY not in f[tree].keys():
            return [(0, 0, n_rows)]
        chunk_ids = f[tree][CHUNK_ID_KEY].array(library="np")
    ranges = []
    start = 0
    for row in range(1, n_rows + 1):
        if row == n_rows or chunk_ids[row] != chunk_ids[start]:
            ranges.append((int(chunk_ids[start]), start, row))
            start = row
    return ranges

### add (or overwrite) the chunk_id column of a data dict
def set_chunk_id(data, chunk_id):
    data[CHUNK_ID_KEY] = np.full(length(data), chunk_id, dtype=np.int64)
    return data

# -----------------------------------------
# writing
# -----------------------------------------

### writer which creates the tree on the first write and extends it afterwards; call close() at the end
# supports 1d arrays, fixed-size nd arrays and (via jagged=[keys]) variable-length lists
# use:
#   writer = TreeWriter(path, tree)
#   writer.write(data)   (as often as needed)
#   writer.close()
class TreeWriter:
    def __init__(self, path, tree=DEFAULT_TREE):
        self.path = path
        self.tree = tree
        self.file = None
        self.n_written = 0
        self.jagged_keys = []
        prepare_output_file(path)

    def write(self, data, *, jagged=()):
        if length(data) == 0:
            return
        if self.file is None:
            self.jagged_keys = list(jagged)
            self.file = uproot.recreate(self.path)
            branch_types = {}
            for key in data:
                if key in self.jagged_keys:
                    branch_types[key] = "var * int64"
                elif data[key].ndim == 1:
                    branch_types[key] = data[key].dtype
                else:
                    branch_types[key] = (data[key].dtype, data[key].shape[1:])
            self.file.mktree(self.tree, branch_types)
            log(f"    created tree \"{self.tree}\" in {self.path}")
        if len(self.jagged_keys) > 0:
            import awkward as ak
            data = dict(data)
            for key in self.jagged_keys:
                rows = []
                for row in data[key]:
                    rows.append(np.asarray(row, dtype=np.int64))
                data[key] = ak.Array(rows)
        self.file[self.tree].extend(data)
        self.n_written += length(data)

    ### write a small one-row tree with run-level numbers next to the data tree
    def write_summary(self, summary, tree=SUMMARY_TREE):
        if self.file is None:
            self.file = uproot.recreate(self.path)
        data = {}
        branch_types = {}
        for key in summary:
            data[key] = np.array([summary[key]])
            branch_types[key] = data[key].dtype
        self.file.mktree(tree, branch_types)
        self.file[tree].extend(data)

    ### store a histogram object next to the trees, so that it can be drawn directly in ROOT (TBrowser, ->Draw())
    # histogram: (bin contents, bin edges) for a TH1D, (bin contents[x, y], x edges, y edges) for a TH2D
    def write_histogram(self, name, histogram):
        if self.file is None:
            self.file = uproot.recreate(self.path)
        parts = []
        for part in histogram:
            parts.append(np.asarray(part, dtype=np.float64))
        self.file[name] = tuple(parts)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

### write a complete data dict as a new ROOT file
# histograms: optional {name: (contents, edges) or (contents, x edges, y edges)}, stored as TH1D / TH2D objects
def write_tree(path, data, tree=DEFAULT_TREE, *, summary=None, jagged=(), histograms=None):
    writer = TreeWriter(path, tree)
    writer.write(data, jagged=jagged)
    if summary is not None:
        writer.write_summary(summary)
    if histograms is not None:
        for name in histograms:
            writer.write_histogram(name, histograms[name])
    writer.close()

# -----------------------------------------
# selective reading (used by the plotting scripts)
# -----------------------------------------

### names of all branches of a tree
def list_branches(path, tree=None):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        return list(f[tree].keys())

### read only the given branches of a tree into memory, returns {key: np.ndarray}
# branches are addressed by their exact name (works for names like "chi2/ndf")
def read_branches(path, keys, tree=None):
    tree = resolve_tree_name(path, tree)
    data = {}
    with uproot.open(path) as f:
        available = f[tree].keys()
        for key in keys:
            if key not in available:
                raise KeyError(f"Branch \"{key}\" not found in tree \"{tree}\" of {path}.")
            data[key] = f[tree][key].array(library="np")
    return to_native_arrays(data)

### read single rows (row numbers) of a tree, returns {key: np.ndarray} with one element per requested row
def read_rows(path, rows, tree=None):
    tree = resolve_tree_name(path, tree)
    n_rows = n_entries(path, tree)
    pieces = []
    for row in rows:
        row = int(row)
        if row < 0 or row >= n_rows:
            raise IndexError(f"Row {row} is outside of tree \"{tree}\" of {path}, which has {n_rows:,} rows.")
        pieces.append(read_entries(path, tree, row, row + 1))
    if len(pieces) == 0:
        return {}
    data = {}
    for key in pieces[0]:
        values = []
        for piece in pieces:
            values.append(piece[key])
        data[key] = np.concatenate(values)
    return data

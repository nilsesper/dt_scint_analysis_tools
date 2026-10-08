###########################################
### ROOT FILE I/O UTILS (uproot based)
###########################################
# All workflow stages exchange data as ROOT files holding one TTree whose branches
# are the keys of the usual data dicts ({key: np.ndarray}).
#
# Conventions (kept from the first ROOT pipeline, so files stay readable by the
# existing analysis scripts):
#   - dt hits are stored in a tree called "dt_hits"
#   - every other object (sl patterns, sl fits, super fits, ...) in a tree called "tree"
#   - branch names are the dict keys, unchanged (also "chi2/ndf"). Because of names like
#     "chi2/ndf", never pass branch names as uproot *expressions*; always read whole
#     trees (as done here) and pick the keys from the returned dict.
#
# Streaming and the "chunk_id" branch:
#   Large files are processed in chunks. Row-wise stages (fits, cuts) do not care where a
#   chunk starts or ends. Stages which combine rows (pattern finding, super patterns, fit
#   groups, muons) do: they only combine rows which are in the same chunk. To keep these
#   stages independent of how a later script happens to read the file, the pattern stage
#   stamps every row with the "chunk_id" of the dt-hit chunk it came from. All later
#   stages carry this branch along and combine rows per chunk_id (see iterate_blocks).

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
DEFAULT_STEP_SIZE = "200 MB"

_START = time.perf_counter()

# colours of the log lines: time in cyan, step description (text in square brackets at the start of a message,
# e.g. "[dt hits -> cell counts]") in yellow, progress counters ("chunk 3 / 10", "block 3 / 10") in magenta.
# Colours are used when the output goes to a terminal. Set the environment variable DT_SCINT_COLOR=1 to force
# them (e.g. for batch logs which are viewed with a colour-aware tool) or DT_SCINT_COLOR=0 to switch them off.
_CYAN, _YELLOW, _MAGENTA, _RESET = "\033[36m", "\033[33m", "\033[35m", "\033[0m"
_STEP_PATTERN = re.compile(r"^(\s*)(\[[^\]]+\])")
_PROGRESS_PATTERN = re.compile(r"\b(?:chunk|block) [\d,]+ ?/ ?[\d,]+")

def _use_color():
    setting = os.environ.get("DT_SCINT_COLOR", "").strip().lower()
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    return sys.stdout.isatty()

### print with elapsed time prefix, always flushed (so it shows up immediately in batch logs)
def log(msg):
    msg = str(msg)
    prefix = f"[{time.perf_counter() - _START:9.1f}s]"
    if _use_color():
        prefix = f"{_CYAN}{prefix}{_RESET}"
        msg = _STEP_PATTERN.sub(lambda m: f"{m.group(1)}{_YELLOW}{m.group(2)}{_RESET}", msg, count=1)
        msg = _PROGRESS_PATTERN.sub(lambda m: f"{_MAGENTA}{m.group(0)}{_RESET}", msg)
    print(f"{prefix} {msg}", flush=True)

### number of rows of a data dict
def length(data):
    if data is None or len(data) == 0:
        return 0
    return len(next(iter(data.values())))

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

### pick the tree to read: the given name, else "tree", else "dt_hits", else the only tree in the file
def resolve_tree_name(path, tree=None):
    with uproot.open(path) as f:
        names = [k.split(";")[0] for k in f.keys(filter_classname="TTree")]
    if tree is not None:
        if tree not in names:
            raise KeyError(f"Tree \"{tree}\" not found in {path}. Available trees: {names}")
        return tree
    for candidate in (DEFAULT_TREE, DT_HITS_TREE):  # (histogram objects in the file are not trees and are ignored here)
        if candidate in names:
            return candidate
    data_trees = [n for n in names if n != SUMMARY_TREE]
    if len(data_trees) == 1:
        return data_trees[0]
    raise KeyError(f"Cannot decide which tree to read in {path}. Available trees: {names}. Pass the tree name explicitly.")

### number of entries of a tree
def n_entries(path, tree=None):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        return f[tree].num_entries

### number of chunks iterate_tree will yield for this tree and step_size (for progress messages)
def n_steps(path, tree=None, *, step_size=DEFAULT_STEP_SIZE):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        t = f[tree]
        n = t.num_entries
        entries_per_step = step_size if isinstance(step_size, int) else t.num_entries_for(step_size)
    return int(np.ceil(n / max(1, entries_per_step)))

### number of blocks iterate_blocks will yield (= number of different chunk_id values, for progress messages)
def n_blocks(path, tree=None, *, block_key=CHUNK_ID_KEY):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        t = f[tree]
        if t.num_entries == 0:
            return 0
        if block_key not in t.keys():
            return 1
        ids = t[block_key].array(library="np")
    return int(1 + np.count_nonzero(ids[1:] != ids[:-1]))

### turn the arrays returned by uproot into plain native numpy arrays
def _clean(chunk):
    out = {}
    for k, v in chunk.items():
        if v.dtype != object and not v.dtype.isnative:
            v = v.astype(v.dtype.newbyteorder("="))
        out[k] = v
    return out

### read a complete tree into memory as {key: np.ndarray}
def read_tree(path, tree=None):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        return _clean(f[tree].arrays(library="np"))

### iterate over a tree in chunks, yields (entry_start, {key: np.ndarray})
# step_size: number of entries (int) or memory size (str, e.g. "200 MB")
def iterate_tree(path, tree=None, *, step_size=DEFAULT_STEP_SIZE):
    tree = resolve_tree_name(path, tree)
    entry_start = 0
    for chunk in uproot.iterate(f"{path}:{tree}", step_size=step_size, library="np"):
        chunk = _clean(chunk)
        n = length(chunk)
        yield entry_start, chunk
        entry_start += n

### iterate_tree with progress numbers: yields (chunk number starting at 1, number of chunks, {key: np.ndarray})
def iterate_chunks(path, tree=None, *, step_size=DEFAULT_STEP_SIZE):
    n_chunks = n_steps(path, tree, step_size=step_size)
    for i_chunk, (_, chunk) in enumerate(iterate_tree(path, tree, step_size=step_size), start=1):
        yield i_chunk, n_chunks, chunk

### command line value of --step_size: a number of rows ("50000") or a memory size ("200 MB")
def parse_step_size(value):
    return int(value) if value.strip().isdigit() else value

### iterate over a tree in blocks of equal "chunk_id", yields (chunk_id, entry_start, {key: np.ndarray})
# Rows with the same chunk_id are contiguous in the file. A block is yielded as a whole,
# no matter where the read steps of this function fall.
# If the tree has no chunk_id branch, the whole tree is one block with chunk_id 0.
def iterate_blocks(path, tree=None, *, block_key=CHUNK_ID_KEY, step_size=DEFAULT_STEP_SIZE):
    tree = resolve_tree_name(path, tree)
    with uproot.open(path) as f:
        has_block_key = block_key in f[tree].keys()
    if not has_block_key:
        data = read_tree(path, tree)
        if length(data) > 0:
            yield 0, 0, data
        return
    pending = []  # list of dict pieces which belong to the current block
    pending_id = None
    pending_start = 0
    for entry_start, chunk in iterate_tree(path, tree, step_size=step_size):
        ids = chunk[block_key]
        n = len(ids)
        if n == 0:
            continue
        bounds = np.concatenate(([0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [n]))
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], bounds[b + 1]
            this_id = ids[lo]
            if pending_id is not None and this_id != pending_id:
                yield int(pending_id), pending_start, _concatenate(pending)
                pending = []
            if len(pending) == 0:
                pending_id = this_id
                pending_start = entry_start + lo
            pending.append({k: v[lo:hi] for k, v in chunk.items()})
    if len(pending) > 0:
        yield int(pending_id), pending_start, _concatenate(pending)

def _concatenate(pieces):
    if len(pieces) == 1:
        return pieces[0]
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0].keys()}

### add (or overwrite) the chunk_id column of a data dict
def set_chunk_id(data, chunk_id):
    data[CHUNK_ID_KEY] = np.full(length(data), chunk_id, dtype=np.int64)
    return data

### writer which creates the tree on first write and extends it afterwards
# supports 1d arrays, fixed-size nd arrays and (via jagged=[keys]) variable-length lists
class TreeWriter:
    def __init__(self, path, tree=DEFAULT_TREE):
        self.path = path
        self.tree = tree
        self.file = None
        self.n_written = 0
        self._jagged = []
        prepare_output_file(path)

    def _to_writable(self, data):
        if len(self._jagged) == 0:
            return data
        import awkward as ak
        out = dict(data)
        for k in self._jagged:
            out[k] = ak.Array([np.asarray(x, dtype=np.int64) for x in data[k]])
        return out

    def write(self, data, *, jagged=()):
        if length(data) == 0:
            return
        if self.file is None:
            self._jagged = list(jagged)
            self.file = uproot.recreate(self.path)
            branch_types = {}
            for k, v in data.items():
                if k in self._jagged:
                    branch_types[k] = "var * int64"
                elif v.ndim == 1:
                    branch_types[k] = v.dtype
                else:
                    branch_types[k] = (v.dtype, v.shape[1:])
            self.file.mktree(self.tree, branch_types)
            log(f"    created tree \"{self.tree}\" in {self.path}")
        self.file[self.tree].extend(self._to_writable(data))
        self.n_written += length(data)

    ### write a small one-row tree with run-level numbers next to the data tree
    def write_summary(self, summary, tree=SUMMARY_TREE):
        if self.file is None:
            self.file = uproot.recreate(self.path)
        data = {k: np.array([v]) for k, v in summary.items()}
        self.file.mktree(tree, {k: v.dtype for k, v in data.items()})
        self.file[tree].extend(data)

    ### store a histogram object next to the trees, so that it can be drawn directly in ROOT (TBrowser, ->Draw())
    # histogram: (bin contents, bin edges) for a TH1D, (bin contents[x, y], x edges, y edges) for a TH2D
    def write_histogram(self, name, histogram):
        if self.file is None:
            self.file = uproot.recreate(self.path)
        self.file[name] = tuple(np.asarray(part, dtype=np.float64) for part in histogram)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

### write a complete data dict as a new ROOT file
# histograms: optional {name: (contents, edges) or (contents, x edges, y edges)}, stored as TH1D / TH2D objects
def write_tree(path, data, tree=DEFAULT_TREE, *, summary=None, jagged=(), histograms=None):
    with TreeWriter(path, tree) as writer:
        writer.write(data, jagged=jagged)
        if summary is not None:
            writer.write_summary(summary)
        for name, histogram in (histograms or {}).items():
            writer.write_histogram(name, histogram)
    return

### convert a jagged branch read back from a file into a list of lists of int
def jagged_to_lists(column):
    return [[int(x) for x in row] for row in column]

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
    with uproot.open(path) as f:
        t = f[tree]
        available = t.keys()
        missing = [k for k in keys if k not in available]
        if len(missing) > 0:
            raise KeyError(f"Branches {missing} not found in tree \"{tree}\" of {path}.")
        return _clean({k: t[k].array(library="np") for k in keys})

### read single rows (entry numbers) of a tree, returns {key: np.ndarray} with one element per requested row
def read_rows(path, rows, tree=None):
    tree = resolve_tree_name(path, tree)
    rows = [int(r) for r in rows]
    with uproot.open(path) as f:
        t = f[tree]
        n = t.num_entries
        bad = [r for r in rows if r < 0 or r >= n]
        if len(bad) > 0:
            raise IndexError(f"Rows {bad} are outside of tree \"{tree}\" of {path}, which has {n:,} rows.")
        pieces = [_clean(t.arrays(entry_start=r, entry_stop=r + 1, library="np")) for r in rows]
    if len(pieces) == 0:
        return {}
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0].keys()}

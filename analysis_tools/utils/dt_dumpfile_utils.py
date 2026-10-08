###########################################
### DUMPFILE: raw data words -> dt hits
###########################################
# A dumpfile is a text file with one 64 bit data word per line. It is read in blocks of lines; the blocks can be
# decoded in parallel. The timestamp of a hit is
#   ts = tdc + 32 * bx + 3564 * 32 * orbit + (orbit counter overflows) * (orbit counter range) * 3564 * 32
# An overflow of the orbit counter shows up as a jump back of the orbit number. Counting the overflows needs all
# earlier blocks, so a block is first decoded on its own (decode_block) and the overflows of the earlier blocks are
# added afterwards, in the order of the file (set_timestamps).
#
# Use:
#   dumpfile, n_blocks = open_dumpfile(file_name, n_lines_to_skip, block_n_lines, n_proc, label)
#   overflow_counter = new_overflow_counter()
#   while True:
#       blocks = decode_next_blocks(dumpfile, block_n_lines, n_blocks_at_once, all_cells, pool)
#       if len(blocks) == 0:
#           break
#       for block in blocks:
#           dt_hits = set_timestamps(block, overflow_counter)
#   dumpfile.close()

import os
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils
from analysis_tools.utils.root_utils import log

# -----------------------------------------

DT_HIT_TS_UNCERTAINTY = np.sqrt((1 / np.sqrt(12)) ** 2 + params.dt_hit_add_ts_unc ** 2)

def count_lines(file_name):
    n_lines = 0
    last_byte = b""
    with open(file_name, "rb") as f:
        while True:
            buffer = f.read(16 * 1024 * 1024)
            if len(buffer) == 0:
                break
            n_lines += buffer.count(b"\n")
            last_byte = buffer[-1:]
    if last_byte != b"" and last_byte != b"\n":
        n_lines += 1  # last line without line break
    return n_lines

### open the dumpfile and skip the first n_lines_to_skip lines (old hits still in the readout buffer)
# returns the open file and the number of blocks of block_n_lines lines which follow
def open_dumpfile(file_name, n_lines_to_skip, block_n_lines, n_proc, label):
    n_lines = count_lines(file_name)
    n_blocks = int(np.ceil(max(0, n_lines - n_lines_to_skip) / block_n_lines))
    log(f"[{label}] {n_lines:,} lines in the dumpfile ({os.path.getsize(file_name) / 1e6:.1f} MB), "
        f"{n_lines_to_skip:,} skipped -> {n_blocks:,} blocks of {block_n_lines:,} lines, n_proc={n_proc}")
    dumpfile = open(file_name, "rb")
    for _ in range(n_lines_to_skip):
        dumpfile.readline()
    return dumpfile, n_blocks

### the next block_n_lines lines as one bytes object (empty at the end of the file)
def read_block(dumpfile, block_n_lines):
    lines = []
    for _ in range(block_n_lines):
        line = dumpfile.readline()
        if line == b"":
            break
        lines.append(line)
    return b"".join(lines)

### number of orbit counter overflows before every word (inside this block)
def count_orbit_overflows(orbit):
    overflows = np.zeros(len(orbit), dtype=np.int64)
    n_overflows = 0
    for i in range(1, len(orbit)):
        if orbit[i - 1] - orbit[i] > params._oc_difference_for_overflow:
            n_overflows += 1
        overflows[i] = n_overflows
    return overflows

### decode the data words of one block and keep the hits of dt cells
# all_cells=False: only cells which are not masked / dead; all_cells=True: every cell of the chamber (testpulse runs)
# returns a dict:
#   "dt_hits": dict of arrays (without the final "ts"), None if the block has no dt hits
#   "ts_in_counter_range": timestamp of every dt hit from tdc, bx and orbit number
#   "overflows_in_block": orbit counter overflows inside this block before every dt hit
#   "n_words", "first_orbit", "last_orbit", "n_overflows_in_block"
def decode_block(block_bytes, all_cells=False):
    words = np.array(block_bytes.split(), dtype=np.uint64)
    n_words = len(words)
    block = {"dt_hits": None, "ts_in_counter_range": None, "overflows_in_block": None, "n_words": n_words,
             "first_orbit": None, "last_orbit": None, "n_overflows_in_block": 0}
    if n_words == 0:
        return block

    # fields of the data words
    fields = {}
    for key in params._htg_keys:
        mask = np.uint64(params._htg_shifted_mask[key])
        shift = np.uint64(params._htg_bitshift[key])
        fields[key] = ((words & mask) >> shift).astype(params._htg_keys[key])

    orbit = fields["oc"].astype(np.int64)
    overflows = count_orbit_overflows(orbit)
    ts_in_counter_range = (fields["tdc"].astype(np.uint64) * np.uint64(derived_params._tdc_to_timestamp)
                           + fields["bx"].astype(np.uint64) * np.uint64(derived_params._bx_to_timestamp)
                           + fields["oc"].astype(np.uint64) * np.uint64(derived_params._orbit_to_timestamp))
    block["first_orbit"] = int(orbit[0])
    block["last_orbit"] = int(orbit[-1])
    block["n_overflows_in_block"] = int(overflows[-1])

    # words of dt cells
    excluded_cells = dt_chamber_utils.excluded_cells()
    selected_words = []
    selected_cells = []
    for i in range(n_words):
        cell = dt_chamber_utils.cell_of_channel(int(fields["ro_ch"][i]), int(fields["ch"][i]))
        if cell is None:
            continue
        if not all_cells and (cell["sl"], cell["ly"], cell["wi"]) in excluded_cells:
            continue
        selected_words.append(i)
        selected_cells.append(cell)
    n_dt_hits = len(selected_words)
    if n_dt_hits == 0:
        return block

    dt_hits = {}
    for key in fields:
        dt_hits[key] = fields[key][selected_words]
    dt_hits["ts"] = np.zeros(n_dt_hits, dtype=params._ts_type)  # set by set_timestamps
    dt_hits["err_ts"] = np.full(n_dt_hits, DT_HIT_TS_UNCERTAINTY, dtype=np.float64)
    for key in params._dt_mapping_keys:
        values = []
        for cell in selected_cells:
            values.append(cell[key])
        dt_hits[key] = np.array(values, dtype=params._dt_mapping_keys[key])
    for key in params._dt_other_keys:
        if key != "ts" and key != "err_ts":
            dt_hits[key] = np.full(n_dt_hits, 0, dtype=params._dt_other_keys[key])  # simulation truth, 0 in data
    block["dt_hits"] = dt_hits
    block["ts_in_counter_range"] = ts_in_counter_range[selected_words]
    block["overflows_in_block"] = overflows[selected_words]
    return block

### read and decode the next n_blocks_at_once blocks; returns the decoded blocks in the order of the file (empty list at the end)
# pool: decode the blocks on the processes of this multiprocessing pool (None: one after the other)
def decode_next_blocks(dumpfile, block_n_lines, n_blocks_at_once, all_cells, pool):
    raw_blocks = []
    for _ in range(n_blocks_at_once):
        raw_block = read_block(dumpfile, block_n_lines)
        if len(raw_block) == 0:
            break
        raw_blocks.append(raw_block)
    if pool is None:
        decoded = []
        for raw_block in raw_blocks:
            decoded.append(decode_block(raw_block, all_cells))
        return decoded
    jobs = []
    for raw_block in raw_blocks:
        jobs.append((raw_block, all_cells))
    return pool.starmap(decode_block, jobs)

### orbit counter overflows of all blocks decoded so far (blocks have to be given in the order of the file)
def new_overflow_counter():
    return {"n_overflows": 0, "last_orbit": None}

### set the final timestamps of the dt hits of a decoded block, returns the dt hits (or None)
def set_timestamps(block, overflow_counter):
    if block["n_words"] == 0:
        return None
    # overflow between the last word of the previous block and the first word of this one
    if overflow_counter["last_orbit"] is not None:
        if overflow_counter["last_orbit"] - block["first_orbit"] > params._oc_difference_for_overflow:
            overflow_counter["n_overflows"] += 1
    dt_hits = block["dt_hits"]
    if dt_hits is not None:
        n_overflows = (block["overflows_in_block"] + overflow_counter["n_overflows"]).astype(np.uint64)
        ts = block["ts_in_counter_range"] + n_overflows * np.uint64(derived_params._orbit_overflow_to_timestamp)
        dt_hits["ts"] = ts.astype(params._ts_type)
    overflow_counter["n_overflows"] += block["n_overflows_in_block"]
    overflow_counter["last_orbit"] = block["last_orbit"]
    return dt_hits

###########################################
### DUMPFILE: raw data words -> dt hits
###########################################
# A dumpfile is a text file with one 64 bit data word per line. It is read in blocks of lines:
#   dumpfile, n_blocks = open_dumpfile(file_name, n_lines_to_skip, block_n_lines, label)
#   overflow_state = {"oc_overflow": 0, "last_oc": None}
#   for i_block in range(n_blocks):
#       hits = data_utils.import_raw_lines(read_lines(dumpfile, block_n_lines), silent=True)   (fields of the data words)
#       dt_hits = dt_hit_utils.extract_dt_hits(hits, overflow_state=overflow_state)          (dt hits with timestamp)
#   dumpfile.close()
# The orbit counter overflows are counted from the start of the file (timestamp_utils.add_timestamp), so the blocks
# have to be processed in the order of the file.

import os
import numpy as np

from analysis_tools.params import params
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
def open_dumpfile(file_name, n_lines_to_skip, block_n_lines, label):
    n_lines = count_lines(file_name)
    n_blocks = int(np.ceil(max(0, n_lines - n_lines_to_skip) / block_n_lines))
    log(f"[{label}] {n_lines:,} lines in the dumpfile ({os.path.getsize(file_name) / 1e6:.1f} MB), "
        f"{n_lines_to_skip:,} skipped -> {n_blocks:,} blocks of {block_n_lines:,} lines")
    dumpfile = open(file_name)
    for _ in range(n_lines_to_skip):
        dumpfile.readline()
    return dumpfile, n_blocks

### the next block_n_lines lines (fewer at the end of the file)
def read_lines(dumpfile, block_n_lines):
    lines = []
    for _ in range(block_n_lines):
        line = dumpfile.readline()
        if line == "":
            break
        if line.strip() != "":
            lines.append(line)
    return lines


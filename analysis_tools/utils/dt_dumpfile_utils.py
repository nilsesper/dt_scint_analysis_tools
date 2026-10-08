###########################################
### DUMPFILE: raw data words -> dt hits
###########################################
# A dumpfile is a text file with one 64 bit data word per line. The file is read in blocks of lines,
# which can be decoded in parallel. The timestamp of a hit is
#   ts = tdc + 32 * bx + 3564 * 32 * orbit + (orbit counter overflows) * (orbit counter range) * 3564 * 32
# Overflows of the orbit counter are found where the orbit number jumps back. Counting them needs all earlier
# blocks, so the decoding works on one block alone and the overflows are added afterwards, in the order of the file.

import collections
import functools
import itertools
import os
import numpy as np

from analysis_tools.params import params, derived_params
from analysis_tools.utils import dt_chamber_utils, parallel_utils, root_utils
from analysis_tools.utils.root_utils import log

# -----------------------------------------

DT_HIT_TS_UNCERTAINTY = np.sqrt((1 / np.sqrt(12)) ** 2 + params.dt_hit_add_ts_unc ** 2)

def count_lines(file_name, buffer_size=16 * 1024 * 1024):
    n_lines, last_byte = 0, b""
    with open(file_name, "rb") as f:
        while buffer := f.read(buffer_size):
            n_lines += buffer.count(b"\n")
            last_byte = buffer[-1:]
    return n_lines + (1 if last_byte not in (b"", b"\n") else 0)

### blocks of block_n_lines lines as bytes (the file is never held in memory as a whole)
def read_line_blocks(file_name, block_n_lines, n_lines_to_skip):
    with open(file_name, "rb") as f:
        for _ in itertools.islice(f, n_lines_to_skip):
            pass
        while block := list(itertools.islice(f, block_n_lines)):
            yield b"".join(block)

### one decoded block: the dt hits (still without "ts") and what is needed to add the orbit counter overflows
DecodedBlock = collections.namedtuple("DecodedBlock", [
    "dt_hits",                # dict of arrays, None if the block has no dt hits
    "ts_in_counter_range",    # timestamp of every dt hit from tdc, bx and orbit number
    "overflows_in_block",     # orbit counter overflows inside this block before every dt hit
    "n_words", "first_orbit", "last_orbit", "n_overflows_in_block",
])

def _orbit_counter_overflows(orbit):
    jumps_back = (orbit[:-1] - orbit[1:]) > params._oc_difference_for_overflow
    return np.concatenate(([0], np.cumsum(jumps_back)))

### decode the data words of one block and keep the hits of dt cells
# all_cells=False: only analysed cells (not masked / dead), all_cells=True: every cell of the chamber (testpulse runs)
def decode_block(block_bytes, all_cells=False):
    words = np.array(block_bytes.split(), dtype=np.uint64)
    n_words = len(words)
    if n_words == 0:
        return DecodedBlock(None, None, None, 0, None, None, 0)
    fields = {k: ((words & np.uint64(params._htg_shifted_mask[k])) >> np.uint64(params._htg_bitshift[k])).astype(dtype)
              for k, dtype in params._htg_keys.items()}
    del words
    orbit = fields["oc"].astype(np.int64)
    overflows = _orbit_counter_overflows(orbit)
    ts_in_counter_range = (fields["tdc"].astype(np.uint64) * np.uint64(derived_params._tdc_to_timestamp)
                           + fields["bx"].astype(np.uint64) * np.uint64(derived_params._bx_to_timestamp)
                           + fields["oc"].astype(np.uint64) * np.uint64(derived_params._orbit_to_timestamp))
    block_info = dict(n_words=n_words, first_orbit=int(orbit[0]), last_orbit=int(orbit[-1]), n_overflows_in_block=int(overflows[-1]))

    tables = dt_chamber_utils.readout_tables()
    ro_ch, ch = fields["ro_ch"].astype(np.intp), fields["ch"].astype(np.intp)
    sl, ly, wi = (tables.cell_keys[k][ro_ch, ch] for k in ("sl", "ly", "wi"))
    accepted_cells = tables.is_chamber_cell if all_cells else tables.is_analysed_cell
    is_dt_hit = tables.is_dt_channel[ro_ch, ch] & tables.has_cell[ro_ch, ch] & accepted_cells[sl, ly, wi]
    n_dt_hits = int(is_dt_hit.sum())
    if n_dt_hits == 0:
        return DecodedBlock(None, None, None, **block_info)

    dt_hits = {k: v[is_dt_hit] for k, v in fields.items()}
    dt_hits["ts"] = np.zeros(n_dt_hits, dtype=params._ts_type)
    dt_hits["err_ts"] = np.full(n_dt_hits, DT_HIT_TS_UNCERTAINTY, dtype=np.float64)
    for k in params._dt_mapping_keys.keys():
        dt_hits[k] = tables.cell_keys[k][ro_ch[is_dt_hit], ch[is_dt_hit]]
    for k, dtype in params._dt_other_keys.items():
        if k not in ("ts", "err_ts"):
            dt_hits[k] = np.full(n_dt_hits, 0, dtype=dtype)  # simulation truth, 0 in data
    return DecodedBlock(dt_hits, ts_in_counter_range[is_dt_hit], overflows[is_dt_hit], **block_info)

### adds the orbit counter overflows of all earlier blocks; blocks have to be given in the order of the file
class OrbitOverflowCounter:
    def __init__(self):
        self.n_overflows = 0
        self.last_orbit = None

    def set_timestamps(self, block):
        if self.last_orbit is not None and (self.last_orbit - block.first_orbit) > params._oc_difference_for_overflow:
            self.n_overflows += 1  # between the last word of the previous block and the first word of this one
        if block.dt_hits is not None:
            n_overflows = (block.overflows_in_block + self.n_overflows).astype(np.uint64)
            ts = block.ts_in_counter_range + n_overflows * np.uint64(derived_params._orbit_overflow_to_timestamp)
            block.dt_hits["ts"] = ts.astype(params._ts_type)
        self.n_overflows += block.n_overflows_in_block
        self.last_orbit = block.last_orbit
        return block.dt_hits

### the dt hits of a dumpfile, block by block in the order of the file: yields (dt hits or None, n_words, i_block, n_blocks)
# n_proc > 1: the blocks are decoded on several processes
def iterate_dt_hits(input_dumpfile, *, n_lines_to_skip, block_n_lines, n_proc, all_cells=False, label="dumpfile -> dt hits"):
    root_utils.check_input_file(input_dumpfile)
    n_lines = count_lines(input_dumpfile)
    n_blocks = int(np.ceil(max(0, n_lines - n_lines_to_skip) / block_n_lines))
    log(f"[{label}] {n_lines:,} lines in the dumpfile ({os.path.getsize(input_dumpfile) / 1e6:.1f} MB), "
        f"{n_lines_to_skip:,} skipped -> {n_blocks:,} blocks of {block_n_lines:,} lines, n_proc={n_proc}")
    overflow_counter = OrbitOverflowCounter()
    blocks = read_line_blocks(input_dumpfile, block_n_lines, n_lines_to_skip)
    with parallel_utils.optional_pool(n_proc) as pool:
        decoded_blocks = parallel_utils.map_in_order(functools.partial(decode_block, all_cells=all_cells), blocks, pool)
        for i_block, block in enumerate(decoded_blocks, start=1):
            if block.n_words == 0:
                yield None, 0, i_block, n_blocks
                continue
            yield overflow_counter.set_timestamps(block), block.n_words, i_block, n_blocks
    if overflow_counter.n_overflows > 0:
        log(f"[{label}] orbit counter overflows found: {overflow_counter.n_overflows:,}")

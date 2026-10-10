#################################################################
### raw dumpfile (.txt) -> raw scint hits (ROOT file, tree "raw_scint_hits")
# raw scint hit = hit of a single SiPM (the mezzanines run without on-FPGA coincidence).
# Decodes the data words, keeps the SiPM channels (params._raw_scint_mapping), adds ly, st, sipm and the timestamp.
# The timestamps are made exactly like those of the dt hits (dumpfile_to_dt_hits.py: same blocks, same lines skipped,
# orbit counter overflows counted the same way), so dt and scint hits of the same dumpfile have the same time scale.
#
# example:
#   python scripts/dumpfile_to_raw_scint_hits.py --input_dumpfile run.txt --raw_scint_hits_file out/run_raw_scint_hits.root
#################################################################

import argparse
import time

from analysis_tools.utils import data_utils, dt_dumpfile_utils, root_utils, scint_hit_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert a raw dumpfile into a ROOT file of raw scint hits (single SiPM hits).")
    parser.add_argument("--input_dumpfile", type=str, required=True, help="input file path: raw dumpfile (.txt)")
    parser.add_argument("--raw_scint_hits_file", type=str, required=True, help="output file path: raw scint hits (.root)")
    parser.add_argument("--n_lines_to_skip", type=int, default=999,
                        help="number of lines at the start of the dumpfile to ignore (old hits still in the readout buffer); "
                             "use the same value as for dumpfile_to_dt_hits.py")
    parser.add_argument("--block_lines", type=int, default=500_000, help="number of dumpfile lines processed at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dumpfile -> raw scint hits"
    root_utils.check_input_file(args.input_dumpfile)
    log(f"[{label}] START \"{args.input_dumpfile}\" -> \"{args.raw_scint_hits_file}\"")

    t_start = time.perf_counter()
    dumpfile, n_blocks = dt_dumpfile_utils.open_dumpfile(args.input_dumpfile, args.n_lines_to_skip, args.block_lines, label)
    overflow_state = {"oc_overflow": 0, "last_oc": None}  # orbit counter overflows, counted from the start of the file
    output_file = root_utils.create_file(args.raw_scint_hits_file)
    n_words_total, n_hits_total = 0, 0
    for i_block in range(n_blocks):
        lines = dt_dumpfile_utils.read_lines(dumpfile, args.block_lines)
        hits = data_utils.import_raw_lines(lines, silent=True)
        raw_scint_hits = scint_hit_utils.extract_raw_scint_hits(hits, overflow_state=overflow_state)
        n_hits = root_utils.length(raw_scint_hits)
        if n_hits > 0:
            root_utils.write_rows(output_file, root_utils.RAW_SCINT_HITS_TREE, raw_scint_hits)
        n_words_total += len(lines)
        n_hits_total += n_hits
        log(f"[{label}] block {i_block + 1:,} / {n_blocks:,}: {len(lines):,} raw hits -> {n_hits:,} raw scint hits "
            f"| totals: {n_words_total:,} raw, {n_hits_total:,} raw scint")
    dumpfile.close()
    output_file.close()

    log(f"[{label}] DONE. {n_words_total:,} raw hits read, {n_hits_total:,} raw scint hits written, "
        f"took {time.perf_counter() - t_start:.1f}s")
    if n_hits_total == 0:
        raise RuntimeError(f"No raw scint hits found in {args.input_dumpfile} -- check params._raw_scint_mapping (readout channels).")

if __name__ == "__main__":
    main()
    log("###### Done.")

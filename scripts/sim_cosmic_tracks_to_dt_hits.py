#################################################################
### simulation: cosmic muon tracks -> dt hits (ROOT file, tree "dt_hits")
# propagates the muons through the chamber and creates one hit per crossed cell (with params._dt_cell_efficiency)
# optional: gaussian noise on the drift times, constant random time offset per wire (mis-calibration)
# the output has the format of dumpfile_to_dt_hits.py and can be given to dt_hits_to_sl_patterns.py
#################################################################

import argparse
import numpy as np
import multiprocessing
import time

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import dt_sim_utils, root_utils

# ---------------------------------------------------------------

### one chunk of sim cosmic muons -> its sim dt hits; runs in its own process if --n_proc > 1
def sim_dt_hits_in_chunk(job):
    cosmic_muons_file, start, stop, chunk_id, noise_ampl, sys_miscalib_ampl, verbose = job
    t_start = time.perf_counter()
    cosmic_muons = root_utils.read_tree(cosmic_muons_file, root_utils.DEFAULT_TREE, start, stop)
    n_sim_muons = root_utils.length(cosmic_muons)
    dt_hits = dt_sim_utils.hits_from_muons(muons=cosmic_muons, noise_ampl=noise_ampl, sys_miscalib_ampl=sys_miscalib_ampl)
    return dt_hits, n_sim_muons, time.perf_counter() - t_start

def main():
    parser = argparse.ArgumentParser(description="Create simulated dt hits from cosmic muon tracks.")
    parser.add_argument("--cosmic_muons_file", type=str, required=True, help="input file path: cosmic muon tracks (.root)")
    parser.add_argument("--dt_hits_file", type=str, required=True, help="output file path: simulated dt hits (.root)")
    parser.add_argument("--ts_noise_amplitude", type=float, default=0, help="sigma of gaussian noise on the hit timestamps in timestamp units")
    parser.add_argument("--sys_miscalib_ampl", type=float, default=0, help="sigma of a constant random time offset per wire in timestamp units")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--chunk_size", type=int, default=20_000, help="number of hits per chunk")
    parser.add_argument("--n_proc", type=int, default=1, help="number of chunks processed at the same time")
    parser.add_argument("--verbose", action="store_true", help="print every pattern found (switches off --n_proc)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    label = "sim cosmic muons -> sim dt hits"
    root_utils.check_input_file(args.cosmic_muons_file)
    log(f"[{label}] START \"{args.cosmic_muons_file}\" -> \"{args.dt_hits_file}\" (chunks of {args.chunk_size:,} hits, n_proc={args.n_proc})")
    n_proc = args.n_proc
    if args.verbose:
        n_proc = 1

    ### the chunks: (input file, first row, row after the last row, chunk_id, settings)
    n_hits = root_utils.number_of_rows(args.cosmic_muons_file, root_utils.DEFAULT_TREE)
    jobs = []
    for start in range(0, n_hits, args.chunk_size):
        stop = min(start + args.chunk_size, n_hits)
        chunk_id = len(jobs) + 1
        jobs.append((args.cosmic_muons_file, start, stop, chunk_id, args.ts_noise_amplitude, args.sys_miscalib_ampl, args.verbose))

    ### process the chunks: with n_proc > 1, n_proc chunks at the same time; the results come in the order of the chunks
    if n_proc > 1:
        pool = multiprocessing.Pool(n_proc)
        results = pool.imap(sim_dt_hits_in_chunk, jobs)
    else:
        results = map(sim_dt_hits_in_chunk, jobs)

    output_file = root_utils.create_file(args.dt_hits_file)
    n_sim_muons_total, n_dt_hits_total = 0, 0
    i_chunk = 0
    for dt_hits, n_sim_muons, seconds in results:
        i_chunk += 1
        root_utils.write_rows(output_file, root_utils.DT_HITS_TREE, dt_hits)
        n_dt_hits = root_utils.length(dt_hits)
        n_sim_muons_total += n_sim_muons
        n_dt_hits_total += n_dt_hits
        log(f"    chunk {i_chunk:,} / {len(jobs):,}: {n_sim_muons:,} sim cosmic muons -> {n_dt_hits:,} sim dt hits ({seconds:.2f}s)")
    output_file.close()
    if n_proc > 1:
        pool.close()
        pool.join()
    log(f"[{label}] DONE. {len(jobs):,} chunks, sim_cosmic_muons={n_sim_muons_total:,}, "
        f"sim_dt_hits={n_dt_hits_total:,}")

if __name__ == "__main__":
    main()
    log("###### Done.")

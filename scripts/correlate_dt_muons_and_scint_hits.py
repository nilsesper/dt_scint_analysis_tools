#################################################################
### dt muons + scint hits -> correlated pairs (dt muon and scint hit of the same muon)
# Every scint hit (in time order) looks for dt muons with |ts_dt - ts_scint| <= --ts_window. The pair is taken only
# if there is exactly one dt muon in the window and this dt muon is not paired yet (see scint_hit_utils.py).
# For every pair the dt muon track is extrapolated to the hit strip: x_at_strip, y_at_strip, dist_to_strip.
# Both files must come from the same dumpfile (same time scale). To use only some dt muons or scint hits, cut the
# files first with apply_cuts.py (scint hits: --tree scint_hits).
#
# example:
#   python scripts/correlate_dt_muons_and_scint_hits.py --dt_muons_file out/run_dt_muons.root \
#          --scint_hits_file out/run_scint_hits.root --corr_file out/run_corr_dt_muons_scint_hits.root
#################################################################

import argparse
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import root_utils, scint_hit_utils
from analysis_tools.utils.root_utils import log

# ---------------------------------------------------------------

TS_TO_SECONDS = 0.78e-9

def main():
    parser = argparse.ArgumentParser(description="Find the scint hit of every dt muon (correlation in time).")
    parser.add_argument("--dt_muons_file", type=str, required=True, help="input file path: dt muons (.root)")
    parser.add_argument("--scint_hits_file", type=str, required=True, help="input file path: scint hits (.root)")
    parser.add_argument("--corr_file", type=str, required=True, help="output file path: correlated pairs (.root)")
    parser.add_argument("--ts_window", type=float, default=params._corr_ts_group_window,
                        help="max |ts scint - ts dt| in timestamp units (default: params._corr_ts_group_window)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args()

    label = "dt muons + scint hits -> correlation"
    root_utils.check_input_file(args.dt_muons_file)
    root_utils.check_input_file(args.scint_hits_file)
    log(f"[{label}] START \"{args.dt_muons_file}\" + \"{args.scint_hits_file}\" -> \"{args.corr_file}\" (window {args.ts_window} TU)")

    dt_muons = root_utils.read_tree(args.dt_muons_file, root_utils.DEFAULT_TREE)
    scint_hits = root_utils.read_tree(args.scint_hits_file, root_utils.SCINT_HITS_TREE)
    n_dt_muons, n_scint_hits = root_utils.length(dt_muons), root_utils.length(scint_hits)
    if n_dt_muons == 0 or n_scint_hits == 0:
        raise RuntimeError(f"Nothing to correlate: {n_dt_muons} dt muons, {n_scint_hits} scint hits.")

    ts_dt = np.asarray(dt_muons["ts"], dtype=np.float64)
    ts_scint = np.asarray(scint_hits["ts"], dtype=np.float64)
    pairs = scint_hit_utils.pair_scint_hits_and_dt_muons(ts_scint, ts_dt, args.ts_window)
    corr = scint_hit_utils.correlation_rows(scint_hits, dt_muons, pairs)
    n_pairs = len(pairs)

    # rates over the time range of the dt muons
    duration = TS_TO_SECONDS * (np.amax(ts_dt) - np.amin(ts_dt))
    output_file = root_utils.create_file(args.corr_file)
    root_utils.write_rows(output_file, root_utils.DEFAULT_TREE, corr)
    root_utils.write_summary(output_file, {
        "duration_seconds": duration, "n_dt_muons": n_dt_muons, "n_scint_hits": n_scint_hits, "n_pairs": n_pairs,
        "ts_window": args.ts_window})
    output_file.close()

    log(f"[{label}] duration {duration:.1f} s: dt muon rate {n_dt_muons / duration:.2f} Hz, scint hit rate "
        f"{n_scint_hits / duration:.2f} Hz, correlation rate {n_pairs / duration:.2f} Hz")
    if n_pairs > 0:
        n_inside = int(np.sum(corr["dist_to_strip"] == 0))
        log(f"[{label}] {n_pairs:,} pairs; delta_ts (scint - dt) median {np.median(corr['delta_ts']):.1f} TU; "
            f"dt track crosses the hit strip in {n_inside:,} of {n_pairs:,} pairs")
    log(f"[{label}] DONE.")

if __name__ == "__main__":
    main()
    log("###### Done.")

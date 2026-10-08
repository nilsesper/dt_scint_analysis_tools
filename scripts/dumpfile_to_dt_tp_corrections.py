#################################################################
### testpulse run: raw dumpfile (.txt) -> timing calibration of all dt cells (ROOT file)
# For a dumpfile recorded with simultaneous testpulses on all channels:
#   - dt hits of all cells of the chamber (masked / dead cells included, no dead time cut)
#   - per cell: histogram of the time inside the orbit (ts_orbit = tdc + 32 * bx, bins of 1 ts unit), position of
#     the first peak (weighted mean); later peaks are ringing of the testpulse circuit and are ignored
#   - the known testpulse delays per frontend connector (params._tp_time_offset) are subtracted
#   - correction per cell: ts_corr = <mean testpulse time of the chamber (or SL)> - <testpulse time of the cell>,
#     the mean is taken over the cells which are not masked / dead,
#     applied with a plus sign: ts_corrected = ts + ts_corr
#
# output (.root): tree "tree" with one row per cell:
#   sl, ly, wi, fe_id                 cell
#   ts_corr, err_ts_corr              the correction (0 for cells without testpulse peak)
#   valid                             1 if a testpulse peak was found
#   masked                            1 for cells in params._dt_wire_mask / _dt_dead_wires (calibrated, but not used for the mean)
#   tp_ts_mean, tp_ts_err             testpulse time of the cell (after the offset correction)
#   tp_ts_mean_raw, tp_ts_err_raw, tp_offset   ... before the offset correction, and the subtracted offset
#   ts_target                         mean testpulse time the cell is aligned to
#   n_hits, n_peak_hits, peak_ts_min, peak_ts_max   hits of the cell, hits in the first peak and its range in ts_orbit
# plus tree "summary" and histograms (TH2D per SL, x = wire, y = layer: ts_corr_sl<N>, tp_ts_mean_sl<N>, n_peak_hits_sl<N>;
# TH1D per SL: ts_orbit_sl<N> of all hits). Output ending .pcl: only the correction dict of the old script.
#
# Use the result with
#   python scripts/dumpfile_to_dt_hits.py ... --dt_tp_corrections_file <this output>
#   python scripts/run_dt_pipeline.py ... --dt_tp_corrections_file <this output>
#
# examples:
#   python scripts/dumpfile_to_dt_tp_corrections.py --input_dumpfile tp_run.txt --dt_tp_corrections_file calib/tp_corrections.root
#   python scripts/dumpfile_to_dt_tp_corrections.py --input_dumpfile tp_run.txt --dt_tp_corrections_file calib/tp_corrections.root \
#          --dt_tp_hits_file calib/tp_dt_hits.root --alignment sl --n_proc 4
#################################################################

import argparse

from analysis_tools.utils.root_utils import log
from analysis_tools.params import params
from analysis_tools.utils import dt_pipeline_utils

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Calculate the timing calibration of all dt cells from a testpulse dumpfile.")
    parser.add_argument("--input_dumpfile", type=str, required=True, help="input file path: raw dumpfile (.txt) recorded with testpulses")
    parser.add_argument("--dt_tp_corrections_file", type=str, required=True,
                        help="output file path: timing calibration (.root; .pcl for the format of the old script)")
    parser.add_argument("--dt_tp_hits_file", type=str, default=None,
                        help="optional output file path: the testpulse dt hits (.root, tree \"dt_hits\", with branch ts_orbit) for inspection")
    parser.add_argument("--alignment", type=str, choices=["chamber", "sl"], default="chamber",
                        help="align all cells to the mean of the full chamber (default, needs aligned testpulses of all SLs) "
                             "or each SL to its own mean (a time offset between the SLs remains)")
    parser.add_argument("--rel_thres", type=float, default=0.2,
                        help="threshold for the peak search, relative to the highest bin of the cell histogram")
    parser.add_argument("--no_offset_correction", action="store_true",
                        help="do not subtract the known testpulse delays per frontend connector (params._tp_time_offset)")
    parser.add_argument("--n_lines_to_skip", type=int, default=None,
                        help=f"number of lines at the start of the dumpfile to ignore (default: params._dumpfile_hits_to_skip = {params._dumpfile_hits_to_skip})")
    parser.add_argument("--n_proc", type=int, default=1, help="number of processes to read the dumpfile (does not change the result)")
    parser.add_argument("--block_lines", type=int, default=500_000, help="number of dumpfile lines processed at once")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)

    dt_pipeline_utils.dumpfile_to_dt_tp_corrections(
        args.input_dumpfile, args.dt_tp_corrections_file, dt_tp_hits_file=args.dt_tp_hits_file, n_lines_to_skip=args.n_lines_to_skip,
        block_n_lines=args.block_lines, n_proc=args.n_proc, rel_thres=args.rel_thres, alignment=args.alignment,
        correct_for_offsets=not args.no_offset_correction,
    )

if __name__ == "__main__":
    main()
    log("###### Done.")

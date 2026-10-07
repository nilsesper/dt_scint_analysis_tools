#################################################################
### simulation: generate cosmic muon tracks (ROOT file)
# muons start in a plane at the bottom of the chamber, uniformly over the chamber area plus a margin,
# with the angular distribution params.cosmic_muon_theta_weight and exponential waiting times
# simulated muons have muon_id >= 1 (data has muon_id = 0)
#
# simulation chain:
#   sim_gen_cosmic_tracks.py -> sim_cosmic_tracks_to_dt_hits.py [-> sim_add_dt_hit_noise.py] [-> sim_add_dt_secondary_hits.py]
#   -> run_dt_pipeline.py --from_stage hit_diff_hist  (with the dt hits file named <prefix>_dt_hits.root)
#################################################################

import argparse
import numpy as np

from analysis_tools.utils.root_utils import log
from analysis_tools.utils import data_utils, dt_utils, muon_utils, plot_utils, root_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate simulated cosmic muon tracks.")
    parser.add_argument("--cosmic_muons_file", type=str, required=True, help="output file path: cosmic muon tracks (.root)")
    parser.add_argument("--duration_s", type=float, default=1000, help="simulated time in seconds")
    parser.add_argument("--margin_mm", type=float, default=1500, help="size of the muon source beyond the chamber edges in mm")
    parser.add_argument("--rate_hz_per_m2", type=float, default=147, help="muon rate through the source plane in Hz/m^2")
    parser.add_argument("--t_start", type=float, default=1000, help="timestamp of the start of the simulation in timestamp units")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random number generator, for reproducible output (default: random)")
    parser.add_argument("--params_file", type=str, default=None,
                        help="parameter file to use instead of analysis_tools/params/params.py (e.g. another readout mapping)")
    args = parser.parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    t_sim = int(args.duration_s / (plot_utils.TS_UNIT_NS * 1e-9))  # in timestamp units
    xrange = [params._dt_chamber["pos"][0] - args.margin_mm, params._dt_chamber["pos"][0] + params._dt_chamber["size"][0] + args.margin_mm]
    yrange = [params._dt_chamber["pos"][1] - args.margin_mm, params._dt_chamber["pos"][1] + params._dt_chamber["size"][1] + args.margin_mm]
    z0 = params._dt_chamber["pos"][2]  # lowest point of chamber (closest to sl 1)
    muon_area = np.abs(xrange[1] - xrange[0]) * np.abs(yrange[1] - yrange[0]) * 1e-6  # m^2
    muon_rate = muon_area * args.rate_hz_per_m2 * plot_utils.TS_UNIT_NS * 1e-9  # 1 / timestamp unit
    n_muons = np.random.poisson(lam=muon_rate * t_sim)
    inter_arrival_times = np.random.exponential(1.0 / muon_rate, n_muons)  # in timestamp units
    ts = args.t_start + np.cumsum(inter_arrival_times)
    log(f"###### Generating {n_muons} cosmic muon tracks over {args.duration_s:g} s = {t_sim} TU on {muon_area:.2f} m^2...")
    cosmic_muons = muon_utils.generate_cosmic_muons(
        n=n_muons, ts=ts, xrange=xrange, yrange=yrange, z0=z0, phirange=[0, 2 * np.pi], thetarange=[0, np.pi / 2], theta_weight=params.cosmic_muon_theta_weight,
    )
    root_utils.write_tree(args.cosmic_muons_file, cosmic_muons)
    log(f"###### Stored {n_muons} cosmic muon tracks in {args.cosmic_muons_file}")

if __name__ == "__main__":
    main()
    log("###### Done.")

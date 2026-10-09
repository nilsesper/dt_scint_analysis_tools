# Output files of the dt workflow

Every script writes one ROOT file. A file holds **trees** (tables: one branch per column, one entry per row) and
sometimes **histograms** (TH1D / TH2D objects to draw directly in ROOT). Look into a file with

```
python scripts/inspect_root_file.py --input_file <file> [--branches "t0,chi2/ndf" --n_rows 10]
```

In python, `root_utils.read_tree(path, tree)` gives a dict `{branch name: numpy array}`.
Units: lengths in mm, times in timestamp units (TU, 1 TU = 0.78 ns), angles in rad. Branches starting with `sim_`
hold the simulation truth; in data they are 0. Coordinate systems: see the header of
`analysis_tools/utils/dt_chamber_utils.py`.

Overview (in the order of the workflow):

| file (as named by `run_dt_pipeline.py`) | made by | trees | histograms |
|---|---|---|---|
| `<prefix>_dt_hits.root` | `dumpfile_to_dt_hits.py` | `dt_hits` | – |
| `<prefix>_hit_diff_hist.root` | `dt_hits_to_hit_diff_hist.py` | `tree`, `summary` | `hit_diff_hist` (TH1D) |
| `<prefix>_cell_counts.root` | `dt_hits_to_cell_counts.py` | `tree`, `summary` | `cell_counts` (TH2D) |
| `<prefix>_sl_patterns.root` | `dt_hits_to_sl_patterns.py` | `tree` | – |
| `<prefix>_sl_fits.root`, `_sl_fits_cut.root` | `sl_patterns_to_sl_fits.py`, `apply_cuts.py` | `tree` | – |
| `<prefix>_super_fits.root`, `_super_fits_cut.root` | `sl_fits_to_super_fits.py`, `apply_cuts.py` | `tree` | – |
| `<prefix>_dt_muons.root` | `super_fits_to_dt_muons.py` | `tree` | – |
| testpulse calibration | `dumpfile_to_dt_tp_corrections.py` | `tree`, `summary` | per SL: `ts_corr_sl<n>`, `tp_ts_mean_sl<n>`, `n_peak_hits_sl<n>` (TH2D), `ts_orbit_sl<n>` (TH1D) |
| simulated muons | `sim_gen_cosmic_tracks.py` | `tree` | – |

`apply_cuts.py` writes the same trees and branches as its input file, only with fewer rows.

---

## dt hits — tree `dt_hits`, one row per hit, sorted by `ts` (within each block of `--block_lines` dumpfile lines)

| branch | meaning |
|---|---|
| `ro_ch`, `ch` | readout channel and channel number of the data word |
| `oc`, `bx`, `tdc` | orbit counter, bunch crossing, TDC count of the data word |
| `ts`, `err_ts` | timestamp `tdc + 32 bx + 3564 * 32 orbit` (+ orbit counter overflows, + timing correction) and its uncertainty |
| `sl`, `ly`, `wi` | cell: superlayer (1-3), layer (0-3), wire |
| `conn_id`, `fe_id`, `ch_id` | connector of the OBDT mapping, frontend connector (index in `params._fe_idx_list`), conductor (0-15) |
| `sim_ts`, `sim_dt`, `sim_dd`, `sim_lat`, `sim_id`, `sim_vd`, `sim_tan_alpha`, `sim_loc_x0`, `sim_x0`, `sim_y0`, `sim_z0`, `sim_theta`, `sim_phi` | simulation truth: muon time, drift time, drift distance, laterality, muon number, drift velocity, slope, local position, muon track |

The testpulse hits (`--dt_tp_hits_file` of `dumpfile_to_dt_tp_corrections.py`) have in addition `ts_orbit = tdc + 32 bx`.

## hit diff hist — time between consecutive hits of the same cell

- histogram `hit_diff_hist` (TH1D): draw this one in ROOT.
- tree `tree`, one row per bin: `edge_low`, `edge_high`, `center`, `hist` (bin content), `err_hist` (sqrt(N), at least 1).
  Drawing the branch `hist` in ROOT shows how often each bin content occurs; use `tree->Draw("hist:center")`.
- tree `summary`, one row: `entries`, `underflow`, `overflow`, `n_hits`.

## cell counts — number of hits per cell

- histogram `cell_counts` (TH2D): x = wire, y = 4 * (sl - 1) + ly.
- tree `tree`, one row per cell: `sl`, `ly`, `wi`, `count`.
- tree `summary`, one row: `duration_seconds`, `ts_min`, `ts_max` (first / last hit), `n_hits`.

## sl patterns — tree `tree`, one row per pattern (4 hits of one superlayer), sorted by `ts3` within each chunk

| branch | meaning |
|---|---|
| `sl`, `pat_type` | superlayer; pattern shape = index in `params._dt_sl_patterns` (0 = "+a", 1 = "-a", ...) |
| `wi0`..`wi3`, `ts0`..`ts3`, `err_ts0`..`err_ts3` | wire, time and time uncertainty of the hit in layer 0..3 |
| `chunk_id` | number of the chunk of dt hits the pattern was found in (1, 2, ...) |
| `sim_...` | simulation truth of the hits (`sim_lat0`..`3`, `sim_dt0`..`3`, `sim_dd0`..`3`, `sim_lat_id` = index of the true laterality in the pattern shape, ...) |

## sl fits — tree `tree`: all branches of the sl patterns, plus per row

| branch | meaning |
|---|---|
| `impossible` | 1: the hit times cannot come from one track, no fit (all fit branches are 0) |
| `laterality` | index of the best laterality in `params._dt_sl_patterns[shape]["laterality"]` |
| `t0`, `x0`, `tan_alpha`, `vd` | fit result of the best laterality: crossing time, track position at the height of the layer-3 wire relative to that wire, slope, drift velocity (mm / TU; fixed unless fitted) |
| `err_...`, `corr_..._...` | uncertainties and covariances of the fit parameters |
| `chi2/ndf` | chi2 / number of degrees of freedom |
| `dt0`..`dt3` | drift time of the hit in layer 0..3 according to the fit |
| `lat<i>_<key>` | the same fit results for every laterality i of the shape (0..3) |

Frame of `x0` and `tan_alpha`: track frame with the wire of layer 3 (`wi3`) as origin, see `dt_chamber_utils.py`.

## super fits — tree `tree`, one row per pair of sl fits of the two phi superlayers

| branch | meaning |
|---|---|
| `row_sl1`, `row_sl3` | rows of the two paired sl fits in the sl fits file (the cut file the super fits were made from) |
| `n_candidates_sl1`, `n_candidates_sl3` | number of sl fits of SL 1 / SL 3 within the time window (`params._muon_tgroup_tolerance`) around the t0 of the SL 1 fit, the paired fit included (only fits which may be paired: passing the cuts of the input file and `--max_chi2` / `--max_alpha_deg`). 1 = no other sl fit could have been taken instead |
| `pat_type_sl<n>`, `wi0_sl<n>`..`wi3_sl<n>` | pattern shape and wires in superlayer n |
| `ts0`..`ts7`, `err_ts0`..`err_ts7` | the 8 hit times: layers 0-3 of SL 1, then layers 0-3 of SL 3 |
| `<sl fit key>_sl<n>` | the sl fit results of superlayer n (`t0_sl1`, `chi2/ndf_sl3`, ...) |
| `chunk_id` | chunk of dt hits the two sl fits come from |
| `sim_...`, `sim_id_mismatch` | simulation truth of the SL 1 pattern; 1 if the two sl fits come from different simulated muons |
| `impossible_super_fits` | 1: no fit possible |
| `t0_super_fits`, `x0_super_fits`, `tan_alpha_super_fits`, `vd_super_fits`, `err_...`, `corr_...`, `chi2/ndf_super_fits` | fit result of the best laterality combination |
| `lat_id1_super_fits`, `lat_id2_super_fits` | laterality index of the SL 1 / SL 3 pattern |
| `dt0_super_fits`..`dt7_super_fits`, `ts_residual_super_fits` (8 values) | drift times and residuals (fitted - measured hit time) of the 8 hits |
| `ref_x_super_fits`, `ref_z_super_fits` | chamber position (x, z) of the reference wire (layer 3 of SL 3); `x0` is relative to it |

The optional `--super_patterns_file` holds the same rows without the `..._super_fits` branches.

## dt muons — tree `tree`, one row per muon, sorted by time within each chunk

| branch | meaning |
|---|---|
| `x0`, `y0`, `z0` | point of the muon track at `z0 = params._muon_reco_z0` (chamber frame) |
| `theta`, `phi` | direction: angle to the z axis, angle in the x-y plane from the x axis |
| `ts` | arrival time (mean of the super fit and theta sl fit t0) |
| `err_...` | uncertainties |
| `n_theta_candidates` | theta sl fits inside the time window around the super fit t0, also those taken by another super fit (> 1: ambiguous, the closest unused one was taken) |
| `n_candidates_sl1`, `n_candidates_sl3` | copied from the super fit: candidates in the two phi superlayers. Unambiguous muons: `n_candidates_sl1 == 1`, `n_candidates_sl3 == 1` and `n_theta_candidates == 1`, e.g. `apply_cuts.py --cuts "n_candidates_sl1,==,1;n_candidates_sl3,==,1;n_theta_candidates,==,1"` |
| `delta_t0` | t0 of the theta sl fit - t0 of the super fit |
| `super_fit_row` | row of the super fit in the (cut) super fits file |
| `sl1_fit_row`, `sl2_fit_row`, `sl3_fit_row` | rows of the three sl fits in the (cut) sl fits file |
| `chunk_id` | chunk of dt hits |
| `sim_...`, `sim_id_mismatch` | simulation truth; 1 if the combined fits come from different simulated muons |

## testpulse calibration — `dumpfile_to_dt_tp_corrections.py`

- tree `tree`, one row per cell of the chamber:

| branch | meaning |
|---|---|
| `sl`, `ly`, `wi`, `fe_id` | cell, frontend connector |
| `ts_corr`, `err_ts_corr` | the correction, applied as `ts -> ts + ts_corr` (0 for cells without testpulse peak) |
| `valid` | 1 if a testpulse peak was found |
| `masked` | 1 for cells in `params._dt_wire_mask` / `_dt_dead_wires` (calibrated, but not used for the target time) |
| `tp_ts_mean`, `tp_ts_err` | testpulse time of the cell after the offset correction |
| `tp_ts_mean_raw`, `tp_ts_err_raw`, `tp_offset` | before the offset correction, and the subtracted offset (`params._tp_time_offset`) |
| `ts_target` | mean testpulse time the cell is aligned to (whole chamber or superlayer) |
| `n_hits`, `n_peak_hits`, `peak_ts_min`, `peak_ts_max` | hits of the cell, hits in the first peak and its range in `ts_orbit` |

- tree `summary`, one row: `n_cells`, `n_valid`, `rel_thres`, `chamber_alignment`, `correct_for_offsets`, `n_raw_hits`, `n_dt_hits`, `n_lines_to_skip`.
- histograms per superlayer n: `ts_corr_sl<n>`, `tp_ts_mean_sl<n>`, `n_peak_hits_sl<n>` (TH2D, x = wire, y = layer), `ts_orbit_sl<n>` (TH1D, time inside the orbit of all hits).
- with an output file name ending `.pcl`: only the dict `{sl: {ly: {wi: {"ts_corr", "err_ts_corr"}}}}` (format of the old testpulse script).

## simulated muons — `sim_gen_cosmic_tracks.py`, tree `tree`, one row per muon

`x0`, `y0`, `z0`, `theta`, `phi`, `ts`: the generated muon track and time; `sim_id`: muon number (1, 2, ...);
the `err_...` and the other `sim_...` branches are 0.
The simulated hits (`sim_cosmic_tracks_to_dt_hits.py`, `sim_add_dt_hit_noise.py`, `sim_add_dt_secondary_hits.py`,
`dt_hits_mask_cells.py`) are dt hit files.

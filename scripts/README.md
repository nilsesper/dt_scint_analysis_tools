# DT workflow with ROOT files

One script per stage. Every script takes its input and output files as arguments; nothing is read from or
written to a fixed location. Run `python <script> --help` for all options.

Before running, the repository has to be on the `PYTHONPATH` (e.g. `export PYTHONPATH=$PWD` in the repository directory).
Needed python packages: numpy, scipy, matplotlib, tqdm, uproot, awkward.

## Stages

```
testpulse dumpfile (.txt)
  └─ dumpfile_to_dt_tp_corrections.py ─► tp_corrections.root     (timing calibration of every cell, see below)

dumpfile (.txt)
  └─ dumpfile_to_dt_hits.py ───────────► dt_hits.root            (optional: --dt_tp_corrections_file tp_corrections.root)
        ├─ dt_hits_to_hit_diff_hist.py ► hit_diff_hist.root      (side product)
        ├─ dt_hits_to_cell_counts.py ──► cell_counts.root        (side product)
        └─ dt_hits_to_sl_patterns.py ──► sl_patterns.root        (applies the dead time cut)
              └─ sl_patterns_to_sl_fits.py ► sl_fits.root        (fixed drift velocity)
                    └─ apply_cuts.py ──────► sl_fits_cut.root
                          └─ sl_fits_to_super_fits.py ► super_fits.root   (two phi superlayers, 8 hits, fixed drift velocity)
                                └─ apply_cuts.py ─────► super_fits_cut.root
                                      └─ super_fits_to_dt_muons.py ► dt_muons.root
                                         (cut super fits + theta sl fits of sl_fits_cut.root)
```

A dt muon is built from one super fit of the two phi superlayers (x-z view) and the sl fit of the theta superlayer
(y-z view) which is closest in time, if their `t0` differ by less than `params._muon_tgroup_tolerance`.

## Step by step

```
D=/path/to/output            # any directory
python scripts/dt_root/dumpfile_to_dt_hits.py      --input_dumpfile /path/to/run.txt --dt_hits_file $D/run_dt_hits.root --n_proc 8
#   with testpulse timing calibration: add --dt_tp_corrections_file /path/to/tp_corrections.root

python scripts/dt_root/dt_hits_to_hit_diff_hist.py --dt_hits_file $D/run_dt_hits.root --hit_diff_hist_file $D/run_hit_diff_hist.root
python scripts/dt_root/dt_hits_to_cell_counts.py   --dt_hits_file $D/run_dt_hits.root --cell_counts_file $D/run_cell_counts.root
python scripts/dt_root/dt_hits_to_sl_patterns.py   --dt_hits_file $D/run_dt_hits.root --sl_patterns_file $D/run_sl_patterns.root --n_proc 8
python scripts/dt_root/sl_patterns_to_sl_fits.py   --sl_patterns_file $D/run_sl_patterns.root --sl_fits_file $D/run_sl_fits.root --n_proc 8
python scripts/dt_root/apply_cuts.py               --input_file $D/run_sl_fits.root --output_file $D/run_sl_fits_cut.root --cuts "impossible,==,0;chi2/ndf,<,20"
python scripts/dt_root/sl_fits_to_super_fits.py    --sl_fits_file $D/run_sl_fits_cut.root --super_fits_file $D/run_super_fits.root --n_proc 8
python scripts/dt_root/apply_cuts.py               --input_file $D/run_super_fits.root --output_file $D/run_super_fits_cut.root \
       --cuts "impossible_super_fits,==,0;chi2/ndf_super_fits,<,20"
python scripts/dt_root/super_fits_to_dt_muons.py   --super_fits_file $D/run_super_fits_cut.root --sl_fits_file $D/run_sl_fits_cut.root --dt_muons_file $D/run_dt_muons.root
```

The super fit uses the fixed drift velocity of `params.py` by default; `--free_vd` makes it a fit parameter
(`--super_fit_free_vd` in `run_dt_pipeline.py`). The super fit result branches carry the suffix `_super_fits`
(e.g. `t0_super_fits`, `chi2/ndf_super_fits`). Another suffix can be chosen with `--suffix` in `sl_fits_to_super_fits.py`
and then has to be given to the later scripts as well (`--suffix`, in `run_dt_pipeline.py` `--super_fit_suffix`).

## Everything in one go

```
python scripts/dt_root/run_dt_pipeline.py --input_dumpfile /path/to/run.txt --output_dir $D --n_proc 8
```

With `--dt_tp_corrections_file /path/to/tp_corrections.root` the testpulse timing calibration is applied in stage `dt_hits`,
so `<prefix>_dt_hits.root` holds the corrected hits and all later stages use them.
The cuts of the two cut stages are set with `--sl_fit_cuts` and `--super_fit_cuts`.

Writes `<prefix>_<stage>.root` for every stage into `--output_dir` (`<prefix>` = dumpfile name without ending, or `--prefix`).
`--from_stage`, `--to_stage` and `--skip_stages` select a part of the chain, e.g. redo everything after the fits:

```
python scripts/dt_root/run_dt_pipeline.py --output_dir $D --prefix run --from_stage sl_fits_cut
```

## Testpulse timing calibration

A dumpfile recorded with simultaneous testpulses on all channels gives the time offset of every cell:

```
python scripts/dt_root/dumpfile_to_dt_tp_corrections.py --input_dumpfile /path/to/tp_run.txt --dt_tp_corrections_file $D/tp_corrections.root \
       --dt_tp_hits_file $D/tp_dt_hits.root       # optional: the testpulse hits, for the single cell plots
python scripts/dt_root/plot_dt_tp_corrections.py --dt_tp_corrections_file $D/tp_corrections.root --store_plots /path/to/plots/tp \
       --dt_tp_hits_file $D/tp_dt_hits.root --cells "1:0:10,2:3:40"
```

- All cells of the chamber are calibrated, also masked / dead ones (no dead time cut). Per cell, the time inside the
  orbit (`tdc + 32 bx`) is histogrammed with bins of 1 TU; the first peak above `--rel_thres` (default 0.2) of the
  highest bin is the testpulse response (later peaks are ringing), its position is the weighted mean.
- The known testpulse delays per frontend connector (`params._tp_time_offset`, e.g. theta testpulse latency, old cables)
  are subtracted (`--no_offset_correction` to switch off).
- `ts_corr = <mean of the chamber> - <testpulse time of the cell>` (`--alignment sl`: mean of the SL instead; then a
  time offset between the SLs remains). The mean is taken over the cells which are not masked / dead. Cells without
  testpulse peak get `ts_corr = 0` (`valid = 0`). The first `params._dumpfile_hits_to_skip` lines are skipped (`--n_lines_to_skip`).
- Output: tree `tree` with one row per cell (`sl, ly, wi, ts_corr, err_ts_corr, valid, masked, tp_ts_mean, tp_ts_err,
  tp_ts_mean_raw, tp_offset, n_hits, n_peak_hits, peak_ts_min, peak_ts_max, ...`), tree `summary`, and ROOT histograms
  `ts_corr_sl<N>`, `tp_ts_mean_sl<N>`, `n_peak_hits_sl<N>` (TH2D, wire vs layer) and `ts_orbit_sl<N>` (TH1D).
  Same numbers as the old `dt_testpulses.py` (which ignored the masked / dead cells).
- Applied with a plus sign: `ts -> ts + ts_corr`, `err_ts -> sqrt(err_ts² + err_ts_corr²)`, oc / bx / tdc recalculated.
  Hits of cells which are not in the calibration file are left uncorrected (with a warning).

Using it: `--dt_tp_corrections_file` in `dumpfile_to_dt_hits.py` / `run_dt_pipeline.py`; for an existing dt hits file
`dt_hits_timing_correction.py --dt_hits_file ... --dt_tp_corrections_file ... --corr_dt_hits_file ...`.
An old calibration pickle (`DT_CORRECTIONS.pcl`) works everywhere as well, or is converted with
`pcl_to_root.py --input_pcl_file DT_CORRECTIONS.pcl --output_file tp_corrections.root`.

## Simulation

```
python scripts/dt_root/sim_gen_cosmic_tracks.py        --cosmic_muons_file $D/sim_muons.root --duration_s 60 --seed 1
python scripts/dt_root/sim_cosmic_tracks_to_dt_hits.py --cosmic_muons_file $D/sim_muons.root --dt_hits_file $D/sim_clean_dt_hits.root --ts_noise_amplitude 3
# optional: random noise hits, secondary hits, dead cells
python scripts/dt_root/sim_add_dt_hit_noise.py         --dt_hits_file $D/sim_clean_dt_hits.root --dt_hits_file_with_noise $D/sim_noise_dt_hits.root --noise_rate_hz 15
python scripts/dt_root/sim_add_dt_secondary_hits.py    --dt_hits_file $D/sim_noise_dt_hits.root --dt_hits_file_with_secondaries $D/sim_sec_dt_hits.root
python scripts/dt_root/dt_hits_mask_cells.py           --dt_hits_file $D/sim_sec_dt_hits.root --masked_dt_hits_file $D/sim_dt_hits.root --cells "3:0:10,3:1:41"
# reconstruction: same chain as for data, starting from the dt hits file <prefix>_dt_hits.root
python scripts/dt_root/run_dt_pipeline.py --output_dir $D --prefix sim --from_stage hit_diff_hist
```

The simulated truth is carried through all files in the `muon_...` branches (`muon_id` >= 1 for simulated muons,
0 for noise and for data). In `dt_muons.root`, `muon_theta`, `muon_phi`, `muon_x0`, ... are the true track parameters
and `muon_id_mismatch` marks muons built from fits of different simulated muons. `singleplot_dt_muon.py --simulation`
draws the true track into the event display.

## Plots

All plotting scripts read the ROOT files directly. Give `--store_plots <directory>` to save the figures
(`--format pdf` for pdf instead of png) and/or `--show_plots` to open them in windows.

```
P=/path/to/plots
# histograms of all basic branches of any file of the workflow
python scripts/dt_root/plot_histograms.py --input_file $D/run_sl_fits.root --store_plots $P/sl_fits
python scripts/dt_root/plot_histograms.py --input_file $D/run_sl_fits.root --store_plots $P/sl_fits_good \
       --cuts "impossible,==,0;chi2/ndf,<,20" --branches "t0,x0,tan_alpha,chi2/ndf" --split_by sl

# dt hits: occupancy / rate maps, rate per wire, low and high occupancy cells, hit time difference
python scripts/dt_root/plot_dt_hits.py --dt_hits_file $D/run_dt_hits.root --hit_diff_hist_file $D/run_hit_diff_hist.root --store_plots $P/dt_hits

# sl fits: drift times, fit residuals, time between fits, rates
python scripts/dt_root/plot_sl_fits.py --sl_fits_file $D/run_sl_fits_cut.root --store_plots $P/sl_fits

# super fits: residuals, drift times, comparison with the two sl fits (T0, slope, position)
python scripts/dt_root/plot_super_fits.py --super_fits_file $D/run_super_fits_cut.root --store_plots $P/super_fits

# dt muons: x-y maps per superlayer, x-z / y-z projections, 3d view, angles, timing
python scripts/dt_root/plot_dt_muons.py --dt_muons_file $D/run_dt_muons.root --store_plots $P/dt_muons

# event display of single sl fits (timestamps with residuals, pattern cells with track)
python scripts/dt_root/singleplot_sl_fit.py --sl_fits_file $D/run_sl_fits.root --rows 500,600 --store_plots $P/single_fits
python scripts/dt_root/singleplot_sl_fit.py --sl_fits_file $D/run_sl_fits.root --n_fits 5 --cuts "impossible,==,0;chi2/ndf,<,2" --store_plots $P/single_fits

# event display of single super fits with their two sl fits (8 timestamps with residuals, cells of both phi sls with tracks)
python scripts/dt_root/singleplot_super_fit.py --super_fits_file $D/run_super_fits_cut.root --rows 0,10 --store_plots $P/single_super_fits

# event display of single dt muons (chamber views with hit cells, sl fits, super fit and global track)
python scripts/dt_root/singleplot_dt_muon.py --sl_fits_file $D/run_sl_fits_cut.root --super_fits_file $D/run_super_fits_cut.root \
       --dt_muons_file $D/run_dt_muons.root --rows 0,7 --store_plots $P/single_muons
```

- `plot_histograms.py` leaves out the branches of the not-selected lateralities (`lat0_...`), bookkeeping branches and
  branches with the same value in every row; `--all_branches` or `--branches` gets them. Whole-number branches get one
  bin per value, all others `--n_bins` bins over the central 99% of the values (`--full_range` for everything).
- "Rows" are the row numbers in the given file, the same numbers the `row_sl*`, `super_fit_row` and `sl*_fit_row` branches refer to.
- `plot_dt_muons.py` marks the cells of `params._dt_wire_mask` and `params._dt_dead_wires` in the maps; give other
  cells with `--mark_cells "sl:ly:wi,..."`. `plot_dt_hits.py` prints the low and high occupancy cells in that format.
- Shared code: `analysis_tools/utils/plot_utils.py` (histograms are drawn with `hist_utils`, geometry with `geoplot_utils`).

## Speed and log output

- `--n_proc N` runs the dumpfile conversion, the pattern search, the sl fits and the super fits on N processes
  (`run_dt_pipeline.py` passes it to all four). The results do not depend on N: the pattern search gives every piece of a superlayer the hits of
  the time window before it, so it finds exactly the patterns of the search on one process.
- Every step prints its progress as `chunk i / n` (dumpfile conversion: `block i / n`), so the total is known from
  the first line on.
- Log lines start with the time since the start of the script. In a terminal the time is shown in cyan and the step
  name (e.g. `[dt hits -> cell counts]`) in yellow. `export DT_SCINT_COLOR=1` forces the colours when the output
  is written to a file, `DT_SCINT_COLOR=0` switches them off.

## Parameter file

All scripts use `analysis_tools/params/params.py` unless another file is given with `--params_file`. Use the same
parameter file for all stages of one run. For the scripts outside `scripts/dt_root/`, which do not have this argument, set
`export DT_SCINT_PARAMS_FILE=/path/to/params_file.py` instead.

## Helpers

```
python scripts/dt_root/inspect_root_file.py --input_file $D/run_sl_fits.root --branches "sl,t0,chi2/ndf"
python scripts/dt_root/pcl_to_root.py --input_pcl_file sim_dt_hits.pcl --output_file sim_dt_hits.root --tree dt_hits --sort_key ts
```

`pcl_to_root.py` converts a .pcl data file of the older scripts into a ROOT file of this workflow.

## File format

- dt hits are in a tree called `dt_hits`, everything else in a tree called `tree`. Branch names are the keys of the
  data dicts used throughout `analysis_tools` (also `chi2/ndf`).
- Read a file in python with `root_utils.read_tree(path)` (returns `{key: np.ndarray}`) or in chunks with
  `root_utils.iterate_tree(path)`.
- `hit_diff_hist.root`: histogram object `hit_diff_hist` (TH1D) to draw directly in ROOT; tree `tree` has one row per
  bin (branch `hist` = bin content), tree `summary` holds entries / underflow / overflow. Drawing the branch `hist` of
  the tree in ROOT shows how often each bin content occurs, not the distribution: use the TH1D or
  `tree->Draw("hist:center")`.
  `cell_counts.root`: histogram object `cell_counts` (TH2D, x = wire, y = 4 * (sl - 1) + ly); tree `tree` has one row
  per cell, tree `summary` holds duration_seconds / ts_min / ts_max.
  Both scripts write the old `.pcl` format instead if the output file name ends with `.pcl`.
- Rows which point to other files: `super_fits.root` has `row_sl1`, `row_sl3` (rows of the two combined sl fits in the
  cut sl fits file). `dt_muons.root` has `super_fit_row` (row in the cut super fits file) and `sl1_fit_row`,
  `sl2_fit_row`, `sl3_fit_row` (rows in the cut sl fits file).
- `dt_muons.root`, other branches: `n_theta_candidates` = number of theta sl fits inside the time window of the super
  fit (more than 1: ambiguous match, the closest was taken), `delta_t0` = theta `t0` - super fit `t0`,
  `muon_id_mismatch` = 1 if the combined fits come from different simulated muons (simulation only).

## Chunks and the `chunk_id` branch

The dt hits are processed in chunks (`--step_size`, default 200 MB). The dead time cut, the pattern search and the
hit difference histogram do not look across the border between two chunks. Every pattern carries the `chunk_id` of
its hit chunk, and this branch is passed on to all later files. The stages which combine rows (super fits, dt muons)
only combine rows of the same `chunk_id`. This way the results depend only on the `--step_size` used for
`dt_hits_to_sl_patterns.py` and not on how the later scripts read their files; `--step_size` and `--n_proc` of the
later stages change speed and memory use only.

## Where the code is

- `analysis_tools/utils/dt_pipeline_utils.py`: one function per stage (the scripts only parse arguments).
- `analysis_tools/utils/root_utils.py`: reading and writing the ROOT files.
- `analysis_tools/utils/dt_utils.py`: the reconstruction functions (pattern search, sl fit, super patterns, super fit,
  `reco_muons_from_super_fits`).



# DT workflow with ROOT files

One script per stage. Every script does its whole stage itself: read the input file, call the reconstruction
functions of `analysis_tools/utils/`, write the output file. Every script takes its input and output files as
arguments; nothing is read from or written to a fixed location. Run `python <script> --help` for all options.
The trees and branches of every output file are listed in [OUTPUT_FILES.md](../OUTPUT_FILES.md).

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
python scripts/dumpfile_to_dt_hits.py      --input_dumpfile /path/to/run.txt --dt_hits_file $D/run_dt_hits.root
#   with testpulse timing calibration: add --dt_tp_corrections_file /path/to/tp_corrections.root

python scripts/dt_hits_to_hit_diff_hist.py --dt_hits_file $D/run_dt_hits.root --hit_diff_hist_file $D/run_hit_diff_hist.root
python scripts/dt_hits_to_cell_counts.py   --dt_hits_file $D/run_dt_hits.root --cell_counts_file $D/run_cell_counts.root
python scripts/dt_hits_to_sl_patterns.py   --dt_hits_file $D/run_dt_hits.root --sl_patterns_file $D/run_sl_patterns.root --n_proc 8
python scripts/sl_patterns_to_sl_fits.py   --sl_patterns_file $D/run_sl_patterns.root --sl_fits_file $D/run_sl_fits.root --n_proc 8
python scripts/apply_cuts.py               --input_file $D/run_sl_fits.root --output_file $D/run_sl_fits_cut.root --cuts "impossible,==,0;chi2/ndf,<,20"
python scripts/sl_fits_to_super_fits.py    --sl_fits_file $D/run_sl_fits_cut.root --super_fits_file $D/run_super_fits.root --n_proc 8
python scripts/apply_cuts.py               --input_file $D/run_super_fits.root --output_file $D/run_super_fits_cut.root \
       --cuts "impossible_super_fits,==,0;chi2/ndf_super_fits,<,20"
python scripts/super_fits_to_dt_muons.py   --super_fits_file $D/run_super_fits_cut.root --sl_fits_file $D/run_sl_fits_cut.root --dt_muons_file $D/run_dt_muons.root
```

The super fit uses the fixed drift velocity of `params.py` by default; `--free_vd` makes it a fit parameter
(`--super_fit_free_vd` in `run_dt_pipeline.py`). The super fit result branches carry the suffix `_super_fits`
(e.g. `t0_super_fits`, `chi2/ndf_super_fits`).

## Everything in one go

```
python scripts/run_dt_pipeline.py --input_dumpfile /path/to/run.txt --output_dir $D --n_proc 8
```

`run_dt_pipeline.py` runs the stage scripts above one after the other, with the same commands (each command is
printed before it runs, so a single stage can be repeated by hand).

With `--dt_tp_corrections_file /path/to/tp_corrections.root` the testpulse timing calibration is applied in stage `dt_hits`,
so `<prefix>_dt_hits.root` holds the corrected hits and all later stages use them.
The cuts of the two cut stages are set with `--sl_fit_cuts` and `--super_fit_cuts`.

Writes `<prefix>_<stage>.root` for every stage into `--output_dir` (`<prefix>` = dumpfile name without ending, or `--prefix`).
`--from_stage`, `--to_stage` and `--skip_stages` select a part of the chain, e.g. redo everything after the fits:

```
python scripts/run_dt_pipeline.py --output_dir $D --prefix run --from_stage sl_fits_cut
```

## Testpulse timing calibration

A dumpfile recorded with simultaneous testpulses on all channels gives the time offset of every cell:

```
python scripts/dumpfile_to_dt_tp_corrections.py --input_dumpfile /path/to/tp_run.txt --dt_tp_corrections_file $D/tp_corrections.root \
       --dt_tp_hits_file $D/tp_dt_hits.root       # optional: the testpulse hits, for the single cell plots
python scripts/plot_dt_tp_corrections.py --dt_tp_corrections_file $D/tp_corrections.root --store_plots /path/to/plots/tp \
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
- Output: one row per cell plus histograms, see [OUTPUT_FILES.md](../OUTPUT_FILES.md).
  Same numbers as the old `dt_testpulses.py` (which ignored the masked / dead cells).
- Applied with a plus sign: `ts -> ts + ts_corr`, `err_ts -> sqrt(err_ts² + err_ts_corr²)`, oc / bx / tdc recalculated.
  Hits of cells which are not in the calibration file are left uncorrected (with a warning).

Using it: `--dt_tp_corrections_file` in `dumpfile_to_dt_hits.py` / `run_dt_pipeline.py`; for an existing dt hits file
`dt_hits_timing_correction.py --dt_hits_file ... --dt_tp_corrections_file ... --corr_dt_hits_file ...`.
An old calibration pickle (`DT_CORRECTIONS.pcl`) works everywhere as well, or is converted with
`pcl_to_root.py --input_pcl_file DT_CORRECTIONS.pcl --output_file tp_corrections.root`.

## Simulation

```
python scripts/sim_gen_cosmic_tracks.py        --cosmic_muons_file $D/sim_muons.root --duration_s 60 --seed 1
python scripts/sim_cosmic_tracks_to_dt_hits.py --cosmic_muons_file $D/sim_muons.root --dt_hits_file $D/sim_clean_dt_hits.root --ts_noise_amplitude 3
# optional: random noise hits, secondary hits, dead cells
python scripts/sim_add_dt_hit_noise.py         --dt_hits_file $D/sim_clean_dt_hits.root --dt_hits_file_with_noise $D/sim_noise_dt_hits.root --noise_rate_hz 15
python scripts/sim_add_dt_secondary_hits.py    --dt_hits_file $D/sim_noise_dt_hits.root --dt_hits_file_with_secondaries $D/sim_sec_dt_hits.root
python scripts/dt_hits_mask_cells.py           --dt_hits_file $D/sim_sec_dt_hits.root --masked_dt_hits_file $D/sim_dt_hits.root --cells "3:0:10,3:1:41"
# reconstruction: same chain as for data, starting from the dt hits file <prefix>_dt_hits.root
python scripts/run_dt_pipeline.py --output_dir $D --prefix sim --from_stage hit_diff_hist
```

The simulated truth is carried through all files in the `sim_...` branches (`sim_id` >= 1 for simulated muons,
0 for noise and for data). In `dt_muons.root`, `sim_theta`, `sim_phi`, `sim_x0`, ... are the true track parameters
and `sim_id_mismatch` marks muons built from fits of different simulated muons. `singleplot_dt_muon.py --simulation`
draws the true track into the event display.

## Plots

All plotting scripts read the ROOT files directly. Give `--store_plots <directory>` to save the figures
(`--format pdf` for pdf instead of png) and/or `--show_plots` to open them in windows.

```
P=/path/to/plots
# histograms of all basic branches of any file of the workflow
python scripts/plot_histograms.py --input_file $D/run_sl_fits.root --store_plots $P/sl_fits
python scripts/plot_histograms.py --input_file $D/run_sl_fits.root --store_plots $P/sl_fits_good \
       --cuts "impossible,==,0;chi2/ndf,<,20" --branches "t0,x0,tan_alpha,chi2/ndf" --split_by sl

# dt hits: occupancy / rate maps, rate per wire, low and high occupancy cells, hit time difference
python scripts/plot_dt_hits.py --cell_counts_file $D/run_cell_counts.root --hit_diff_hist_file $D/run_hit_diff_hist.root --store_plots $P/dt_hits

# sl fits: drift times, fit residuals, time between fits, rates
python scripts/plot_sl_fits.py --sl_fits_file $D/run_sl_fits_cut.root --store_plots $P/sl_fits

# super fits: residuals, drift times, comparison with the two sl fits (T0, slope, position)
python scripts/plot_super_fits.py --super_fits_file $D/run_super_fits_cut.root --store_plots $P/super_fits

# dt muons: x-y maps per superlayer, x-z / y-z projections, 3d view, angles, timing
python scripts/plot_dt_muons.py --dt_muons_file $D/run_dt_muons.root --store_plots $P/dt_muons

# event display of single sl fits (timestamps with residuals, pattern cells with track)
python scripts/singleplot_sl_fit.py --sl_fits_file $D/run_sl_fits.root --rows 500,600 --store_plots $P/single_fits
python scripts/singleplot_sl_fit.py --sl_fits_file $D/run_sl_fits.root --n_fits 5 --cuts "impossible,==,0;chi2/ndf,<,2" --store_plots $P/single_fits

# event display of single super fits with their two sl fits (8 timestamps with residuals, cells of both phi sls with tracks)
python scripts/singleplot_super_fit.py --super_fits_file $D/run_super_fits_cut.root --rows 0,10 --store_plots $P/single_super_fits

# event display of single dt muons (chamber views with hit cells, sl fits, super fit and global track)
python scripts/singleplot_dt_muon.py --sl_fits_file $D/run_sl_fits_cut.root --super_fits_file $D/run_super_fits_cut.root \
       --dt_muons_file $D/run_dt_muons.root --rows 0,7 --store_plots $P/single_muons
```

- `plot_histograms.py` leaves out the branches of the not-selected lateralities (`lat0_...`), bookkeeping branches and
  branches with the same value in every row; `--all_branches` or `--branches` gets them. Whole-number branches get one
  bin per value, all others `--n_bins` bins over the central 99% of the values (`--full_range` for everything).
- "Rows" are the row numbers in the given file, the same numbers the `row_sl*`, `super_fit_row` and `sl*_fit_row` branches refer to.
- `plot_dt_muons.py` marks the cells of `params._dt_wire_mask` and `params._dt_dead_wires` in the maps; give other
  cells with `--mark_cells "sl:ly:wi,..."`. `plot_dt_hits.py` prints the low and high occupancy cells in that format.
- Shared code: `analysis_tools/utils/plot_utils.py`. One histogram as one plot file is one call:
  `plot_utils.plot_histogram(values, "name", args, xlabel=..., title=...)` (bins, bars with error bars, info box, saving).
  The chamber is drawn with `plot_utils.draw_chamber(ax, "phi" or "theta", cell_colors={(sl, ly, wi): colour})`.

## Chunks and parallel processing

Large files are not read at once, but in **chunks** of `--chunk_size` rows (the dumpfile: in blocks of `--block_lines` lines).
In the scripts this is always the same simple loop:

```python
n_rows = root_utils.number_of_rows(input_file, "tree")
n_chunks = (n_rows + chunk_size - 1) // chunk_size
for i_chunk in range(n_chunks):
    start = i_chunk * chunk_size
    stop = min(start + chunk_size, n_rows)
    rows = root_utils.read_tree(input_file, "tree", start, stop)
    ...                                                  # process the rows of this chunk
    root_utils.write_rows(output_file, "tree", result)   # appended to the output tree
```

- Sorting: the dt hits are sorted by time when they are written (within each block of the dumpfile). The pattern search
  sorts the hits of each superlayer by time again, the pairing of sl fits to super fits sorts by `t0`, the muon
  reconstruction sorts by `t0` and writes the muons sorted by time. So inside a chunk everything is processed in time
  order; only at the border between two chunks (or two dumpfile blocks) a few patterns / pairs can be lost.
- The dead time cut and the pattern search only see the hits of one chunk of dt hits (`--chunk_size` of
  `dt_hits_to_sl_patterns.py`, default 1,000,000 hits): patterns made of hits of two chunks are not found. Every
  pattern gets the number of its chunk in the branch `chunk_id`, and all later files keep it. The super fits and the
  dt muons only combine rows with the same `chunk_id`, so the results depend on the chunk size of the pattern search only.
- **Parallel processing** (`--n_proc N` of `dt_hits_to_sl_patterns.py`, `sl_patterns_to_sl_fits.py`,
  `sl_fits_to_super_fits.py`): the chunks are independent of each other, so N chunks are processed at the same time,
  each one in its own process (python `multiprocessing.Pool`). The results come back in the order of the chunks and
  are written one after the other; they are the same as with `--n_proc 1`. A file with only one chunk is processed
  by one process (e.g. a short run in `sl_fits_to_super_fits.py`, which works per `chunk_id`).

## Log output

- Every step prints its progress as `chunk i / n` (dumpfile conversion: `block i / n`).
- Log lines start with the time since the start of the script. In a terminal the time is shown in cyan, the step
  name (e.g. `[dt hits -> cell counts]`) in yellow and the progress counter in magenta. `export DT_SCINT_COLOR=1`
  forces the colours when the output is written to a file, `DT_SCINT_COLOR=0` switches them off.

## Parameter file

All scripts use `analysis_tools/params/params.py` unless another file is given with `--params_file` (or
`export DT_SCINT_PARAMS_FILE=/path/to/params_file.py`). Use the same parameter file for all stages of one run.

## Chamber geometry

`params._dt_chamber` describes the chamber (mm): per superlayer the orientation (`phi`: wires along y, measures x;
`theta`: wires along x, measures y), the size of a cell, and per layer the wire range and the lower corner of the
cell of wire 0 (`cell_0`). The cells of a layer follow each other along the measured axis. `pos` / `size` of the
chamber and the superlayers are only used for drawing. A different chamber type or cell layout only needs a new
`_dt_chamber` (and the readout mapping) in a parameter file.

`analysis_tools/utils/dt_chamber_utils.py` builds everything else from it (cells, wire positions, ...). Its header
describes all coordinate systems with sketches: the chamber frame, the track frame of the fits (origin at the wire of
layer 3 of the pattern) and the muon parameters.

## Helpers

```
python scripts/inspect_root_file.py --input_file $D/run_sl_fits.root --branches "sl,t0,chi2/ndf"
python scripts/pcl_to_root.py --input_pcl_file sim_dt_hits.pcl --output_file sim_dt_hits.root --tree dt_hits --sort_key ts
```

`pcl_to_root.py` converts a .pcl data file of the older scripts into a ROOT file of this workflow.

## File format

All files, trees, branches and histograms: [OUTPUT_FILES.md](../OUTPUT_FILES.md). Read a file in python with
`root_utils.read_tree(path, "tree")` (returns `{branch name: numpy array}`; dt hits: tree `"dt_hits"`).

## Where the code is

`scripts/`: one script per stage (read, process, write) and the plotting scripts.

`analysis_tools/utils/`:
- `dt_dumpfile_utils.py`: reading the dumpfile block by block (the words are decoded with `data_utils.import_raw_lines`)
- `timestamp_utils.py`: timestamps from tdc / bx / orbit counter with orbit counter overflows (`add_timestamp`), back to
  oc / bx / tdc (`remap_htg_timestamp`), time inside the orbit, sorting by time
- `dt_calibration_utils.py`: applying a testpulse calibration, calibration from a testpulse run
- `dt_hit_utils.py`: dt hits from the data words (`extract_dt_hits`), dead time cut, time between hits of a cell, hits per cell
- `dt_pattern_utils.py`: pattern search
- `dt_fit_utils.py`: sl fits, pairing of the phi sl fits to super patterns, super fits (fit model, fit, best laterality)
- `dt_muon_reco_utils.py`: dt muons from super fits and theta sl fits
- `dt_sim_utils.py`, `muon_utils.py`: simulated muons, hits, noise and secondary hits
- `dt_chamber_utils.py`: superlayers / layers / wires, readout channel lookup, cell positions, coordinate systems, tracks
- `root_utils.py`: reading and writing ROOT files, `log`
- `plot_utils.py`: one-call histogram plots, drawing the chamber, figure output
- `hist_utils.py`: histogram calculation and drawing, peak finding (used by `plot_utils` and the testpulse calibration)
- `data_utils.py`: reading raw dumpfile words, cuts (`cut_data`, `parse_cuts`), sorting, merging, pickle files

`analysis_tools/params/`: `params.py` (all settings), `derived_params.py` (readout lookup tables, unit conversions).

________________________________________________________________________________________________________

# DT workflow

## Cosmic track simulation
Cosmic muons have muon_id >= 1.

Generate:
python scripts/sim/gen_cosmic_tracks.py --cosmic_muons_file data_files/sim_muons_fulldt.pcl

Plot:
python scripts/sim/plot_cosmic_tracks.py --show_plots --cosmic_muons_file data_files/sim_muons_fulldt.pcl

## Cosmic tracks -> Simulated DT hits

python scripts/sim/cosmic_tracks_to_dt_hits.py --cosmic_muons_file data_files/sim_muons_fulldt.pcl --dt_hits_file data_files/sim_muons_dt_hits.pcl

One can add noise and mis-calibration to the timestamps:
...

One can add more features to the hits:
Seconday hits (2nd hit in same cell in given time interval after 1st hit with given probability):
...
Noise hits (Poisson distributed with given rate):
...

## Dumpfile -> DT hits
Also apply dead time to DT hits of same channel.
Data has muon_id = 0.

python scripts/dt/dumpfile_to_dt_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/dt_cosm_7.txt --dt_hits_file data_files/dt_cosm_7_hits.pcl

Plot:
python scripts/dt/plot_dt_hits.py --show_plots --dt_hits_file data_files/ddt_cosm_7_hits.pcl

### DT hits, apply timing calibration
Apply testpulse-derived timing calibration to hit timestamps.

python scripts/dt/dt_hits_timing_correction.py --dt_hits_file thesis_data_files/dt_data_run/dt_hits_nodeadtime.pcl --dt_tp_corrections_file thesis_data_files/dt_testpulse_run/dt_tp_corrections.pcl --corr_dt_hits_file thesis_data_files/dt_data_run/dt_hits_nodeadtime_corr.pcl

## DT hits -> SL patterns
For data can apply testpulse timing correction file.
Do not write argument if no correction is asked (e.g. for simulation).

python scripts/dt/dt_hits_to_sl_patterns.py --dt_hits_file data_files/dt_cosm_7_hits.pcl --sl_patterns_file data_files/dt_cosm_7_patterns.pcl --dt_tp_corrections_file data_files/dt_tp_corrections_6.pcl

For simulation (to match only hits of similar muon_id):
python scripts/dt/dt_hits_to_sl_patterns.py --dt_hits_file data_files/sim_muons_dt_hits.pcl --sl_patterns_file data_files/sim_muons_sl_fits_realmuons_noparambounds.pcl --simulation_only_muon_patterns

Plot:
python scripts/dt/plot_sl_patterns.py --show_plots --sl_patterns_file data_files/dt_cosm_7_patterns.pcl
Plot for simulation:
python scripts/dt/plot_sl_patterns.py --show_plots --sl_patterns_file data_files/sim_muons_sl_patterns_realmuons.pcl --simulation

## SL patterns -> SL fits

python scripts/dt/sl_patterns_to_sl_fits.py --sl_patterns_file data_files/dt_cosm_7_patterns.pcl --sl_fits_file data_files/dt_cosm_7_fits.pcl

Apply cuts to reject bad or unphysical fits:
python scripts/general/apply_cuts.py --input_data_file data_files/sim_muons_sl_fits_realmuons_parambounds.pcl --cut_data_file data_files/sim_muons_sl_fits_realmuons_parambounds_aftercuts.pcl --cuts "chi2/ndf,<,1;x0,<=,21;x0,>=,-21;dt0,>=,0;dt0,<=,params._dt_max_drift_time;dt1,>=,0;dt1,<=,params._dt_max_drift_time;dt2,>=,0;dt2,<=,params._dt_max_drift_time;dt3,>=,0;dt3,<=,params._dt_max_drift_time"

Plot:
python scripts/dt/plot_sl_fits.py --show_plots --sl_fits_file data_files/dt_cosm_7_fits.pcl
Plot for simulation:
python scripts/dt/plot_sl_fits.py --show_plots --sl_fits_file data_files/sim_muons_sl_fits_realmuons_noparambounds.pcl --simulation

## SL fits -> SL fit groups (per SL)

python scripts/dt/sl_fits_to_sl_fit_groups.py --sl_fits_file data_files/combined_2_sl_fits_slaligned_aftercuts.pcl --sl_fit_groups_file data_files/combined_2_sl_fit_groups.pcl

Plot:
...

## SL fit groups (per SL) -> DT muons

python scripts/dt/sl_fit_groups_to_dt_muons.py --sl_fits_file thesis_data_files/dt_data_run/sl_fits_aftercuts.pcl --sl_fit_groups_file thesis_data_files/dt_data_run/sl_fit_groups.pcl --dt_muons_file thesis_data_files/dt_data_run/dt_muons.pcl

Plot:
python scripts/dt/plot_dt_muons.py --show_plots --dt_muons_file data_files/dt_cosm_7_dt_muons.pcl
Plot for simulation:
python scripts/dt/plot_dt_muons.py --show_plots --dt_muons_file data_files/sim_muons_dt_muons.pcl --simulation


________________________________________________________________________________________________________

# Scintillator workflow

## Dumpfile -> Raw scint hits
Single SiPM hits, if no on-FPGA coincidence.
Also apply dead time to SiPM hits of same channel.
Data has muon_id = 0.

python scripts/scint/dumpfile_to_raw_scint_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/sipm_cosmics_40.txt --raw_scint_hits_file data_files/sipm_cosm_40_raw_hits.pcl

Plot:
python scripts/scint/plot_raw_scint_hits.py --raw_scint_hits_file data_files/sipm_cosm_40_raw_hits.pcl --show_plots

## Raw scint hits -> Scint hits

python scripts/scint/raw_scint_hits_to_scint_hits.py --raw_scint_hits_file data_files/sipm_cosm_41_raw_hits.pcl --scint_hits_file data_files/sipm_cosm_41_hits.pcl

## Dumpfile -> Scint hits
Scintillator strip hits, if active on-FPGA coincidence.
Data has muon_id = 0.

python scripts/scint/dumpfile_to_scint_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/sipm_cosmics_43.txt --scint_hits_file data_files/sipm_cosm_43_hits.pcl

Plot:
python scripts/scint/plot_scint_hits.py --scint_hits_file data_files/sipm_cosm_44_hits.pcl --show_plots

## Scint hits -> Scint areas
Scintillator pixel, offline coincidence of crossing scintillator strips.

python scripts/scint/scint_hits_to_areas.py --scint_hits_file data_files/sipm_cosm_44_hits.pcl --scint_areas_file data_files/sipm_cosm_44_areas.pcl

Plot:
python scripts/scint/plot_scint_areas.py --scint_areas_file data_files/sipm_cosm_44_areas.pcl --show_plots

________________________________________________________________________________________________________

# Combined workflow

## Dumpfile -> DT hits + Raw scint hits

python scripts/combined/dumpfile_to_dt_and_raw_scint_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/combined_2.txt --dt_hits_file data_files/combined_2_dt_hits.pcl --raw_scint_hits_file data_files/combined_2_raw_scint_hits.pcl

Raw scint hits: Do next steps as above.
- Raw scint hits -> Scint hits

## Dumpfile -> DT hits + Scint hits

DT hits: Do next steps as above.
- DT hits -> SL patterns
- SL patterns -> SL fits (and apply cuts)
- SL fits -> SL fit groups
- SL fit groups -> DT muons

Scint hits: Do next steps as above.
- Scint hits -> Scint areas

## Correlate DT muons + Scint areas

python scripts/combined/correlate_dt_muons_and_scint_areas.py --dt_muons_file data_files/combined_2_dt_muons.pcl --scint_areas_file data_files/combined_2_scint_areas.pcl

________________________________________________________________________________________________________


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

One can add features to the hits:
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

## DT hits -> SL patterns
For data can apply testpulse timing correction file.
Do not write argument if no correction is asked.

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

## SL fits -> DT muons

python scripts/dt/sl_fits_to_dt_muons.py --sl_fits_file data_files/dt_cosm_7_fits.pcl --dt_muons_file data_files/dt_cosm_7_dt_muons.pcl

Plot:
python scripts/dt/plot_dt_muons.py --show_plots --dt_muons_file data_files/dt_cosm_7_dt_muons.pcl
Plot for simulation:
python scripts/dt/plot_dt_muons.py --show_plots --dt_muons_file data_files/sim_muons_dt_muons.pcl --simulation

## DT muons -> Acceptance corrected DT muons
Correct angular acceptance for different muon angles.
Use correction extracted from simulated dataset for this.

...

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



Plot:


## Dumpfile -> Scint hits
Scintillator strip hits, if active on-FPGA coincidence.
Data has muon_id = 0.


Plot:


## Scint hits -> Scint areas
Scintillator pixel, offline coincidence of crossing scintillator strips.


Plot:

________________________________________________________________________________________________________

# Combined workflow

...

________________________________________________________________________________________________________


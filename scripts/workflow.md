> Note: the DT chain, the DT simulation and the DT plots run on ROOT files, see [dt_root/README.md](dt_root/README.md).
> This file describes the parts which still work on .pcl files: scintillator, DT-scintillator correlation and testpulses.
> They have not been ported to ROOT files yet. The DT steps named below ("DT hits -> SL patterns" etc.) refer to the
> ROOT workflow; its files can not be read directly by the scripts in `scripts/combined/`.

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

________________________________________________________________________________________________________

# Testpulse workflow

## DT timing calibration
Creates the timing corrections per wire (`DT_CORRECTIONS.pcl`) used by `scripts/dt_root/dt_hits_timing_correction.py`.

python scripts/testpulses/dt_testpulses.py --inputfile /path/to/testpulse_dumpfile.txt --dt_tp_timing_file /path/to/dt_tp_timing.pcl --dt_tp_corrections_file /path/to/DT_CORRECTIONS.pcl

Plot:
python scripts/testpulses/plot_dt_testpulses.py --help

## Scintillator timing calibration
python scripts/testpulses/scint_calib_testpulses.py --inputfile /path/to/testpulse_dumpfile.txt --create_calib --calib_dir /path/to/calibration_files

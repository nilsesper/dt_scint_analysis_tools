# dt_scint_analysis_tools
Standalone python code to analyze DT chamber testpulse and cosmics data recorded triggerless with Phase-2 electronics
(OBDT boards, slow control and timing box / HTG box, dumpfile format).  
Developed for [Nils' Master's thesis](https://github.com/nilsesper/masters-thesis), extended during [Justus' Bachelor's thesis](https://github.com/JustusT04/Bachelorarbeit), and now merged by Nils with the help of AI tools.  
Integrates the CMS DT MB1 chamber geometry with a simplified triggerless track reconstruction algorithm.  
(The SiPM scintillator analysis and combination of DT tracks and scintillator hits, which was used in Nils' Master's thesis is not part of the current version repository, and may be added in the future.)  

## Workflow

**raw dumpfile -> dt hits -> sl patterns -> sl fits -> cut sl fits -> super fits of the two phi superlayers -> cut super fits -> dt muons (super fit + theta sl fit)**  
**plus testpulse timing calibration, simplified simulation, and plotting utilities**  
One script per step in `scripts/`; every script takes its input and output files as arguments.  
Commands and code overview: [scripts/README.md](scripts/README.md); trees and branches of all output files:  
[OUTPUT_FILES.md](OUTPUT_FILES.md); coordinate systems: header of `analysis_tools/utils/dt_chamber_utils.py`.  

## Setup

- miniconda environment: `conda_env.yml` (`update_conda_env.sh`, `export_conda_env.sh`)
- `source env.sh` activates the environment and puts the repository on the `PYTHONPATH` (or `export PYTHONPATH=$PWD` in the repository directory)
- required python packages: numpy, scipy, matplotlib, tqdm, uproot, awkward

## Structure

- `analysis_tools/params` holds the parameters:
    -`params.py`: all settings (data format, chamber geometry, readout mapping, calibration, reconstruction, simulation, branches, plotting)
    - `derived_params.py`: lookup tables and unit conversions
- `analysis_tools/utils/` holds the code base:
    - one file per topic: dumpfile decoding, calibration, hits, pattern search, fits, muons, chamber, ROOT files, plotting
- `scripts/` holds the scripts to be executed by the user:
    - one script per workflow stage: read the input file, process, write the output file
    - also contains the plotting scripts

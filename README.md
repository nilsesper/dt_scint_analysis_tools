# dt_scint_analysis_tools
Standalone python code to analyze DT chamber testpulse and cosmics data, and SiPM-based scintillator detector with ZYNQ / lpGBT readout.  
Supports dumpfile data format recorded with Phase-2 electronics and slow control and timing box / HTG box.  
Developed for [Nils' Master's thesis](https://github.com/nilsesper/masters-thesis).  
Integrates CMS DT MB1 chamber geometry, and simplified triggerless reconstruction algorithm.  


## Workflows

- **DT workflow with ROOT files (current):** raw dumpfile -> dt hits -> sl patterns -> sl fits -> sl refits / super fits /
  sl fit groups -> dt muons, one script per stage in `scripts/dt_root/`. All scripts take their input and output files as
  arguments. Commands: [scripts/dt_root/README.md](scripts/dt_root/README.md).
- **Older workflow on .pcl files:** [scripts/workflow.md](scripts/workflow.md) (`scripts/dt/`, `scripts/scint/`,
  `scripts/combined/`, `scripts/sim/`). Simulation, scintillator, testpulse calibration and the plotting scripts still use it.

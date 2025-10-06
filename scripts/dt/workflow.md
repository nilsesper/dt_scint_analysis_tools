# DT reconstruction

## Dumpfile

## DT hits
```
python scripts/dt/dumpfile_to_dt_hits.py --input_dumpfile ~/masterarbeit/zynq_read-out_software/output/dt_cosm_1_split/xaa --dt_hits_file data_files/dt_cosm_1_xaa.pcl
```
Plotting:
```
python scripts/dt/plot_dt_hits.py --dt_hits_file data_files/dt_cosm_1_xaa_hits.pcl --sl_fits_file data_files/dt_cosm_1_xaa_fits.pcl
```

## SL fits
```
python scripts/dt/dt_hits_to_sl_fits.py --dt_hits_file data_files/dt_cosm_1_xaa_hits.pcl --sl_fits_file data_files/dt_cosm_1_xaa_fits.pcl
```
Plotting:
```
```





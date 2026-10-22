###########################################
### TRACK FITS: holds the correction functions for the dt reconstruction
###########################################

import numpy as np

from analysis_tools.params import params, derived_params

### timing correction function for signal propagation along the wire
def ts_sig_prop_corr(x_wire): # x_wire in mm
    # ts_corr = delta_x / v_sig_prop , delta_x = h * tan_alpha
    ts_corr = x_wire * 1e-3 / params.v_sig_prop * 1e9 / 0.78 # convert to TS units
    return ts_corr

### timing correction function for track fit angle
def ts_fit_angle_corr(tan_alpha): # tan alpha of track fit
    # ts_corr was measured empirically using perpendicular sl as reference
    # was fitted to parabola with parameters a0, a1, a2
    ts_corr = (params.angle_corr_a0 + params.angle_corr_a1 * tan_alpha + params.angle_corr_a2 * tan_alpha**2) / 0.78 # convert to ts units
    return ts_corr


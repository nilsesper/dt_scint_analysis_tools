#################################################################
### angular acceptance of patterns
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

# ++++ copied from analysis code ++++

### dt hit patterns per superlayer
# reference is on top (highest z coordinate i.e. ly 3)
# higher wi index towards right -->
# ly  [+A]     ref              [-A]     ref         
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | - | O | - | -     - | - | O | - | - | -
# 1   | - | - | O | - | - |     | - | - | O | - | - |
# 0   - | - | - | O | - | -     - | - | O | - | - | -
# possible lateralities (l=-1, r=+1) ly 3-0:
# llrl rlrl llrr rlrr           rrlr lrlr rrll lrll
# lateralities ly 0-3:
# lrll lrlr rrll rrlr           rlrr rlrl llrr llrl
#
# ly  [+B]     ref              [-B]     ref         
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | O | - | - | -     - | - | - | O | - | -
# 1   | - | - | O | - | - |     | - | - | O | - | - |
# 0   - | - | - | O | - | -     - | - | O | - | - | -
# possible lateralities (l=-1, r=+1) ly 3-0:
# lrrr lrrl lrll                rlll rllr rlrr
# lateralities ly 0-3:
# rrrl lrrl llrl                lllr rllr rrlr
# 
# ly  [+C]     ref              [-C]     ref         
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | - | O | - | -     - | - | O | - | - | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | - | O | - | -     - | - | O | - | - | -
# possible lateralities (l=-1, r=+1) ly 3-0:
# lllr rllr rrlr                rrrl lrrl llrl
# lateralities ly 0-3:
# rlll rllr rlrr                lrrr lrrl lrll

cell_width = 42 # mm
cell_height = 13 # mm
rad_to_deg = 180/np.pi
# tan(alpha) = delta_x / delta_z, alpha > 0 for track towards bottom right
# alpha_min: smallest angle
# alpha_max: largest angle
# measured w.r.t. muon passing at wire height in cell
patterns = {
    "+a": { # in units of cell_width (for x) and cell_height (for z)
        "1_delta_x": 0.5, # delta_x for alpha bound 1
        "1_delta_z": 1, # delta_z for alpha bound 1
        "2_delta_x": -0.5, # delta_x for alpha bound 2
        "2_delta_z": 3, # delta_z for alpha bound 2
    },
    "-a": {
        "1_delta_x": 0.5,
        "1_delta_z": 3,
        "2_delta_x": -0.5,
        "2_delta_z": 1,
    },
    "+b": {
        "1_delta_x": 0.5,
        "1_delta_z": 1,
        "2_delta_x": 0,
        "2_delta_z": 4,
    },
    "-b": {
        "1_delta_x": 0,
        "1_delta_z": 4,
        "2_delta_x": -0.5,
        "2_delta_z": 1,
    },
    "+c": {
        "1_delta_x": 0.5,
        "1_delta_z": 1,
        "2_delta_x": 0,
        "2_delta_z": 4,
    },
    "-c": {
        "1_delta_x": 0,
        "1_delta_z": 4,
        "2_delta_x": -0.5,
        "2_delta_z": 1,
    },
}

for pat_idx, (pat_type, pat) in enumerate(patterns.items()):
    # calculate
    tan_alpha_1 = -(pat["1_delta_x"]*cell_width) / (pat["1_delta_z"]*cell_height)
    tan_alpha_2 = -(pat["2_delta_x"]*cell_width) / (pat["2_delta_z"]*cell_height)
    tan_alpha_min = np.amin( [tan_alpha_1, tan_alpha_2] )
    tan_alpha_max = np.amax( [tan_alpha_1, tan_alpha_2] )
    alpha_min = np.arctan( tan_alpha_min )
    alpha_max = np.arctan( tan_alpha_max )
    # print
    print(f"pattern {pat_idx} ({pat_type}):")
    print(f"  tan_alpha_min = {tan_alpha_min}")
    print(f"  tan_alpha_max = {tan_alpha_max}")
    print(f"  alpha_min = {alpha_min} rad = {alpha_min*rad_to_deg} deg")
    print(f"  alpha_max = {alpha_max} rad = {alpha_max*rad_to_deg} deg")














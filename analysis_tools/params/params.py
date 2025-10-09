###############################
### CONSTANTS & PARAMETERS
###############################

import numpy as np
import matplotlib as mpl

# -----------------------------------------

### htg box data
# dumpfile data field masks
_htg_shifted_mask = {
    "ch":                                                            0b0000000000000000000000000000000000000000000000000000000011111111, # bits 7:0 [8]
    "bx":                                                0b0000000000000000000000000000000000000000000011111111111100000000, # bits 19:8 [12]
    "tdc":                                          0b0000000000000000000000000000000000000001111100000000000000000000, # bits 24:20 [5]
    "oc":                    #0b111111111111111100000000000000000000000000000000, # bits 47:32 [16]
    0b0000001111111111111111111111111100000000000000000000000000000000, # bits 57:32 [26]
    "ro_ch": 0b1111110000000000000000000000000000000000000000000000000000000000, # bits 63:58 [6]

    # for 16 ch htg: 0b1111110000000000000000000000000000000000000000000000000000000000
    # for 4 ch htg:  0b111110000000000000000000000000000000000000000000000000000000000
}
"""
_htg_mask = {
    "ch": 0b11111111, # 8 bit
    "bx": 0b111111111111,  # 12 bit
    "tdc": 0b11111, # 5 bit
    "oc": 0b1111111111111111, # 16 bit
    "ro_ch": 0b111111, # 6 bit
}
"""
# no of ro_chs in htg firmware
_htg_n_ro_chs = 28
# dumpfile data fields bitshift
_htg_bitshift = {
    "ch": 0,
    "bx": 8,
    "tdc": 20,
    "oc": 32,
    "ro_ch": 58, # 4 ch htg: 60, 16 ch htg: 58 
}
# htg data keys & data types
_htg_keys = {
    "ch": np.uint8,
    "bx": np.uint16,
    "tdc": np.uint8,
    "oc": np.uint32,
    "ro_ch": np.uint8,
}

### dumpfile import
# no of hits to skip in dumpfile since expect them to be old hits still in htg buffer
_dumpfile_hits_to_skip = 50000

### timestamp conversion
_lhc_tdc_count = 32 # max value of TDC + 1  (i.e. conversion factor: _lhc_tdc_count TDC = 1 BX)
_lhc_bunch_count = 3564 # max value of BX + 1 (i.e. conversion factor: _lhc_bunch_count BX = 1 ORBIT)
#_lhc_orbit_count = 65536 # 2^16, max value of ORBIT + 1
_lhc_orbit_count = 2**26 # = 67108864 = 2^26, max value of ORBIT + 1
_ts_type = np.uint64 # data type of timestamp field in hits
_ts_float_type = np.float64 # for fitting, ts type as float

_oc_difference_for_overflow = 50000 # difference between oc and last oc for oc overflow to be triggered

### dt specific
# fe conn idx list (key "fe_id")
_fe_idx_list = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "6A", "6B", "7A", "7B", "8A", "8B", "9A", "9B", "10A", "10B", "11A", "11B", "12A", "12B", "13A", "13B", "14A", "14B"] # idx -> fe conn str

### dt hit format:
### - htg origin: {"oc", "bx", "tdc", "ch"}
### - added dt mapping: {"sl": 1-3 superlayer, "ly": 0-3 layer in sl, "wi": 0-XX wire in layer, "conn_id": idx of fe_conn_name "J35" in fe_mapping dict, "fe_id": _fe_idx_list[fe] for fe name "1A" etc., "ch_id": 0-15 for each fec}
### - added timestamp: {"ts": converted timestamp from oc,bx,tdc}
_dt_mapping_keys = { # {key: dtype}
    "sl": np.uint8,
    "ly": np.uint8,
    "wi": np.uint8,
    "conn_id": np.uint8,
    "fe_id": np.uint8,
    "ch_id": np.uint8,
}
_dt_other_keys = {
    "ts": _ts_type,
    "muon_ts": _ts_type,
    "dt": np.uint16, # drift time (in ts units)
    "dd": np.float64, # drift distance (in mm)
    "muon_id": np.uint64, # id / idx of correlated muon
    "hit_lat": np.int8, # hit laterality -1 (l) left of wire, +1 (r) right of wire
}

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
# 
# ly  [+D]     ref              [-D]     ref         
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | - | O | - | -     - | - | O | - | - | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | - | - | O | -     - | O | - | - | - | -
# possible lateralities (l=-1, r=+1) ly 3-0:
# llll rrrr llrr rrll lllr rrrl      rrrr llll rrll llrr rrrl lllr
# lateralities ly 0-3:
# llll rrrr rrll llrr rlll lrrr      rrrr llll llrr rrll lrrr rlll
# 
# (!) depends on ly_indent = _dt_chamber["sls"][sl]["ly_indent"]
# with ly_indent = [True, False, True, False] (order is ly 0-3) we have: (z axis goes upwards)
# ly  rel_wi   -2  -1   0   1         
# 3           | - | - | O | - | - 
# 2           - | - | - | - | - | -  
# 1           | - | - | - | - | - 
# 0           - | - | - | - | - | -  
#     rel_wi     -2  -1   0   1   
#
# use layer 3 (top layer) as reference where relative wire index is fixed to 0
_dt_sl_patterns = { # pat_type key in sl patterns is idx of key, i.e. "+a"=0, "-a"=1 etc.
    # order in lists: ly 0,1,2,3
    "+a": {
        "rel_wis": [0,0,0,0], # list of relative wire index of layers 0-3
        # laterality ly 0-3: lrll lrlr rrll rrlr
        "laterality": ([-1,1,-1,-1], [-1,1,-1,1], [1,1,-1,-1], [1,1,-1,1], ) # list of possible lateralities (muon left l=-1 / right r=+1 of wire) for this pattern for layers 0-3
    },
    "-a":{
        "rel_wis": [-1,0,-1,0],
        # laterality ly 0-3: rlrr rlrl llrr llrl
        "laterality": ([1,-1,1,1], [1,-1,1,-1], [-1,-1,1,1], [-1,-1,1,-1], )
    },
    "+b": {
        "rel_wis": [0,0,-1,0],
        # laterality ly 0-3: rrrl lrrl llrl     # before: lllr rllr rrlr
        "laterality": ([1,1,1,-1], [-1,1,1,-1], [-1,-1,1,-1], )     # before: "laterality": ([-1,-1,-1,1], [1,-1,-1,1], [1,1,-1,1], )
    },
    "-b": {
        "rel_wis": [-1,0,0,0],
        # laterality ly 0-3: lllr rllr rrlr     # before: rrrl lrrl llrl
        "laterality": ([-1,-1,-1,1], [1,-1,-1,1], [1,1,-1,1], )     # before: "laterality": ([1,1,1,-1], [-1,1,1,-1], [-1,-1,1,-1], )
    },
    "+c": {
        "rel_wis": [0,1,0,0],
        # laterality ly 0-3: rlll rllr rlrr
        "laterality": ([1,-1,-1,-1], [1,-1,-1,1], [1,-1,1,1], )
    },
    "-c": {
        "rel_wis": [-1,-1,-1,0],
        # laterality ly 0-3: lrrr lrrl lrll
        "laterality": ([-1,1,1,1], [-1,1,1,-1], [-1,1,-1,-1], )
    },
    ### FOR NOW REJECT "OUTER" +-d PATTERNS: problems due to rrll llrr ambiguity...
    #"+d": {
    #    "rel_wis": [1,1,0,0],
    #    # laterality ly 0-3: llll rrrr rrll llrr rlll lrrr lllr rrrl
    #    "laterality": ([-1,-1,-1,-1], [1,1,1,1], [1,1,-1,-1], [-1,-1,1,1], [1,-1,-1,-1], [-1,1,1,1],) #[-1,-1,-1,1], [1,1,1,-1] )
    #},
    #"-d": {
    #    "rel_wis": [-2,-1,-1,0],
    #    # laterality ly 0-3: rrrr llll llrr rrll lrrr rlll lllr rrrl
    #    "laterality": ([1,1,1,1], [-1,-1,-1,-1], [1,1,-1,-1], [-1,-1,1,1], [-1,1,1,1], [1,-1,-1,-1],) #[-1,-1,-1,1], [1,1,1,-1] )
    #},
}

### dt fake hit patterns (to check for number of noise-induced patterns)
# reference is on top (highest z coordinate i.e. ly 3)
# higher wi index towards right -->
# ly  [+fA]    ref              [-fA]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | O | - | - | - | -     - | - | - | - | O | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | O | - | - | -     - | - | - | O | - | -
#
# ly  [+fB]    ref              [-fB]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | O | - | - | -     - | - | - | O | - | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | - | O | - | -     - | - | O | - | - | -
#
# ly  [+fC]    ref              [-fC]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | O | - | - | - | -     - | - | - | - | O | -
# 1   | - | - | O | - | - |     | - | - | O | - | - |
# 0   - | O | - | - | - | -     - | - | - | - | O | -
#
# (!) depends on ly_indent = _dt_chamber["sls"][sl]["ly_indent"]
# with ly_indent = [True, False, True, False] (order is ly 0-3) we have: (z axis goes upwards)
# ly  rel_wi   -2  -1   0   1         
# 3           | - | - | O | - | - 
# 2           - | - | - | - | - | -  
# 1           | - | - | - | - | - 
# 0           - | - | - | - | - | -  
#     rel_wi     -2  -1   0   1   
#
# use layer 3 (top layer) as reference where relative wire index is fixed to 0
_dt_sl_fake_patterns = { # pat_type key in sl patterns is idx of key, i.e. "+a"=0, "-a"=1 etc.
    # order in lists: ly 0,1,2,3
    "+fa": {
        "rel_wis": [-1,0,-2,0], # list of relative wire index of layers 0-3
    },
    "-fa":{
        "rel_wis": [0,-1,1,0],
    },
    "+fb": {
        "rel_wis": [0,1,-1,0],
    },
    "-fb": {
        "rel_wis": [-1,-1,0,0],
    },
    "+fc": {
        "rel_wis": [-2,0,-2,0],
    },
    "-fc": {
        "rel_wis": [1,0,1,0],
    },
}

# sl_pattern keys
_sl_pattern_keys = { # {key: dtype}
    "sl": np.uint8, # sl of pattern in dt chamber
    "pat_type": np.uint8, # index of string name of pattern (index of key of _dt_sl_patterns)
    "ts0": _ts_type, # timestamp of ly 0 wire of pattern
    "wi0": np.uint8, # wire index of ly 0 wire of pattern
    "ts1": _ts_type, # timestamp of ly 1 wire of pattern
    "wi1": np.uint8, # wire index of ly 0 wire of pattern
    "ts2": _ts_type, # timestamp of ly 2 wire of pattern
    "wi2": np.uint8, # wire index of ly 0 wire of pattern
    "ts3": _ts_type, # timestamp of ly 3 wire of pattern
    "wi3": np.uint8, # wire index of ly 0 wire of pattern
    "muon_ts": _ts_type, # timestamp of simulated correlated muon
    #"muon_x0": np.float64, # x0 of simulated correlated muon
    #"muon_tan_alpha": np.float64, # tan_alpha of simulated correlated muon
    "muon_id": np.uint64, # id / idx of correlated muon
}

# sl fit keys (fit list also also keeps sl_pattern_keys)
_sl_fit_keys = { # {key: dtype}
    "laterality": np.uint8, # idx of selected laterality [] in _dt_sl_patterns 
    "t0": np.uint64, # t0 fit param
    "x0": np.float64, # x0 fit param
    "tan_alpha": np.float64, # tan(alpha) fit param
    "chi2/ndf": np.float64, # reduced chi2 value
    "theta_proj": np.float64, # "projected" theta angle from tan_alpha (just for first checks, not used in further analysis)
}

## when reconstructing hits

# dt drift velocity
_drift_velocity = 54.5 # unit: um / ns = 10^-6 / 10 ^-9 m/s = 10^3 m/s
_dt_cell_width = 42 # mm

# --- dt
# apply dead time for all channels individually (if value > 0)
_dt_ts_individual_dead_time = 0 #600 #600 #1250 #800 #0 # in ts units
# timestamp window in which hits of sl must lie in order to be counted as pattern
_dt_sl_patterns_ts_window = 520 #int(400 / 0.78) # in same unit as timestamp (0.78 ns)
# acceptance interval for dt sl pattern grouping
_t0_acceptance_interval = 100 # max temporal distance of t0 values of dt sl pattern fits that should be grouped together, in ts units
_xproj_acceptance_interval = 50 # max spatial distance of 2 phi muon sl fits along the x axis, when projecting one to the other sl (delta_z(1-2) = z(sl=3.ly=3.wi=wi3_1) - z(sl=3.ly=3.wi=wi3_2)), in mm
_dt_max_drift_time = int((_dt_cell_width*1e-3/2) / (_drift_velocity*1e3) / 0.78e-9) # max drift time measured from time of muon arrival t0 in the sl pattern fit
_dt_t0_tolerance = 10 # tolerance which the muon arrival time t0 should take in the sl pattern fit, between ts_min of 4 hits in superlayer & ts_min+_dt_max_drift_time
_dt_tan_alpha_range = [-100, 100] # allowed range of tan alpha sl pattern fit parameter
_err_ts = 5 #1 # error of timestamps for fitting (in ts_units)

# --- scint
# acceptance interval for scintillator hits (2 sipm coincidence of strips) -> muon areas (2 strip coincidence) grouping
_scintillator_ts_acceptance_interval = 625 #64 #1250 #64 #32 # max temporal distance of ts values of scintillator that should be grouped together, in ts units
# 1280 = 1 us , 64 = 50 ns , 500 ~ 391 ns , 16 = 12.5 ns , 32 = 25 ns
# acceptance interval for raw scintillator hits (single sipm hits) -> scintillator hits (2 sipm coincidence of strips) grouping
_raw_scintillator_ts_acceptance_interval = _scintillator_ts_acceptance_interval # in ts units
# apply dead time for all channels individually (if value > 0)
_raw_scintillator_ts_individual_dead_time = 0 # 0, 64, 1250 # in ts units

## when simulating muon hits

# global time delay for scintillator hits by muons (scint ts = muon ts + _scintillator_delay)
_scintillator_hit_delay = 10 # timestamp units

### scintillator specific

### scintillator hit format:
### - htg origin: {"oc", "bx", "tdc", "ch"}
### - added scintillator mapping: {"ly": 0-1 scintillator layer, "st": 0-15 strip in layer, "sipm": 0-1 id of sipm of this strip (if data without strip coicidence is given), "ch_id": idx of coinc ch name in dict}
### - added timestamp: {"ts": converted timestamp from oc,bx,tdc}
## scint hits: coincidence of 2 sipms of strip
_scint_mapping_keys = {
    "ly": np.uint8,
    "st": np.uint8,
    "ch_id": np.uint8,
}
_scint_other_keys = {
    "ts": _ts_type,
    "muon_ts": _ts_type,
    "xhit": np.float64, # relative hit position: x_hit = xhit + xleft(lower x coord of strip) (in mm)
    "muon_id": np.uint64, # id / idx of correlated muon
    "sipm_delta_ts": _ts_type, # ts difference between two sipm hits of the strip
    "st_delta_last_ts0": _ts_type, # time difference to last scint hit for sipm0 hit ts
    "st_delta_last_ts1": _ts_type, # time difference to last scint hit for sipm1 hit ts
    "st_delta_last_ts": _ts_type, # time difference to last scint hit (mean ts)
}
## raw scint hits: single sipm hits
_raw_scint_mapping_keys = {
    "ly": np.uint8,
    "st": np.uint8,
    "sipm": np.uint8,
    "ch_id": np.uint8,
}
_raw_scint_other_keys = {
    "ts": _ts_type,
    "muon_ts": _ts_type,
    "xhit": np.float64, # relative hit position: x_hit = xhit + xleft(lower x coord of strip) (in mm)
    "muon_id": np.uint64, # id / idx of correlated muon
    "delta_last_ts":_ts_type, # difference in ts units to last hit
}

## use custom coordinate frame
# x axis: along theta wires, phi sl granularity
# y axis: along phi wires, theta sl granularity
# z axis: vertical axis (positive direction up from SL1 to SL 3)
# global origin: at corner of smallest coordinates of SL1

### all length units are mm, except where explicitly given

### orientaion (coordinate frame slicing) explanation
# indices of "real" coordinates (x:0, y:1, z:2) for the given orientation
# for the 2D plotted coorinates (x, y)
_orientation = {
    "phi": (0, 2), # plot (x,z) - phi wires into screen, theta wires parallel to x axis
    "theta": (1, 2), # plot (y,z) - theta wires into screen, phi wires parallel to y axis
}

### colors and colormaps
# fill: fill color
# edge: edge/line color
_color_info = {
    "fill": "white",
    "edge": "black",
    "sl": {
        "fill": "white",
        "edge": "black",
    },
    "honeycomb": {
        "fill": "white",
        "edge": "black",
    },
    "cell": {
        "edge": "black",
        None: "lightgray",
        "wire": "black",
        "side_view": "lightgray",
        "cmap": mpl.colormaps["Reds"],
    },
    "muon": {
        "linewidth": 1.5,
        "markersize": 50,
    }
}

### dt & scintillator muon correlation
# correlate muon object (dt reco) and muon area (scintillator reco)
_correlation_ts_window = 128 # = 100 ns, 1280 = 1 us, temporal correlation window, in ts units
_correlation_xy_window = 15 # spatial correlation window, in mm

### muon object specific
# a muon descibes a muon track
_muon_obj_keys = {
    "x0": np.float64, # reference point (x0,y0,z0), in mm
    "y0": np.float64,
    "z0": np.float64,
    "theta": np.float64, # theta angle (angle relative to z axis), in rad
    "phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad
    "ts": _ts_type, # timestamp of muon arrival (assume velocity is infinite, therefore during propagation no time passes, is alright here)
    "muon_id": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
}

### muon area object specific
# a muon area is an x-y rectangle at given z position which describes the area where a muon has been
_muon_area_obj_keys = {
    "z0": np.float64, # z reference point, in mm
    "xmin": np.float64, # smallest allowed x point, in mm -> (x,y) rectangle
    "xmax": np.float64, # largest allowed x point, in mm
    "ymin": np.float64, # smallest allowed y point, in mm
    "ymax": np.float64, # largest allowed y point, in mm 
    "xcenter": np.float64, # center x position of area, in mm
    "ycenter": np.float64, # center y position of area, in mm
    "ts": _ts_type, # timestamp of muon arrival
    "muon_id": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
    "pixel": np.uint16, # pixel index of scintillator pixel corresponding to this muon area
    "ly_delta_ts": np.uint64, # ts difference between two hits in the layers
}

### muon correlation object: holds info of 1 muon & 1 muon area
# has same keys as muon with "m_" prefix
# has same kes as muon area with "a_" prefix
_muon_corr_obj_keys = {
    # muon keys
    "x0": np.float64, # reference point (x0,y0,z0), in mm
    "y0": np.float64,
    "z0": np.float64,
    "theta": np.float64, # theta angle (angle relative to z axis), in rad
    "phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad
    "ts_muon": _ts_type, # timestamp of muon arrival (assume velocity is infinite, therefore during propagation no time passes, is alright here)
    "muon_id": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
    # muon area keys (z0 same as muon)
    "xmin": np.float64, # smallest allowed x point, in mm -> (x,y) rectangle
    "xmax": np.float64, # largest allowed x point, in mm
    "ymin": np.float64, # smallest allowed y point, in mm
    "ymax": np.float64, # largest allowed y point, in mm 
    "xcenter": np.float64, # center x position of area, in mm
    "ycenter": np.float64, # center y position of area, in mm
    "ts_area": _ts_type, # timestamp of muon arrival
    "muon_id_area": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
    "pixel": np.uint16, # pixel index of scintillator pixel corresponding to this muon area
    # advances / analysis keys
    "delta_ts": np.int64, # ts difference = ts_muon - ts_area
    "delta_xcenter": np.float64, # x deviation = x_muon - x_center_area
    "delta_ycenter": np.float64, # y deviation = x_muon - x_center_area
    "ts_orbit": np.int64, # ts difference of this ts (oc=this_orbit, bx=this_bx, tdc=this_tdc) wrt beginning of orbit (oc=this_orbit, bx=0, tdc=0)
}

### cosmic muon theta weight for a given theta value
# muon flux ~ cos(theta)^(n-1), n ~ 3 (when assuming flat earth) -> https://arxiv.org/pdf/1606.06907
# theta in range [0, pi]
def cosmic_muon_theta_weight(theta):
    norm = np.pi/2 # integral cos²(x) from 0 to pi = pi / 2
    return np.cos(theta)**2 * 1/norm # normalized to 1 for integral from 0 to pi

### general: data key symbols & units for plotting
_key_symbols = {
    "x0": "$x_0$",
    "y0": "$y_0$",
    "z0": "$z_0$",
    "xmin": "$x_\\text{min}$",
    "xmax": "$x_\\text{max}$",
    "ymin": "$y_\\text{min}$",
    "ymax": "$y_\\text{max}$",
    "xcenter": "$x_\\text{center}$",
    "ycenter": "$y_\\text{center}$",
    "ts": "$T$",
    "ts_muon": "$T$",
    "ts_area": "$T$",
    "t0": "$T_0$",
    "phi": "$\\phi$",
    "theta": "$\\theta$",
    "ch": "Channel",
    "ro_ch": "Readout channel",
    "tdc": "$TDC$",
    "bx": "$BX$",
    "oc": "$OC$",
    "wi": "Wire",
    "ly": "Layer",
    "sl": "Superlayer",
    "st": "Strip",
    "pixel": "Scintillator pixel",
    "delta_ts": "$T_\\text{muon}-T_\\text{scint}$",
    "delta_xcenter": "$x_{0,\\text{muon}}-x_{\\text{center},\\text{scint}}$",
    "delta_ycenter": "$y_{0,\\text{muon}}-y_{\\text{center},\\text{scint}}$",
    "ts_orbit": "$T_\\text{orbit}$",
    "ly_delta_ts": "$\\Delta T_\\text{layers}$",
    "sipm_delta_ts": "$\\Delta T_\\text{SiPMs}$",
    "st_delta_last_ts0": "$\\Delta T_\\text{strip, SiPM 0}$",
    "st_delta_last_ts1": "$\\Delta T_\\text{strip, SiPM 1}$",
    "st_delta_last_ts": "$\\Delta T_\\text{strip}$",
    "laterality": "Laterality",
    "t0": "$T_0$",
    "tan_alpha": "$\\text{tan}\\alpha$",
    "chi2/ndf": "$\\chi^2/n_\\text{df}$",
    "pat_type": "SL pattern type",
    "theta_proj": "$\\theta_\\text{proj}$",
    "wi3": "Wire, Layer 3",
    "ts3": "$T_\\text{Wire, Layer 3}$",
    "muon_ts": "$T_\\text{muon}$",
}
_key_units = {
    "x0": "mm",
    "y0": "mm",
    "z0": "mm",
    "xmin": "mm",
    "xmax": "mm",
    "ymin": "mm",
    "ymax": "mm",
    "xcenter": "mm",
    "ycenter": "mm",
    "ts": "TU", # timestamp unit: "$0.78\;\\text{ns}$",
    "ts_muon": "TU",
    "ts_area": "TU",
    "t0": "TU",
    "phi": "rad",
    "theta": "rad",
    "ch": "",
    "ro_ch": "",
    "tdc": "TDCU", # tdc units
    "bx": "BXU", # bx units
    "oc": "OCU", # oc units
    "wi": "",
    "ly": "",
    "sl": "",
    "st": "",
    "pixel": "",
    "delta_ts": "TU",
    "delta_xcenter": "mm",
    "delta_ycenter": "mm",
    "ts_orbit": "TU",
    "ly_delta_ts": "TU",
    "sipm_delta_ts": "TU",
    "st_delta_last_ts0": "TU",
    "st_delta_last_ts1": "TU",
    "st_delta_last_ts": "TU",
    "laterality": "",
    "t0": "TU",
    "tan_alpha": "",
    "chi2/ndf": "",
    "pat_type": "",
    "theta_proj": "rad",
    "wi3": "",
    "ts3": "TU",
    "muon_ts": "TU",
}

###############################
### HARDWARE SETUP
###############################

### dt chamber properties: {type: type of chamber (mb1), sls: {sl_id: {type: sl type (phi/theta), n_lys: no. of layers, n_wis: no. of wires, offset_ly: [true if wi of this ly is shifted towards higher wi, for all lys]}}}
# single cell properties (mm)
_cell_w_spacer = 0 # estimated only from sketch = 1.2
_cell_h_spacer = 1.5
_cell_width = 42-_cell_w_spacer # size of cell air volume
_cell_height = 13-_cell_h_spacer # size of cell air volume
_cell_offset = (_cell_width+_cell_w_spacer)/2
_cell_wire_radius = 0.5 # only for illustration (real radius much smaller)
_cell_wire_width = 0.5 # only for illustration (real width much smaller)
# full dt chamber property map
_dt_chamber = {
    "name": "MB w/1/s -z R",
    "sls": {
        1: {
            "orient": "phi",
            "n_lys": 4,
            "n_wis": 49,
            "offset_ly": [True, False, True, False], # for ly 0,1,2,3: True means shifted to right i.e. towards higher wi idx
            "size": (2126., 2513., 53.5),
            "pos": (1.8, 0., 0.), # corner with smallest coordinates of this sl, *RELATIVE TO* base point of chamber point with smallest coordinates
            "ch_pos": (22.9, 86., 0.), # corner with smallest coordinates of first cell (ly=0,wi=0), *RELATIVE TO* sl point with smallest coordinates
            "ch_spacer": (_cell_w_spacer, _cell_w_spacer, _cell_h_spacer), # size of spacer between layers/chambers
            "ch_size": (_cell_width, 2341., _cell_height), # size of cell
            "ch_offset": (_cell_offset, 0., 0.), # offset of cell between alternating layers
            "wi_radius": _cell_wire_radius, # wire radius to be displayed (much larger than real wire radius)
            "wi_linewidth": _cell_wire_width, # linewidth of side view of wire
        },
        2: {
            "orient": "theta",
            "n_lys": 4,
            "n_wis": 57,
            "offset_ly": [True, False, True, False],
            "size": (2172., 2462.4, 53.5),
            "pos": (0, 25.3, 181.5),
            "ch_pos": (86., 24.6, 0.),
            "ch_spacer": (_cell_w_spacer, _cell_w_spacer, _cell_h_spacer), # (0., _cell_w_spacer, _cell_h_spacer),
            "ch_size": (2000., _cell_width, _cell_height),
            "ch_offset": (0., _cell_offset, 0.),
            "wi_radius": _cell_wire_radius,
            "wi_linewidth": _cell_wire_width,
        },
        3: {
            "orient": "phi",
            "n_lys": 4,
            "n_wis": 49,
            "offset_ly": [True, False, True, False],
            "size": (2126., 2513., 53.5),
            "pos": (21.0-1.8, 0., 235.),
            "ch_pos": (22.9, 86., 0.),
            "ch_spacer": (_cell_w_spacer, _cell_w_spacer, _cell_h_spacer), # (_cell_w_spacer, 0., _cell_h_spacer)
            "ch_size": (_cell_width, 2341., _cell_height),
            "ch_offset": (_cell_offset, 0., 0.),
            "wi_radius": _cell_wire_radius,
            "wi_linewidth": _cell_wire_width,
        },
    },
    "n_sl": 3,
    "honeycomb": {
        "size": (2033., 2458., 128.),
        "pos": (30.7, 27.5, 53.5), # corner with smallest coordinates of honeycomb, *RELATIVE TO* base point of chamber point with smallest coordinates
    },
    "size": (2172, 2513., 288.5),
    "pos": (0, 0., 0.), # point with smallest coordinates of dt chamber
}

### obdt mappings: {fe_conn_name: {chs: (ch list), fe: fec name, sl: superlayer}}, fe conns sorted in order
_obdt_phi_1_fe_mapping = { # need to mask connectors J25, J26, J27
    'J23': {"label": 11, "sl": 1, "fe": "6A", "chs": (158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 204, 205, 206, 207)},
    'J24': {"label": 12, "sl": 1, "fe": "6B", "chs": ( 62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77)},
    'J25': {"label": 13, "sl": 1, "fe": "7A", "chs": (110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125)},
    'J26': {"label": 14, "sl": 1, "fe": None, "chs": (186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201)},
    'J27': {"label": 15, "sl": 1, "fe": None, "chs": (224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239)},
    'J28': {"label": 10, "sl": 1, "fe": "5B", "chs": (126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141)},
    'J29': {"label":  9, "sl": 1, "fe": "5A", "chs": ( 46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61)},
    'J30': {"label":  8, "sl": 1, "fe": "4B", "chs": ( 78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93)},
    'J31': {"label":  7, "sl": 1, "fe": "4A", "chs": (170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185)},
    'J32': {"label":  6, "sl": 1, "fe": "3B", "chs": (202, 203,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13)},
    'J33': {"label":  1, "sl": 1, "fe": "1A", "chs": (142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157)},
    'J34': {"label":  2, "sl": 1, "fe": "1B", "chs": ( 30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45)},
    'J35': {"label":  3, "sl": 1, "fe": "2A", "chs": ( 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109)},
    'J36': {"label":  4, "sl": 1, "fe": "2B", "chs": (208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223)},
    'J37': {"label":  5, "sl": 1, "fe": "3A", "chs": ( 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29)},
}
_obdt_phi_2_fe_mapping = { # need to mask connectors J25, J26, J27
    'J23': {"label": 11, "sl": 3, "fe": "6A", "chs": (158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 204, 205, 206, 207)},
    'J24': {"label": 12, "sl": 3, "fe": "6B", "chs": ( 62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77)},
    'J25': {"label": 13, "sl": 3, "fe": "7A", "chs": (110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125)},
    'J26': {"label": 14, "sl": 3, "fe": None, "chs": (186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201)},
    'J27': {"label": 15, "sl": 3, "fe": None, "chs": (224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239)},
    'J28': {"label": 10, "sl": 3, "fe": "5B", "chs": (126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141)},
    'J29': {"label":  9, "sl": 3, "fe": "5A", "chs": ( 46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61)},
    'J30': {"label":  8, "sl": 3, "fe": "4B", "chs": ( 78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93)},
    'J31': {"label":  7, "sl": 3, "fe": "4A", "chs": (170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185)},
    'J32': {"label":  6, "sl": 3, "fe": "3B", "chs": (202, 203,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13)},
    'J33': {"label":  1, "sl": 3, "fe": "1A", "chs": (142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157)},
    'J34': {"label":  2, "sl": 3, "fe": "1B", "chs": ( 30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45)},
    'J35': {"label":  3, "sl": 3, "fe": "2A", "chs": ( 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109)},
    'J36': {"label":  4, "sl": 3, "fe": "2B", "chs": (208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223)},
    'J37': {"label":  5, "sl": 3, "fe": "3A", "chs": ( 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29)},
}
_obdt_theta_1_fe_mapping = {
    'jin1a': {"label": "Jin1", "sl": 2, "fe": "3A", "chs": ( 33,  34,  31,  30,  35,  28,  32,  29,  26,  24,  27,   5,  25,   3,   4,   0)},
    'jin1b': {"label": "Jin1", "sl": 2, "fe": "3B", "chs": (  2,   1,  94,   7,   9, 155,  63,  65, 153,  17,  16,  20,  18,  19,  22,  21)},
    'jin2a': {"label": "Jin2", "sl": 2, "fe": "4A", "chs": (157, 151, 150, 227,  11,   8,   6,  10, 154, 152, 226, 146, 147, 122, 120, 116)},
    'jin2b': {"label": "Jin2", "sl": 2, "fe": "4B", "chs": (224, 133, 118, 119, 117, 115, 101, 100, 102,  99, 107, 105, 103, 106, 214, 211)},
    'jin3a': {"label": "Jin3", "sl": 2, "fe": "7A", "chs": (210, 218, 220, 225, 215, 104, 199, 213, 212, 201, 202, 196, 198, 200, 121, 124)},
    'jin3b': {"label": "Jin3", "sl": 2, "fe": "7B", "chs": ( 45, 194, 125, 126, 123, 139, 140, 144, 148, 149, 222, 193, 203, 192, 223,  46)},
    'jin4a': {"label": "Jin4", "sl": 2, "fe": "6A", "chs": (216, 197, 195,  54,  47,  49,  44,  48,  38, 108, 109, 111, 110,  36, 112, 113)},
    'jin4b': {"label": "Jin4", "sl": 2, "fe": "6B", "chs": ( 37, 114, 132, 128, 129, 130, 127, 131,  51,  50, 208,  52,  53,  41,  39, 205)},
    'jin5a': {"label": "Jin5", "sl": 2, "fe": "1A", "chs": ( 23,  14,  12,  15,  13, 221, 219, 217, 207, 209,  93, 206,  62,  69,  80,  79)},
    'jin5b': {"label": "Jin5", "sl": 2, "fe": "1B", "chs": ( 76, 204,  42,  59,  40,  43, 180, 181, 178, 176, 177, 179, 162, 160, 182, 161)},
    'jin6a': {"label": "Jin6", "sl": 2, "fe": "2A", "chs": (135, 183, 138, 163, 185, 184, 167, 164, 145, 137, 134, 136, 170, 142, 143, 141)},
    'jin6b': {"label": "Jin6", "sl": 2, "fe": "2B", "chs": (171, 156, 158, 159,  66,  90,  92,  91,  64,  68,  95,  97,  70, 173,  96,  98)},
    'jin7a': {"label": "Jin7", "sl": 2, "fe": "5A", "chs": ( 72,  67, 165,  71,  61,  73,  82,  81,  83,  75,  78, 166, 168,  77,  56,  74)},
    'jin7b': {"label": "Jin7", "sl": 2, "fe": "5B", "chs": (169,  58,  60,  57,  86,  55,  85,  87,  89, 172, 174, 187, 175,  84,  88, 186)},
    'jin8a': {"label": "Jin8", "sl": 2, "fe": "8A", "chs": (191, 190, 188, 189, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239)},
}

## dt chamber testpulses
# testpulse timing offset correction
# {sl: {fe: additional delay (in ts units), through longer cables or different tp latency}}
_old_tp_cable_add_latency = 8 / 0.78 # ts units
_theta_tp_add_latency = 0 / 0.78 # ts units
# the value in the _tp_time_offset map will be subtracted from the extracted tp timestamps / time positions
# (i.e. the values in this map here describe the time it takes "longer" than 0 offset)
_tp_time_offset_err = 1 # error on offset correction, in ts units
_tp_time_offset = { # in ts units
    1: { # phi sl 1
        "1A": 0,
        "1B": 0,
        "2A": 0,
        "2B": 0,
        "3A": 0,
        "3B": 0,
        "4A": 0,
        "4B": 0,
        "5A": 0,
        "5B": 0,
        "6A": 0,
        "6B": 0,
        "7A": 0,
    },
    2: { # theta sl
        "1A": _theta_tp_add_latency + 0,
        "1B": _theta_tp_add_latency + 0,
        "2A": _theta_tp_add_latency + 0,
        "2B": _theta_tp_add_latency + 0,
        "3A": _theta_tp_add_latency + 0,
        "3B": _theta_tp_add_latency + 0,
        "4A": _theta_tp_add_latency + 0,
        "4B": _theta_tp_add_latency + 0,
        "5A": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "5B": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "6A": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "6B": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "7A": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "7B": _theta_tp_add_latency + _old_tp_cable_add_latency,
        "8A": _theta_tp_add_latency + _old_tp_cable_add_latency,
    },
    3: { # phi sl 2
        "1A": 0,
        "1B": 0,
        "2A": 0,
        "2B": 0,
        "3A": 0,
        "3B": 0,
        "4A": 0,
        "4B": 0,
        "5A": 0,
        "5B": 0,
        "6A": 0,
        "6B": 0,
        "7A": 0,
    }
}

### scintillator properties: {type: type of scintillator (hodoscope), lys: {ly_id: {type: layer type (strips), orient: orientation of strips (parallel to phi/theta sl)}}}
# single strip properties (mm)
_strip_width = 30.
_strip_height = 5.
_strip_length = 500.
_strip_w_spacer = 20/15
_strip_h_spacer = 0.
# full scintillator
_scintillator = {
    "type": "hodoscope",
    "lys": {
        0: {
            "type": "strips",
            "orient": "phi",
            "size": (0., 0., 0.),
            "pos": (10., 10., 20.), # corner with smallest coordinates of this layer, *RELATIVE TO* base point of chamber point with smallest coordinates
            "n_sts": 16, # no of strips
            "ch_pos": (0., 0., 0.), # corner with smallest coordinates of first strip (st=0), *RELATIVE TO* ly point with smallest coordinates
            "ch_spacer": (_strip_w_spacer, _strip_w_spacer, _strip_h_spacer), # size of spacer between strips
            "ch_size": (_strip_width, _strip_length, _strip_height), # size of strip
        },
        1: {
            "type": "strips",
            "orient": "theta",
            "size": (0., 0., 0.),
            "pos": (10., 10., 5.), 
            "n_sts": 16,
            "ch_pos": (0., 0., 0.),
            "ch_spacer": (_strip_w_spacer, _strip_w_spacer, _strip_h_spacer),
            "ch_size": (_strip_length, _strip_width, _strip_height),
        },
    },
    "n_lys": 2,
    "size": (520., 520., 30.),
    "pos": (100., 100., -100.), # point with smallest coordinates of scintillator
}
### mezzanine scintillator mapping: {coinc_ch_name: {ch: ch id, ly: scint layer, st: scint strip}}
## scint hits
# configuration with strip coincidence
_mezzanine_1_fe_mapping_strip_coinc = {
    f"coinc_ch_{i}": {"ly": 0, "st": i, "ch": i} for i in range(0, 8) # ly0
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": i, "ch": 16+i} for i in range(0, 8) # ly1
}
_mezzanine_2_fe_mapping_strip_coinc = {
    f"coinc_ch_{i}": {"ly": 0, "st": 8+i, "ch": i} for i in range(0, 8) # ly0
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": 8+i, "ch": 16+i} for i in range(0, 8) # ly1
}
## raw scint hits
# configuration without any coincidence
_mezzanine_1_fe_mapping_no_coinc = {
    f"coinc_ch_{i}": {"ly": 0, "st": i, "ch": i, "sipm": 0} for i in range(0, 8) # ly0, sipm0
} | {
    f"coinc_ch_{8+i}": {"ly": 0, "st": i, "ch": 8+i, "sipm": 1} for i in range(0, 8) # ly0, sipm1
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": i, "ch": 16+i, "sipm": 0} for i in range(0, 8) # ly1, sipm0
} | {
    f"coinc_ch_{24+i}": {"ly": 1, "st": i, "ch": 24+i, "sipm": 1} for i in range(0, 8) # ly1, sipm1
}
_mezzanine_2_fe_mapping_no_coinc = {
    f"coinc_ch_{i}": {"ly": 0, "st": 8+i, "ch": i, "sipm": 0} for i in range(0, 8) # ly0, sipm0
} | {
    f"coinc_ch_{8+i}": {"ly": 0, "st": 8+i, "ch": 8+i, "sipm": 1} for i in range(0, 8) # ly0, sipm1
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": 8+i, "ch": 16+i, "sipm": 0} for i in range(0, 8) # ly1, sipm0
} | {
    f"coinc_ch_{24+i}": {"ly": 1, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) # ly1, sipm1
}
# map of masked channels in detector (noisy/dead), if for this (ly, st) the sipm is listed here, the other sipm hits are used as strip hit (scint hit) without sipm coincidence
# if one does not list it here, then the strip will be dead when one of its sipms is masked
# if both sipms are masked, do not put it here since the strip is dead anyway
_scint_masked_sipms = { # {ly: {st: sipm}} which is masked, use only other sipm
    0: {
        5: 1, # mez1 ro_ch27 ch6
        #6: 0-1, # mez1 ro_ch27 ch13-14
    },
    1: {
        #5: 1, # mez1 ro_ch27 ch29
        6: 1, # mez1 ro_ch27 ch30
    },
}

### mezzanine input channel mapping (for timing calibration)
# mapping of input channel = coincidence channel (for timing calibration there is no coincidence programmed)
# to the fpga pins & banks
mezzanine_input_mapping = { # ch id = idx of sipm_p/n signal in fw = input ch idx in fw  -->  {silkscreen (e.g. A4), sipm quad id = thres dac id (e.g. 3), thres dac ch (e.g. B)}
	 0: {'silkscreen': 'A1', 'quad_id': 0, 'thres_dac_ch': 'D', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 0},
	 1: {'silkscreen': 'A2', 'quad_id': 1, 'thres_dac_ch': 'D', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 1},
	 2: {'silkscreen': 'A3', 'quad_id': 2, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 0},
	 3: {'silkscreen': 'A4', 'quad_id': 3, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 1},
	 4: {'silkscreen': 'A5', 'quad_id': 4, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 2},
	 5: {'silkscreen': 'A6', 'quad_id': 5, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 3},
	 6: {'silkscreen': 'A7', 'quad_id': 6, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 4},
	 7: {'silkscreen': 'A8', 'quad_id': 7, 'thres_dac_ch': 'D', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 5},
	 8: {'silkscreen': 'B1', 'quad_id': 0, 'thres_dac_ch': 'C', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 2},
	 9: {'silkscreen': 'B2', 'quad_id': 1, 'thres_dac_ch': 'C', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 3},
	10: {'silkscreen': 'B3', 'quad_id': 2, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 6},
	11: {'silkscreen': 'B4', 'quad_id': 3, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 7},
	12: {'silkscreen': 'B5', 'quad_id': 4, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 8},
	13: {'silkscreen': 'B6', 'quad_id': 5, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 9},
	14: {'silkscreen': 'B7', 'quad_id': 6, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 10},
	15: {'silkscreen': 'B8', 'quad_id': 7, 'thres_dac_ch': 'C', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 11},
	16: {'silkscreen': 'C1', 'quad_id': 0, 'thres_dac_ch': 'B', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 4},
	17: {'silkscreen': 'C2', 'quad_id': 1, 'thres_dac_ch': 'B', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 5},
	18: {'silkscreen': 'C3', 'quad_id': 2, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 12},
	19: {'silkscreen': 'C4', 'quad_id': 3, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 13},
	20: {'silkscreen': 'C5', 'quad_id': 4, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 14},
	21: {'silkscreen': 'C6', 'quad_id': 5, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 15},
	22: {'silkscreen': 'C7', 'quad_id': 6, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 16},
	23: {'silkscreen': 'C8', 'quad_id': 7, 'thres_dac_ch': 'B', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 17},
	24: {'silkscreen': 'D1', 'quad_id': 0, 'thres_dac_ch': 'A', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 6},
	25: {'silkscreen': 'D2', 'quad_id': 1, 'thres_dac_ch': 'A', 'fpga_bank': 65, 'pin_inverted': True, 'mctt_instance': 1, 'mctt_input_ch': 7},
	26: {'silkscreen': 'D3', 'quad_id': 2, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 18},
	27: {'silkscreen': 'D4', 'quad_id': 3, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 19},
	28: {'silkscreen': 'D5', 'quad_id': 4, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 20},
	29: {'silkscreen': 'D6', 'quad_id': 5, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': True, 'mctt_instance': 0, 'mctt_input_ch': 21},
	30: {'silkscreen': 'D7', 'quad_id': 6, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 22},
	31: {'silkscreen': 'D8', 'quad_id': 7, 'thres_dac_ch': 'A', 'fpga_bank': 66, 'pin_inverted': False, 'mctt_instance': 0, 'mctt_input_ch': 23},
}

### hardware setup
## dt mapping: {ro_ch: obdt_mapping}
_dt_mapping = {
    8: _obdt_phi_1_fe_mapping, # obdt1_phi: dt sl1 (phi)
    10: _obdt_phi_2_fe_mapping, # obdt2_phi: dt sl3 (phi)
    14: _obdt_theta_1_fe_mapping, # obdt3_theta: dt sl2 (theta)
}
## scintillator mapping: {ro_ch: mezzanine_mapping}
# coincidence strips = 2 sipm coincidence hits
_scint_mapping = {
    27: _mezzanine_1_fe_mapping_strip_coinc, # mez1: scint ly0-1, st0-7
    26: _mezzanine_2_fe_mapping_strip_coinc, # mez2: scint ly0-1, st8-15
}
# no coincidence = raw sipm hits
_raw_scint_mapping = {
    27: _mezzanine_1_fe_mapping_no_coinc, # mez1: ly0-1, st0-7
    26: _mezzanine_2_fe_mapping_no_coinc, # mez2: ly0-1, st8-15
}

# ro_ch labels
_ro_ch_labels = {
    8: "ob1",
    10: "ob2",
    14: "ob3",
    27: "mez1",
    26: "mez2",
}

### muon reconstruction z position
# reco muon z0 value (select base z value for reco muon)
_muon_reco_z0 = _scintillator["pos"][2] # in mm





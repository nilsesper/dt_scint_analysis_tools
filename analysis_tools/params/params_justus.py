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
    "oc": np.uint64,
    "ro_ch": np.uint8,
}

### dumpfile import
# no of hits to skip in dumpfile since expect them to be old hits still in htg buffer
_dumpfile_hits_to_skip = 50000 #50000

### timestamp conversion
_lhc_tdc_count = 32 # max value of TDC + 1  (i.e. conversion factor: _lhc_tdc_count TDC = 1 BX)
_lhc_bunch_count = 3564 # max value of BX + 1 (i.e. conversion factor: _lhc_bunch_count BX = 1 ORBIT)
_lhc_orbit_count = 65536 # 2^16, max value of ORBIT + 1
#_lhc_orbit_count = 2**26 # = 67108864 = 2^26, max value of ORBIT + 1
_ts_type = np.float64 # data type of timestamp field in hits
_ts_float_type = np.float64 # for fitting, ts type as float

#_oc_difference_for_overflow = 50000 # difference between oc and last oc for oc overflow to be triggered
_oc_difference_for_overflow = 10000 # difference between oc and last oc for oc overflow to be triggered

### general
rad_to_deg = 180/np.pi
deg_to_rag = np.pi/180

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
    "err_ts": np.float64,
    # simulation keys
    "muon_ts": _ts_type,
    "muon_dt": np.float64, # drift time (in ts units)
    "muon_dd": np.float64, # drift distance (in mm)
    "muon_id": np.uint64, # id / idx of correlated muon
    "muon_lat": np.int8, # hit laterality -1 (l) left of wire, +1 (r) right of wire
    "muon_tan_alpha": np.float64, # simulated correlated muon tan_alpha (sl projection)
    "muon_loc_x0": np.float64, # simulated correlated muon x0 (sl projection)
    "muon_vd": np.float64, # vd of simulated muon hit
    # simulation muon keys
    "muon_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "muon_y0": np.float64, # of sim muon
    "muon_z0": np.float64, # of sim muon
    "muon_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "muon_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
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

## sl meantimer method
# patterns, lateralities and additional information
#   relations:
#   x_prime = h*tan_alpha
#   xd = td*vd
#   d/2 = vd / t_max
#   (1) xd_sign[3]*xd[3] + xd_sign[2]*xd[2] + x_prime_sign[0]*x_prime = d/2
#   (2) xd_sign[2]*xd[2] + xd_sign[1]*xd[1] + x_prime_sign[1]*x_prime = d/2
#   (3) xd_sign[1]*xd[1] + xd_sign[0]*xd[0] + x_prime_sign[2]*x_prime = d/2
_meantimer_patterns = { # order in lists: ly 0,1,2,3
    0: {
        "name": "+A",
        "rel_wis": (0,0,0,0),
        "lateralities": {
            0: {
                "laterality": (-1,1,-1,-1),
                "xd_sign": (1,1,1,-1),
                "x_prime_sign": (1,-1,1),
            },
            1: {
                "laterality": (-1,1,-1,1),
                "xd_sign": (1,1,1,1),
                "x_prime_sign": (1,-1,1),
            },
            2: {
                "laterality": (1,1,-1,-1),
                "xd_sign": (-1,1,1,-1),
                "x_prime_sign": (1,-1,1),
            },
            3: {
                "laterality": (1,1,-1,1),
                "xd_sign": (-1,1,1,1),
                "x_prime_sign": (1,-1,1),
            },
        },
    },
}

### dt fake hit patterns (to check for number of noise-induced patterns)
# reference is on top (highest z coordinate i.e. ly 3)
# higher wi index towards right -->
# ly  [+fA]    ref              [-fA]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | O | - | - | - | -     - | - | - | - | O | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | O | - | - | -     - | - | - | O | - | -
# rel_wi_0-3: -1 1 -2 0         0 -1 1 0
#
# ly  [+fB]    ref              [-fB]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | - | O | - | - | -     - | - | - | O | - | -
# 1   | - | - | - | O | - |     | - | O | - | - | - |
# 0   - | - | - | O | - | -     - | - | O | - | - | -
# rel_wi_0-3: 0 1 -1 0          -1 -1 0 0
#
# ly  [+fC]    ref              [-fC]    ref            (f: "fake" pattern)
# 3   | - | - | O | - | - |     | - | - | O | - | - |
# 2   - | O | - | - | - | -     - | - | - | - | O | -
# 1   | - | - | O | - | - |     | - | - | O | - | - |
# 0   - | O | - | - | - | -     - | - | - | - | O | -
# rel_wi_0-3: -2 0 -2 0         1 0 1 0
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
        "rel_wis": [-1,1,-2,0], # list of relative wire index of layers 0-3
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
    "err_ts0": np.float64, # timestamp error of ly 0 wire of pattern
    "wi0": np.uint8, # wire index of ly 0 wire of pattern
    "ts1": _ts_type, # timestamp of ly 1 wire of pattern
    "err_ts1": np.float64, # timestamp error of ly 1 wire of pattern
    "wi1": np.uint8, # wire index of ly 1 wire of pattern
    "ts2": _ts_type, # timestamp of ly 2 wire of pattern
    "err_ts2": np.float64, # timestamp error of ly 2 wire of pattern
    "wi2": np.uint8, # wire index of ly 2 wire of pattern
    "ts3": _ts_type, # timestamp of ly 3 wire of pattern
    "err_ts3": np.float64, # timestamp error of ly 3 wire of pattern
    "wi3": np.uint8, # wire index of ly 3 wire of pattern
    # simulation keys
    "muon_ts": _ts_type, # ts of correlated muon
    "muon_dt0": np.float64, # drift time (in ts units) for sim muon hit in ly 0
    "muon_dt1": np.float64, # drift time (in ts units) for sim muon hit in ly 1
    "muon_dt2": np.float64, # drift time (in ts units) for sim muon hit in ly 2
    "muon_dt3": np.float64, # drift time (in ts units) for sim muon hit in ly 3
    "muon_dd0": np.float64, # drift distance (in mm) for sim muon hit in ly 0
    "muon_dd1": np.float64, # drift distance (in mm) for sim muon hit in ly 1
    "muon_dd2": np.float64, # drift distance (in mm) for sim muon hit in ly 2
    "muon_dd3": np.float64, # drift distance (in mm) for sim muon hit in ly 3
    "muon_id": np.uint64, # id / idx of correlated muon
    "muon_lat0": np.int8, # lat of correlated muon hit in ly0
    "muon_lat1": np.int8, # lat of correlated muon hit in ly1
    "muon_lat2": np.int8, # lat of correlated muon hit in ly2
    "muon_lat3": np.int8, # lat of correlated muon hit in ly3
    "muon_lat_id": np.int8, # hit laterality -1 (l) left of wire, +1 (r) right of wire
    "muon_tan_alpha": np.float64, # simulated correlated muon tan_alpha (sl projection)
    "muon_x0_loc": np.float64, # simulated correlated muon x0 (sl projection)
    "muon_vd": np.float64, # vd of simulated muon hit
    # simulation muon keys
    "muon_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "muon_y0": np.float64, # of sim muon
    "muon_z0": np.float64, # of sim muon
    "muon_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "muon_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
}

# sl fit keys (fit list also also keeps sl_pattern_keys)
_sl_fit_keys = { # {key: dtype}
    # best fit
    "impossible": np.float64, # impossible True = 1, False = 0
    "laterality": np.uint8, # idx of selected laterality [] in _dt_sl_patterns 
    "t0": np.float64, # t0 fit param
    "err_t0": np.float64, # error from fit
    "x0": np.float64, # x0 fit param
    "err_x0": np.float64, # error from fit
    "tan_alpha": np.float64, # tan(alpha) fit param
    "err_tan_alpha": np.float64, # error from fit
    "vd": np.float64, # drift velocity (in mm/ts unit) fit param
    "err_vd": np.float64, # error from fit

    # correlations
    "corr_t0_x0": np.float64,
    "corr_t0_tan_alpha": np.float64,
    "corr_t0_vd": np.float64,
    "corr_x0_tan_alpha": np.float64,
    "corr_x0_vd": np.float64,
    "corr_tan_alpha_vd": np.float64,

    "chi2/ndf": np.float64, # reduced chi2 value

    "dt0": np.float64, # estimated drift time t0-ts for ly0
    "dt1": np.float64, # estimated drift time t0-ts for ly1
    "dt2": np.float64, # estimated drift time t0-ts for ly2
    "dt3": np.float64, # estimated drift time t0-ts for ly3
} # sl fits also have pattern keys already i.e. "muon_XXX" keys
# other laterality fits
_sl_fit_other_keys = {
    f"lat{i}_{k}": np.float64 for k in ["impossible", "t0", "err_t0", "x0", "err_x0", "tan_alpha", "err_tan_alpha", "vd", "err_vd", "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd", "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd", "chi2/ndf", "dt0", "dt1", "dt2", "dt3"] for i in range(4)
}

# sl fit group keys (superlayer-level concatenation of sl fits close in time)
_sl_fit_group_keys = {
    "tgroup": np.float64, # 
    "n_fits": np.uint64,
    "sl": np.uint8, # superlayer of this group
    "idcs": [], # list of sl_fits indices of sl fits belonging to this group
}


# super pattern fit keys (fit list also also keeps super_pattern keys)
_super_pattern_fit_keys = { # {key: dtype}
    # best fit
    "impossible": np.float64, # impossible True = 1, False = 0
    "laterality_sl1": np.uint8, # idx of selected laterality [] in _dt_sl_patterns 
    "laterality_sl3": np.uint8, # idx of selected laterality [] in _dt_sl_patterns 
    "t0": np.float64, # t0 fit param
    "err_t0": np.float64, # error from fit
    "x0": np.float64, # x0 fit param
    "err_x0": np.float64, # error from fit
    "tan_alpha": np.float64, # tan(alpha) fit param
    "err_tan_alpha": np.float64, # error from fit
    "vd": np.float64, # drift velocity (in mm/ts unit) fit param
    "err_vd": np.float64, # error from fit
    "x_pos": np.float64, # x position of relative wire
    "z_pos": np.float64, # z position of relative wire

    # correlations
    "corr_t0_x0": np.float64,
    "corr_t0_tan_alpha": np.float64,
    "corr_t0_vd": np.float64,
    "corr_x0_tan_alpha": np.float64,
    "corr_x0_vd": np.float64,
    "corr_tan_alpha_vd": np.float64,

    "chi2/ndf": np.float64, # reduced chi2 value

    "dt0": np.float64, # estimated drift time t0-ts for ly0
    "dt1": np.float64, # estimated drift time t0-ts for ly1
    "dt2": np.float64, # estimated drift time t0-ts for ly2
    "dt3": np.float64, # estimated drift time t0-ts for ly3
    "dt4": np.float64, # estimated drift time t0-ts for ly4 here the new SL begins
    "dt5": np.float64, # estimated drift time t0-ts for ly5
    "dt6": np.float64, # estimated drift time t0-ts for ly6
    "dt7": np.float64, # estimated drift time t0-ts for ly7
} # sl fits also have pattern keys already i.e. "muon_XXX" keys

_other_super_pattern_keys = _sl_fit_other_keys
## --------- when reconstructing hits

# dt drift velocity
_drift_velocity = 54.5 #53 #54.5 #53 #54.5 #53 #50.8 #54.5 # initial value: 54.5 # unit: um / ns = 10^-6 / 10 ^-9 m/s = 10^3 m/s
_dt_cell_width = 42 # mm, width of dt cell = 2x max drift distance
# when allowing changes of vd in fit, give vd param bounds:
#_drift_velocity_min = 10 # um/ns old value

#_drift_velocity_min = 40    # for checking bad fits cause
#_drift_velocity_max = 100 # um/ns
_drift_velocity_min = 20    # for checking bad fits cause
_drift_velocity_max = 100 # um/ns


### --- dt

#_dt_max_drift_time = (_dt_cell_width*1e-3/2) / (_drift_velocity*1e3) / 0.78e-9 # max drift time measured from time of muon arrival t0 in the sl pattern fit, in ts units
# use larger max drift time for floating parameter refit
_dt_max_drift_time = 600 / 0.78e-9
_dt_max_drift_time_vd_min = (_dt_cell_width*1e-3/2) / (_drift_velocity_min*1e3) / 0.78e-9 # max drift time for vdmin

## --- dumpfile -> dt hits
# apply dead time for all channels individually (if value > 0)
_dt_ts_individual_dead_time = 600 #1000 #600 #1250 #800 #0 # in ts units

# additional uncertainty assiged to dt ts because of geometry (different path lengths for charges if not on height of wire)
# assumed 5 ns = 6 tu uncertainty
dt_hit_add_ts_unc = 3 # in tu was set to 6

## --- dt hits -> sl patterns
# timestamp window in which hits of sl must lie in order to be counted as pattern
_t0_tolerance = 0 # tolerance of t0 beyond max drift time bound
# 
# if vd as fit parameter: use max drift time possible with lower vdmin bound
_dt_sl_patterns_ts_window_fit_vd = _dt_max_drift_time_vd_min + _t0_tolerance # in same unit as timestamp (0.78 ns)
# if vd fixed: use reference drift time
_dt_sl_patterns_ts_window = _dt_max_drift_time + _t0_tolerance # in same unit as timestamp (0.78 ns)

# --- sl patterns -> sl fits

# curve fit:
# allowed range of alpha sl pattern fit parameter
# alpha < 0 means towards bottom left (because z axis goes up)
_dt_pattern_alpha_range = { # pat_idx : [alpha_min, alpha_max] in rad
    0: [-1.0164888305933455 , 0.4939413689195812], # +a
    1: [-0.4939413689195812 , 1.0164888305933455], # -a
    2: [-1.0164888305933455 , 0], # +b
    3: [0 , 1.0164888305933455], # -b
    4: [-1.0164888305933455 , 0], # +c
    5: [0 , 1.0164888305933455], # -c
}
#_dt_pattern_alpha_range = {i: [-np.pi/2, np.pi/2] for i in range(6)} # uncomment if no alpha range should be used

# meantimer method:
_meantimer_tolerance_t0 = 1 # t0 (= t_muon) tolerance between different meantimer equations to accept laterality, in timestamp units
_meantimer_tolerance_tan_alpha = 0.03 # tan_alpha tolerance between different meantimer equations to accept laterality, in rad
## sl fit / pattern timing correction for different superlayers
# relative time calibration between superlayers (different obdt boards), will be applied with positive sign i.e. t0_corr = t0_before + _sl_time_offset[sl]
_sl_time_offset = {
    1: 0,
    2: 0, #0,
    3: 0,
}

## --- sl fits -> sl fit groups
_sl_fit_group_ts_tolerance = _dt_max_drift_time

## --- sl fit groups -> dt muons
_muon_tgroup_tolerance = 20/0.78 # time interval in which sl fit groups of different sls are combined to "muon"
_muon_n_fits_max = 1 # max n_fits in sl fit groups selected for "muon"
_muon_slphi_tan_alpha_tolerance = 0.05 # max deviation of tan_alpha for both sl fits in phi sl was set to 0.1
_muon_slphi_xproj_tolerance = 30 # max deviation of x_proj (projected sl fit track position at z=_muon_reco_z0) for both sl fits in phi sl
_muon_chi2_ndf_max = 10 #10 # max chi2 of muon sl fits
# reco muon z0 value (select base z value for reco muon)
_muon_reco_z0 = 144 #170 #_scintillator["pos"][2] # in mm

### --- scint

_raw_scint_hits_grouping_ts_tolerance = 100 # max temporal distance of raw scint groups, in tu

# acceptance interval for scintillator hits (2 sipm coincidence of strips) -> muon areas (2 strip coincidence) grouping
_scintillator_ts_acceptance_interval = 32 #10 #32 #32 #32 #1024 #32 #625 #64 #1250 #64 #32 # max temporal distance of ts values of scintillator that should be grouped together, in ts units
# 1280 = 1 us , 64 = 50 ns , 500 ~ 391 ns , 16 = 12.5 ns , 32 = 25 ns

# acceptance interval for raw scintillator hits (single sipm hits) -> scintillator hits (2 sipm coincidence of strips) grouping
_raw_scintillator_ts_acceptance_interval = _scintillator_ts_acceptance_interval # in ts units
# apply dead time for all channels individually (if value > 0)
_raw_scintillator_ts_individual_dead_time = 0 #500 #100 #100 # 0, 64, 1250 # in ts units
# dead time for scintillator strips
_scintillator_ts_individual_dead_time = 0 #500 # in ts units
## timestamp isolation of scint areas - to remove crosstalk hits
#_scint_area_clear_interval_down = 20 #200 # in ts units, isolation wrt last hit (extendable dead time)
#_scint_area_clear_interval_up = 0 #200 # in ts units, isolation wrt next hit

### --- dt and scint combination / correlation
_corr_ts_group_window = 1000 # in tu, timestamp window of scint and dt hit grouping

## --------- when simulating muon hits
# global time delay for scintillator hits by muons (scint ts = muon ts + _scintillator_delay)
_scintillator_hit_delay = 0 # timestamp units
# dt single cell hit efficiency
_dt_cell_efficiency = 96.93 * 1e-2 # from cms 2024 performance

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
    "err_ts": np.float64,
    "muon_ts": _ts_type,
    "xhit": np.float64, # relative hit position: x_hit = xhit + xleft(lower x coord of strip) (in mm)
    "muon_id": np.uint64, # id / idx of correlated muon
    "sipm_delta_ts": _ts_type, # ts difference between two sipm hits of the strip
    "sipm_delta_ts_signed": _ts_type,
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
    "err_ts": np.float64,
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
    # errors
    "err_x0": np.float64,
    "err_y0": np.float64,
    "err_z0": np.float64,
    "err_theta": np.float64,
    "err_phi": np.float64,
    "err_ts": np.float64,
    # other
    "muon_id": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
    "sl1_fit_group": np.uint64, # fit group idx used for sl 1
    "sl2_fit_group": np.uint64, # fit group idx used for sl 2
    "sl3_fit_group": np.uint64, # fit group idx used for sl 3
    # simulation muon keys
    "muon_ts": np.float64, # timestamp, in tu - of sim muon
    "muon_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "muon_y0": np.float64, # of sim muon
    "muon_z0": np.float64, # of sim muon
    "muon_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "muon_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
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
    "st0": np.uint8, # st idx of hit in ly0
    "st1": np.uint8, # st idx of hit in ly1
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
# theta in range [0, pi/2]
def cosmic_muon_theta_weight(theta):
    norm = 2/3 # integral from 0 to pi
    return np.cos(theta)**2 * np.sin(theta) * 1/norm # normalized to 1 for integral from 0 to pi/2
    # sin(theta) is solid angle weighting factor

### flat theta weight for a given theta value
# theta in range [0, pi/2]
def flat_theta_weight(theta):
    norm = 2 # integral from 0 to pi
    return np.sin(theta) * 1/norm # normalized to 1 for integral from 0 to pi/2
    # sin(theta) is solid angle weighting factor

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
    "muon_dt": "$t_\\text{d,muon}$",
    "muon_dd": "$x_\\text{d,muon}$",
    "muon_lat": "$\\text{Laterality}_\\text{muon}$",
    "muon_x0_loc": "$x0_\\text{loc, muon}$",
    "muon_tan_alpha": "$\\text{tan}\\alpha_\\text{muon}$",
    "muon_ts": "$T_\\text{0, muon}$",
    "muon_dt0": "$t_\\text{drift, ly 0, muon}$",
    "muon_dt1": "$t_\\text{drift, ly 1, muon}$",
    "muon_dt2": "$t_\\text{drift, ly 2, muon}$",
    "muon_dt3": "$t_\\text{drift, ly 3, muon}$",
    "muon_dd0": "$x_\\text{drift, ly 0, muon}$",
    "muon_dd1": "$x_\\text{drift, ly 1, muon}$",
    "muon_dd2": "$x_\\text{drift, ly 2, muon}$",
    "muon_dd3": "$x_\\text{drift, ly 3, muon}$",
    "muon_id": "Muon ID",
    "muon_x0": "$x0_\\text{muon}$",
    "muon_lat0": "Laterality hit$_\\text{ly 0, muon}$",
    "muon_lat1": "Laterality hit$_\\text{ly 1, muon}$",
    "muon_lat2": "Laterality hit$_\\text{ly 2, muon}$",
    "muon_lat3": "Laterality hit$_\\text{ly 3, muon}$",
    "muon_lat_id": "Pattern laterality$_\\text{muon}$",
    "muon_vd": "$v_\\text{drift, muon}$",
    "vd": "$v_\\text{drift}$",
    "dt": "$t_\\text{drift}$",
    "dt0": "$t_\\text{drift, ly 0}$",
    "dt1": "$t_\\text{drift, ly 1}$",
    "dt2": "$t_\\text{drift, ly 2}$",
    "dt3": "$t_\\text{drift, ly 3}$",
    "tgroup": "$T_\\text{group}$",
    "n_fits": "$N_\\text{fits}$",
    "err_ts": "$\\sigma_T$",
    "err_t0": "$\\sigma_{T_0}$",
    "err_x0": "$\\sigma_{x_0}$",
    "err_tan_alpha": "$\\sigma_{\\tan\\alpha}$",
    "err_vd": "$\\sigma_{v_D}$",
    "corr_t0_x0": "$\\text{cov}(T_0,\\; x_0)$",
    "corr_t0_tan_alpha": "$\\text{cov}(T_0,\\; \\tan\\alpha)$", 
    "corr_t0_vd": "$\\text{cov}(T_0,\\; v_D)$", 
    "corr_x0_tan_alpha": "$\\text{cov}(x_0,\\; \\tan\\alpha)$", 
    "corr_x0_vd": "$\\text{cov}(x_0,\\; v_D)$", 
    "corr_tan_alpha_vd": "$\\text{cov}(\\tan\\alpha,\\; v_D)$",
    "impossible": "Impossible to fit",
    "err_x0": "$\\sigma_{x_0}$",
    "err_y0": "$\\sigma_{y_0}$",
    "err_z0": "$\\sigma_{z_0}$",
    "err_phi": "$\\sigma_{\\phi}$",
    "err_theta": "$\\sigma_{\\theta}$",
    "n_hits": "$N_\\text{hits}$",
    "n_hits_nodupl": "$N_\\text{hit channels}$",
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
    "muon_dt": "TU",
    "muon_dd": "mm",
    "muon_lat": "",
    "muon_x0_loc": "mm",
    "muon_tan_alpha": "",
    "muon_dt": "TU",
    "muon_dd": "mm",
    "muon_lat": "",
    "muon_x0": "mm",
    "muon_tan_alpha": "",
    "muon_ts": "TU",
    "muon_dt0": "TU",
    "muon_dt1": "TU",
    "muon_dt2": "TU",
    "muon_dt3": "TU",
    "muon_dd0": "mm",
    "muon_dd1": "mm",
    "muon_dd2": "mm",
    "muon_dd3": "mm",
    "muon_id": "Muon ID",
    "muon_lat0": "",
    "muon_lat1": "",
    "muon_lat2": "",
    "muon_lat3": "",
    "muon_lat_id": "",
    "muon_vd": "mm/TU",
    "vd": "mm/TU",
    "dt": "TU",
    "dt0": "TU",
    "dt1": "TU",
    "dt2": "TU",
    "dt3": "TU",
    "tgroup": "TU",
    "n_fits": "",
    "err_ts": "TU",
    "err_t0": "TU",
    "err_x0": "mm",
    "err_tan_alpha": "",
    "err_vd": "mm/TU",
    "corr_t0_x0": "TU mm",
    "corr_t0_tan_alpha": "TU", 
    "corr_t0_vd": "mm", 
    "corr_x0_tan_alpha": "mm", 
    "corr_x0_vd": "mm${}^2$/TU", 
    "corr_tan_alpha_vd": "mm/TU",
    "impossible": "",
    "err_x0": "mm",
    "err_y0": "mm",
    "err_z0": "mm",
    "err_ts": "TU",
    "err_phi": "rad",
    "err_theta": "rad",
    "n_hits": "",
    "n_hits_nodupl": "",
}

###############################
### HARDWARE SETUP
###############################

""" #### OLD WRONG CHAMBER GEOMETRY
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
            "ch_size": (_cell_width, 2379., _cell_height), # size of cell  # #2379 #2341
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
            "ch_pos": (86., 22.9, 0.), # 24.6
            "ch_spacer": (_cell_w_spacer, _cell_w_spacer, _cell_h_spacer), # (0., _cell_w_spacer, _cell_h_spacer),
            "ch_size": (2038., _cell_width, _cell_height), # #2000 #2038
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
            "pos": (_cell_offset+1.8, 0., 235.), # 21.0-1.8
            "ch_pos": (22.9, 86., 0.),
            "ch_spacer": (_cell_w_spacer, _cell_w_spacer, _cell_h_spacer), # (_cell_w_spacer, 0., _cell_h_spacer)
            "ch_size": (_cell_width, 2379., _cell_height), # #2379 #2341
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
"""

##################

### my coordinate frame:
# x: along theta wires & along phi cells/cut
# y: along phi wires & along theta cells/cut
# z: top-bottom direction
## units in mm

## dt chamber:
#   (highest z)
# sl3 (phi)
# sl2 (theta)
# honeycomb
# sl1 (phi)
#   (lowest z)

### dt chamber properties: {type: type of chamber (mb1), sls: {sl_id: {type: sl type (phi/theta), n_lys: no. of layers, n_wis: no. of wires, offset_ly: [true if wi of this ly is shifted towards higher wi, for all lys]}}}
# single cell properties (mm)
_cell_w_spacer = 0 # estimated only from sketch = 1.2
_cell_h_spacer = 1.5
_cell_width = 42-_cell_w_spacer # size of cell air volume
_cell_height = 13-_cell_h_spacer # size of cell air volume
_cell_offset = (_cell_width+_cell_w_spacer)/2
_cell_wire_radius = 0.5 # only for illustration (real radius much smaller)
_cell_wire_width = 0.5 # only for illustration (real width much smaller)

#### CHAMBER GEOMETRY FROM CMSSW

### st1 (mb1) wh0 se4: cmssw
### position:
#   x_this = - x_cms * 10
#   y_this = z_cms * 10
#   z_this = y_cms * 10
### size:
#   x_this = x_cms * 10
#   y_this = z_cms * 10
#   z_this = y_cms * 10

### calculate x10 from cmssw, since they use cm and I use mm

### this coord frame:
# ( X: phi cell granularity , Y: theta cell granularity , Z: chamber height )

# chosen so that sl 1 (phi) wi 0 bottom corner is at coord frame origin (x=0, y=0, z=0)  [ on the corner where sl 2 (theta) has wi 0 ]
global_shift = (-388.4 -21, -65 , -4175.5) #(-391.2 , -8.5, -4311.75)
cmssw_layershift = (0, 0, 1.5 * 13)
cmssw_wireshift_sl1 = (26.4/2, 113.1/2, 1.4/2)
cmssw_wireshift_sl2 = (113.1/2, 26.4/2, 1.4/2)
cmssw_wireshift_sl3 = (26.4/2, 113.1/2, 1.4/2)
cmssw_chamber_pos = (391.2, 8.5, 4311.75 - 362/2)
cmssw_chamber_size = (2180, 2511, 362)

cmssw_sl1_size = (2126.4, 2511, 53.5)
cmssw_sl1_pos = (375.2-cmssw_chamber_pos[0]-cmssw_layershift[0] , 8.5-cmssw_chamber_pos[1]-cmssw_layershift[1] , 4194.25-cmssw_chamber_pos[2]-cmssw_layershift[2] )
cmssw_sl2_size = (2170, 2462.4, 53.5)
# in cmssw the offsets are = 0, but seems off
sl2_y_offset = (cmssw_sl1_size[1]-cmssw_sl2_size[1])/2
sl2_x_offset = (cmssw_sl1_size[0]-cmssw_sl2_size[0])/2
cmssw_sl2_pos = (396.2-cmssw_chamber_pos[0]-cmssw_layershift[0]+sl2_x_offset , 8.5-cmssw_chamber_pos[1]-cmssw_layershift[1]+sl2_y_offset , 4375.75-cmssw_chamber_pos[2]-cmssw_layershift[2] )
cmssw_sl3_size = (2126.4, 2511, 53.5)
cmssw_sl3_pos = (396.2-cmssw_chamber_pos[0]-cmssw_layershift[0] , 8.5-cmssw_chamber_pos[1]-cmssw_layershift[1] , 4429.25-cmssw_chamber_pos[2]-cmssw_layershift[2] )

cmssw_sl1_ly1_n_wi = 49
cmssw_sl1_ly1_min_wi = 0
cmssw_sl1_ly1_max_wi = 48
cmssw_sl1_ly1_pos = ( 375.2-cmssw_chamber_pos[0]-cmssw_sl1_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl1_pos[1], 4174.75-cmssw_chamber_pos[2]-cmssw_sl1_pos[2],)
cmssw_sl1_ly1_size = (2059.3, 2398, 11.5)
cmssw_sl1_ly2_n_wi = 50
cmssw_sl1_ly2_min_wi = 0
cmssw_sl1_ly2_max_wi = 49
cmssw_sl1_ly2_pos = ( 375.2-cmssw_chamber_pos[0]-cmssw_sl1_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl1_pos[1], 4187.75-cmssw_chamber_pos[2]-cmssw_sl1_pos[2],)
cmssw_sl1_ly2_size = (2101.3, 2398, 11.5)
cmssw_sl1_ly3_n_wi = 49
cmssw_sl1_ly3_min_wi = 0
cmssw_sl1_ly3_max_wi = 48
cmssw_sl1_ly3_pos = ( 375.2-cmssw_chamber_pos[0]-cmssw_sl1_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl1_pos[1], 4200.75-cmssw_chamber_pos[2]-cmssw_sl1_pos[2],)
cmssw_sl1_ly3_size = (2059.3, 2398, 11.5)
cmssw_sl1_ly4_n_wi = 48
cmssw_sl1_ly4_min_wi = 1
cmssw_sl1_ly4_max_wi = 48
cmssw_sl1_ly4_pos = ( 375.2-cmssw_chamber_pos[0]-cmssw_sl1_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl1_pos[1], 4213.75-cmssw_chamber_pos[2]-cmssw_sl1_pos[2],)
cmssw_sl1_ly4_size = (2017.3, 2398, 11.5)

cmssw_sl2_ly1_n_wi = 57
cmssw_sl2_ly1_min_wi = 0
cmssw_sl2_ly1_max_wi = 56
cmssw_sl2_ly1_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl2_pos[0]+sl2_x_offset, 8.5-cmssw_chamber_pos[1]-cmssw_sl2_pos[1]+sl2_y_offset, 4356.25-cmssw_chamber_pos[2]-cmssw_sl2_pos[2],)
cmssw_sl2_ly1_size = (2057, 2395.3, 11.5)
cmssw_sl2_ly2_n_wi = 58
cmssw_sl2_ly2_min_wi = 0
cmssw_sl2_ly2_max_wi = 57
cmssw_sl2_ly2_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl2_pos[0]+sl2_x_offset, 8.5-cmssw_chamber_pos[1]-cmssw_sl2_pos[1]+sl2_y_offset, 4369.25-cmssw_chamber_pos[2]-cmssw_sl2_pos[2],)
cmssw_sl2_ly2_size = (2057, 2437.3, 11.5)
cmssw_sl2_ly3_n_wi = 57
cmssw_sl2_ly3_min_wi = 0
cmssw_sl2_ly3_max_wi = 56
cmssw_sl2_ly3_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl2_pos[0]+sl2_x_offset, 8.5-cmssw_chamber_pos[1]-cmssw_sl2_pos[1]+sl2_y_offset, 4382.25-cmssw_chamber_pos[2]-cmssw_sl2_pos[2],)
cmssw_sl2_ly3_size = (2057, 2395.3, 11.5)
cmssw_sl2_ly4_n_wi = 56
cmssw_sl2_ly4_min_wi = 1
cmssw_sl2_ly4_max_wi = 56
cmssw_sl2_ly4_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl2_pos[0]+sl2_x_offset, 8.5-cmssw_chamber_pos[1]-cmssw_sl2_pos[1]+sl2_y_offset, 4395.25-cmssw_chamber_pos[2]-cmssw_sl2_pos[2],)
cmssw_sl2_ly4_size = (2057, 2353.3, 11.5)

cmssw_sl3_ly1_n_wi = 49
cmssw_sl3_ly1_min_wi = 0
cmssw_sl3_ly1_max_wi = 48
cmssw_sl3_ly1_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl3_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl3_pos[1], 4409.75-cmssw_chamber_pos[2]-cmssw_sl3_pos[2],)
cmssw_sl3_ly1_size = (2059.3, 2398, 11.5)
cmssw_sl3_ly2_n_wi = 50
cmssw_sl3_ly2_min_wi = 0
cmssw_sl3_ly2_max_wi = 49
cmssw_sl3_ly2_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl3_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl3_pos[1], 4422.75-cmssw_chamber_pos[2]-cmssw_sl3_pos[2],)
cmssw_sl3_ly2_size = (2101.3, 2398, 11.5)
cmssw_sl3_ly3_n_wi = 49
cmssw_sl3_ly3_min_wi = 0
cmssw_sl3_ly3_max_wi = 48
cmssw_sl3_ly3_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl3_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl3_pos[1], 4435.75-cmssw_chamber_pos[2]-cmssw_sl3_pos[2],)
cmssw_sl3_ly3_size = (2059.3, 2398, 11.5)
cmssw_sl3_ly4_n_wi = 48
cmssw_sl3_ly4_min_wi = 1
cmssw_sl3_ly4_max_wi = 48
cmssw_sl3_ly4_pos = ( 396.2-cmssw_chamber_pos[0]-cmssw_sl3_pos[0], 8.5-cmssw_chamber_pos[1]-cmssw_sl3_pos[1], 4448.75-cmssw_chamber_pos[2]-cmssw_sl3_pos[2],)
cmssw_sl3_ly4_size = (2017.3, 2398, 11.5)

_dt_chamber = {
    "name": "MB w/1/s -z R",
    "sls": {
        1: {
            "orient": "phi",
            "n_lys": 4,
            "pos": cmssw_sl1_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of chamber point with smallest coordinates
            "size": cmssw_sl1_size,
            "lys": {
                0: {
                    "n_wis": cmssw_sl1_ly1_n_wi,
                    "min_wi": cmssw_sl1_ly1_min_wi,
                    "max_wi": cmssw_sl1_ly1_max_wi,
                    "pos": cmssw_sl1_ly1_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl1_ly1_size,
                    "ch_pos": (21+cmssw_wireshift_sl1[0], 0.+cmssw_wireshift_sl1[1], 0.+cmssw_wireshift_sl1[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                1: {
                    "n_wis": cmssw_sl1_ly2_n_wi,
                    "min_wi": cmssw_sl1_ly2_min_wi,
                    "max_wi": cmssw_sl1_ly2_max_wi,
                    "pos": cmssw_sl1_ly2_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl1_ly2_size,
                    "ch_pos": (0.+cmssw_wireshift_sl1[0], 0.+cmssw_wireshift_sl1[1], 0.+cmssw_wireshift_sl1[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                2: {
                    "n_wis": cmssw_sl1_ly3_n_wi,
                    "min_wi": cmssw_sl1_ly3_min_wi,
                    "max_wi": cmssw_sl1_ly3_max_wi,
                    "pos": cmssw_sl1_ly3_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl1_ly3_size,
                    "ch_pos": (21+cmssw_wireshift_sl1[0], 0.+cmssw_wireshift_sl1[1], 0.+cmssw_wireshift_sl1[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                3: {
                    "n_wis": cmssw_sl1_ly4_n_wi,
                    "min_wi": cmssw_sl1_ly4_min_wi,
                    "max_wi": cmssw_sl1_ly4_max_wi,
                    "pos": cmssw_sl1_ly4_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl1_ly4_size,
                    "ch_pos": (0.+cmssw_wireshift_sl1[0], 0.+cmssw_wireshift_sl1[1], 0.+cmssw_wireshift_sl1[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
            },
            "wi_radius": _cell_wire_radius, # wire radius to be displayed (much larger than real wire radius)
            "wi_linewidth": _cell_wire_width, # linewidth of side view of wire
        },
        2: {
            "orient": "theta",
            "n_lys": 4,
            "pos": cmssw_sl2_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of chamber point with smallest coordinates
            "size": cmssw_sl2_size,
            "lys": {
                0: {
                    "n_wis": cmssw_sl2_ly1_n_wi,
                    "min_wi": cmssw_sl2_ly1_min_wi,
                    "max_wi": cmssw_sl2_ly1_max_wi,
                    "pos": cmssw_sl2_ly1_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl2_ly1_size,
                    "ch_pos": (0.+cmssw_wireshift_sl2[0], 21+cmssw_wireshift_sl2[1], 0.+cmssw_wireshift_sl2[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (2057, 42, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                1: {
                    "n_wis": cmssw_sl2_ly2_n_wi,
                    "min_wi": cmssw_sl2_ly2_min_wi,
                    "max_wi": cmssw_sl2_ly2_max_wi,
                    "pos": cmssw_sl2_ly2_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl2_ly2_size,
                    "ch_pos": (0.+cmssw_wireshift_sl2[0], 0.+cmssw_wireshift_sl2[1], 0.+cmssw_wireshift_sl2[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (2057, 42, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                2: {
                    "n_wis": cmssw_sl2_ly3_n_wi,
                    "min_wi": cmssw_sl2_ly3_min_wi,
                    "max_wi": cmssw_sl2_ly3_max_wi,
                    "pos": cmssw_sl2_ly3_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl2_ly3_size,
                    "ch_pos": (0.+cmssw_wireshift_sl2[0], 21+cmssw_wireshift_sl2[1], 0.+cmssw_wireshift_sl2[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (2057, 42, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                3: {
                    "n_wis": cmssw_sl2_ly4_n_wi,
                    "min_wi": cmssw_sl2_ly4_min_wi,
                    "max_wi": cmssw_sl2_ly4_max_wi,
                    "pos": cmssw_sl2_ly4_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl2_ly4_size,
                    "ch_pos": (0.+cmssw_wireshift_sl2[0], 0.+cmssw_wireshift_sl2[1], 0.+cmssw_wireshift_sl2[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (2057, 42, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
            },
            "wi_radius": _cell_wire_radius, # wire radius to be displayed (much larger than real wire radius)
            "wi_linewidth": _cell_wire_width, # linewidth of side view of wire
        },
        3: {
            "orient": "phi",
            "n_lys": 4,
            "pos": cmssw_sl3_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of chamber point with smallest coordinates
            "size": cmssw_sl3_size,
            "lys": {
                0: {
                    "n_wis": cmssw_sl3_ly1_n_wi,
                    "min_wi": cmssw_sl3_ly1_min_wi,
                    "max_wi": cmssw_sl3_ly1_max_wi,
                    "pos": cmssw_sl3_ly1_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl3_ly1_size,
                    "ch_pos": (21+cmssw_wireshift_sl3[0], 0.+cmssw_wireshift_sl3[1], 0.+cmssw_wireshift_sl3[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                1: {
                    "n_wis": cmssw_sl3_ly2_n_wi,
                    "min_wi": cmssw_sl3_ly2_min_wi,
                    "max_wi": cmssw_sl3_ly2_max_wi,
                    "pos": cmssw_sl3_ly2_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl3_ly2_size,
                    "ch_pos": (0.+cmssw_wireshift_sl3[0], 0.+cmssw_wireshift_sl3[1], 0.+cmssw_wireshift_sl3[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                2: {
                    "n_wis": cmssw_sl3_ly3_n_wi,
                    "min_wi": cmssw_sl3_ly3_min_wi,
                    "max_wi": cmssw_sl3_ly3_max_wi,
                    "pos": cmssw_sl3_ly3_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl3_ly3_size,
                    "ch_pos": (21+cmssw_wireshift_sl3[0], 0.+cmssw_wireshift_sl3[1], 0.+cmssw_wireshift_sl3[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
                3: {
                    "n_wis": cmssw_sl3_ly4_n_wi,
                    "min_wi": cmssw_sl3_ly4_min_wi,
                    "max_wi": cmssw_sl3_ly4_max_wi,
                    "pos": cmssw_sl3_ly4_pos, # corner with smallest coordinates of this sl, *RELATIVE TO* base point of layer point with smallest coordinates
                    "size": cmssw_sl3_ly4_size,
                    "ch_pos": (0.+cmssw_wireshift_sl3[0], 0.+cmssw_wireshift_sl3[1], 0.+cmssw_wireshift_sl3[2]), # lowest coordinate position of wire wi=0 of this ly
                    "ch_size": (42, 2398, 13), # size of wire in all directions
                    "ch_spacer": (0., 0., 0.), # size of spacer between channels (in all directions)
                },
            },
            "wi_radius": _cell_wire_radius, # wire radius to be displayed (much larger than real wire radius)
            "wi_linewidth": _cell_wire_width, # linewidth of side view of wire
        },
    },
    "n_sl": 3,
    "honeycomb": {
        "pos": (0., 0., 0.), #(30.7, 27.5, 53.5), # corner with smallest coordinates of honeycomb, *RELATIVE TO* base point of chamber point with smallest coordinates
        "size": (0., 0., 0.), #(2033., 2458., 128.),
    },
    "pos": (cmssw_chamber_pos[0]+global_shift[0], cmssw_chamber_pos[1]+global_shift[1], cmssw_chamber_pos[2]+global_shift[2]), # point with smallest coordinates of dt chamber
    "size": (cmssw_chamber_size[0], cmssw_chamber_size[1], cmssw_chamber_size[2]), 
}

#print(f"chamber_pos = {_dt_chamber["pos"]}")

### obdt mappings: {fe_conn_name: {chs: (ch list), fe: fec name, sl: superlayer}}, fe conns sorted in order
_obdt_phi_1_fe_mapping = { # need to mask connectors J26, J27
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
_obdt_phi_2_fe_mapping = { # need to mask connectors J26, J27
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
_theta_tp_add_latency = 116 / 0.78 # ts units
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

### dt wires to mask manually
"""
_dt_wire_mask = { # sl: ly: [wire_ids]
    1: {
        0: [7, 35],
        1: [10, 24],
        2: [0, 12, 19, 23, 33, 38],
        3: [5],
    },
    2: {
        0: [1, 11, 17, 18, 49],
        1: [0, 8, 33, 36],
        2: [15],
        3: [1, 19],
    },
    3: {
        0: [32, 40],
        1: [24, 48],
        2: [],
        3: [31],
    }

}
"""
_dt_wire_mask = { # sl: ly: [wire_ids]
    1: {
        0: [1, 2, 16, 38],
        1: [10],
        2: [37, 48],
        3: [1, 30, 32],
    },
    2: {
        0: [18, 19, 40, 44],
        1: [6, 14, 20],
        2: [56],
        3: [],
    },
    3: {
        0: [20, 38],
        1: [44, 36],
        2: [],
        3: [37],
    }

}


### scintillator properties: {type: type of scintillator (hodoscope), lys: {ly_id: {type: layer type (strips), orient: orientation of strips (parallel to phi/theta sl)}}}
# single strip properties (mm)
_strip_width = 30.
_strip_height = 5.
_strip_length = 500.
_strip_w_spacer = 20/15
_strip_h_spacer = 0.
# scint position
scint_size = (-520., -520., 40.) 
# MEASUREMENT: sl1 bottom left edge to scint top left edge (smallest x,y coordinates, largest z coordinate)
scint_edge_to_sl1_edge = ( 760, None, -990 )
# MEASUREMENT: sl2 bottom left edge to scint top left edge (smallest x,y coordinates, largest z coordinate)
scint_edge_to_sl2_edge = ( None, 1170, None )
# translate to used global coord frame
scint_pos = (
    scint_edge_to_sl1_edge[0]-scint_size[0]+(cmssw_sl1_pos[0]-cmssw_sl1_ly1_pos[0]),
    scint_edge_to_sl2_edge[1]-scint_size[1]+(cmssw_sl2_pos[1]-cmssw_sl2_ly1_pos[1]),
    scint_edge_to_sl1_edge[2]-scint_size[2]+(cmssw_sl1_pos[2]-cmssw_sl1_ly1_pos[2])
) # shift from sl1 casing edge to wi0 ly0 cell edge - which is the coordinate origin
# full scintillator
_scintillator = {
    "type": "hodoscope",
    "lys": {
        1: {
            "type": "strips",
            "orient": "phi", # strip segmentation (width) along x axis
            "size": (0., 0., 0.),
            "pos": (-10., -10., 20+(10-_strip_height/2)),  # corner with smallest coordinates of this layer, *RELATIVE TO* base point of chamber point with smallest coordinates
            "n_sts": 16, # no of strips
            "ch_pos": (0., 0., 0.), # corner with smallest coordinates of first strip (st=0), *RELATIVE TO* ly point with smallest coordinates
            "ch_spacer": (-_strip_w_spacer, -_strip_w_spacer, _strip_h_spacer), # size of spacer between strips
            "ch_size": (-_strip_width, -_strip_length, _strip_height), # size of strip
        },
        0: {
            "type": "strips",
            "orient": "theta", # strip segmentation (width) along y axis
            "size": (0., 0., 0.),
            "pos": (-10., -10., 0+(10-_strip_height/2)),
            "n_sts": 16,
            "ch_pos": (0., 0., 0.),
            "ch_spacer": (-_strip_w_spacer, -_strip_w_spacer, _strip_h_spacer),
            "ch_size": (-_strip_length, -_strip_width, _strip_height),
        },
    },
    "n_lys": 2,
    "size": scint_size,
    # dt chamber coordinates: sl 1 (phi) wi 0 is at (0, 0, 0), on the side where sl 2 (theta) has wi 0
    #"pos": (2189 -950 , 2511 -720 , -980), # point with smallest coordinates of scintillator
    "pos": scint_pos,
}
### mezzanine scintillator mapping: {coinc_ch_name: {ch: ch id, ly: scint layer, st: scint strip}}
## scint hits
# configuration with strip coincidence
_mezzanine_1_fe_mapping_strip_coinc = {
    f"coinc_ch_{i}": {"ly": 0, "st": i, "ch": i} for i in range(0, 16) # ly0
}
_mezzanine_2_fe_mapping_strip_coinc = {
    f"coinc_ch_{i}": {"ly": 1, "st": i, "ch": i} for i in range(0, 16) # ly1
}
## raw scint hits
# configuration without any coincidence
"""
### before 31-10-2025
_mezzanine_1_fe_mapping_no_coinc = { # ly 0-1, st 0-7
    f"coinc_ch_{i}": {"ly": 0, "st": i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 1, "st": i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 0, "st": i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
_mezzanine_2_fe_mapping_no_coinc = { # ly 0-1, st 8-15
    f"coinc_ch_{i}": {"ly": 0, "st": 8+i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 1, "st": 8+i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": 8+i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 0, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
#"""
""" until 09-11-2025
_mezzanine_1_fe_mapping_no_coinc = { # ly 0
    f"coinc_ch_{i}": {"ly": 0, "st": 0+i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 0, "st": 8+i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 0, "st": 0+i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 0, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
_mezzanine_2_fe_mapping_no_coinc = { # ly 1
    f"coinc_ch_{i}": {"ly": 1, "st": 0+i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 1, "st": 8+i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": 0+i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 1, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
#"""
_mezzanine_1_fe_mapping_no_coinc = { # one sipm of all strips
    f"coinc_ch_{i}": {"ly": 0, "st": 0+i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 0, "st": 8+i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 1, "st": 0+i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 1, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
_mezzanine_2_fe_mapping_no_coinc = { # one sipm of all strips
    f"coinc_ch_{i}": {"ly": 1, "st": 0+i, "ch": i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{8+i}": {"ly": 1, "st": 8+i, "ch": 8+i, "sipm": 0} for i in range(0, 8) 
} | {
    f"coinc_ch_{16+i}": {"ly": 0, "st": 0+i, "ch": 16+i, "sipm": 1} for i in range(0, 8) 
} | {
    f"coinc_ch_{24+i}": {"ly": 0, "st": 8+i, "ch": 24+i, "sipm": 1} for i in range(0, 8) 
}
# map of masked channels in detector (noisy/dead), if for this (ly, st) the sipm is listed here, the other sipm hits are used as strip hit (scint hit) without sipm coincidence
# if one does not list it here, then the strip will be dead when one of its sipms is masked
# if both sipms are masked, do not put it here since the strip is dead anyway
_scint_masked_sipms = { # {ly: {st: sipm}} which is masked, use only other sipm
    0: {
        #st: 0 for st in range(16)
    },
    1: {
        #st: 1 for st in range(16)
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
# ref pos (lowest x,y,z point of scint)
scint_ref_pos = (
    scint_edge_to_sl1_edge[0]+(cmssw_sl1_pos[0]-cmssw_sl1_ly1_pos[0]),
    scint_edge_to_sl2_edge[1]+(cmssw_sl2_pos[1]-cmssw_sl2_ly1_pos[1]),
    scint_edge_to_sl1_edge[2]-scint_size[2]+(cmssw_sl1_pos[2]-cmssw_sl1_ly1_pos[2])
)
#print(f"scint_ref_pos = {scint_ref_pos}")

### hardware setup
## dt mapping: {ro_ch: obdt_mapping}
_dt_mapping = {
    14: _obdt_phi_1_fe_mapping, # obdt1_phi: dt sl1 (phi)
    28: _obdt_phi_2_fe_mapping, # obdt2_phi: dt sl3 (phi)
    26: _obdt_theta_1_fe_mapping, # obdt3_theta: dt sl2 (theta)
}
## scintillator mapping: {ro_ch: mezzanine_mapping}
# coincidence strips = 2 sipm coincidence hits
_scint_mapping = {
    #27: _mezzanine_1_fe_mapping_strip_coinc, # mez1: scint ly0-1, st0-7
    #25: _mezzanine_2_fe_mapping_strip_coinc, # mez2: scint ly0-1, st8-15
}
# no coincidence = raw sipm hits
_raw_scint_mapping = {
    #27: _mezzanine_1_fe_mapping_no_coinc, # mez1: ly0-1, st0-7
    #25: _mezzanine_2_fe_mapping_no_coinc, # mez2: ly0-1, st8-15
}

# ro_ch labels
_ro_ch_labels = {
    26: "ob1",
    28: "ob2",
    14: "ob3",
    #27: "mez1",
    #25: "mez2",
}

#### plotting
_legend_alpha = 0.7
_hist_info_alpha = 0.7
_info_font_size = 12





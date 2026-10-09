###############################
### PARAMETERS
###############################
# All settings of the analysis: data format, chamber geometry, readout mapping, calibration, reconstruction,
# simulation, branch definitions and plotting. Values which follow from these are in derived_params.py, the chamber
# geometry built from _dt_chamber (and the description of all coordinate systems) is in analysis_tools/utils/dt_chamber_utils.py.
# Another parameter file with the same names can be used with --params_file (see analysis_tools/params_file_selection.py).
# Units: lengths in mm, times in timestamp units (TU, 1 TU = 0.78 ns = 1 TDC count), angles in rad.

import numpy as np
import matplotlib as mpl

rad_to_deg = 180 / np.pi

###############################
### READOUT DATA FORMAT (HTG box dumpfile)
###############################

# fields of a 64 bit data word: mask and shift
_htg_shifted_mask = {
    "ch":    0b0000000000000000000000000000000000000000000000000000000011111111, # bits 7:0 [8]
    "bx":    0b0000000000000000000000000000000000000000000011111111111100000000, # bits 19:8 [12]
    "tdc":   0b0000000000000000000000000000000000000001111100000000000000000000, # bits 24:20 [5]
    "oc":    0b0000001111111111111111111111111100000000000000000000000000000000, # bits 57:32 [26]
    "ro_ch": 0b1111110000000000000000000000000000000000000000000000000000000000, # bits 63:58 [6]
}
_htg_bitshift = {
    "ch": 0,
    "bx": 8,
    "tdc": 20,
    "oc": 32,
    "ro_ch": 58,
}
_htg_keys = {
    "ch": np.uint8,
    "bx": np.uint16,
    "tdc": np.uint8,
    "oc": np.uint64,
    "ro_ch": np.uint8,
}

# lines at the start of a dumpfile with old hits still in the HTG buffer (default of the testpulse calibration)
_dumpfile_hits_to_skip = 50000

# time counters: ts = tdc + _lhc_tdc_count * (bx + _lhc_bunch_count * (orbit + _lhc_orbit_count * orbit counter overflows))
_lhc_tdc_count = 32 # TDC counts per BX
_lhc_bunch_count = 3564 # BX per orbit
#_lhc_orbit_count = 2**26 # range of the orbit counter
_lhc_orbit_count = 65536 # range of the orbit counter
# _oc_difference_for_overflow = 50_000 # the orbit counter has overflowed when it jumps back by more than this
_oc_difference_for_overflow = 10_000 # the orbit counter has overflowed when it jumps back by more than this
_ts_type = np.float64 # data type of timestamps
_ts_float_type = np.float64 # data type of timestamps in fits

###############################
### DT CHAMBER GEOMETRY
###############################
# Chamber frame (mm): x is measured by the phi superlayers (their wires run along y), y by the theta superlayer
# (wires along x), z points up from SL 1 to SL 3.
# Numbers from the CMSSW geometry of MB1, wheel 0, sector 4 (x = -x_cms, y = z_cms, z = y_cms, cm -> mm), shifted so
# that the lower corner of cell 0 of SL 1 layer 0 is at about (0, 0, 0).
# A cell is a box of cell_size; in a layer the cells of wires min_wi .. max_wi follow each other along the measured
# axis, without gaps, starting with the cell of wire 0 at cell_0 (its lower corner).
_dt_chamber = {
    "name": "MB1 (CMSSW: wheel 0, sector 4)",
    "pos": (-18.2, -56.5, -44.75),  # lower corner of the chamber box (drawing only)
    "size": (2180, 2511, 362),
    "sls": {
        1: {
            "orient": "phi",
            "pos": (-34.2, -56.5, -0.75),  # lower corner of the superlayer box (drawing only)
            "size": (2126.4, 2511, 53.5),
            "cell_size": (42, 2398, 13),  # (x, y, z)
            "lys": {
                0: {"min_wi": 0, "max_wi": 48, "cell_0": (0.0, 0.05, -0.05)},
                1: {"min_wi": 0, "max_wi": 49, "cell_0": (-21.0, 0.05, 12.95)},
                2: {"min_wi": 0, "max_wi": 48, "cell_0": (0.0, 0.05, 25.95)},
                3: {"min_wi": 1, "max_wi": 48, "cell_0": (-21.0, 0.05, 38.95)},
            },
        },
        2: {
            "orient": "theta",
            "pos": (-35.0, -32.2, 180.75),  # lower corner of the superlayer box (drawing only)
            "size": (2170, 2462.4, 53.5),
            "cell_size": (2057, 42, 13),  # (x, y, z)
            "lys": {
                0: {"min_wi": 0, "max_wi": 56, "cell_0": (21.55, 2.0, 181.45)},
                1: {"min_wi": 0, "max_wi": 57, "cell_0": (21.55, -19.0, 194.45)},
                2: {"min_wi": 0, "max_wi": 56, "cell_0": (21.55, 2.0, 207.45)},
                3: {"min_wi": 1, "max_wi": 56, "cell_0": (21.55, -19.0, 220.45)},
            },
        },
        3: {
            "orient": "phi",
            "pos": (-13.2, -56.5, 234.25),  # lower corner of the superlayer box (drawing only)
            "size": (2126.4, 2511, 53.5),
            "cell_size": (42, 2398, 13),  # (x, y, z)
            "lys": {
                0: {"min_wi": 0, "max_wi": 48, "cell_0": (21.0, 0.05, 234.95)},
                1: {"min_wi": 0, "max_wi": 49, "cell_0": (0.0, 0.05, 247.95)},
                2: {"min_wi": 0, "max_wi": 48, "cell_0": (21.0, 0.05, 260.95)},
                3: {"min_wi": 1, "max_wi": 48, "cell_0": (0.0, 0.05, 273.95)},
            },
        },
    },
}

###############################
### READOUT MAPPING
###############################

# frontend connectors, index = "fe_id"
_fe_idx_list = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "6A", "6B", "7A", "7B", "8A", "8B", "9A", "9B", "10A", "10B", "11A", "11B", "12A", "12B", "13A", "13B", "14A", "14B"]
# a frontend connector serves 4 wires x 4 layers: conductor ch_id (0-15) -> wire 4 * fe_id + ch_id // 4, layer below
_fe_conductor_layers = (3, 1, 2, 0) # layer of conductor ch_id % 4

# OBDT boards: {connector: {"sl": superlayer, "fe": frontend connector (None: not connected), "chs": OBDT channel per conductor}}
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

# readout channel (ro_ch) of the HTG box -> OBDT board
_dt_mapping = {
    10: _obdt_phi_1_fe_mapping, # OBDT 1, phi: SL 1
     8: _obdt_phi_2_fe_mapping, # OBDT 2, phi: SL 3
    28: _obdt_theta_1_fe_mapping, # OBDT 3, theta: SL 2
}

# cells left out of the analysis: {sl: {ly: [wires]}}
_dt_wire_mask = { # sl: ly: [wire_ids]
    1: {
        0: [],
        1: [],
        2: [],
        3: [],
    },
    2: {
        0: [],
        1: [],
        2: [],
        3: [],
    },
    3: {
        0: [],        
        1: [],        
        2: [],        
        3: [],
    }
}
_dt_dead_wires = { # sl: ly: [wire_ids] these wires are dead, no signal is expected from them, they are masked in the analysis
    1: {
        0: [],
        1: [],
        2: [],
        3: [],
    },
    2: {
        0: [],
        1: [],
        2: [],
        3: [],
    },
    3: {
        0: [],
        1: [],
        2: [],
        3: [],
    }

}

###############################
### TESTPULSE TIMING CALIBRATION
###############################
# known extra delay of the testpulses per frontend connector, subtracted from the measured testpulse times
_old_tp_cable_add_latency = 8 / 0.78 # longer (old) testpulse cables
_theta_tp_add_latency = 116 / 0.78 # testpulse latency of the theta OBDT
_tp_time_offset_err = 1 # uncertainty of the offsets
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

###############################
### RECONSTRUCTION
###############################

## drift
_drift_velocity = 54.5 # um/ns
_drift_velocity_min = 20 # bounds of the drift velocity when it is a fit parameter, um/ns
_drift_velocity_max = 100
_dt_cell_width = 42 # = 2 x max drift distance
_dt_max_drift_time = 600 / 0.78
_dt_max_drift_time_vd_min = (_dt_cell_width*1e-3/2) / (_drift_velocity_min*1e3) / 0.78e-9 # max drift time for _drift_velocity_min

## dt hits
_dt_ts_individual_dead_time = 600 # dead time of a readout channel (0: no dead time cut)
dt_hit_add_ts_unc = 3 # time uncertainty of a hit in addition to the TDC binning (different drift paths)

## sl patterns: 4 hits of one superlayer within the max drift time
_t0_tolerance = 0 # tolerance of t0 beyond the max drift time
_dt_sl_patterns_ts_window_fit_vd = _dt_max_drift_time_vd_min + _t0_tolerance # free drift velocity
_dt_sl_patterns_ts_window = _dt_max_drift_time + _t0_tolerance # fixed drift velocity

# Pattern shapes: which 4 cells (one per layer) a muon can cross in a superlayer.
#   "rel_wis":    wire of layer 0, 1, 2, 3 relative to the wire of layer 3 (the reference cell, "O" in the sketches)
#   "laterality": the possible lateralities of the 4 hits (layer 0, 1, 2, 3): -1 = track at smaller h than the wire
#                 (left in the sketches), +1 = larger h (right)
# pat_type (branch of the sl patterns) = index of the shape in this dict: "+a" = 0, "-a" = 1, ...
# The wires of layers 0 and 2 are half a cell to the right of the wires of layers 1 and 3 (in all superlayers).
# Cells of the pattern = O, layer 3 on top:
#
# ly  [+A]                         [-A]
# 3   | - | - | O | - | - |        | - | - | O | - | - |
# 2     | - | - | O | - | - |        | - | O | - | - | - |
# 1   | - | - | O | - | - |        | - | - | O | - | - |
# 0     | - | - | O | - | - |        | - | O | - | - | - |
#
# ly  [+B]                         [-B]
# 3   | - | - | O | - | - |        | - | - | O | - | - |
# 2     | - | O | - | - | - |        | - | - | O | - | - |
# 1   | - | - | O | - | - |        | - | - | O | - | - |
# 0     | - | - | O | - | - |        | - | O | - | - | - |
#
# ly  [+C]                         [-C]
# 3   | - | - | O | - | - |        | - | - | O | - | - |
# 2     | - | - | O | - | - |        | - | O | - | - | - |
# 1   | - | - | - | O | - |        | - | O | - | - | - |
# 0     | - | - | O | - | - |        | - | O | - | - | - |
#
# (the wider shapes +D / -D are switched off: they cannot distinguish the lateralities rrll and llrr)
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

## sl fits: allowed track angle per pattern shape
_dt_pattern_alpha_range = { # pat_idx : [alpha_min, alpha_max] in rad
    0: [-1.0164888305933455 , 0.4939413689195812], # +a
    1: [-0.4939413689195812 , 1.0164888305933455], # -a
    2: [-1.0164888305933455 , 0], # +b
    3: [0 , 1.0164888305933455], # -b
    4: [-1.0164888305933455 , 0], # +c
    5: [0 , 1.0164888305933455], # -c
}

## super fits and dt muons
_muon_tgroup_tolerance = 20/0.78 # max t0 difference of fits which are combined (super patterns, muons)
_muon_slphi_tan_alpha_tolerance = 0.05 # max tan_alpha difference of the two phi sl fits of a super pattern
_muon_slphi_xproj_tolerance = 30 # max difference of the track positions of the two phi sl fits at z = _muon_reco_z0
_muon_reco_z0 = 144 # z where the muons are given

###############################
### SIMULATION
###############################

_dt_cell_efficiency = 96.93 * 1e-2 # from CMS 2024 performance

### angular distribution of cosmic muons (theta in [0, pi/2])
# flux ~ cos(theta)^(n-1), n ~ 3 (flat earth, https://arxiv.org/pdf/1606.06907), times sin(theta) for the solid angle
def cosmic_muon_theta_weight(theta):
    norm = 2/3
    return np.cos(theta)**2 * np.sin(theta) * 1/norm

###############################
### BRANCHES OF THE ROOT FILES {name: dtype}
###############################
# "sim_..." branches: simulation truth (0 in data)

# dt hits: the data word fields (_htg_keys) and
_dt_mapping_keys = {
    "sl": np.uint8,
    "ly": np.uint8,
    "wi": np.uint8,
    "conn_id": np.uint8, # index of the connector in the OBDT mapping
    "fe_id": np.uint8, # index in _fe_idx_list
    "ch_id": np.uint8, # conductor of the frontend connector, 0-15
}
_dt_other_keys = {
    "ts": _ts_type,
    "err_ts": np.float64,
    # simulation keys
    "sim_ts": _ts_type,
    "sim_dt": np.float64, # drift time (in ts units)
    "sim_dd": np.float64, # drift distance (in mm)
    "sim_id": np.uint64, # id / idx of correlated muon
    "sim_lat": np.int8, # hit laterality -1 (l) left of wire, +1 (r) right of wire
    "sim_tan_alpha": np.float64, # simulated correlated muon tan_alpha (sl projection)
    "sim_loc_x0": np.float64, # simulated correlated muon x0 (sl projection)
    "sim_vd": np.float64, # vd of simulated muon hit
    # simulation muon keys
    "sim_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "sim_y0": np.float64, # of sim muon
    "sim_z0": np.float64, # of sim muon
    "sim_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "sim_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
}

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
    "sim_ts": _ts_type, # ts of correlated muon
    "sim_dt0": np.float64, # drift time (in ts units) for sim muon hit in ly 0
    "sim_dt1": np.float64, # drift time (in ts units) for sim muon hit in ly 1
    "sim_dt2": np.float64, # drift time (in ts units) for sim muon hit in ly 2
    "sim_dt3": np.float64, # drift time (in ts units) for sim muon hit in ly 3
    "sim_dd0": np.float64, # drift distance (in mm) for sim muon hit in ly 0
    "sim_dd1": np.float64, # drift distance (in mm) for sim muon hit in ly 1
    "sim_dd2": np.float64, # drift distance (in mm) for sim muon hit in ly 2
    "sim_dd3": np.float64, # drift distance (in mm) for sim muon hit in ly 3
    "sim_id": np.uint64, # id / idx of correlated muon
    "sim_lat0": np.int8, # lat of correlated muon hit in ly0
    "sim_lat1": np.int8, # lat of correlated muon hit in ly1
    "sim_lat2": np.int8, # lat of correlated muon hit in ly2
    "sim_lat3": np.int8, # lat of correlated muon hit in ly3
    "sim_lat_id": np.int8, # hit laterality -1 (l) left of wire, +1 (r) right of wire
    "sim_tan_alpha": np.float64, # simulated correlated muon tan_alpha (sl projection)
    "sim_x0_loc": np.float64, # simulated correlated muon x0 (sl projection)
    "sim_vd": np.float64, # vd of simulated muon hit
    # simulation muon keys
    "sim_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "sim_y0": np.float64, # of sim muon
    "sim_z0": np.float64, # of sim muon
    "sim_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "sim_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
}

# sl fits: the sl pattern branches and the result of the best laterality
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
}
# ... and the results of every laterality
_sl_fit_other_keys = {
    f"lat{i}_{k}": np.float64 for k in ["impossible", "t0", "err_t0", "x0", "err_x0", "tan_alpha", "err_tan_alpha", "vd", "err_vd", "corr_t0_x0", "corr_t0_tan_alpha", "corr_t0_vd", "corr_x0_tan_alpha", "corr_x0_vd", "corr_tan_alpha_vd", "chi2/ndf", "dt0", "dt1", "dt2", "dt3"] for i in range(4)
}

# dt muons
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
    "sim_id": np.uint64, # id / idx of correlated muon (used to compare simulation + reconstruction)
    # simulation muon keys
    "sim_ts": np.float64, # timestamp, in tu - of sim muon
    "sim_x0": np.float64, # reference point (x0,y0,z0), in mm - of sim muon
    "sim_y0": np.float64, # of sim muon
    "sim_z0": np.float64, # of sim muon
    "sim_theta": np.float64, # theta angle (angle relative to z axis), in rad - of sim muon
    "sim_phi": np.float64, # phi angle (angle relative to x axis, between x and y axis), in rad - of sim muon
}

###############################
### PLOTTING
###############################

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

_wire_draw_radius = 0.5 # drawn size of a wire (much larger than the real one)
_wire_draw_linewidth = 0.5
_plot_z_margin = 11.5 # space above / below the hits in the event displays
_legend_alpha = 0.7
_hist_info_alpha = 0.7
_info_font_size = 12

### axis labels: symbol and unit of a branch
_key_symbols = {
    "x0": "$x_0$",
    "y0": "$y_0$",
    "z0": "$z_0$",
    "ts": "$T$",
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
    "ts_orbit": "$T_\\text{orbit}$",
    "laterality": "Laterality",
    "t0": "$T_0$",
    "tan_alpha": "$\\text{tan}\\alpha$",
    "chi2/ndf": "$\\chi^2/n_\\text{df}$",
    "pat_type": "SL pattern type",
    "wi3": "Wire, Layer 3",
    "ts3": "$T_\\text{Wire, Layer 3}$",
    "sim_dt": "$t_\\text{d,sim}$",
    "sim_dd": "$x_\\text{d,sim}$",
    "sim_lat": "$\\text{Laterality}_\\text{sim}$",
    "sim_x0_loc": "$x0_\\text{loc, sim}$",
    "sim_tan_alpha": "$\\text{tan}\\alpha_\\text{sim}$",
    "sim_ts": "$T_\\text{0, sim}$",
    "sim_dt0": "$t_\\text{drift, ly 0, sim}$",
    "sim_dt1": "$t_\\text{drift, ly 1, sim}$",
    "sim_dt2": "$t_\\text{drift, ly 2, sim}$",
    "sim_dt3": "$t_\\text{drift, ly 3, sim}$",
    "sim_dd0": "$x_\\text{drift, ly 0, sim}$",
    "sim_dd1": "$x_\\text{drift, ly 1, sim}$",
    "sim_dd2": "$x_\\text{drift, ly 2, sim}$",
    "sim_dd3": "$x_\\text{drift, ly 3, sim}$",
    "sim_id": "Simulated muon ID",
    "sim_x0": "$x0_\\text{sim}$",
    "sim_lat0": "Laterality hit$_\\text{ly 0, sim}$",
    "sim_lat1": "Laterality hit$_\\text{ly 1, sim}$",
    "sim_lat2": "Laterality hit$_\\text{ly 2, sim}$",
    "sim_lat3": "Laterality hit$_\\text{ly 3, sim}$",
    "sim_lat_id": "Pattern laterality$_\\text{sim}$",
    "sim_vd": "$v_\\text{drift, sim}$",
    "vd": "$v_\\text{drift}$",
    "dt": "$t_\\text{drift}$",
    "dt0": "$t_\\text{drift, ly 0}$",
    "dt1": "$t_\\text{drift, ly 1}$",
    "dt2": "$t_\\text{drift, ly 2}$",
    "dt3": "$t_\\text{drift, ly 3}$",
    "err_ts": "$\\sigma_T$",
    "err_t0": "$\\sigma_{T_0}$",
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
}

_key_units = {
    "x0": "mm",
    "y0": "mm",
    "z0": "mm",
    "ts": "TU", # timestamp unit: "$0.78\;\\text{ns}$",
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
    "ts_orbit": "TU",
    "laterality": "",
    "t0": "TU",
    "tan_alpha": "",
    "chi2/ndf": "",
    "pat_type": "",
    "wi3": "",
    "ts3": "TU",
    "sim_x0_loc": "mm",
    "sim_dt": "TU",
    "sim_dd": "mm",
    "sim_lat": "",
    "sim_x0": "mm",
    "sim_tan_alpha": "",
    "sim_ts": "TU",
    "sim_dt0": "TU",
    "sim_dt1": "TU",
    "sim_dt2": "TU",
    "sim_dt3": "TU",
    "sim_dd0": "mm",
    "sim_dd1": "mm",
    "sim_dd2": "mm",
    "sim_dd3": "mm",
    "sim_id": "",
    "sim_lat0": "",
    "sim_lat1": "",
    "sim_lat2": "",
    "sim_lat3": "",
    "sim_lat_id": "",
    "sim_vd": "mm/TU",
    "vd": "mm/TU",
    "dt": "TU",
    "dt0": "TU",
    "dt1": "TU",
    "dt2": "TU",
    "dt3": "TU",
    "err_t0": "TU",
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
}

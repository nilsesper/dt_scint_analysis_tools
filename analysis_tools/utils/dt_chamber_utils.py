###########################################
### DT CHAMBER: cells, readout channels, positions
###########################################
# Everything about the chamber is built from params._dt_chamber (geometry) and params._dt_mapping (readout).
# Lengths in mm.
#
# ---------------------------------------------------------------------------------------------------------------
# 1. CHAMBER FRAME (x, y, z)
# ---------------------------------------------------------------------------------------------------------------
# All positions in params._dt_chamber are given in this frame. z points up. The chamber has 3 superlayers (SL):
#
#       z [mm]   side view, y points into the page
#        ^
#    287 |   +-------------------------------------------------+
#        |   | SL 3 (phi):   wires along y  -> measures x      |
#    234 |   +-------------------------------------------------+
#    234 |   +-------------------------------------------------+
#        |   | SL 2 (theta): wires along x  -> measures y      |
#    181 |   +-------------------------------------------------+
#        |
#     53 |   +-------------------------------------------------+
#        |   | SL 1 (phi):   wires along y  -> measures x      |
#      0 |   +-------------------------------------------------+
#        +-------------------------------------------------------> x [mm]   (0 ... about 2100)
#
# A hit only tells in which cell (sl, ly, wi) a muon passed, and when. A phi superlayer therefore measures the track
# in the x-z plane ("phi view"), the theta superlayer in the y-z plane ("theta view"). In this code "h" is the axis
# a superlayer measures: h = x for a phi superlayer, h = y for the theta superlayer.
#
# One superlayer, seen along its wires (o = wire in the middle of the cell):
#
#       z
#       ^   layer 3   |  o  |  o  |  o  |  o  |  o  |
#       |   layer 2      |  o  |  o  |  o  |  o  |  o  |      a cell is 42 mm wide (along h) and 13 mm high (along z)
#       |   layer 1   |  o  |  o  |  o  |  o  |  o  |         layers 1 and 3 are shifted by half a cell (21 mm)
#       |   layer 0      |  o  |  o  |  o  |  o  |  o  |      against layers 0 and 2
#       +-------------------------------------------------> h
#
# Drift: the muon passes the cell at the distance d from the wire. The hit comes at ts = t0 + d / vd, with t0 the time
# the muon crossed the chamber and vd the drift velocity. The hit does not tell on which side of the wire the muon
# passed: this is the "laterality" (-1: track at smaller h than the wire, +1: at larger h).
#
# ---------------------------------------------------------------------------------------------------------------
# 2. TRACK FRAME (h, z) of a fit
# ---------------------------------------------------------------------------------------------------------------
# The fits do not use the chamber frame, but a frame whose origin is a reference wire:
#   - sl fit:    the wire of layer 3 of the pattern (branch "wi3"),
#   - super fit: the wire of layer 3 of the pattern in the upper phi superlayer (SL 3).
# In this frame a straight track is
#       h(z) = x0 + z * tan_alpha
# x0 = position of the track at the height of the reference wire, relative to the reference wire (|x0| <= 21 mm)
# tan_alpha = slope dh / dz (alpha = angle to the vertical in the view of the superlayer)
# and the hit in the cell with the wire at (h_wire, z_wire) comes at ts = t0 + laterality * (h(z_wire) - h_wire) / vd.
#
# Back to the chamber frame (track_position_in_chamber):
#       h_chamber(z) = h_ref + x0 + (z - z_ref) * tan_alpha        with (h_ref, z_ref) = reference wire in the chamber frame
# For super fits (h_ref, z_ref) is stored in the branches "ref_x", "ref_z".
#
# ---------------------------------------------------------------------------------------------------------------
# 3. MUONS
# ---------------------------------------------------------------------------------------------------------------
# A reconstructed (or simulated) muon is a straight line in the chamber frame: the point (x0, y0, z0) and the
# direction theta (angle to the z axis) and phi (angle in the x-y plane, measured from the x axis).
# Its slope in the x-z plane is tan(theta) * cos(phi), in the y-z plane tan(theta) * sin(phi).
# Reconstructed muons are given at z0 = params._muon_reco_z0.
#
# A box (cell, superlayer, chamber) is a dict {"low": [x, y, z], "high": [x, y, z], "center": [x, y, z]};
# the center of a cell is the position of its wire.

import copy
import numpy as np

from analysis_tools.params import params, derived_params

# -----------------------------------------

X = 0
Y = 1
Z = 2
MEASURED_AXIS = {"phi": X, "theta": Y}

# -----------------------------------------
# superlayers, layers, wires
# -----------------------------------------

def superlayers():
    return list(params._dt_chamber["sls"].keys())

def phi_superlayers():
    result = []
    for sl in superlayers():
        if params._dt_chamber["sls"][sl]["orient"] == "phi":
            result.append(sl)
    return result

def theta_superlayer():
    for sl in superlayers():
        if params._dt_chamber["sls"][sl]["orient"] == "theta":
            return sl
    return None

def layers(sl):
    return list(params._dt_chamber["sls"][sl]["lys"].keys())

def wires(sl, ly):
    layer = params._dt_chamber["sls"][sl]["lys"][ly]
    return list(range(layer["min_wi"], layer["max_wi"] + 1))

### all cells of the chamber as list of (sl, ly, wi)
def chamber_cells():
    cells = []
    for sl in superlayers():
        for ly in layers(sl):
            for wi in wires(sl, ly):
                cells.append((sl, ly, wi))
    return cells

### cells left out of the analysis (params._dt_wire_mask and params._dt_dead_wires) as a set of (sl, ly, wi)
def excluded_cells():
    cells = set()
    for table in [params._dt_wire_mask, params._dt_dead_wires]:
        for sl in table:
            for ly in table[sl]:
                for wi in table[sl][ly]:
                    cells.add((sl, ly, wi))
    return cells

### {sl: {ly: {wi: copy of content}}} for every cell of the chamber
def chamber_map(content):
    result = {}
    for sl in superlayers():
        result[sl] = {}
        for ly in layers(sl):
            result[sl][ly] = {}
            for wi in wires(sl, ly):
                result[sl][ly][wi] = copy.deepcopy(content)
    return result

# -----------------------------------------
# readout channels
# -----------------------------------------

### cell of a readout channel: {"sl", "ly", "wi", "conn_id", "fe_id", "ch_id"}, or None if the channel has no cell
def cell_of_channel(ro_ch, ch):
    if ro_ch not in derived_params._dt_remap_table:
        return None
    if ch not in derived_params._dt_remap_table[ro_ch]:
        return None
    return derived_params._dt_remap_table[ro_ch][ch]

### readout keys (ro_ch, ch, fe_id, conn_id, ch_id) of a cell
def readout_keys_of_cell(sl, ly, wi):
    channel = derived_params._dt_inverted_remap_table[sl][ly][wi]
    keys = {}
    for key in ["ro_ch", "ch", "fe_id", "conn_id", "ch_id"]:
        keys[key] = channel[key]
    return keys

### frontend connector name (e.g. "5A") of a cell
def fe_name_of_cell(sl, ly, wi):
    return params._fe_idx_list[derived_params._dt_inverted_remap_table[sl][ly][wi]["fe_id"]]

# -----------------------------------------
# positions in the chamber frame
# -----------------------------------------

def orientation(sl):
    return params._dt_chamber["sls"][sl]["orient"]

### axis measured by a superlayer: X for phi, Y for theta
def measured_axis(sl):
    return MEASURED_AXIS[orientation(sl)]

def box_size(box, axis):
    return box["high"][axis] - box["low"][axis]

### box from its lower corner and its size
def corner_box(low, size):
    high = []
    center = []
    for axis in range(len(low)):
        high.append(low[axis] + size[axis])
        center.append(low[axis] + size[axis] / 2)
    return {"low": list(low), "high": high, "center": center}

### all cells: CELLS[sl][ly][wi] = box; the cells of a layer follow each other along the measured axis,
# starting with the cell of wire 0 at "cell_0" (lower corner) of params._dt_chamber
def build_cells():
    cells = {}
    for sl in superlayers():
        superlayer = params._dt_chamber["sls"][sl]
        axis = measured_axis(sl)
        size = superlayer["cell_size"]
        cells[sl] = {}
        for ly in layers(sl):
            cells[sl][ly] = {}
            for wi in wires(sl, ly):
                low = list(superlayer["lys"][ly]["cell_0"])
                low[axis] = low[axis] + wi * size[axis]
                cells[sl][ly][wi] = corner_box(low, size)
    return cells

CELLS = build_cells()

def cell(sl, ly, wi):
    return CELLS[sl][ly][wi]

def is_cell(sl, ly, wi):
    return wi in CELLS[sl][ly]

### position of the wire of a cell: (h, z), with h along the axis the superlayer measures
def wire_position(sl, ly, wi):
    box = cell(sl, ly, wi)
    return box["center"][measured_axis(sl)], box["center"][Z]

### z of the wires of a layer
def layer_z(sl, ly):
    first_wire = wires(sl, ly)[0]
    return cell(sl, ly, first_wire)["center"][Z]

### width of a cell along the axis the superlayer measures (42 mm)
def cell_width(sl):
    return params._dt_chamber["sls"][sl]["cell_size"][measured_axis(sl)]

### outline of a superlayer / the chamber (from "pos" and "size" in params._dt_chamber, only used for drawing)
def box_from_pos_size(pos, size):
    low, high, center = [], [], []
    for axis in range(3):
        low.append(np.amin([pos[axis], pos[axis] + size[axis]]))
        high.append(np.amax([pos[axis], pos[axis] + size[axis]]))
        center.append(np.mean([pos[axis], pos[axis] + size[axis]]))
    return {"low": low, "high": high, "center": center}

def superlayer_box(sl):
    return box_from_pos_size(params._dt_chamber["sls"][sl]["pos"], params._dt_chamber["sls"][sl]["size"])

def chamber_box():
    return box_from_pos_size(params._dt_chamber["pos"], params._dt_chamber["size"])

### (lowest, highest) value along an axis covered by all superlayers
def superlayers_range(axis):
    lows, highs = [], []
    for sl in superlayers():
        lows.append(superlayer_box(sl)["low"][axis])
        highs.append(superlayer_box(sl)["high"][axis])
    return np.amin(lows), np.amax(highs)

### the upper of the two phi superlayers (its layer 3 is the reference of the super fits)
def find_top_phi_superlayer():
    sl_1, sl_2 = phi_superlayers()
    if layer_z(sl_1, 3) >= layer_z(sl_2, 3):
        return sl_1
    return sl_2

TOP_PHI_SUPERLAYER = find_top_phi_superlayer()

# -----------------------------------------
# tracks (see "2. TRACK FRAME" and "3. MUONS" above)
# -----------------------------------------

### position of the wire (sl, ly, wi) in the track frame with the reference wire (ref_sl, layer 3, ref_wi): (h, z)
def position_in_track_frame(sl, ly, wi, ref_sl, ref_wi):
    h, z = wire_position(sl, ly, wi)
    h_ref, z_ref = wire_position(ref_sl, 3, ref_wi)
    return h - h_ref, z - z_ref

### allowed range of x0 for the laterality of the reference cell: the half of the cell on that side of the wire
def x0_range(ref_sl, laterality_ref):
    half_width = cell_width(ref_sl) / 2
    if laterality_ref == -1:
        return -half_width, 0.0
    if laterality_ref == 1:
        return 0.0, half_width
    return 0.0, 0.0

### track h(z) = x0 + z * tan_alpha in the track frame, and its uncertainty
def track_position(z, x0, tan_alpha):
    return x0 + z * tan_alpha

def err_track_position(z, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    return np.sqrt(err_x0**2 + z**2 * err_tan_alpha**2 + 2 * z * corr_x0_tan_alpha)

### the same track in the chamber frame at height z; (h_ref, z_ref) = reference wire in the chamber frame
def track_position_in_chamber(h_ref, z_ref, x0, tan_alpha, z):
    return track_position(z - z_ref, x0, tan_alpha) + h_ref

def err_track_position_in_chamber(z_ref, z, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    return err_track_position(z - z_ref, err_x0, err_tan_alpha, corr_x0_tan_alpha)

### a muon (x0, y0, z0, theta, phi) seen in the view of orient ("phi": x-z, "theta": y-z): its position at height z
def muon_track_position(orient, z, x0, y0, z0, theta, phi):
    if orient == "phi":
        return x0 + np.tan(theta) * np.cos(phi) * (z - z0)
    return y0 + np.tan(theta) * np.sin(phi) * (z - z0)

def err_muon_track_position(orient, z, x0, y0, z0, theta, phi, err_x0, err_y0, err_z0, err_theta, err_phi):
    if orient == "phi":
        err_base = err_x0
        tan_alpha = np.tan(theta) * np.cos(phi)
        err_tan_alpha = np.sqrt((1 / np.cos(theta)**2 * np.cos(phi))**2 * err_theta**2 + (np.tan(theta) * (-np.sin(phi)))**2 * err_phi**2)
    else:
        err_base = err_y0
        tan_alpha = np.tan(theta) * np.sin(phi)
        err_tan_alpha = np.sqrt((1 / np.cos(theta)**2 * np.cos(phi))**2 * err_theta**2 + (np.tan(theta) * (np.cos(phi)))**2 * err_phi**2)
    return np.sqrt(err_base**2 + (z - z0)**2 * err_tan_alpha**2 + (-tan_alpha)**2 * err_z0**2)

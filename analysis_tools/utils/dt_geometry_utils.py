###########################################
### DT CHAMBER GEOMETRY
###########################################
# Built from params._dt_chamber. Three frames are used:
#
# chamber frame (x, y, z): see params.py. A phi superlayer measures x, the theta superlayer measures y; the 2d view
#   of a superlayer ("phi": x-z, "theta": y-z) shows the axis it measures horizontally and z vertically.
#
# pattern frame (h, z): the 4 cells of an sl pattern, h along the measured axis, with the wire of the layer 3 cell at
#   (0, 0). Cells are addressed by layer and wire relative to the wire of layer 3 ("rel_wi"). All superlayers have the
#   same layout, so the pattern frame is the same everywhere; sl fits give x0 (at z = 0) and tan_alpha in this frame.
#
# super pattern frame (x, z): chamber frame shifted so that SUPER_FRAME_ORIGIN (the wire FRAME_REFERENCE_WIRE of the
#   topmost layer of the phi superlayers) is at (0, 0). Each super fit is done in this frame shifted once more, to the
#   topmost wire of its own pattern ("ref_x", "ref_z" are stored with the fit).
#
# Tracks (all frames): a straight track is h(z) = x0 + z * tan_alpha. The hit times it gives (the fit model) are in
# dt_track_fit_utils.
#
# A box (cell, superlayer, chamber) is a dict {"low": [...], "high": [...], "center": [...]} with one value per axis
# (cell: the center is the wire position).

import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils

# -----------------------------------------

X = 0
Y = 1
Z = 2
MEASURED_AXIS = {"phi": X, "theta": Y}
VIEW_AXES = {"phi": (X, Z), "theta": (Y, Z)}  # (horizontal, vertical) axis of the 2d view of a superlayer
FRAME_REFERENCE_WIRE = 10  # wire used to place the pattern frame and the super pattern frame (any wire of the chamber would do)

def orientation(sl):
    return params._dt_chamber["sls"][sl]["orient"]

def measured_axis(sl):
    return MEASURED_AXIS[orientation(sl)]

def view_axes(orient):
    return VIEW_AXES[orient]

def box_size(box, axis):
    return box["high"][axis] - box["low"][axis]

# -----------------------------------------
# chamber frame
# -----------------------------------------

### box from its lower corner and its size
def corner_box(low, size):
    high = []
    center = []
    for axis in range(len(low)):
        high.append(low[axis] + size[axis])
        center.append(low[axis] + size[axis] / 2)
    return {"low": list(low), "high": high, "center": center}

### all cells: CELLS[sl][ly][wi] = box
def build_cells():
    cells = {}
    for sl in dt_chamber_utils.superlayers():
        superlayer = params._dt_chamber["sls"][sl]
        axis = MEASURED_AXIS[superlayer["orient"]]
        size = superlayer["cell_size"]
        cells[sl] = {}
        for ly in dt_chamber_utils.layers(sl):
            cells[sl][ly] = {}
            for wi in dt_chamber_utils.wires(sl, ly):
                low = list(superlayer["lys"][ly]["cell_0"])
                low[axis] = low[axis] + wi * size[axis]  # the cells of a layer follow each other along the measured axis
                cells[sl][ly][wi] = corner_box(low, size)
    return cells

CELLS = build_cells()

def cell(sl, ly, wi):
    return CELLS[sl][ly][wi]

def is_cell(sl, ly, wi):
    return wi in CELLS[sl][ly]

### position of the wire of a cell in the 2d view of its superlayer: (along the measured axis, z)
def wire_position(sl, ly, wi):
    box = cell(sl, ly, wi)
    return box["center"][measured_axis(sl)], box["center"][Z]

### z of the wires of a layer
def layer_z(sl, ly):
    first_wire = dt_chamber_utils.wires(sl, ly)[0]
    return cell(sl, ly, first_wire)["center"][Z]

### box of the superlayer or chamber outline (from pos and size in params.py)
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
    for sl in dt_chamber_utils.superlayers():
        lows.append(superlayer_box(sl)["low"][axis])
        highs.append(superlayer_box(sl)["high"][axis])
    return np.amin(lows), np.amax(highs)

# -----------------------------------------
# pattern frame
# -----------------------------------------

### cells of the pattern frame: PATTERN_CELLS[ly][rel_wi] = box over (h, z), rel_wi from -2 to 2
# taken from the first phi superlayer around FRAME_REFERENCE_WIRE, shifted so that the wire of layer 3 is at (0, 0)
def build_pattern_cells():
    sl = dt_chamber_utils.phi_superlayers()[0]
    axis = measured_axis(sl)
    size = params._dt_chamber["sls"][sl]["cell_size"]
    reference = cell(sl, 3, FRAME_REFERENCE_WIRE)
    boxes = {}
    for ly in dt_chamber_utils.layers(sl):
        boxes[ly] = {}
        for rel_wi in range(-2, 3):
            box = cell(sl, ly, FRAME_REFERENCE_WIRE + rel_wi)
            low = [box["low"][axis] - reference["low"][axis], box["low"][Z] - reference["low"][Z]]
            boxes[ly][rel_wi] = corner_box(low, [size[axis], size[Z]])
    origin = boxes[3][0]["center"]  # the wire of layer 3
    pattern_cells = {}
    for ly in boxes:
        pattern_cells[ly] = {}
        for rel_wi in boxes[ly]:
            box = boxes[ly][rel_wi]
            shifted = {"low": [], "high": [], "center": []}
            for corner in ["low", "high", "center"]:
                for i in range(2):
                    shifted[corner].append(np.float64(box[corner][i]) - origin[i])
            pattern_cells[ly][rel_wi] = shifted
    return pattern_cells

PATTERN_CELLS = build_pattern_cells()

def pattern_cell(ly, rel_wi):
    return PATTERN_CELLS[ly][rel_wi]

def pattern_layer_z(ly):
    return PATTERN_CELLS[ly][0]["center"][1]

### an sl fit (pattern frame of its pattern, wire wi_layer3 in layer 3): track position in the chamber frame along the
# measured axis at height z, and its uncertainty
def sl_track_position(sl, wi_layer3, x0, tan_alpha, z):
    wire_h, wire_z = wire_position(sl, 3, wi_layer3)
    return track_position(z=-wire_z + z, x0=x0, tan_alpha=tan_alpha) + wire_h

def err_sl_track_position(sl, wi_layer3, x0, tan_alpha, z, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    wire_h, wire_z = wire_position(sl, 3, wi_layer3)
    return err_track_position(z=-wire_z + z, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)

# -----------------------------------------
# super pattern frame
# -----------------------------------------

def find_top_phi_superlayer():
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    if cell(sl_1, 3, FRAME_REFERENCE_WIRE)["center"][Z] >= cell(sl_2, 3, FRAME_REFERENCE_WIRE)["center"][Z]:
        return sl_1
    return sl_2

TOP_PHI_SUPERLAYER = find_top_phi_superlayer()
SUPER_FRAME_ORIGIN = (cell(TOP_PHI_SUPERLAYER, 3, FRAME_REFERENCE_WIRE)["center"][X], cell(TOP_PHI_SUPERLAYER, 3, FRAME_REFERENCE_WIRE)["center"][Z])

### wire position (x, z) of a cell of a phi superlayer in the super pattern frame
def super_frame_position(sl, ly, wi):
    box = cell(sl, ly, wi)
    return box["center"][X] - SUPER_FRAME_ORIGIN[0], box["center"][Z] - SUPER_FRAME_ORIGIN[1]

### range of x0 (track position at the height of the topmost layer) in the super pattern frame: the half of the cell of
# wire wi_top (layer 3 of the top phi superlayer) given by the laterality there
def super_frame_x0_range(wi_top, laterality_top):
    box = cell(TOP_PHI_SUPERLAYER, 3, wi_top)
    low = box["low"][X] - SUPER_FRAME_ORIGIN[0]
    high = box["high"][X] - SUPER_FRAME_ORIGIN[0]
    wire = box["center"][X] - SUPER_FRAME_ORIGIN[0]
    if laterality_top == -1:
        return low, wire
    if laterality_top == 1:
        return wire, high
    return wire, wire

# -----------------------------------------
# straight tracks
# -----------------------------------------

def track_position(z, x0, tan_alpha):
    return x0 + z * tan_alpha

def err_track_position(z, x0, tan_alpha, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    return np.sqrt(err_x0**2 + z**2 * err_tan_alpha**2 + 2 * z * corr_x0_tan_alpha)

### a muon (x0, y0, z0, theta, phi) seen in the 2d view of orient: position along the horizontal axis at height z
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

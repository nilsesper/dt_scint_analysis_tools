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
# super pattern frame (x, z): chamber frame shifted so that SUPER_FRAME_ORIGIN (the wire of the topmost layer of the
#   phi superlayers, at FRAME_REFERENCE_WIRE) is at (0, 0). Each super fit is done in this frame shifted once more,
#   to the topmost wire of its own pattern ("ref_x", "ref_z" are stored with the fit).
#
# Track model (all frames): a straight track h(z) = x0 + z * tan_alpha crossing at time t0 gives a hit in the cell with
# the wire at (h_cell, z) at ts = t0 + laterality * (x0 + z * tan_alpha - h_cell) / vd.

import dataclasses
import functools
import numpy as np

from analysis_tools.params import params
from analysis_tools.utils import dt_chamber_utils

# -----------------------------------------

X, Y, Z = 0, 1, 2
MEASURED_AXIS = {"phi": X, "theta": Y}
VIEW_AXES = {"phi": (X, Z), "theta": (Y, Z)}  # (horizontal, vertical) axis of the 2d view of a superlayer
FRAME_REFERENCE_WIRE = 10  # wire used to place the pattern frame and the super pattern frame (any wire of the chamber would do)

### a box: lower corner, upper corner, centre (for a cell: the wire), each one value per axis
@dataclasses.dataclass(frozen=True)
class Box:
    low: tuple
    high: tuple
    center: tuple

    def size(self, axis):
        return self.high[axis] - self.low[axis]

def orientation(sl):
    return params._dt_chamber["sls"][sl]["orient"]

def measured_axis(sl):
    return MEASURED_AXIS[orientation(sl)]

def view_axes(orient):
    return VIEW_AXES[orient]

# -----------------------------------------
# chamber frame
# -----------------------------------------

def _corner_box(low, size):
    return Box(low=tuple(low), high=tuple(low[a] + size[a] for a in range(len(low))), center=tuple(low[a] + size[a] / 2 for a in range(len(low))))

@functools.cache
def _cells():
    cells = {}
    for sl, superlayer in params._dt_chamber["sls"].items():
        axis, size = MEASURED_AXIS[superlayer["orient"]], superlayer["cell_size"]
        for ly, layer in superlayer["lys"].items():
            for wi in range(layer["min_wi"], layer["max_wi"] + 1):
                low = list(layer["cell_0"])
                low[axis] = low[axis] + wi * size[axis]
                cells.setdefault(sl, {}).setdefault(ly, {})[wi] = _corner_box(low, size)
    return cells

def cell(sl, ly, wi):
    return _cells()[sl][ly][wi]

def is_cell(sl, ly, wi):
    return wi in _cells()[sl][ly]

### position of the wire of a cell in the 2d view of its superlayer: (along the measured axis, z)
def wire_position(sl, ly, wi):
    c = cell(sl, ly, wi)
    return c.center[measured_axis(sl)], c.center[Z]

def layer_z(sl, ly):
    return cell(sl, ly, params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]).center[Z]

def _box_from_pos_size(pos, size):
    low = tuple(np.amin([pos[a], pos[a] + size[a]]) for a in range(3))
    high = tuple(np.amax([pos[a], pos[a] + size[a]]) for a in range(3))
    return Box(low=low, high=high, center=tuple(np.mean([pos[a], pos[a] + size[a]]) for a in range(3)))

def superlayer_box(sl):
    return _box_from_pos_size(params._dt_chamber["sls"][sl]["pos"], params._dt_chamber["sls"][sl]["size"])

def chamber_box():
    return _box_from_pos_size(params._dt_chamber["pos"], params._dt_chamber["size"])

### range along an axis covered by all superlayers
def superlayers_range(axis):
    boxes = [superlayer_box(sl) for sl in dt_chamber_utils.superlayers()]
    return np.amin([b.low[axis] for b in boxes]), np.amax([b.high[axis] for b in boxes])

# -----------------------------------------
# pattern frame
# -----------------------------------------

def _pattern_reference():
    return dt_chamber_utils.phi_superlayers()[0], 3, FRAME_REFERENCE_WIRE

### cell of layer ly and wire rel_wi (relative to the wire of layer 3) in the pattern frame, as Box over (h, z)
@functools.cache
def pattern_cell(ly, rel_wi):
    sl, ref_ly, ref_wi = _pattern_reference()
    axis = measured_axis(sl)
    size = params._dt_chamber["sls"][sl]["cell_size"]

    def corner_box(ly, rel_wi):
        c, ref = cell(sl, ly, ref_wi + rel_wi), cell(sl, ref_ly, ref_wi)
        return _corner_box([c.low[axis] - ref.low[axis], c.low[Z] - ref.low[Z]], [size[axis], size[Z]])

    origin = corner_box(ref_ly, 0).center  # the reference wire
    box = corner_box(ly, rel_wi)
    return Box(low=tuple(np.float64(v) - o for v, o in zip(box.low, origin)), high=tuple(np.float64(v) - o for v, o in zip(box.high, origin)),
               center=tuple(np.float64(v) - o for v, o in zip(box.center, origin)))

def pattern_layer_z(ly):
    return pattern_cell(ly, 0).center[1]

### an sl fit (pattern frame of its pattern, wire wi_layer3 in layer 3): track position in the chamber frame along the
# measured axis at height z, and its uncertainty
def sl_track_position(sl, wi_layer3, x0, tan_alpha, z):
    wire_h, wire_z = wire_position(sl, 3, wi_layer3)
    return track_position(z=-wire_z + z, x0=x0, tan_alpha=tan_alpha) + wire_h

def err_sl_track_position(sl, wi_layer3, x0, tan_alpha, z, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    _, wire_z = wire_position(sl, 3, wi_layer3)
    return err_track_position(z=-wire_z + z, x0=x0, tan_alpha=tan_alpha, err_x0=err_x0, err_tan_alpha=err_tan_alpha, corr_x0_tan_alpha=corr_x0_tan_alpha)

# -----------------------------------------
# super pattern frame
# -----------------------------------------

def _top_phi_superlayer():
    sl_1, sl_2 = dt_chamber_utils.phi_superlayers()
    return sl_1 if cell(sl_1, 3, FRAME_REFERENCE_WIRE).center[Z] >= cell(sl_2, 3, FRAME_REFERENCE_WIRE).center[Z] else sl_2

TOP_PHI_SUPERLAYER = _top_phi_superlayer()
SUPER_FRAME_ORIGIN = (cell(TOP_PHI_SUPERLAYER, 3, FRAME_REFERENCE_WIRE).center[X], cell(TOP_PHI_SUPERLAYER, 3, FRAME_REFERENCE_WIRE).center[Z])

### wire position (x, z) of a cell of a phi superlayer in the super pattern frame
@functools.cache
def super_frame_position(sl, ly, wi):
    c = cell(sl, ly, wi)
    return c.center[X] - SUPER_FRAME_ORIGIN[0], c.center[Z] - SUPER_FRAME_ORIGIN[1]

### range of x0 (track position at the height of the topmost layer) in the super pattern frame: the half of the cell of
# wire wi_top (layer 3 of the top phi superlayer) given by the laterality there
@functools.cache
def super_frame_x0_range(wi_top, laterality_top):
    c = cell(TOP_PHI_SUPERLAYER, 3, wi_top)
    low, high, wire = c.low[X] - SUPER_FRAME_ORIGIN[0], c.high[X] - SUPER_FRAME_ORIGIN[0], c.center[X] - SUPER_FRAME_ORIGIN[0]
    return (low if laterality_top == -1 else wire), (high if laterality_top == 1 else wire)

# -----------------------------------------
# track model
# -----------------------------------------

def hit_time(x_cell, t0, x0, tan_alpha, z, laterality, vd):
    return (x0 + z * tan_alpha - x_cell) * laterality / vd + t0

def err_hit_time(x_cell, t0, x0, tan_alpha, z, laterality, vd, *, err_t0, err_x0, err_tan_alpha, err_vd, corr_t0_x0, corr_t0_tan_alpha, corr_x0_tan_alpha,
                 corr_t0_vd, corr_x0_vd, corr_tan_alpha_vd):
    d_t0, d_x0, d_tan_alpha = 1, laterality / vd, z * laterality / vd
    d_vd = -(x0 + z * tan_alpha - x_cell) * laterality / vd**2
    return np.sqrt(
          d_t0**2 * err_t0**2 + d_x0**2 * err_x0**2 + d_tan_alpha**2 * err_tan_alpha**2 + d_vd**2 * err_vd**2
        + 2 * d_t0 * d_x0 * corr_t0_x0 + 2 * d_t0 * d_tan_alpha * corr_t0_tan_alpha + 2 * d_t0 * d_vd * corr_t0_vd
        + 2 * d_x0 * d_tan_alpha * corr_x0_tan_alpha + 2 * d_x0 * d_vd * corr_x0_vd + 2 * d_tan_alpha * d_vd * corr_tan_alpha_vd
    )

def track_position(z, x0, tan_alpha):
    return x0 + z * tan_alpha

def err_track_position(z, x0, tan_alpha, err_x0, err_tan_alpha, corr_x0_tan_alpha):
    return np.sqrt(err_x0**2 + z**2 * err_tan_alpha**2 + 2 * z * corr_x0_tan_alpha)

### a muon (x0, y0, z0, theta, phi) seen in the 2d view of orient: position along the horizontal axis at height z
def muon_track_position(orient, z, x0, y0, z0, theta, phi):
    base = x0 if orient == "phi" else y0
    tan_alpha = np.tan(theta) * np.cos(phi) if orient == "phi" else np.tan(theta) * np.sin(phi)
    return base + tan_alpha * (z - z0)

def err_muon_track_position(orient, z, x0, y0, z0, theta, phi, err_x0, err_y0, err_z0, err_theta, err_phi):
    err_base = err_x0 if orient == "phi" else err_y0
    if orient == "phi":
        tan_alpha = np.tan(theta) * np.cos(phi)
        err_tan_alpha = np.sqrt((1 / np.cos(theta)**2 * np.cos(phi))**2 * err_theta**2 + (np.tan(theta) * (-np.sin(phi)))**2 * err_phi**2)
    else:
        tan_alpha = np.tan(theta) * np.sin(phi)
        err_tan_alpha = np.sqrt((1 / np.cos(theta)**2 * np.cos(phi))**2 * err_theta**2 + (np.tan(theta) * (np.cos(phi)))**2 * err_phi**2)
    return np.sqrt(err_base**2 + (z - z0)**2 * err_tan_alpha**2 + (-tan_alpha)**2 * err_z0**2)

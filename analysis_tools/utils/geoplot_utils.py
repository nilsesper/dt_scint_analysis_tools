###########################################
### DRAWING THE DT CHAMBER
###########################################
# Patches (matplotlib) of the chamber in the 2d view of a superlayer orientation ("phi": x-z, "theta": y-z), and of
# the cells of an sl pattern in the pattern frame. cell_data: {sl: {ly: {wi: {"color", "text"}}}} (dt_chamber_utils.cell_display_map)

import copy
import matplotlib.patches as pat

import analysis_tools.params.params as params
from analysis_tools.utils import dt_chamber_utils, dt_geometry_utils as geometry

# -----------------------------------------

def _rectangle(box, h_axis, v_axis, **kwargs):
    return pat.Rectangle((box.low[h_axis], box.low[v_axis]), width=box.size(h_axis), height=box.size(v_axis), **kwargs)

### one cell: in the view of its own superlayer orientation a box with the wire as dot; in the other view the cells of a
# layer lie behind each other, so only the first cell is drawn, with the wire as line
def cell_pat(orient, sl, ly, wi, *, wire=False, cell_data=None, transparent=False):
    h_axis, v_axis = geometry.view_axes(orient)
    box = geometry.cell(sl, ly, wi)
    cell_data = cell_data if cell_data is not None else {"color": params._color_info["cell"][None], "text": ""}
    zorder = 2 if cell_data["color"] != params._color_info["cell"][None] else 1
    if not transparent:
        edgecolor, facecolor, facecolor_side = params._color_info["cell"]["edge"], cell_data["color"], params._color_info["cell"]["side_view"]
        wire_color, wire_side_color = params._color_info["cell"]["wire"], params._color_info["cell"]["wire"]
    else:
        edgecolor, facecolor, facecolor_side = cell_data["color"], "none", "none"
        wire_color, wire_side_color = "white", "none"
    patches = []
    if orient == geometry.orientation(sl):
        patches.append(_rectangle(box, h_axis, v_axis, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder))
        if wire:
            patches.append(pat.Circle((box.center[h_axis], box.center[v_axis]), radius=params._wire_draw_radius, edgecolor=None, facecolor=wire_color))
    elif wi == params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]:
        patches.append(_rectangle(box, h_axis, v_axis, edgecolor=edgecolor, facecolor=facecolor_side))
        if wire:
            patches.append(pat.Polygon([(box.low[h_axis], box.center[v_axis]), (box.high[h_axis], box.center[v_axis])], linewidth=params._wire_draw_linewidth,
                                       edgecolor=wire_side_color, facecolor=None, closed=False, visible=True))
    return patches

def layer_pat(orient, sl, ly, *, wire=False, cell_data=None, transparent=False):
    return [patch for wi in dt_chamber_utils.wires(sl, ly)
            for patch in cell_pat(orient=orient, sl=sl, ly=ly, wi=wi, wire=wire, cell_data=cell_data[wi], transparent=transparent)]

def superlayer_pat(orient, sl, *, wire=False, cell_data=None, transparent=False):
    h_axis, v_axis = geometry.view_axes(orient)
    edgecolor, facecolor = (params._color_info["sl"]["edge"], params._color_info["sl"]["fill"]) if not transparent else ("white", "none")
    sl_pos, sl_size = params._dt_chamber["sls"][sl]["pos"], params._dt_chamber["sls"][sl]["size"]
    patches = [pat.Rectangle((sl_pos[h_axis], sl_pos[v_axis]), width=sl_size[h_axis], height=sl_size[v_axis], facecolor=facecolor, edgecolor=edgecolor)]
    for ly in dt_chamber_utils.layers(sl):
        patches.extend(layer_pat(orient=orient, sl=sl, ly=ly, wire=wire, cell_data=cell_data[ly], transparent=transparent))
    return patches

def chamber_pat(orient, *, wire=False, cell_data=None, transparent=False):
    return [patch for sl in dt_chamber_utils.superlayers()
            for patch in superlayer_pat(orient=orient, sl=sl, wire=wire, cell_data=cell_data[sl], transparent=transparent)]

def chamber_ax(ax, orient, cell_data, *, wire=False, transparent=False):
    for patch in chamber_pat(orient=orient, wire=wire, cell_data=cell_data, transparent=transparent):
        ax.add_patch(patch)
    return ax

### one cell of the pattern frame (layer ly, wire rel_wi relative to the wire of layer 3)
def cell_pat_rel_wi(ly, rel_wi, *, wire=False, cell_data=None):
    box = geometry.pattern_cell(ly, rel_wi)
    cell_data = cell_data if cell_data is not None else {"color": params._color_info["cell"][None], "text": ""}
    patches = [_rectangle(box, 0, 1, edgecolor=params._color_info["cell"]["edge"], facecolor=cell_data["color"])]
    if wire:
        patches.append(pat.Circle((box.center[0], box.center[1]), radius=params._wire_draw_radius, edgecolor=None, facecolor=params._color_info["cell"]["wire"]))
    return patches

### the cells around an sl pattern (layer 3: the reference cell), the cells of pattern shape pat_name in colour
SL_PATTERN_CELLS_TO_DRAW = {3: [0], 2: [-1, 0], 1: [-1, 0, 1], 0: [-2, -1, 0, 1]}

def empty_sl_pattern_ax(ax, pat_name, *, wire=False):
    for ly, rel_wis in SL_PATTERN_CELLS_TO_DRAW.items():
        for rel_wi in rel_wis:
            cell_data = {"color": params._color_info["cell"][None], "text": ""}
            if params._dt_sl_patterns[pat_name]["rel_wis"][ly] == rel_wi:
                cell_data["color"] = "aqua"
            for patch in cell_pat_rel_wi(ly=ly, rel_wi=rel_wi, wire=wire, cell_data=copy.deepcopy(cell_data)):
                ax.add_patch(patch)
    return ax

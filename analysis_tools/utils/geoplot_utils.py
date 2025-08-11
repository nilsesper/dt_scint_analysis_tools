###########################################
### GEOMETRY PLOTTING UTILS
###########################################

import numpy as np
import copy
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params
import analysis_tools.utils.muon_utils as muon_utils

# -----------------------------------------

########### helper functions:

###--------- draw full dt chamber

# draw one cell as list of patches
def cell_pat(orient, sl, ly, wi, *, wire=False, cell_data=None): # sliced cell data for this cell {color, text}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # emtpy cell struct if none is given
    if cell_data == None: cell_data = {"color": params._color_info["cell"][None], "text": ""}
    ## if the orientation matches the wire direction, draw all cells
    if orient == params._dt_chamber["sls"][sl]["orient"]:
        # cell color
        #cell_struct.get_cell_value(sl, ly, wi)
        cell_color = cell_data["color"] #params._color_info["cell"][None]
        #if value != None: cell_color = cmap(norm(value))
        # cell position
        #cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][x_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
        #pos_x = params._dt_chamber["sls"][sl]["pos"][x_axis]+params._dt_chamber["sls"][sl]["ch_pos"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis]+wi*(params._dt_chamber["sls"][sl]["ch_size"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis])+cell_offset
        #size_x = params._dt_chamber["sls"][sl]["ch_size"][x_axis]
        #pos_y = params._dt_chamber["sls"][sl]["pos"][y_axis]+params._dt_chamber["sls"][sl]["ch_pos"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis]+ly*(params._dt_chamber["sls"][sl]["ch_size"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis])
        #size_y = params._dt_chamber["sls"][sl]["ch_size"][y_axis]
        #patches.append( pat.Rectangle((pos_x, pos_y), width=size_x, height=size_y, edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        patches.append( pat.Rectangle((derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0], derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), width=(derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0]), height=(derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        # wire (dot)
        if wire:
            #wire_x = pos_x+(params._dt_chamber["sls"][sl]["ch_size"][x_axis])/2
            #wire_y = pos_y+(params._dt_chamber["sls"][sl]["ch_size"][y_axis])/2
            wire_x = derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0]+(params._dt_chamber["sls"][sl]["ch_size"][x_axis])/2
            wire_y = derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]+(params._dt_chamber["sls"][sl]["ch_size"][y_axis])/2
            patches.append( pat.Circle((wire_x, wire_y), radius=params._dt_chamber["sls"][sl]["wi_radius"], edgecolor=None, facecolor=params._color_info["cell"]["wire"] ) )
    ## if orientation is flipped, only outline the cells
    # in order to prevent overlaying objects in the plot only draw wi=0 for each layer from the side
    # always display the cells with None value, since behind the drawn wi=0 are many others and it is simpler to not draw any data
    elif (orient in ["theta", "phi"]) and (wi == 0):
        # cell color
        cell_color = params._color_info["cell"]["side_view"]
        # cell position
        #cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][x_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
        #pos_x = params._dt_chamber["sls"][sl]["pos"][x_axis]+(params._dt_chamber["sls"][sl]["ch_pos"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis])+wi*(params._dt_chamber["sls"][sl]["ch_size"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis])+cell_offset
        #size_x = params._dt_chamber["sls"][sl]["ch_size"][x_axis]
        #pos_y = params._dt_chamber["sls"][sl]["pos"][y_axis]+(params._dt_chamber["sls"][sl]["ch_pos"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis])+ly*(params._dt_chamber["sls"][sl]["ch_size"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis])
        #size_y = params._dt_chamber["sls"][sl]["ch_size"][y_axis]
        #patches.append( pat.Rectangle((pos_x, pos_y), width=size_x, height=size_y, edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        patches.append( pat.Rectangle((derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0], derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), width=(derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0]), height=(derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        # wire (line)
        if wire:
            patches.append( pat.Polygon([(derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0], derived_params._dt_cell_coordinates[sl][ly][wi][3+y_axis]), (derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][1], derived_params._dt_cell_coordinates[sl][ly][wi][3+y_axis])], linewidth=params._dt_chamber["sls"][sl]["wi_linewidth"], edgecolor=params._color_info["cell"]["edge"], facecolor=None, closed=False, visible=True) ) # params._color_info["cell"]["wire"]
    return patches # return list of mpl patches

# draw one layer as list of patches
def layer_pat(orient, sl, ly, *, wire=False, cell_data=None): # sliced cell data for this layer {wi: {color, text}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # cells
    for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
        patches.extend( cell_pat(orient=orient, sl=sl, ly=ly, wi=wi, wire=wire, cell_data=cell_data[wi]) )
    return patches # return list of mpl patches

# draw one superlayer as list of patches
def superlayer_pat(orient, sl, *, wire=False, cell_data=None): # sliced cell data for this superlayer {ly: {wi: {color, text}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # sl casing
    patches.append( pat.Rectangle((params._dt_chamber["sls"][sl]["pos"][x_axis], params._dt_chamber["sls"][sl]["pos"][y_axis]), width=params._dt_chamber["sls"][sl]["size"][x_axis], height=params._dt_chamber["sls"][sl]["size"][y_axis], edgecolor=params._color_info["sl"]["edge"], facecolor=params._color_info["sl"]["fill"]) )
    # layers
    for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
        patches.extend( layer_pat(orient=orient, sl=sl, ly=ly, wire=wire, cell_data=cell_data[ly]) )
    return patches # return list of mpl patches

# draw one chamber as list of patches
def chamber_pat(orient, *, wire=False, cell_data=None):  # sliced cell data for this chamber {sl: {ly: {wi: {color, text}}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # chamber casing
    patches.append( pat.Rectangle((params._dt_chamber["pos"][x_axis], params._dt_chamber["pos"][y_axis]), width=params._dt_chamber["size"][x_axis], height=params._dt_chamber["size"][y_axis], edgecolor=params._color_info["edge"], facecolor=params._color_info["fill"]) )
    # honeycomb
    patches.append( pat.Rectangle((params._dt_chamber["honeycomb"]["pos"][x_axis], params._dt_chamber["honeycomb"]["pos"][y_axis]), width=params._dt_chamber["honeycomb"]["size"][x_axis], height=params._dt_chamber["honeycomb"]["size"][y_axis], edgecolor=params._color_info["honeycomb"]["edge"], facecolor=params._color_info["honeycomb"]["fill"]) )
    # superlayers
    for sl in range(1,params._dt_chamber["n_sl"]+1):
        patches.extend( superlayer_pat(orient=orient, sl=sl, wire=wire, cell_data=cell_data[sl]) )
    return patches # return list of mpl patches

### draw dt chamber into existing ax (subplot)
# cell_data: {sl: {ly: {wi: {"color": color, "text": text}}}}
def chamber_ax(ax, orient, cell_data, *, wire=False):
    patches = chamber_pat(orient=orient, wire=wire, cell_data=cell_data)
    for patch in patches:
        ax.add_patch(patch)
    return ax

### draw muon track into existing ax (subplot)
# zrange: [zmin, zmax] of shown muon track
def muon_ax(ax, orient, muon, *, color="tab:blue"):
    z0 = muon["z0"]
    zstep = 10
    (x1, y1, z1) = muon_utils.propagate_muon(muon=muon, z=z0+zstep)
    _y0 = z0
    _y1 = z1
    if orient == "phi":
        _x0 = muon["x0"]
        _x1 = x1
    else:
        _x0 = muon["y0"]
        _x1 = y1
    ax.axline((_x0, _y0), (_x1, _y1), c=color)
    return ax

###--------- draw sl pattern fit

# draw one cell as list of patches for sl pattern fit
def cell_pat_rel_wi(ly, rel_wi, *, wire=False, cell_data=None): # sliced cell data for this cell {color, text}
    patches = []
    # emtpy cell struct if none is given
    if cell_data == None: cell_data = {"color": params._color_info["cell"][None], "text": ""}
    # cell color
    cell_color = cell_data["color"] #params._color_info["cell"][None]
    patches.append( pat.Rectangle((derived_params._sl_pattern_coordinates[ly][rel_wi][0][0], derived_params._sl_pattern_coordinates[ly][rel_wi][1][0]), width=(derived_params._sl_pattern_coordinates[ly][rel_wi][0][1]-derived_params._sl_pattern_coordinates[ly][rel_wi][0][0]), height=(derived_params._sl_pattern_coordinates[ly][rel_wi][1][1]-derived_params._sl_pattern_coordinates[ly][rel_wi][1][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
    # wire (dot)
    if wire:
        wire_x = derived_params._sl_pattern_coordinates[ly][rel_wi][2]
        wire_y = derived_params._sl_pattern_coordinates[ly][rel_wi][3]
        patches.append( pat.Circle((wire_x, wire_y), radius=params._dt_chamber["sls"][1]["wi_radius"], edgecolor=None, facecolor=params._color_info["cell"]["wire"] ) )
    return patches # return list of mpl patches

### draw empty dt pattern
sl_pat_cells_to_draw = [
    {"ly": 3, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [0]
] + [
    {"ly": 2, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [-1,0]
] + [
    {"ly": 1, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [-1,0,1]
] + [
    {"ly": 0, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [-2,-1,0,1]
]
def empty_sl_pattern_ax(ax, pat_name, *, wire=False): 
    sl_pat_cells_to_draw_colored = copy.deepcopy(sl_pat_cells_to_draw)
    for i in range(len(sl_pat_cells_to_draw_colored)):
        ly = sl_pat_cells_to_draw_colored[i]["ly"]
        rel_wi = sl_pat_cells_to_draw_colored[i]["rel_wi"]
        if params._dt_sl_patterns[pat_name]["rel_wis"][ly] == rel_wi:
            sl_pat_cells_to_draw_colored[i]["cell_data"]["color"] = "aqua"
    patches = []
    for cell in sl_pat_cells_to_draw_colored:
        patches.extend( cell_pat_rel_wi(ly=cell["ly"], rel_wi=cell["rel_wi"], wire=wire, cell_data=cell["cell_data"]) ) # hardcode sl, since all sls the same
    for patch in patches:
        ax.add_patch(patch)
    return ax

### draw dt sl pattern with fit
def sl_fit_ax(ax, sl_dt_fits, pattern_id, *, wire=False):
    sl = sl_dt_fits["sl"][pattern_id]
    pat_type = sl_dt_fits["pat_type"][pattern_id]
    pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
    ax = empty_sl_pattern_ax(ax, pat_name, wire=wire)
    return ax

### draw dt sl fit muon
def sl_muon_fit_ax(ax, sl_dt_fits, pattern_id, *, wire=False, color="red"):
    # plot muon track
    _z0 = derived_params._sl_pattern_coordinates[3][0][3] # z_cell (wire position) of ly=3
    _z1 = derived_params._sl_pattern_coordinates[2][0][3] # z_cell (wire position) of ly=2
    x0_fit, tan_alpha_fit = sl_dt_fits["x0"][pattern_id], sl_dt_fits["tan_alpha"][pattern_id]
    ax.axline((derived_params.f_x_muon(z=_z0, x0=x0_fit, tan_alpha=tan_alpha_fit), _z0), (derived_params.f_x_muon(z=_z1, x0=x0_fit, tan_alpha=tan_alpha_fit), _z1), c=color)
    # plot muon hits
    #for ly in 
    return ax

### draw sl projection of muon object
def sl_muon_proj_ax(ax, muons, muon_id, *, wire=False, color="tab:green"):
    return ax










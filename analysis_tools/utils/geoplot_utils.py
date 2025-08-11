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
def muon_ax(ax, orient, muons, muon_id, *, color="tab:blue"):
    z0 = muons["z0"][muon_id]
    zstep = 10
    (x1, y1, z1) = muon_utils.propagate_muon(muons=muons, muon_id=muon_id, z=z0+zstep)
    _y0 = z0
    _y1 = z1
    if orient == "phi":
        _x0 = muons["x0"][muon_id]
        _x1 = x1
    else:
        _x0 = muons["y0"][muon_id]
        _x1 = y1
    ax.axline((_x0, _y0), (_x1, _y1), c=color)
    return ax

########### macro functions:












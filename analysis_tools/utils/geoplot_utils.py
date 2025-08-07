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

# -----------------------------------------

### plot patches of chamber
@mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12})
def plot_chamber_patches(patches, orient, *, show=True, store=False):
    #fig, [ax, cax] = plt.subplots(2, 1, height_ratios=[1, 0.05], figsize=(12,4))
    #plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.6)
    for patch in patches:
        ax.add_patch( patch )
    ax.margins(x=0.05, y=0.05)
    #ax.invert_yaxis()
    #ax.set_aspect(1, adjustable='box', anchor='C')
    ## colormap
    #fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax, orientation="horizontal")
    #cax.set_xlabel("Hits")
    # text labels
    axbox = ax.get_position()
    x_topleft = axbox.p0[0]
    x_topright, y_topleft = axbox.p1[0], axbox.p1[1]
    ax.text(x_topleft, y_topleft+0.02, "CMS", transform=plt.gcf().transFigure, fontweight="bold")
    ax.text(x_topleft+0.04, y_topleft+0.02, "Private work", transform=plt.gcf().transFigure, fontstyle="italic", fontsize=10)
    description = params._dt_chamber["name"]
    if orient == "theta":
        description += ", $\\theta$ view"
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    elif orient == "phi":
        description += ", $\\phi$ view"
        ax.set_xlabel("$y$ [mm]")
        ax.set_ylabel("$z$ [mm]")
    ax.text(x_topright, y_topleft+0.02, description, transform=plt.gcf().transFigure, horizontalalignment="right")
    #fig.tight_layout()
    # show and/or store plot
    if show == True: fig.show()
    if store != False: fig.savefig(store)

########### helper functions:

# draw one cell
def cell(orient, sl, ly, wi, *, wire=False, cell_data=None): # sliced cell data for this cell {color, text}
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
        cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][x_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
        pos_x = params._dt_chamber["sls"][sl]["pos"][x_axis]+(params._dt_chamber["sls"][sl]["ch_pos"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis])+wi*(params._dt_chamber["sls"][sl]["ch_size"][x_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][x_axis])+cell_offset
        size_x = params._dt_chamber["sls"][sl]["ch_size"][x_axis]
        pos_y = params._dt_chamber["sls"][sl]["pos"][y_axis]+(params._dt_chamber["sls"][sl]["ch_pos"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis])+ly*(params._dt_chamber["sls"][sl]["ch_size"][y_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][y_axis])
        size_y = params._dt_chamber["sls"][sl]["ch_size"][y_axis]
        patches.append( pat.Rectangle((pos_x, pos_y), width=size_x, height=size_y, edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        # wire (line)
        if wire:
            patches.append( pat.Polygon([(pos_x, pos_y+params._dt_chamber["sls"][sl]["ch_size"][y_axis]/2), (pos_x+params._dt_chamber["sls"][sl]["ch_size"][x_axis], pos_y+params._dt_chamber["sls"][sl]["ch_size"][y_axis]/2)], linewidth=params._dt_chamber["sls"][sl]["wi_linewidth"], edgecolor=params._color_info["cell"]["edge"], facecolor=None, closed=False, visible=True) ) # params._color_info["cell"]["wire"]
    return patches

# draw one layer
def layer(orient, sl, ly, *, wire=False, cell_data=None): # sliced cell data for this layer {wi: {color, text}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # cells
    for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
        patches.extend( cell(orient=orient, sl=sl, ly=ly, wi=wi, wire=wire, cell_data=cell_data[wi]) )
    return patches

# draw one superlayer
def superlayer(orient, sl, *, wire=False, cell_data=None): # sliced cell data for this superlayer {ly: {wi: {color, text}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # sl casing
    patches.append( pat.Rectangle((params._dt_chamber["sls"][sl]["pos"][x_axis], params._dt_chamber["sls"][sl]["pos"][y_axis]), width=params._dt_chamber["sls"][sl]["size"][x_axis], height=params._dt_chamber["sls"][sl]["size"][y_axis], edgecolor=params._color_info["sl"]["edge"], facecolor=params._color_info["sl"]["fill"]) )
    # layers
    for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
        patches.extend( layer(orient=orient, sl=sl, ly=ly, wire=wire, cell_data=cell_data[ly]) )
    return patches

# draw one chamber
def chamber(orient, *, wire=False, cell_data=None):  # sliced cell data for this chamber {sl: {ly: {wi: {color, text}}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # chamber casing
    patches.append( pat.Rectangle((params._dt_chamber["pos"][x_axis], params._dt_chamber["pos"][y_axis]), width=params._dt_chamber["size"][x_axis], height=params._dt_chamber["size"][y_axis], edgecolor=params._color_info["edge"], facecolor=params._color_info["fill"]) )
    # honeycomb
    patches.append( pat.Rectangle((params._dt_chamber["honeycomb"]["pos"][x_axis], params._dt_chamber["honeycomb"]["pos"][y_axis]), width=params._dt_chamber["honeycomb"]["size"][x_axis], height=params._dt_chamber["honeycomb"]["size"][y_axis], edgecolor=params._color_info["honeycomb"]["edge"], facecolor=params._color_info["honeycomb"]["fill"]) )
    # superlayers
    for sl in range(1,params._dt_chamber["n_sl"]+1):
        patches.extend( superlayer(orient=orient, sl=sl, wire=wire, cell_data=cell_data[sl]) )
    return patches

########### macro functions:

### plot chamber with data
# cell_data: {sl: {ly: {wi: {"color": color, "text": text}}}}
def plot_cells(orient, cell_data, *, wire=False):
    print(f"Plotting chamber map...")
    #c = cell_struct.deep_copy()
    #c.restrict(sl=sl, ly=ly, wi=wi, orient=orient)
    #cmap, norm = c.color_map(norm=norm)
    chamber_patches = chamber(orient=orient, wire=wire, cell_data=cell_data)
    plot_chamber_patches(patches=chamber_patches, orient=orient, show=True, store=False)










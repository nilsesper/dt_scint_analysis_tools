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

###--------- draw full dt chamber (global coord frame)

# draw one cell as list of patches
def cell_pat(orient, sl, ly, wi, *, wire=False, cell_data=None): # sliced cell data for this cell {color, text}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # emtpy cell struct if none is given
    if cell_data == None: cell_data = {"color": params._color_info["cell"][None], "text": ""}
    ## if the orientation matches the wire direction, draw all cells
    if orient == params._dt_chamber["sls"][sl]["orient"]:
        cell_color = cell_data["color"] #params._color_info["cell"][None]
        patches.append( pat.Rectangle((derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0], derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), width=(derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][x_axis][0]), height=(derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][1]-derived_params._dt_cell_coordinates[sl][ly][wi][y_axis][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
        if wire:
            wire_x = derived_params._dt_cell_coordinates[sl][ly][wi][x_axis+3]#[0]+(params._dt_chamber["sls"][sl]["ch_size"][x_axis])/2
            wire_y = derived_params._dt_cell_coordinates[sl][ly][wi][y_axis+3]#[0]+(params._dt_chamber["sls"][sl]["ch_size"][y_axis])/2
            patches.append( pat.Circle((wire_x, wire_y), radius=params._dt_chamber["sls"][sl]["wi_radius"], edgecolor=None, facecolor=params._color_info["cell"]["wire"] ) )
    ## if orientation is flipped, only outline the cells
    # in order to prevent overlaying objects in the plot only draw wi=0 for each layer from the side
    # always display the cells with None value, since behind the drawn wi=0 are many others and it is simpler to not draw any data
    elif (orient in ["theta", "phi"]) and (wi == params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"]):
        # cell color
        cell_color = params._color_info["cell"]["side_view"]
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
    for wi in range(params._dt_chamber["sls"][sl]["lys"][ly]["min_wi"], params._dt_chamber["sls"][sl]["lys"][ly]["max_wi"]+1):
        patches.extend( cell_pat(orient=orient, sl=sl, ly=ly, wi=wi, wire=wire, cell_data=cell_data[wi]) )
    return patches # return list of mpl patches

# draw one superlayer as list of patches
def superlayer_pat(orient, sl, *, wire=False, cell_data=None): # sliced cell data for this superlayer {ly: {wi: {color, text}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # sl casing
    patches.append( pat.Rectangle((params._dt_chamber["pos"][x_axis]+params._dt_chamber["sls"][sl]["pos"][x_axis], params._dt_chamber["pos"][y_axis]+params._dt_chamber["sls"][sl]["pos"][y_axis]), width=params._dt_chamber["sls"][sl]["size"][x_axis], height=params._dt_chamber["sls"][sl]["size"][y_axis], edgecolor=params._color_info["sl"]["edge"], facecolor=params._color_info["sl"]["fill"]) )
    # layers
    for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
        patches.extend( layer_pat(orient=orient, sl=sl, ly=ly, wire=wire, cell_data=cell_data[ly]) )
    return patches # return list of mpl patches

# draw one chamber as list of patches
def chamber_pat(orient, *, wire=False, cell_data=None):  # sliced cell data for this chamber {sl: {ly: {wi: {color, text}}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # chamber casing
    #patches.append( pat.Rectangle((params._dt_chamber["pos"][x_axis], params._dt_chamber["pos"][y_axis]), width=params._dt_chamber["size"][x_axis], height=params._dt_chamber["size"][y_axis], edgecolor=params._color_info["edge"], facecolor=params._color_info["fill"]) )
    # honeycomb
    #patches.append( pat.Rectangle((params._dt_chamber["pos"][x_axis]+params._dt_chamber["honeycomb"]["pos"][x_axis], params._dt_chamber["pos"][y_axis]+params._dt_chamber["honeycomb"]["pos"][y_axis]), width=params._dt_chamber["honeycomb"]["size"][x_axis], height=params._dt_chamber["honeycomb"]["size"][y_axis], edgecolor=params._color_info["honeycomb"]["edge"], facecolor=params._color_info["honeycomb"]["fill"]) )
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

###--------- draw full scintillator chamber  (global coord frame)

# draw one scintillator strip as list of patches
def scintillator_strip_pat(orient, ly, st, *, cell_data=None): # sliced cell data for this cell {color, text}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # emtpy cell struct if none is given
    if cell_data == None: cell_data = {"color": params._color_info["cell"][None], "text": ""}
    ## if the orientation matches the wire direction, draw all cells
    if orient == params._scintillator["lys"][ly]["orient"]:
        cell_color = cell_data["color"] #params._color_info["cell"][None]
        patches.append( pat.Rectangle((derived_params._scintillator_strip_coordinates[ly][st][x_axis][0], derived_params._scintillator_strip_coordinates[ly][st][y_axis][0]), width=(derived_params._scintillator_strip_coordinates[ly][st][x_axis][1]-derived_params._scintillator_strip_coordinates[ly][st][x_axis][0]), height=(derived_params._scintillator_strip_coordinates[ly][st][y_axis][1]-derived_params._scintillator_strip_coordinates[ly][st][y_axis][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
    ## if orientation is flipped, only outline the cells
    # in order to prevent overlaying objects in the plot only draw wi=0 for each layer from the side
    # always display the cells with None value, since behind the drawn wi=0 are many others and it is simpler to not draw any data
    elif (orient in ["theta", "phi"]) and (st == 0):
        # cell color
        cell_color = params._color_info["cell"]["side_view"]
        patches.append( pat.Rectangle((derived_params._scintillator_strip_coordinates[ly][st][x_axis][0], derived_params._scintillator_strip_coordinates[ly][st][y_axis][0]), width=(derived_params._scintillator_strip_coordinates[ly][st][x_axis][1]-derived_params._scintillator_strip_coordinates[ly][st][x_axis][0]), height=(derived_params._scintillator_strip_coordinates[ly][st][y_axis][1]-derived_params._scintillator_strip_coordinates[ly][st][y_axis][0]), edgecolor=params._color_info["cell"]["edge"], facecolor=cell_color) )
    return patches # return list of mpl patches

# draw one scintillator layer as list of patches
def scintillator_layer_pat(orient, ly, *, cell_data=None): # sliced cell data for this layer {wi: {color, text}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # strips
    for st in range(params._scintillator["lys"][ly]["n_sts"]):
        patches.extend( scintillator_strip_pat(orient=orient, ly=ly, st=st, cell_data=cell_data[st]) )
    return patches # return list of mpl patches

# draw one scintillator as list of patches
def scintillator_pat(orient, *, cell_data=None):  # sliced cell data for this chamber {sl: {ly: {wi: {color, text}}}}
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1] # chamber axis projection (x,y,z) to 2D plot axis (x,y)
    patches = []
    # chamber casing
    patches.append( pat.Rectangle((params._scintillator["pos"][x_axis], params._scintillator["pos"][y_axis]), width=params._scintillator["size"][x_axis], height=params._scintillator["size"][y_axis], edgecolor=params._color_info["edge"], facecolor=params._color_info["fill"]) )
    # layers
    for ly in range(params._scintillator["n_lys"]):
        patches.extend( scintillator_layer_pat(orient=orient, ly=ly, cell_data=cell_data[ly]) )
    return patches # return list of mpl patches

### draw scintillator into existing ax (subplot)
# cell_data: {ly: {st: {"color": color, "text": text}}}
def scintillator_ax(ax, orient, cell_data):
    patches = scintillator_pat(orient=orient, cell_data=cell_data)
    for patch in patches:
        ax.add_patch(patch)
    return ax

###--------- draw other things (global coord frame)

### draw muon track into existing ax (subplot)
# zrange: [zmin, zmax] of shown muon track
def muon_ax(ax, orient, muons, muon_idx, *, color="tab:blue", label=""):
    z0 = derived_params._dt_cell_coordinates[3][3][1][5] if orient == "phi" else derived_params._dt_cell_coordinates[2][3][1][5]
    z1 = derived_params._dt_cell_coordinates[1][0][1][5] if orient == "phi" else derived_params._dt_cell_coordinates[2][0][1][5]
    (x0, y0, z0) = muon_utils.propagate_muon(muons=muons, z=z0, idx=muon_idx)
    (x1, y1, z1) = muon_utils.propagate_muon(muons=muons, z=z1, idx=muon_idx)
    _y0 = z0
    _y1 = z1
    if orient == "phi":
        _x0 = x0
        _x1 = x1
    else:
        _x0 = y0
        _x1 = y1
    ax.axline((_x0, _y0), (_x1, _y1), c=color, linewidth=params._color_info["muon"]["linewidth"], label=label)
    return ax

### draw dt hits (if laterality + drift distance + muon id is known)
# for given muon id
def cell_hits_ax(ax, orient, dt_hits, muon_id, *, color="tab:green"):
    n_hits = len(dt_hits["ch"])
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    x_pts, y_pts = [], []
    for i in range(n_hits):
        if dt_hits["muon_id"][i] != muon_id: continue
        sl, ly, wi = dt_hits["sl"][i], dt_hits["ly"][i], dt_hits["wi"][i]
        if params._dt_chamber["sls"][sl]["orient"] != orient: continue
        dd, hit_lat = dt_hits["dd"][i], dt_hits["hit_lat"][i] # hit_lat: hit laterality, dd drift distance
        x_pts.append( dd*hit_lat + derived_params._dt_cell_coordinates[sl][ly][wi][3+x_axis] )
        y_pts.append( derived_params._dt_cell_coordinates[sl][ly][wi][3+y_axis] )
    x_pts, y_pts = np.array(x_pts), np.array(y_pts)
    ax.scatter(x_pts, y_pts, marker=".", color=color, s=params._color_info["muon"]["markersize"])
    return ax

### draw dt sl fit muon
def chamber_muon_fit_ax(ax, orient, sl_dt_fits, pattern_idx, *, color="red", label=""):
    _z0 = derived_params._sl_pattern_coordinates[3][0][3] # z_cell (wire position) of ly=3
    _z1 = derived_params._sl_pattern_coordinates[2][0][3] # z_cell (wire position) of ly=2
    x0_fit, tan_alpha_fit = sl_dt_fits["x0"][pattern_idx], sl_dt_fits["tan_alpha"][pattern_idx]
    # get info about sl position of muon
    sl = sl_dt_fits["sl"][pattern_idx]
    orient_sl = params._dt_chamber["sls"][sl]["orient"]
    if orient_sl != orient: return ax # skip if wrong orientation
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    # transform coordinates from local coordinate frame (with (0,0) at center (wire) position of cell ly=3, rel_wi=0) into global coordinate frame of dt chamber (used in params.py file)
    base_wi = sl_dt_fits["wi3"][pattern_idx] # wi idx of ly=3 (base wi)
    _coord_transform = [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ]
    # calculate muon track in coord frame
    x0, y0 = derived_params.f_x_muon(z=_z0, x0=x0_fit, tan_alpha=tan_alpha_fit) + _coord_transform[0], _z0 + _coord_transform[1]
    x1, y1 = derived_params.f_x_muon(z=_z1, x0=x0_fit, tan_alpha=tan_alpha_fit) + _coord_transform[0], _z1 + _coord_transform[1]
    # draw line
    ax.axline((x0, y0), (x1, y1), c=color, linewidth=params._color_info["muon"]["linewidth"], label=label)
    return ax

### draw scint hits (if xhit + muon_id is known)
# for given muon id
def scint_hits_ax(ax, orient, scint_hits, muon_id, *, color="tab:blue"):
    n_hits = len(scint_hits["ch"])
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    x_pts, y_pts = [], []
    for i in range(n_hits):
        if scint_hits["muon_id"][i] != muon_id: continue
        ly, st = scint_hits["ly"][i], scint_hits["st"][i]
        if params._scintillator["lys"][ly]["orient"] != orient: continue
        xhit = scint_hits["xhit"][i]
        x_pts.append( xhit + derived_params._scintillator_strip_coordinates[ly][st][x_axis][0] ) # x_hit = xhit + x(left / smaller point of scint st)
        y_pts.append( derived_params._scintillator_strip_coordinates[ly][st][3+y_axis] )
        #print("cell_hits_ax", dd, hit_lat, derived_params._dt_cell_coordinates[sl][ly][wi][3+x_axis], "=", x_pts)
    x_pts, y_pts = np.array(x_pts), np.array(y_pts)
    ax.scatter(x_pts, y_pts, marker=".", color=color, s=params._color_info["muon"]["markersize"])
    return ax

### draw muon area (area of muon hit reco from scintillator)
# for given muon_id
def scint_muon_area_ax(ax, orient, scint_muon_areas, muon_id, *, color="red"):
    n_hits = len(scint_muon_areas["ts"])
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    patches = []
    for i in range(n_hits):
        if scint_muon_areas["muon_id"][i] != muon_id: continue
        xmin, xmax = (scint_muon_areas["xmin"][i], scint_muon_areas["xmax"][i]) if (x_axis == 0) else (scint_muon_areas["ymin"][i], scint_muon_areas["ymax"][i])
        z0 = scint_muon_areas["z0"][i]
        # draw line (side view of muon area)
        patches.append( pat.Polygon([(xmin, z0), (xmax, z0)], edgecolor=color, facecolor=None, closed=False, visible=True, linewidth=params._color_info["muon"]["linewidth"]) )
    for patch in patches:
        ax.add_patch(patch)
    return ax

###--------- draw sl pattern fit (local sl pattern coord frame)

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
    {"ly": 3, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [0]         ] + [
    {"ly": 2, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [-1,0]      ] + [
    {"ly": 1, "rel_wi": rel_wi, "cell_data": {"color": params._color_info["cell"][None], "text": ""}} for rel_wi in [-1,0,1]    ] + [
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

### draw dt sl pattern
def sl_fit_ax(ax, sl_dt_fits, pattern_id, *, wire=False):
    sl = sl_dt_fits["sl"][pattern_id]
    pat_type = sl_dt_fits["pat_type"][pattern_id]
    pat_name = list(params._dt_sl_patterns.keys())[pat_type] # extract pattern name e.g. "+a"
    ax = empty_sl_pattern_ax(ax, pat_name, wire=wire)
    return ax

### draw dt sl pattern and pattern fit
def sl_muon_fit_ax(ax, sl_dt_fits, pattern_id, *, color="red"):
    # plot local sl fit
    _z0 = derived_params._sl_pattern_coordinates[3][0][3] # z_cell (wire position) of ly=3
    _z1 = derived_params._sl_pattern_coordinates[2][0][3] # z_cell (wire position) of ly=2
    x0_fit, tan_alpha_fit = sl_dt_fits["x0"][pattern_id], sl_dt_fits["tan_alpha"][pattern_id]
    ax.axline((derived_params.f_x_muon(z=_z0, x0=x0_fit, tan_alpha=tan_alpha_fit), _z0), (derived_params.f_x_muon(z=_z1, x0=x0_fit, tan_alpha=tan_alpha_fit), _z1), c=color, linewidth=params._color_info["muon"]["linewidth"])
    return ax

### draw sl projection of muon object
# give no of sl to project to
def sl_muon_proj_ax(ax, muons, sl_dt_fits, pattern_id, *, color="tab:green"):
    sl = sl_dt_fits["sl"][pattern_id]
    muon_id = sl_dt_fits["muon_id"][pattern_id]
    orient = params._dt_chamber["sls"][sl]["orient"]
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    z0 = derived_params._dt_cell_coordinates[sl][3][0][y_axis+3] if orient == "phi" else derived_params._dt_cell_coordinates[sl][3][0][y_axis+3]
    z1 = derived_params._dt_cell_coordinates[sl][0][0][y_axis+3] if orient == "phi" else derived_params._dt_cell_coordinates[sl][0][0][y_axis+3]
    (x0, y0, z0) = muon_utils.propagate_muon(muons=muons, z=z0, muon_id=muon_id)
    (x1, y1, z1) = muon_utils.propagate_muon(muons=muons, z=z1, muon_id=muon_id)
    _y0 = z0
    _y1 = z1
    if orient == "phi":
        _x0 = x0
        _x1 = x1
    else:
        _x0 = y0
        _x1 = y1
    # transform coordinates from global coordinate frame of dt chamber (used in params.py file) into local coordinate frame (with (0,0) at center (wire) position of cell ly=3, rel_wi=0)
    base_wi = sl_dt_fits["wi3"][pattern_id] # wi idx of ly=3 (base wi)
    _coord_transform = [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ]
    ax.axline((_x0-_coord_transform[0], _y0-_coord_transform[1]), (_x1-_coord_transform[0], _y1-_coord_transform[1]), c=color, linewidth=params._color_info["muon"]["linewidth"])
    return ax

### draw dt hits (if laterality + drift distance + muon id is known) into sl projection
# for given muon id
def sl_dt_hits_proj_ax(ax, dt_hits, sl_dt_fits, pattern_id, *, color="tab:green", other_lat=False):
    n_hits = len(dt_hits["ch"])
    sl = sl_dt_fits["sl"][pattern_id]
    orient = params._dt_chamber["sls"][sl]["orient"]
    muon_id = sl_dt_fits["muon_id"][pattern_id]
    x_axis, y_axis = params._orientation[orient][0], params._orientation[orient][1]
    # for correct hits
    x_pts, y_pts = [], []
    for i in range(n_hits):
        if dt_hits["muon_id"][i] != muon_id: continue
        hit_sl, hit_ly, hit_wi = dt_hits["sl"][i], dt_hits["ly"][i], dt_hits["wi"][i]
        if sl != hit_sl: continue
        dd, hit_lat = dt_hits["dd"][i], dt_hits["hit_lat"][i] # hit_lat: hit laterality, dd drift distance
        x_pts.append( dd*hit_lat + derived_params._dt_cell_coordinates[sl][hit_ly][hit_wi][3+x_axis] )
        y_pts.append( derived_params._dt_cell_coordinates[hit_sl][hit_ly][hit_wi][3+y_axis] )
    x_pts, y_pts = np.array(x_pts), np.array(y_pts)
    # for hits with other laterality
    if other_lat:
        x_pts2, y_pts2 = [], []
        for i in range(n_hits):
            if dt_hits["muon_id"][i] != muon_id: continue
            hit_sl, hit_ly, hit_wi = dt_hits["sl"][i], dt_hits["ly"][i], dt_hits["wi"][i]
            if sl != hit_sl: continue
            dd, hit_lat = dt_hits["dd"][i], dt_hits["hit_lat"][i] # hit_lat: hit laterality, dd drift distance
            x_pts2.append( -dd*hit_lat + derived_params._dt_cell_coordinates[sl][hit_ly][hit_wi][3+x_axis] )
            y_pts2.append( derived_params._dt_cell_coordinates[hit_sl][hit_ly][hit_wi][3+y_axis] )
        x_pts2, y_pts2 = np.array(x_pts2), np.array(y_pts2)
    # transform coordinates from global coordinate frame of dt chamber (used in params.py file) into local coordinate frame (with (0,0) at center (wire) position of cell ly=3, rel_wi=0)
    base_wi = sl_dt_fits["wi3"][pattern_id] # wi idx of ly=3 (base wi)
    _coord_transform = [ derived_params._dt_cell_coordinates[sl][3][base_wi][x_axis+3], derived_params._dt_cell_coordinates[sl][3][base_wi][y_axis+3] ]
    ax.scatter(x_pts-_coord_transform[0], y_pts-_coord_transform[1], marker=".", color=color, s=params._color_info["muon"]["markersize"])
    if other_lat:
        ax.scatter(x_pts2-_coord_transform[0], y_pts2-_coord_transform[1], marker=".", color="tab:gray", s=params._color_info["muon"]["markersize"])
    return ax










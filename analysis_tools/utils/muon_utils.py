###########################################
### MUON RECONSTRUCTION / DUMMY DATA UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.math_utils as math_utils
import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params

# -----------------------------------------

### propagate all muons to given z coordinate
def propagate_muons(muons, z): # propagate spherical coordinates
    x = muons["x0"] + (z-muons["z0"]) * np.cos(muons["phi"]) * np.tan(muons["theta"])
    y = muons["y0"] + (z-muons["z0"]) * np.sin(muons["phi"]) * np.tan(muons["theta"])
    return (x,y,z)

### propagate one muon to given z coordinate
# if muon_id given: use muon with given muon_id
# if idx given: use muon with given index in list
def propagate_muon(muons, z, muon_id=None, idx=None): # propagate spherical coordinates
    if idx == None and muon_id == None:
        raise Exception("Specify either \"muon_id\" or \"idx\" argument!")
    if muon_id != None and idx != None:
        raise Exception("Only \"muon_id\" or \"idx\" argument must be given, not both!")
    if muon_id != None: # search idx of muon in list which has specifed muon_id
        idx = np.argwhere(muons["muon_id"] == muon_id)[0][0]
    x = muons["x0"][idx] + (z-muons["z0"][idx]) * np.cos(muons["phi"][idx]) * np.tan(muons["theta"][idx])
    y = muons["y0"][idx] + (z-muons["z0"][idx]) * np.sin(muons["phi"][idx]) * np.tan(muons["theta"][idx])
    return (x,y,z)

### generate random cosmic muons
# with spawnpoint range same for all muons: xrange = [xmin, xmax], yrange = [ymin, ymax], z0
# generate n muons
# pass separate timestamp for all muons i.e. ts = [ts[i] for i in range(n)]
def generate_cosmic_muons(n, ts, xrange, yrange, z0, *, silent=False, thetarange=[0, np.pi/2], phirange=[0,2*np.pi]):
    if not silent: print(f"Generating {n} cosmic muons...")
    muons = {k: np.full(n, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    muons["x0"] = np.random.uniform(low=xrange[0], high=xrange[1], size=n).astype(dtype=params._muon_obj_keys["x0"])
    muons["y0"] = np.random.uniform(low=yrange[0], high=yrange[1], size=n).astype(dtype=params._muon_obj_keys["y0"]) # x,y uniformly distributed inside xrange, yrange
    muons["z0"] = np.full(n, z0, dtype=params._muon_obj_keys["z0"])
    muons["theta"] = math_utils.draw_from_pdf(pdf=params.cosmic_muon_theta_weight, val_range=thetarange, n=n, dtype=params._muon_obj_keys["theta"]) # theta distributed according to distribution
    muons["phi"] = np.random.uniform(low=phirange[0], high=phirange[1], size=n).astype(dtype=params._muon_obj_keys["phi"]) # phi uniformly distributed
    muons["ts"] = np.array(ts).astype(dtype=params._muon_obj_keys["ts"])
    muons["muon_id"] = np.arange(0, n, dtype=params._muon_obj_keys["muon_id"])
    return muons

### cut muons by specifying geometrical area (xmin, xmax, ymin, ymax, z0) it has to pass through
# (ignore timestamp, accept all timestamps)
def cut_muons_by_area(muons, xmin, xmax, ymin, ymax, z0, *, silent=False):
    n_muons = len(muons["ts"])
    mask = np.full(n_muons, True)
    if not silent: print(f"Cutting {n_muons} muons to geometrical area x=({xmin}, {xmax}) y=({ymin}, {ymax}) z={z0}...")
    # propagate muons and check if in min/max range -> populate mask
    (x,y,z) = propagate_muons(muons, z=z0)
    mask &= (x >= xmin)
    mask &= (x <= xmax)
    mask &= (y >= ymin)
    mask &= (x <= ymax)
    # apply mask on muons
    cut_muons = {}
    for name in muons.keys():
        cut_muons[name] = copy.deepcopy(muons[name][mask])
    n_cut_muons = len(cut_muons["ts"])
    if not silent:
        if n_muons > 0: print(f"Cut flow: {n_cut_muons} / {n_muons} = {n_cut_muons/n_muons}")
        else: print(f"Cut flow: {n_cut_muons} / {n_muons}")
    return cut_muons

### correlate muons & muon areas
# correlate 1 muon area object (of scintillator) & 1 muon object (of dt chamber)
# do correlation according to:
#   - timestamp (timestamps of muon & muon area must have abs difference <= params._correlation_ts_window)
#   - position (propagate muon to same z as muon area, then:
#               x position of muon must be within [xmin-params._correlation_xy_window, ymin+params._correlation_xy_window], same for y)
# alignment offset is space-time offset between DT muons and scintillator muon_areas:
#   alignment_offset in space-time = (x0, y0, z0, t0)
#   - x0, y0, z0: spatial alignment offset between specified scintillator position & scintillator position that the DT muons are propagated to
#   - t0: temporal alignment offset between scintillator timestamp & DT muon timestamp
#   per definition: X0 = X(DT muon) - X(scintillator)   for X = [x0,y0,z0,t0]
#   -> X(scintillator)_after_alignment != X(DT)
#   -> X(scintillator)_after_alignment = X(scintillator)_before_alignment + X0
def correlate_muons_and_muon_areas(muons, muon_areas, alignment_offset=(0., 0., 0., 0.), *, silent=False):
    n_muons = data_utils.length(muons)
    n_muon_areas = data_utils.length(muon_areas)
    # sort both muons & muon areas by timestamp
    muons = timestamp_utils.sort_by_timestamp(hits=muons, silent=silent)
    muon_areas = timestamp_utils.sort_by_timestamp(hits=muon_areas, silent=silent)
    # copy objects
    muons = copy.deepcopy(muons)
    muon_areas = copy.deepcopy(muon_areas)
    ### apply alignment constants to muon areas
    # X(scintillator)_after_alignment = X(scintillator)_before_alignment + X0
    x_alignment, y_alignment, z_alignment = alignment_offset[0:3]
    ts_alignment = np.int64(np.round(alignment_offset[3], 0))
    for ia in range(n_muon_areas):
        muon_areas["xmin"][ia] = muon_areas["xmin"][ia] + x_alignment
        muon_areas["xmax"][ia] = muon_areas["xmax"][ia] + x_alignment
        muon_areas["ymin"][ia] = muon_areas["ymin"][ia] + y_alignment
        muon_areas["ymax"][ia] = muon_areas["ymax"][ia] + y_alignment
        muon_areas["z0"][ia] = muon_areas["z0"][ia] + z_alignment
        muon_areas["ts"][ia] = muon_areas["ts"][ia] + ts_alignment
    ### continue with these corrected/aligned muon areas
    ### correlate muons & muon areas in time (correlate 2 objects which have timestamps with a difference <= params._correlation_ts_window)
    # collect correlated indices of muons & muon areas
    correlated_indices = [] # [(im = muon index, ia = muon area index) for correlated muons]
    # go step by step through muon (area) objects
    ia = 0 # current muon area object index
    im = 0 # current muon object index
    while True:
        ### muon & muon area object list boundaries:
        # break if index out of range
        if im >= n_muons: break # stop as soon as no more muon are there
        if ia >= n_muon_areas: break # stop as soon as no more muon areas are there
        ### correlation in time (respect alignment)
        ts_m = muons["ts"][im] # timestamp of current muon
        ts_a = muon_areas["ts"][ia] # timestamp of current muon area
        # if current muon is much later than current muon area, go to next muon area
        if ts_m > ts_a and ts_m-ts_a > params._correlation_ts_window: 
            ia += 1
            continue
        # if current muon is much earlier than current muon area, go to next muon
        if ts_m < ts_a and ts_a-ts_m > params._correlation_ts_window:
            im += 1
            continue
        ### correlation in space
        # propagate muon to same z as muon area
        z0 = muon_areas["z0"][ia]
        xmin, xmax = muon_areas["xmin"][ia], muon_areas["xmax"][ia]
        ymin, ymax = muon_areas["ymin"][ia], muon_areas["ymax"][ia]
        (xm, ym, zm) = propagate_muon(muons=muons, z=z0, idx=im)
        # if muon is not within coordinates of muon area plus/minus specified xy tolerance, go to next muon & next muon area 
        # !!!! THIS IS NOT OPTIMAL YET !!!!
        if xm < xmin-params._correlation_xy_window or xm > xmax+params._correlation_xy_window or ym < ymin-params._correlation_xy_window or ym > ymax+params._correlation_xy_window:
            im += 1
            ia += 1
            continue
        ### store correlated objects
        # if correlation conditions are met, the loop was not continued
        # # use current indices of muon & muon area
        # then increment both indices, so each muon & muon area can only be correlated once
        correlated_indices.append((im, ia)) # (muon index, muon area index)
        im += 1
        ia += 1
    ### store in muon correlations object
    n_muon_correlations = len(correlated_indices)
    muon_correlations = {k: np.full(n_muon_correlations, 0, dtype=v) for k,v in params._muon_corr_obj_keys.items()}
    for i in range(n_muon_correlations):
        (im, ia) = correlated_indices[i] # extract muon & muon area index
        # propagate muon to same z as muon area
        z0 = muon_areas["z0"][ia]
        xmin, xmax = muon_areas["xmin"][ia], muon_areas["xmax"][ia]
        ymin, ymax = muon_areas["ymin"][ia], muon_areas["ymax"][ia]
        (xm, ym, zm) = propagate_muon(muons=muons, z=z0, idx=im)
        # fill muon indices
        for k in ["theta", "phi", "muon_id"]:
            muon_correlations[k][i] = muons[k][im]
        muon_correlations["x0"][i], muon_correlations["y0"][i] = xm, ym
        muon_correlations["ts_muon"][i] = muons["ts"][im]
        # fill muon area keys
        for k in ["xmin", "xmax", "ymin", "ymax", "xcenter", "ycenter", "pixel"]:
            muon_correlations[k][i] = muon_areas[k][ia]
        muon_correlations["ts_area"][i] = muon_areas["ts"][ia]
        muon_correlations["muon_id_area"][i] = muon_areas["muon_id"][ia]
    ### add advanced / analysis key values
    muon_correlations["delta_ts"] = np.int64(muon_correlations["ts_muon"]) - np.int64(muon_correlations["ts_area"])
    muon_correlations["delta_xcenter"] = muon_correlations["x0"] - muon_correlations["xcenter"]
    muon_correlations["delta_ycenter"] = muon_correlations["y0"] - muon_correlations["ycenter"]
    return muon_correlations





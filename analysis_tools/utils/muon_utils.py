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
def propagate_muon(muons, muon_id, z): # propagate spherical coordinates
    x = muons["x0"][muon_id] + (z-muons["z0"][muon_id]) * np.cos(muons["phi"][muon_id]) * np.tan(muons["theta"][muon_id])
    y = muons["y0"][muon_id] + (z-muons["z0"][muon_id]) * np.sin(muons["phi"][muon_id]) * np.tan(muons["theta"][muon_id])
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
#   - timestamp (acceptance window params._correlation_ts_window)
#   - position (?)
def correlate_muons_and_muon_areas(muons, muon_areas, *, silent=False):
    n_muons = data_utils.length(muons)
    n_muon_areas = data_utils.length(muon_areas)
    # sort both muons & muon areas by timestamp
    muons = timestamp_utils.sort_by_timestamp(hits=muons, silent=silent)
    muon_areas = timestamp_utils.sort_by_timestamp(hits=muon_areas, silent=silent)
    ### correlate muons & muon areas in time (correlate 2 objects which have timestamps with a difference <= params._correlation_ts_window)
    # collect correlated indices of muons & muon areas
    ts_correlated_indices = [] # [(im = muon index, ia = muon area index) for correlated muons]
    # go step by step through muon (area) objects
    ia = 0 # current muon area object index
    im = 0 # current muon object index
    while True:
        ts_m = muons["ts"][im] # timestamp of current muon
        ts_a = muon_areas["ts"][ia] # timestamp of current muon area
        # if current muon is much later than current muon area, go to next muon area
        if ts_m > ts_a and ts_m-ts_a > params._correlation_ts_window: 
            ia += 1
        # if current muon is much earlier than current muon area, go to next muon
        if ts_m < ts_a and ts_a-ts_m > params._correlation_ts_window:
            im += 1
        # if current muon and muon area are within time window, "correlate them together" and go to next muon & muon area
        # each muon & muon area can therefore only be correlated once
        if (ts_m > ts_a and ts_m-ts_a <= params._correlation_ts_window) or (ts_m < ts_a and ts_a-ts_m <= params._correlation_ts_window):
            ts_correlated_indices.append((im, ia))
            im += 1
            ia += 1
        # muon & muon area object list boundaries: break if index out of range
        if im >= n_muons: break # stop as soon as no more muon are there
        if ia >= n_muon_areas: break # stop as soon as no more muon areas are there
    print(ts_correlated_indices)
    return ts_correlated_indices




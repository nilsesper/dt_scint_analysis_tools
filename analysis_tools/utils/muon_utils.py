###########################################
### MUON RECONSTRUCTION / DUMMY DATA UTILS
###########################################

import numpy as np
import copy

import analysis_tools.utils.math_utils as math_utils
import analysis_tools.utils.data_utils as data_utils

import analysis_tools.params.params as params

# -----------------------------------------

### propagate all muons to given z coordinate
def propagate_muons(muons, z): # propagate spherical coordinates
    x = muons["x0"] + (z-muons["z0"]) * np.cos(muons["phi"]) * np.tan(muons["theta"]) #* np.tan(muons["theta"])
    y = muons["y0"] + (z-muons["z0"]) * np.sin(muons["phi"]) * np.tan(muons["theta"]) #* np.tan(muons["theta"])
    return (x,y,z)


### generate random cosmic muons
# with spawnpoint range same for all muons: xrange = [xmin, xmax], yrange = [ymin, ymax], z0
# generate n muons
# pass separate timestamp for all muons i.e. ts = [ts[i] for i in range(n)]
def generate_cosmic_muons(n, ts, xrange, yrange, z0, *, silent=False, thetarange=[0, np.pi/2], phirange=[0,2*np.pi], theta_weight=params.cosmic_muon_theta_weight):
    if not silent: print(f"Generating {n:,} cosmic muons...")
    muons = {k: np.full(n, 0, dtype=v) for k,v in params._muon_obj_keys.items()}
    muons["x0"] = np.random.uniform(low=xrange[0], high=xrange[1], size=n).astype(dtype=params._muon_obj_keys["x0"])
    muons["y0"] = np.random.uniform(low=yrange[0], high=yrange[1], size=n).astype(dtype=params._muon_obj_keys["y0"]) # x,y uniformly distributed inside xrange, yrange
    muons["z0"] = np.full(n, z0, dtype=params._muon_obj_keys["z0"])
    muons["theta"] = math_utils.draw_from_pdf(pdf=theta_weight, val_range=thetarange, n=n, dtype=params._muon_obj_keys["theta"]) # theta distributed according to distribution
    muons["phi"] = np.random.uniform(low=phirange[0], high=phirange[1], size=n).astype(dtype=params._muon_obj_keys["phi"]) # phi uniformly distributed
    muons["ts"] = np.array(ts).astype(dtype=params._muon_obj_keys["ts"])
    muons["sim_id"] = np.arange(1, n+1, dtype=params._muon_obj_keys["sim_id"]) # sim_id starts at 1 (0 in data and for noise hits)
    return muons


### change muon base point (x,y,z) for new given z
def change_muon_base_point(muons, z_new, *, silent=False):
    n_muons = data_utils.length(muons)
    (x,y,z) = propagate_muons(muons, z=z_new)
    new_muons = copy.deepcopy(muons)
    new_muons["z0"] = np.full(n_muons, z_new)
    new_muons["x0"] = x
    new_muons["y0"] = y
    return new_muons






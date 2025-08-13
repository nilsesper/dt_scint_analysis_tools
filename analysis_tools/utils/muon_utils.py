###########################################
### MUON RECONSTRUCTION / DUMMY DATA UTILS
###########################################

import numpy as np
import copy
import os.path
from tqdm import tqdm

import analysis_tools.params.params as params
import analysis_tools.params.derived_params as derived_params
import analysis_tools.utils.timestamp_utils as timestamp_utils
import analysis_tools.utils.math_utils as math_utils

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










###############################
### DERIVED CONSTANTS & PARAMETERS
###############################
# collects "derived parameters", which depend on the content of params.py

import analysis_tools.params.params as params
import numpy as np

# -----------------------------------------

### ro_chs lists
# list of ro_chs used for dt chamber
_dt_ro_chs = list(params._dt_mapping.keys())
# list of ro_chs used for scintillator
_scint_ro_chs = list(params._scint_mapping.keys())

### ch lists
# list of allowed channels for dt chamber, by ro_ch
_dt_chs_by_ro_ch = {}
for ro_ch in _dt_ro_chs:
    ch_list = []
    for fe_conn_name, fe_mapping in params._dt_mapping[ro_ch].items():
        ch_list.extend(fe_mapping["chs"])
    _dt_chs_by_ro_ch[ro_ch] = ch_list
# list of allowed channels for scintillator, by ro_ch
_scint_chs_by_ro_ch = {}
for ro_ch in _scint_ro_chs:
    ch_list = []
    for ch_name, ch_mapping in params._scint_mapping[ro_ch].items():
        ch_list.append(ch_mapping["ch"])
    _scint_chs_by_ro_ch[ro_ch] = ch_list

### dumpfile keys
_dumpfile_keys = list(params._htg_keys.keys())

### generate dt remapping table: {ro_ch: {ch: {dt_keys: mapping value}}
_fecable2wireoffset = lambda fe: 4*((int(fe[:-1])-1)*2 + (1 if fe[-1]=='B' else 0)) # remapping function: fe connector (XA/B) -> wire offset
_dt_remap_table = {}
for ro_ch in _dt_ro_chs:
    _dt_remap_table[ro_ch] = {}
    for conn_id, (fe_connector_name, fe) in enumerate(params._dt_mapping[ro_ch].items()):
        fe_id = params._fe_idx_list.index(fe["fe"])
        sl = fe["sl"]
        # translate to ly, wi
        wireoffset = _fecable2wireoffset(fe["fe"])
        # fill remap table
        for ch_id in range(16):
            # conductors in the cable come in this order of layers: 4(outer), 2, 3, 1(inner)
            ly = [3,1,2,0][ch_id & 3]
            wi = (ch_id >> 2) + wireoffset
            ch = fe["chs"][ch_id]
            _dt_remap_table[ro_ch][ch] = {
                "conn_id": conn_id, # idx of conn name "J35" in fe_mapping dict
                "fe_id": fe_id, # idx of fe conn name "1A" in order starting at 1A
                "ch_id": ch_id, # ch id wrt connector 0-15, i.e. idx of ch in "chs" list of fe_mapping dict
                "sl": sl, # superlayer 1-3
                "ly": ly, # layer
                "wi": wi, # wire
            }
# generate inverted dt remapping table: {sl: {ly: {wi: {ch, ro_ch, conn_id, fe_id, ch_id}}}}
_dt_inverted_remap_table = {}
for ro_ch in _dt_ro_chs:
    for ch in _dt_remap_table[ro_ch].keys():
        sl = _dt_remap_table[ro_ch][ch]["sl"]
        ly = _dt_remap_table[ro_ch][ch]["ly"]
        wi = _dt_remap_table[ro_ch][ch]["wi"]
        if sl not in _dt_inverted_remap_table.keys():
            _dt_inverted_remap_table[sl] = {}
        if ly not in _dt_inverted_remap_table[sl].keys():
            _dt_inverted_remap_table[sl][ly] = {}
        _dt_inverted_remap_table[sl][ly][wi] = {
            "ch": ch,
            "ro_ch": ro_ch,
        }
        for k in ["conn_id", "fe_id", "ch_id"]:
            _dt_inverted_remap_table[sl][ly][wi][k] = _dt_remap_table[ro_ch][ch][k]

### generate scint remapping table: {ro_ch: {ch: {scint_keys: mapping value}}
_scint_keys = list(params._scint_mapping_keys.keys()) # ly, st, ch_id
_scint_remap_table = {}
for ro_ch in _scint_ro_chs:
    _scint_remap_table[ro_ch] = {}
    for ch_id, (ch_name, ch_mapping) in enumerate(params._scint_mapping[ro_ch].items()):
        ly = ch_mapping["ly"]
        st = ch_mapping["st"]
        ch = ch_mapping["ch"]
        _scint_remap_table[ro_ch][ch] = {
            "ch_id": ch_id, # idx of coinc channel name in mapping dict
            "ly": ly, # layer id
            "st": st, # strip id
        }
# generate inverted scint remapping table: {ly: {st: {ch, ro_ch, ch_id}}}
_scint_inverted_remap_table = {}
for ro_ch in _scint_ro_chs:
    for ch in _scint_remap_table[ro_ch].keys():
        ly = _scint_remap_table[ro_ch][ch]["ly"]
        st = _scint_remap_table[ro_ch][ch]["st"]
        if ly not in _scint_inverted_remap_table.keys():
            _scint_inverted_remap_table[ly] = {}
        _scint_inverted_remap_table[ly][st] = {
            "ch": ch,
            "ro_ch": ro_ch,
        }
        for k in ["ch_id"]:
            _scint_inverted_remap_table[ly][st][k] = _scint_remap_table[ro_ch][ch][k]

### timestamp conversion
_tdc_to_timestamp = 1
_bx_to_timestamp = np.uint64(params._lhc_tdc_count * _tdc_to_timestamp)
_orbit_to_timestamp = np.uint64(params._lhc_bunch_count * _bx_to_timestamp)
_orbit_overflow_to_timestamp = np.uint64(params._lhc_orbit_count * _orbit_to_timestamp)
# 1 timestamp unit = 1 TDC = 0.78 ns 
_ts_unit = 0.78 # ns
_max_ts_value = np.iinfo(params._ts_type).max # max value of ts

### drift velocity conversion
# conversion from um / ns = 10^3 m/s to mm / ts_unit
# ts_unit = _ts_unit = 0.78 ns
_drift_velocity_mm_per_timestamp = ( params._drift_velocity * (_ts_unit) * (1e-3) ) # unit calc: mm/tsu = um/ns * 0.78*ns/tsu * 1e-3*mm/um
# final unit: [_drift_velocity_mm_per_timestamp] = mm / ts_unit

### dt sl patterns
# idx of pattern name is key pat_type
_dt_sl_pattern_names = list(params._dt_sl_patterns.keys())

### dt chamber geometry --> in global coord frame
# calculate positions of center axis (height of wires) for all cells
# allows to easily check if muon has hit chamber
# _dt_cell_coordinates = {sl: {ly: {wi: [[xmin, xmax], [ymin, ymax], [zmin, zmax], x_center_pos, y_center_pos, z_center_pos]}}}
_dt_cell_coordinates = {}
for sl in params._dt_chamber["sls"].keys():
    _dt_cell_coordinates[sl] = {}
    for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
        _dt_cell_coordinates[sl][ly] = {}
        for wi in range(params._dt_chamber["sls"][sl]["n_wis"]):
            _dt_cell_coordinates[sl][ly][wi] = []
            if params._dt_chamber["sls"][sl]["orient"] == "phi": # phi wires along y
                # idx = 0: x axis (axis = 0)
                coord_axis = 0
                cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][coord_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
                pos_x = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+wi*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])+cell_offset
                size_x = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_x, pos_x+size_x])
                # idx = 1: y axis (axis = 1) ==> ALL CELLS LOOK THE SAME FOR PHI SL ALONG Y
                coord_axis = 1
                pos_y = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]
                size_y = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_y, pos_y+size_y])
                # idx = 2: z axis (axis = 2)
                coord_axis = 2
                pos_z = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+ly*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])
                size_z = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_z, pos_z+size_z])
                # idx = 3: x center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_x+size_x/2)
                # idx = 4: y center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_y+size_y/2)
                # idx = 5: z center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_z+size_z/2)
            elif params._dt_chamber["sls"][sl]["orient"] == "theta": # theta wires along x
                # idx = 0: x axis (axis = 0) ==> ALL CELLS LOOK THE SAME FOR THETA SL ALONG X
                coord_axis = 0
                pos_x = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]
                size_x = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_x, pos_x+size_x])
                # idx = 1: y axis (axis = 1)
                coord_axis = 1
                cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][coord_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
                pos_y = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+wi*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])+cell_offset
                size_y = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_y, pos_y+size_y])
                # idx = 2: z axis (axis = 2)
                coord_axis = 2
                pos_z = params._dt_chamber["pos"][coord_axis]+params._dt_chamber["sls"][sl]["pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_pos"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+ly*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])
                size_z = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
                _dt_cell_coordinates[sl][ly][wi].append([pos_z, pos_z+size_z])
                # idx = 3: x center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_x+size_x/2)
                # idx = 4: y center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_y+size_y/2)
                # idx = 5: z center pos
                _dt_cell_coordinates[sl][ly][wi].append(pos_z+size_z/2)

### dt sl pattern geometry
# only the marked cells are "valid"
# ly(z) wi(x) 0   1   2   3        
#               *this is the reference cell, with rel_wi=0
# 3   |   |   |*!*|   |   | 
# 2     |   | - | - |   |    
# 1   |   | - | - | - |   |
# 0     | - | - | - | - |    
# z axis goes up, x axis goes right
# rel_wi is wire idx relative to reference cell (ly3), and has valid range of [-2,-1,0,1,2]
_sl_pattern_coordinates = {} # coordinates of sub-coord frame used to fit dt sl patterns: {ly: {rel_wi: [[xmin, xmax], [zmin, zmax], x_center_pos, z_center_pos]}}, only 2 axes since fit is in x-z-projection !!
for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
    _sl_pattern_coordinates[ly] = {}
    for rel_wi in range(-2,2+1):
        _sl_pattern_coordinates[ly][rel_wi] = []
        sl = 1 # choose phi sl
        # x/y axis projection
        coord_axis = 0
        cell_offset = params._dt_chamber["sls"][sl]["ch_offset"][coord_axis] if params._dt_chamber["sls"][sl]["offset_ly"][ly] else 0
        pos_x = params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+rel_wi*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])+cell_offset
        size_x = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
        # z axis
        coord_axis = 2
        pos_z = params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis]+ly*(params._dt_chamber["sls"][sl]["ch_size"][coord_axis]+params._dt_chamber["sls"][sl]["ch_spacer"][coord_axis])
        size_z = params._dt_chamber["sls"][sl]["ch_size"][coord_axis]
        # fill data into coord map
        _sl_pattern_coordinates[ly][rel_wi].append([pos_x, pos_x+size_x]) # idx = 0: [xmin, xmax]
        _sl_pattern_coordinates[ly][rel_wi].append([pos_z, pos_z+size_z]) # idx = 0: [zmin, zmax]
        _sl_pattern_coordinates[ly][rel_wi].append(pos_x+size_x/2) # idx = 2: x center pos
        _sl_pattern_coordinates[ly][rel_wi].append(pos_z+size_z/2) # idx = 3: z center pos
# transform coordinate system from (0,0) at bottom of cell ly=3, rel_wi=0 to (0,0) at center of cell ly=3, rel_wi=0
_sl_pattern_coordinates_transform = [_sl_pattern_coordinates[3][0][2], _sl_pattern_coordinates[3][0][3]]
for ly in range(params._dt_chamber["sls"][sl]["n_lys"]):
    for rel_wi in range(-2,2+1):
        for i in [0,1]:
            for j in [0,1]:
                _sl_pattern_coordinates[ly][rel_wi][i][j] = _sl_pattern_coordinates[ly][rel_wi][i][j] - _sl_pattern_coordinates_transform[i]
        for i in [2,3]:
            _sl_pattern_coordinates[ly][rel_wi][i] = _sl_pattern_coordinates[ly][rel_wi][i] - _sl_pattern_coordinates_transform[i-2]

### dt sl pattern fit function
# use coordinates defined above
# fit measured timestamps to linear muon track
# alpha: angle in x-z-plane wrt downward facing muon
# x0: starting x position of muon at z = z(ly=3, rel_wi=0) = _sl_pattern_coordinates[ly=3][rel_wi=0][3 (z center pos)]
# muon track: x(z) = x0 + z*tan(alpha)
# drift times: x_drift = v_drift*t_drift  <=>  t_d = x_d/v_d
# measured timestamps: ts(ly=0,1,2,3) = t_drift(ly=0,1,2,3) + t0  (t0 = const for all 4 layers but free parameter for each new muon, since no trigger!)
# assume vdrift = const. with value given in params.py
# i want to fit the function ts(ly, wi) i.e. the timestamp values depending on the layer & wire 
# use local coordinate frame where reference cell (rel_wi=0, ly=3) is at (x=0, z=0) in the center
#   z_cell = (3-ly)*(-_cell_height)  because z axis goes up but ly goes down, therefore minus
#   x_cell = rel_wi*#####   where rel_wi(ly=X) = wi_X-wi_3 relative wire wrt wi 3 i.e. wire in ly3
# x(z) = x_cell + lat*x_drift  where lataterality = -1 (l = left of wire) or +1 (r = right of wire)
# => x(z) = x_cell + lat*v_d*t_d = x_cell + lat*v_d*(ts-t0)  !=   x0 + z*tan(alpha)
# use x_cell as x parameter and ts as y parameter for fit i.e. reassemble equation:
# <=> ts(x_cell) = (x0 + z*tan(alpha) - x_cell) * lat/v_d  + t0
# use functional format compatible with curve_fit i.e. f(x, ...)
# free parameters of function: t0, x0, tan(alpha) = tan_alpha
def f_ts_fit(x_cell, t0, x0, tan_alpha, z, laterality):
    # units: [ts] = 0.78ns = ts_unit, [x_cell] = mm
    ts_fit = (x0 + z * tan_alpha - x_cell) * laterality / _drift_velocity_mm_per_timestamp + t0
    return ts_fit

### dt sl pattern fitted muon line: x(z)
# x(z) = x0 + z*tan(alpha)
def f_x_muon(z, x0, tan_alpha):
    x_muon = x0 + z*tan_alpha
    return x_muon

### scintillator geometry --> in global coord frame
# calculate positions of center axis for all strips
# allows to easily check if muon has hit scintillator
# _dt_cell_coordinates = {ly: {st: [[xmin, xmax], [ymin, ymax], [zmin, zmax], x_center_pos, y_center_pos, z_center_pos]}}
_scintillator_strip_coordinates = {}
for ly in range(params._scintillator["n_lys"]):
    _scintillator_strip_coordinates[ly] = {}
    for st in range(params._scintillator["lys"][ly]["n_sts"]):
        _scintillator_strip_coordinates[ly][st] = []
        if params._scintillator["lys"][ly]["orient"] == "phi": # strips along y
            # idx = 0: x axis (axis = 0)
            coord_axis = 0
            pos_x = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]+st*(params._scintillator["lys"][ly]["ch_size"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis])
            size_x = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_x, pos_x+size_x])
            # idx = 1: y axis (axis = 1) ==> ALL CELLS LOOK THE SAME FOR PHI SL ALONG Y
            coord_axis = 1
            pos_y = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]
            size_y = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_y, pos_y+size_y])
            # idx = 2: z axis (axis = 2)
            coord_axis = 2
            pos_z = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]+ly*(params._scintillator["lys"][ly]["ch_size"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis])
            size_z = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_z, pos_z+size_z])
            # idx = 3: x center pos
            _scintillator_strip_coordinates[ly][st].append(pos_x+size_x/2)
            # idx = 4: y center pos
            _scintillator_strip_coordinates[ly][st].append(pos_y+size_y/2)
            # idx = 5: z center pos
            _scintillator_strip_coordinates[ly][st].append(pos_z+size_z/2)
        elif params._scintillator["lys"][ly]["orient"] == "theta": # strips along x
            # idx = 0: x axis (axis = 0) ==> ALL CELLS LOOK THE SAME FOR THETA SL ALONG X
            coord_axis = 0
            pos_x = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]
            size_x = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_x, pos_x+size_x])
            # idx = 1: y axis (axis = 1)
            coord_axis = 1
            pos_y = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]+st*(params._scintillator["lys"][ly]["ch_size"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis])
            size_y = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_y, pos_y+size_y])
            # idx = 2: z axis (axis = 2)
            coord_axis = 2
            pos_z = params._scintillator["pos"][coord_axis]+params._scintillator["lys"][ly]["pos"][coord_axis]+params._scintillator["lys"][ly]["ch_pos"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis]+ly*(params._scintillator["lys"][ly]["ch_size"][coord_axis]+params._scintillator["lys"][ly]["ch_spacer"][coord_axis])
            size_z = params._scintillator["lys"][ly]["ch_size"][coord_axis]
            _scintillator_strip_coordinates[ly][st].append([pos_z, pos_z+size_z])
            # idx = 3: x center pos
            _scintillator_strip_coordinates[ly][st].append(pos_x+size_x/2)
            # idx = 4: y center pos
            _scintillator_strip_coordinates[ly][st].append(pos_y+size_y/2)
            # idx = 5: z center pos
            _scintillator_strip_coordinates[ly][st].append(pos_z+size_z/2)

### scintillator sensitive area
# xmin of first strip, xmax of last strip), same for y axis, mean z of scintillator
# to be used for cutting reco muons from dt chamber to the muons where one can expect a scintillator hit
_scintillator_sensitive_coordinates = [] # [[xmin, xmax], [ymin, ymax], zmean] of full scintillator detector area
xmin, xmax, ymin, ymax, z0 = [], [], [], [], []
for ly in range(params._scintillator["n_lys"]):
    for st in range(params._scintillator["lys"][ly]["n_sts"]):
        if params._scintillator["lys"][ly]["orient"] == "phi":
            xmin.append(_scintillator_strip_coordinates[ly][st][0][0])
            xmax.append(_scintillator_strip_coordinates[ly][st][0][1])
        elif params._scintillator["lys"][ly]["orient"] == "theta":
            ymin.append(_scintillator_strip_coordinates[ly][st][1][0])
            ymax.append(_scintillator_strip_coordinates[ly][st][1][1])
        z0.append(_scintillator_strip_coordinates[ly][st][5])
_scintillator_sensitive_coordinates = [ [np.amin(xmin), np.amax(xmax)], [np.amin(ymin), np.amax(ymax)], np.mean(z0) ]

### generate all scintillator "pixels" i.e. muon areas by crossing 2 layers
# and then assign an integer index to all of them
# for easier handling: use pixel_index instead of always using [[xmin, xmax], [ymin, ymax], z]
_scint_pixel_coordinates = {} # scintillator pixel coordinates: {pixel_index: muon area [[xmin, xmax], [ymin, ymax], z]
_scint_pixel_mapping = {} # scintillator (ly0_st, ly1_st) mapping to pixel index: {(st of ly0, st of ly1): pixel_index}
xmin, xmax, ymin, ymax, z0, ly0_st, ly1_st = [], [], [], [], [], [], []
# partially hardcoded, only works for 2 layer scintillator setup...
for st0 in range(params._scintillator["lys"][0]["n_sts"]):
    for st1 in range(params._scintillator["lys"][1]["n_sts"]):
        if params._scintillator["lys"][0]["orient"] == "phi":
            xmin.append(_scintillator_strip_coordinates[0][st0][0][0])
            xmax.append(_scintillator_strip_coordinates[0][st0][0][1])
            ymin.append(_scintillator_strip_coordinates[1][st1][1][0])
            ymax.append(_scintillator_strip_coordinates[1][st1][1][1])
        elif params._scintillator["lys"][0]["orient"] == "theta":
            ymin.append(_scintillator_strip_coordinates[0][st0][0][0])
            ymax.append(_scintillator_strip_coordinates[0][st0][0][1])
            xmin.append(_scintillator_strip_coordinates[1][st1][1][0])
            xmax.append(_scintillator_strip_coordinates[1][st1][1][1])
        z0.append( np.mean([_scintillator_strip_coordinates[0][st0][5], _scintillator_strip_coordinates[1][st1][5]]) )
        ly0_st.append(st0)
        ly1_st.append(st1)
for pixel_index in range(len(ly0_st)):
    _scint_pixel_coordinates[pixel_index] = [ [xmin, xmax], [ymin, ymax], z0 ]
    _scint_pixel_mapping[(ly0_st[pixel_index], ly1_st[pixel_index])] = pixel_index

### mezzanine input channel mapping (for timing calibration)
# mapping of input channel = coincidence channel (for timing calibration there is no coincidence programmed)
# to the fpga pins & banks
mezzanine_input_bank_mapping = {} # {bank no: [input chs]}
fpga_banks = list(set([v["fpga_bank"] for v in params.mezzanine_input_mapping.values()]))
for bank in fpga_banks:
    mezzanine_input_bank_mapping[bank] = []
    for k,v in params.mezzanine_input_mapping.items():
        if v["fpga_bank"] == bank:
            mezzanine_input_bank_mapping[bank].append(k)

### color wheel for plotting
# give idx and return tab color
def color_wheel(i):
    color_list = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
        "tab:olive"
    ]
    i = i % len(color_list)
    return color_list[i]


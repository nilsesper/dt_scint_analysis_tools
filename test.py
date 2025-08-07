#########################
# generate dummy data dumpfile
#########################

import numpy as np

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils

filename = "dumpfiles/dummy_data.txt"

hit_list = [
    # wrong hits
    { "ro_ch": 7, "ch": 55, "oc": 0, "bx": 0, "tdc": 0, },
    { "ro_ch": 8, "ch": 254, "oc": 0, "bx": 20, "tdc": 0, },
    { "ro_ch": 24, "ch": 42, "oc": 0, "bx": 30, "tdc": 20, },
    # dt hits
    { "ro_ch": 8, "ch": 186, "oc": 1, "bx": 50, "tdc": 0, },
    { "ro_ch": 8, "ch": 187, "oc": 1, "bx": 50, "tdc": 3, },
    { "ro_ch": 8, "ch": 189, "oc": 1, "bx": 50, "tdc": 10, },
    { "ro_ch": 8, "ch": 188, "oc": 1, "bx": 50, "tdc": 25, },
    # scint hits
    { "ro_ch": 24, "ch": 1, "oc": 2, "bx": 5, "tdc": 3, },
    { "ro_ch": 24, "ch": 9, "oc": 2, "bx": 19, "tdc": 7, },
    { "ro_ch": 25, "ch": 0, "oc": 2, "bx": 23, "tdc": 9, },
    { "ro_ch": 25, "ch": 3, "oc": 2, "bx": 55, "tdc": 15, },
    { "ro_ch": 25, "ch": 3, "oc": 1, "bx": 55, "tdc": 15, }, # overflow
]

dummy_hits = dummy_gen.hit_list_to_hits(hit_list)
dummy_gen.write_to_dumpfile(file_name=filename, hits=dummy_hits)

dumpfile_hits = data_utils.import_raw(file_name=filename)
print("dumpfile_hits =",dumpfile_hits)

dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits)
dt_hits = timestamp_utils.add_timestamp(hits=dt_hits)
dt_hits = timestamp_utils.sort_by_timestamp(hits=dt_hits)
print("dt_hits =",dt_hits)

scint_hits = scint_utils.extract_scint_hits(hits=dumpfile_hits)
scint_hits = timestamp_utils.add_timestamp(hits=scint_hits)
scint_hits = timestamp_utils.sort_by_timestamp(hits=scint_hits)
print("scint_hits =",scint_hits)

dt_sl_patterns = dt_utils.find_sl_patterns(hits=dt_hits)
print("dt_sl_patterns =",dt_sl_patterns)
















#########################
# generate dummy data dumpfile
#########################

import numpy as np

from analysis_tools.utils import dummy_gen, data_utils, dt_utils

filename = "dumpfiles/dummy_data.txt"

hit_list = [
    # non dt hits
    { "ro_ch": 7, "ch": 55, "oc": 0, "bx": 0, "tdc": 0, },
    { "ro_ch": 8, "ch": 254, "oc": 0, "bx": 20, "tdc": 0, },
    # dt hits
    { "ro_ch": 8, "ch": 186, "oc": 1, "bx": 50, "tdc": 0, },
    { "ro_ch": 8, "ch": 187, "oc": 1, "bx": 52, "tdc": 3, },
    { "ro_ch": 8, "ch": 189, "oc": 1, "bx": 55, "tdc": 10, },
    { "ro_ch": 8, "ch": 188, "oc": 1, "bx": 55, "tdc": 25, },
]

dummy_hits = dummy_gen.hit_list_to_hits(hit_list)
dummy_gen.write_to_dumpfile(file_name=filename, hits=dummy_hits)

dumpfile_hits = data_utils.import_raw(file_name=filename)
print("dumpfile_hits =",dumpfile_hits)

dt_hits = dt_utils.extract_dt_hits(hits=dumpfile_hits)
print("dt_hits =",dt_hits)


















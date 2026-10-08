###########################################
### CUTS ON ROWS OF DATA
###########################################
# A cut is (key, operator, value), see data_utils.cut_data for the operators. All cuts are AND-ed.

from analysis_tools.params import params

# -----------------------------------------

### "key1,operator1,value1;key2,operator2,value2;..." -> [(key, operator, value)]
# a value is a number or a parameter name written as "params._name"
def parse_cuts(cuts_str):
    cuts = []
    if cuts_str is None or cuts_str.strip() == "":
        return cuts
    for cut_str in cuts_str.split(";"):
        if cut_str.strip() == "":
            continue
        parts = cut_str.split(",")
        if len(parts) != 3:
            raise ValueError(f"Cannot read cut \"{cut_str}\". Expected format: key,operator,value")
        key, operator, value = [p.strip() for p in parts]
        value = getattr(params, value.split("params.")[1]) if value.startswith("params.") else float(value)
        cuts.append((key, operator, value))
    return cuts

def check_cut_keys(cuts, data, file):
    for key, _, _ in cuts:
        if key not in data:
            raise KeyError(f"Cut key \"{key}\" is not a branch of {file}.")

###########################################
### UNRELATED / GENERAL MATH UTILITY
###########################################

import numpy as np
import copy
import os.path

# -----------------------------------------

### draw one random number from given pdf
# pdf: NORMALIZED pdf of values
# range: acceptable range of values
# n: how many values should be drawn (return np.array)
# dtype: specify dtype of np array
def draw_from_pdf(pdf, val_range, n=1, *, dtype=np.float64):
    x_arr = np.full(n, 0, dtype=dtype)
    for i in range(n):
        x, y = np.random.uniform(low=val_range[0], high=val_range[1]), np.random.uniform(low=0, high=1)
        pdf_val = pdf(x)
        while y > pdf_val: # only when y [0,1] is below pdf value for drawn x value, accept this value
            x, y = np.random.uniform(low=val_range[0], high=val_range[1]), np.random.uniform(low=0, high=1)
            pdf_val = pdf(x)
        x_arr[i] = x
    return x_arr

### calculate mean with error & std
def calculate_mean_std(data, err_data):
    n_data = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    err_mean = std / np.sqrt(n_data)
    return mean, std, err_mean

### return latex scientific notation for float
def latex_float(f, afterpoint_digits=2):
    float_str = f"{f:.{afterpoint_digits+1}g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

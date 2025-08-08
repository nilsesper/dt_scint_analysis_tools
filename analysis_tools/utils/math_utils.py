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
def draw_from_pdf(pdf, range):
    x, y = np.random.uniform(low=range[0], high=range[1]), np.random.uniform(low=0, high=1)
    pdf_val = pdf(x)
    while y > pdf_val: # only when y [0,1] is below pdf value for drawn x value, accept this value
        x, y = np.random.uniform(low=range[0], high=range[1]), np.random.uniform(low=0, high=1)
        pdf_val = pdf(x)
    return x




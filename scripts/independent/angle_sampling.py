#################################################################
### mc angle sampling
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

## angular functions

# (tan_alpha_x, tan_alpha_y) -> (phi, theta)
def tan_alpha_to_angles(tan_alpha_x, tan_alpha_y):
    #phi = np.atan2( tan_alpha_y , tan_alpha_x )
    ##phi = np.arctan( tan_alpha_y / tan_alpha_x )
    #phi_ = np.atan2( tan_alpha_y , tan_alpha_x )
    #phi_ = np.arctan( tan_alpha_y / tan_alpha_x )
    #theta = np.arctan( tan_alpha_y / np.sin(phi_) )
    #theta = np.atan2( tan_alpha_y , np.sin(phi_) )

    phi_reco_prelim = np.atan2( tan_alpha_y, tan_alpha_x )
    phi_periodicity = 2*np.pi
    phi = phi_reco_prelim - phi_periodicity*(phi_reco_prelim//phi_periodicity)
    theta = np.arctan( tan_alpha_x / np.cos(phi) ) #print( np.arctan( tan_alpha_x / np.cos(phi_reco) ) , np.arctan( tan_alpha_y / np.sin(phi_reco) ) )

    return phi, theta

# (phi, theta) -> (tan_alpha_x, tan_alpha_y)
def angles_to_tan(phi, theta):
    tan_alpha_x = np.tan(theta) * np.cos(phi)
    tan_alpha_y = np.tan(theta) * np.sin(phi)
    return tan_alpha_x, tan_alpha_y

gaus_sigma = 0.4
def pdf_cos2(x):
    norm = np.pi/2 # integral cos²(x) from 0 to pi = pi / 2
    return np.cos(x)**2 * 1/norm # normalized to 1 for integral from 0 to pi



# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 12}) #'font.sans-serif': 'Arial',
def main():
    n_samples = 100000
    n_bins = 500

    ### tan alpha to angles

    # sample tan alpha
    #tan_alpha_x = np.random.normal(loc=0.0, scale=gaus_sigma, size=n_samples)
    tan_alpha_x = np.random.laplace(loc=0.0, scale=gaus_sigma, size=n_samples)
    #tan_alpha_x = np.random.uniform(low=0, high=2, size=n_samples)
    
    #tan_alpha_y = np.random.normal(loc=0.0, scale=gaus_sigma, size=n_samples)
    tan_alpha_y = np.random.laplace(loc=0.0, scale=gaus_sigma, size=n_samples)
    #tan_alpha_y = np.random.uniform(low=0, high=2, size=n_samples)

    alpha_x = np.arctan(tan_alpha_x)
    alpha_y = np.arctan(tan_alpha_y)
    
    tan_alpha_bins = np.linspace(-3, 3, n_bins)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(tan_alpha_x, bins=tan_alpha_bins, label="tan_alpha_x", histtype="step")
    ax.hist(tan_alpha_y, bins=tan_alpha_bins, label="tan_alpha_y", histtype="step")
    ax.hist(alpha_x, bins=tan_alpha_bins, label="alpha_x", histtype="step")
    ax.hist(alpha_y, bins=tan_alpha_bins, label="alpha_y", histtype="step")
    ax.legend()
    fig.tight_layout()
    fig.show()

    # propagate to phi and theta
    phi, theta = tan_alpha_to_angles(tan_alpha_x=tan_alpha_x, tan_alpha_y=tan_alpha_y)

    phi_bins = np.linspace(-0.5, 7, n_bins)
    theta_bins = np.linspace(-0.2, 1.7, n_bins)
    # plot phi
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(phi, bins=phi_bins)
    ax.set_xlabel("phi")
    fig.tight_layout()
    fig.show()
    # plot theta
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(theta, bins=theta_bins)
    ax.set_xlabel("theta")
    fig.tight_layout()
    fig.show()



    ### plot angle function values
    n_pixels = 100
    tan_alpha_x = np.linspace(-2,2,n_pixels)
    tan_alpha_y = np.linspace(-2,2,n_pixels)
    phi, theta = [[0 for ix in range(len(tan_alpha_x))] for iy in range(len(tan_alpha_y))], [[0 for ix in range(len(tan_alpha_x))] for iy in range(len(tan_alpha_y))]
    for ix in range(len(tan_alpha_x)):
        for iy in range(len(tan_alpha_y)):
            phi[iy][ix], theta[iy][ix] = tan_alpha_to_angles(tan_alpha_x[ix], tan_alpha_y[iy])
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=phi, extent=[min(tan_alpha_x), max(tan_alpha_x), min(tan_alpha_y), max(tan_alpha_y)])
    ax.set_title("phi")
    ax.set_xlabel("tan_alpha_x")
    ax.set_ylabel("tan_alpha_y")
    ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    im_obj = ax.imshow(X=theta, extent=[min(tan_alpha_x), max(tan_alpha_x), min(tan_alpha_y), max(tan_alpha_y)])
    ax.set_title("theta")
    ax.set_xlabel("tan_alpha_x")
    ax.set_ylabel("tan_alpha_y")
    ax.legend()
    plt.colorbar(im_obj)
    fig.tight_layout()
    fig.show()



    ### angles to tan alpha

    phi = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
    theta = math_utils.draw_from_pdf(pdf=pdf_cos2, val_range=[0, np.pi/2], n=n_samples)

    phi_bins = np.linspace(-0.5, 7, n_bins)
    theta_bins = np.linspace(-0.2, 1.7, n_bins)
    # plot phi
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(phi, bins=phi_bins)
    ax.set_xlabel("phi")
    fig.tight_layout()
    fig.show()
    # plot theta
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(theta, bins=theta_bins)
    ax.set_xlabel("theta")
    fig.tight_layout()
    fig.show()

    # propagate to tan alpha
    tan_alpha_x, tan_alpha_y = angles_to_tan(phi=phi, theta=theta)
    tan_alpha_bins = np.linspace(-3, 3, n_bins)
    # plot tan alpha x
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(tan_alpha_x, bins=tan_alpha_bins)
    ax.set_xlabel("tan_alpha_x")
    fig.tight_layout()
    fig.show()
    # plot tan alpha y
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.hist(tan_alpha_y, bins=tan_alpha_bins)
    ax.set_xlabel("tan_alpha_y")
    fig.tight_layout()
    fig.show()



    input("Press enter to exit.")
    exit()




if __name__ == "__main__":
    main()
    print(f"###### Done.")




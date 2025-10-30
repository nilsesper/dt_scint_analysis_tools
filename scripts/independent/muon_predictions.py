#################################################################
### muon predictions
# based on approximations from CosmicMuon-Shukla ( https://doi.org/10.1142/S0217751X18501750 )
#################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import copy
import argparse

from analysis_tools.utils import dummy_gen, data_utils, dt_utils, scint_utils, timestamp_utils, geoplot_utils, muon_utils, math_utils, hist_utils, process_utils, combination_utils
from analysis_tools.params import params, derived_params

# ---------------------------------------------------------------

n = 3
E0 = 23.78 # GeV
eps = 2000

I0 = 65.2 # m^-2 s^-1 sr^-1 integrated over all E
theta_at_I0 = 0 # rad
E_min_at_I0 = 1 # GeV
E_max_at_I0 = 10000 # GeV
n_E_steps_at_I0 = 10000

E_I0_integration_range = np.logspace(np.log10(E_min_at_I0), np.log10(E_max_at_I0), n_E_steps_at_I0)

# integration

n_E_steps = 10000
E_thres = 1 # GeV
E_max = 100000 # GeV
E_integration_range = np.logspace(np.log10(E_thres), np.log10(E_max), n_E_steps)

n_theta_steps = 10000
theta_min = 0
theta_max = np.pi/2 - 0.01
theta_integration_range = np.linspace(theta_min, theta_max, n_theta_steps)

ref_area = 1 # m^2

# dependencies

def intensity_dependency(E, theta):
    return (E+E0)**(-n)*(1+E/eps)**(-1)*np.cos(theta)**(n-1)
def integrated_intensity_dependency(E_range, theta):
    return np.trapezoid(y=intensity_dependency(E=E_range, theta=theta), x=E_range)
# normalize properly
def intensity(E, theta):
    N = 1 / integrated_intensity_dependency(E_range=E_I0_integration_range, theta=theta_at_I0)
    return I0 * N * intensity_dependency(E, theta)

# plotting

E_range = np.logspace(np.log10(1), np.log10(1000), 1000) # GeV
theta_range = np.linspace(0, np.pi/2-0.1, 1000) # rad

n_values = 1000
theta_values = np.linspace(0, np.pi/2-0.1, n_values) # rad
E_values = np.logspace(np.log10(1), np.log10(100), n_values) # GeV

# main function
@mpl.rc_context({'font.family': 'sans-serif', 'font.size': 30}) #'font.sans-serif': 'Arial',
def main():

    ### plotting

    # energy dependence
    normalization = mcolors.Normalize(vmin=np.amin(theta_values), vmax=np.amax(theta_values)) # Normalize()
    colormap = cm.jet
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    for i in range(n_values):
        ax.plot(E_range, intensity(E=E_range, theta=theta_values[i]), color=colormap(normalization(theta_values[i])))
    #ax.legend()
    scalarmappaple = cm.ScalarMappable(norm=normalization, cmap=colormap)
    scalarmappaple.set_array(theta_values)
    cb = fig.colorbar(scalarmappaple, ax=ax)
    #ax.set_title("Atmospheric muon flux: Energy dependence")
    cb.set_label("$\\theta$ [rad]")
    ax.set_xlabel("$E$ [GeV]")
    ax.set_ylabel(f"$I\\;(E,\\;\\theta)$  [m$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$ GeV$^{{-1}}$]")
    ax.set_yscale("log")
    ax.set_xscale("log")
    #ax.set_ylim(ymin=5e-4)
    fig.tight_layout()
    fig.show()

    # angular dependence
    normalization = mcolors.LogNorm(vmin=np.amin(E_values), vmax=np.amax(E_values))
    colormap = cm.jet
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    for i in range(n_values):
        ax.plot(theta_range, intensity(E=E_values[i], theta=theta_range), color=colormap(normalization(E_values[i])))
    #ax.legend()
    scalarmappaple = cm.ScalarMappable(norm=normalization, cmap=colormap)
    scalarmappaple.set_array(theta_values)
    cb = fig.colorbar(scalarmappaple, ax=ax)
    #ax.set_title("Atmospheric muon flux: Zenith angle dependence")
    cb.set_label("$E$ [GeV]")
    ax.set_xlabel("$\\theta$ [rad]")
    ax.set_ylabel(f"$I\\;(E,\\;\\theta)$  [m$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$ GeV$^{{-1}}$]")
    #ax.set_yscale("log")
    #ax.set_ylim(ymin=5e-4)
    #ax.set_xscale("log")
    fig.tight_layout()
    fig.show()

    ### total rate estimation

    # integrate over all energies >= E_thres (numerically)
    print(f"E integration range = {E_thres} GeV to {E_max} GeV")
    print(f"theta integration range = {theta_min} rad to {theta_max} rad")
    intensity_E_integrated = np.zeros(n_theta_steps)
    for i in range(n_theta_steps):
        intensity_E_integrated[i] = np.trapezoid(y=intensity(E=E_integration_range, theta=theta_integration_range[i]), x=E_integration_range)
        if i % 1000 == 0:
            print(f"integreated I over E for theta = {theta_integration_range[i]} rad = {intensity_E_integrated[i]} m^-2 s^-1 sr^-1")

    # additionally integrate over all angles theta 0 to np.pi/2 - small_value
    intensity_integrated = np.trapezoid(y=intensity_E_integrated*2*np.pi*np.sin(theta_integration_range), x=theta_integration_range)
    print(f"integreated I over E and theta = {intensity_integrated} m^-2 s^-1")

    ref_rate = intensity_integrated*ref_area
    print(f"reference rate = {ref_rate} Hz / {ref_area} m^2")




    input("Press enter to exit.")
    exit()




if __name__ == "__main__":
    main()
    print(f"###### Done.")




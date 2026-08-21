import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Factor_air = 1.0
FACTOR_ARGON = 1.382  # F-201CV-2K0
FACTOR_CO2 = 0.7423   # F-201CV-200

# --- Uncertainty on the conversion factors -------------------------------
# Assumed 1 % relative (calibration/reading) uncertainty on each factor.
FACTOR_REL_ERR = 0.01  # 1 %
sigma_FACTOR_ARGON = FACTOR_REL_ERR * FACTOR_ARGON
sigma_FACTOR_CO2 = FACTOR_REL_ERR * FACTOR_CO2

# Chamber dimensions
l = 2.00  # m
w = 2.55  # m
h = 0.013 * 4 * 3  # m  cellheight * vertical cells per SL * number of SLs


#####################################
max_gas_flow = 10  # l/h
ar_concentration = 0.845  # fraction of argon in the gas mixture (=84.5 %)

# --- Uncertainty on the argon concentration -------------------------------
# The mixing panel only accepts the argon percentage with one decimal place
# (e.g. 84.5 %), so every value that is set is rounded to the nearest 0.1 %.
# Treating this as a rounding/quantisation error, the resolution is
# 0.1 % = 0.001 (as a fraction) and the resulting standard uncertainty is
# taken as half of that resolution (worst-case / half-LSB approach):
ar_concentration_resolution = 0.001          # 0.1 % expressed as a fraction
sigma_ar_concentration = ar_concentration_resolution / 2   # +/- 0.05 %-points

# (If you prefer to treat the rounding error as uniformly distributed
#  between -0.05 and +0.05 %-points, use resolution/(2*sqrt(3)) instead,
#  i.e. sigma_ar_concentration = ar_concentration_resolution / (2*np.sqrt(3)) )

co2_concentration = 1 - ar_concentration  # fraction of CO2 in the gas
# co2 = 1 - ar, and "1" has no uncertainty, so the error is identical
sigma_co2_concentration = sigma_ar_concentration

ar_flow = max_gas_flow * ar_concentration  # l/h
co2_flow = max_gas_flow * co2_concentration  # l/h

# Uncertainty on the flows coming from the rounding of the argon percentage
# (max_gas_flow is assumed to be set independently/exactly here)
sigma_ar_flow = max_gas_flow * sigma_ar_concentration
sigma_co2_flow = max_gas_flow * sigma_co2_concentration

ar_flow_air_eq = ar_flow / FACTOR_ARGON  # l/h Air equivalent
co2_flow_air_eq = co2_flow / FACTOR_CO2  # l/h Air equivalent

# Combine the flow uncertainty and the 1 % factor uncertainty in quadrature
# (standard Gaussian error propagation for a ratio z = x / y):
#   (sigma_z / z)^2 = (sigma_x / x)^2 + (sigma_y / y)^2
rel_err_ar_flow_air_eq = np.sqrt(
    (sigma_ar_flow / ar_flow) ** 2 + (sigma_FACTOR_ARGON / FACTOR_ARGON) ** 2
)
rel_err_co2_flow_air_eq = np.sqrt(
    (sigma_co2_flow / co2_flow) ** 2 + (sigma_FACTOR_CO2 / FACTOR_CO2) ** 2
)

sigma_ar_flow_air_eq = ar_flow_air_eq * rel_err_ar_flow_air_eq
sigma_co2_flow_air_eq = co2_flow_air_eq * rel_err_co2_flow_air_eq

print("\nMaximal Gas Flows")
print("-----------------------")
print(f"Argon: {ar_flow:.2f} +/- {sigma_ar_flow:.3f} l/h")
print(f"CO2:   {co2_flow:.2f} +/- {sigma_co2_flow:.3f} l/h")
print(f"Argon-Air_eq: {ar_flow_air_eq:.2f} +/- {sigma_ar_flow_air_eq:.3f} l/h")
print(f"CO2-Air_eq:   {co2_flow_air_eq:.2f} +/- {sigma_co2_flow_air_eq:.3f} l/h")
print(f"\nArgonpercentage: {100*ar_concentration:.2f} +/- {100*sigma_ar_concentration:.3f} %")
print(f"Total-Gasflow: {max_gas_flow:.2f} l/h")

# ===========================================================================
# Actual gas mix resulting from the AIR-EQUIVALENT values entered on the MFCs
# ===========================================================================
# You don't set the real gas flow directly - you set the air-equivalent flow
# on each mass-flow-controller (the numbers above), and the controller
# converts that to a real flow using its conversion factor:
#     real_flow = air_eq_value_set * FACTOR
# Two things can now make the *real* mix deviate from the intended 84.5 %:
#   1) the display/setpoint resolution of the air-eq value you can dial in
#   2) the 1% uncertainty on each conversion factor (independent for Ar/CO2)

# Argon air-eq can only be dialed in to ONE decimal place (l/h) -> resolution 0.1 l/h
AR_AIR_EQ_RESOLUTION = 0.1  # l/h
# CO2 resolution assumed the same one-decimal limit - CHANGE THIS if the CO2
# MFC actually allows finer/coarser steps than argon.
CO2_AIR_EQ_RESOLUTION = 0.1  # l/h

sigma_ar_air_eq_round = AR_AIR_EQ_RESOLUTION / 2   # half-LSB rounding error
sigma_co2_air_eq_round = CO2_AIR_EQ_RESOLUTION / 2

# Values as actually set on the machine (rounded to the settable resolution)
ar_flow_air_eq_set = round(ar_flow_air_eq / AR_AIR_EQ_RESOLUTION) * AR_AIR_EQ_RESOLUTION
co2_flow_air_eq_set = round(co2_flow_air_eq / CO2_AIR_EQ_RESOLUTION) * CO2_AIR_EQ_RESOLUTION

# Resulting real (true) gas flows delivered to the chamber
ar_flow_true = ar_flow_air_eq_set * FACTOR_ARGON
co2_flow_true = co2_flow_air_eq_set * FACTOR_CO2

# --- statistical (quadrature) error on the true flows: z = x*y ---
rel_err_ar_true = np.sqrt(
    (sigma_ar_air_eq_round / ar_flow_air_eq_set) ** 2 + (sigma_FACTOR_ARGON / FACTOR_ARGON) ** 2
)
rel_err_co2_true = np.sqrt(
    (sigma_co2_air_eq_round / co2_flow_air_eq_set) ** 2 + (sigma_FACTOR_CO2 / FACTOR_CO2) ** 2
)
sigma_ar_flow_true = ar_flow_true * rel_err_ar_true
sigma_co2_flow_true = co2_flow_true * rel_err_co2_true

# --- resulting real gas mix fraction f = ar / (ar + co2) ---
total_flow_true = ar_flow_true + co2_flow_true
ar_concentration_true = ar_flow_true / total_flow_true

# error propagation on the ratio (partial derivatives of f wrt ar and co2)
df_dar = co2_flow_true / total_flow_true ** 2
df_dco2 = -ar_flow_true / total_flow_true ** 2
sigma_ar_concentration_true = np.sqrt(
    (df_dar * sigma_ar_flow_true) ** 2 + (df_dco2 * sigma_co2_flow_true) ** 2
)

deviation_pp = (ar_concentration_true - ar_concentration) * 100  # percentage points

# --- worst-case (linear, non-statistical) bound: factors err in opposite directions ---
ar_flow_wc_high = ar_flow_air_eq_set * (FACTOR_ARGON + sigma_FACTOR_ARGON)
co2_flow_wc_low = co2_flow_air_eq_set * (FACTOR_CO2 - sigma_FACTOR_CO2)
ar_conc_wc_high = ar_flow_wc_high / (ar_flow_wc_high + co2_flow_wc_low)

ar_flow_wc_low = ar_flow_air_eq_set * (FACTOR_ARGON - sigma_FACTOR_ARGON)
co2_flow_wc_high = co2_flow_air_eq_set * (FACTOR_CO2 + sigma_FACTOR_CO2)
ar_conc_wc_low = ar_flow_wc_low / (ar_flow_wc_low + co2_flow_wc_high)

print("\nResulting real gas mix (from air-equivalent values set on the machine)")
print("------------------------------------------------------------------------")
print(f"Air-eq set  Argon: {ar_flow_air_eq_set:.2f} l/h   CO2: {co2_flow_air_eq_set:.2f} l/h")
print(f"Real flow   Argon: {ar_flow_true:.3f} +/- {sigma_ar_flow_true:.3f} l/h")
print(f"Real flow   CO2:   {co2_flow_true:.3f} +/- {sigma_co2_flow_true:.3f} l/h")
print(f"\nReal Argon fraction: {100*ar_concentration_true:.3f} +/- {100*sigma_ar_concentration_true:.3f} %")
print(f"Deviation from nominal ({100*ar_concentration:.2f} %): {deviation_pp:+.3f} pp (1 sigma)")
print(f"\nWorst-case (factor errors opposite sign) Argon fraction range:")
print(f"  {100*ar_conc_wc_low:.3f} % ... {100*ar_conc_wc_high:.3f} %"
      f"  (i.e. {100*ar_concentration:.2f} % {100*(ar_conc_wc_high-ar_concentration):+.3f} pp / "
      f"{100*(ar_conc_wc_low-ar_concentration):+.3f} pp)")

volume = l * w * h  # m^3
l_to_m3 = 1 / 1000  # l/m^3
htoday = 24  # h/d
safetyfactor = 2  # Safety factor indicating how many times the chamber volume should be flushed
flush_time = volume / (max_gas_flow * l_to_m3)  # h
flush_time_safe = flush_time * safetyfactor  # h
print("\nFlushing time")
print("----------------")
print(f"Room volume: {volume:.2f} m^3")
print("\nFlushing Time")
print(f"Flushing-Time without Safety: {volume/(max_gas_flow * l_to_m3):.2f} h")
print(f"Flushing-Time with Safety: {flush_time_safe:.2f} h")

print(f"\nFlushing-Time without Safety in days: {flush_time/htoday:.2f} d")
print(f"Flushing-Time with Safety in days: {flush_time_safe/htoday:.2f} d")


days = np.array([0, 4, 18, 20, 21, 24, 27])
ar = np.array([195, 170, 102, 93, 87, 75, 57])
co2 = np.array([61, 60, 59, 59, 58, 56, 55])

# exponentielles Modell
def exp_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

# Ableitung = Druckverlust pro Tag
def pressure_loss(x, a, b):
    return a * b * np.exp(-b * x)

# Fit
params_ar, _ = curve_fit(exp_fit, days, ar, maxfev=10000)
params_co2, _ = curve_fit(exp_fit, days, co2, maxfev=10000)

a_ar, b_ar, c_ar = params_ar
a_co2, b_co2, c_co2 = params_co2

# Zeitachse inkl. 20 Tage Extrapolation
t = np.linspace(0, 47, 200)

# Druckverlust bar/Tag
loss_ar = pressure_loss(t, a_ar, b_ar)
loss_co2 = pressure_loss(t, a_co2, b_co2)

plt.figure(figsize=(8, 5))
plt.plot(t, loss_ar, label="Argon Druckverlust")
plt.plot(t, loss_co2, label="CO\u2082 Druckverlust")

plt.xlabel("Zeit [Tage]")
plt.ylabel("Druckverlust [bar/Tag]")
plt.grid()
plt.legend()
plt.savefig("druckverlust.png", dpi=150)
plt.show()

# Beispiele:
print(f"Argon Druckverlust am Tag 0: {pressure_loss(0,a_ar,b_ar):.2f} bar/Tag")
print(f"Argon Druckverlust am Tag 27: {pressure_loss(27,a_ar,b_ar):.2f} bar/Tag")
print(f"Argon Druckverlust am Tag 47: {pressure_loss(47,a_ar,b_ar):.2f} bar/Tag")

print(f"CO2 Druckverlust am Tag 0: {pressure_loss(0,a_co2,b_co2):.2f} bar/Tag")
print(f"CO2 Druckverlust am Tag 27: {pressure_loss(27,a_co2,b_co2):.2f} bar/Tag")
print(f"CO2 Druckverlust am Tag 47: {pressure_loss(47,a_co2,b_co2):.2f} bar/Tag")
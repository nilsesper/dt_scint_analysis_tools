import numpy as np
import matplotlib.pyplot as plt
Factor_air = 1.0
FACTOR_ARGON = 1.382 #F-201CV-2K0
FACTOR_CO2 = 0.7423  #F-201CV-200
#Chamber dimensions
l = 2.00 #m
w = 2.55 #m
h = 0.013*4*3 #m cellheight * vertical cells per SL *number of SLs
"""
#air_argon = float(input("Argon-MFC [l/h Luftäquivalent]: "))
#argon_percent = float(input("Gewünschter Argonanteil [%]: "))
air_argon = 6.12 # l/h Air equivalent
argon_percent = 85 # percentage of argon in the gas mixture (per volume)
x = argon_percent / 100.0

air_co2 = ((1 - x) / x) * (FACTOR_ARGON / FACTOR_CO2) * air_argon

# actual gas flows in l/h
gas_argon = air_argon * FACTOR_ARGON
gas_co2 = air_co2 * FACTOR_CO2

total_gas_flow = gas_argon + gas_co2

print("\nSettings")
print("-------------")
print(f"Argon-Air_eq: {air_argon:.2f} l/h Air equivalent")
print(f"CO2-Air_eq:   {air_co2:.2f} l/h Air equivalent")

print("\nResulting Gas Flows")
print("-----------------------")
print(f"Argon: {gas_argon:.2f} l/h")
print(f"CO2:   {gas_co2:.2f} l/h")

argon_fraction = gas_argon / (gas_argon + gas_co2)
print(f"\nArgonpercentage: {100*argon_fraction:.2f} %")
print(f"Total-Gasflow: {total_gas_flow:.2f} l/h")

volume = l * w * h # m^3
l_to_m3 = 1/1000 # l/m^3
htoday = 24 # h/d
safetyfactor = 2 # Safety factor indicating how many times the chamber volume should be flushed
flush_time = volume/(total_gas_flow * l_to_m3) # h
flush_time_safe = flush_time * safetyfactor # h
print("\nFlushing time")
print("----------------")
print(f"Room volume: {volume:.2f} m^3")
print ("\nFlushing Time")
print(f"Flushing-Time without Safety: {volume/(total_gas_flow * l_to_m3):.2f} h")
print(f"Flushing-Time with Safety: {flush_time_safe:.2f} h")

print(f"\nFlushing-Time without Safety in days: {flush_time/htoday:.2f} d")
print (f"Flushing-Time with Safety in days: {flush_time_safe/htoday:.2f} d")
"""


#####################################
max_gas_flow = 10 # l/h
ar_concentration = 0.87 # fraction of argon in the gas mixture
co2_concentration = 1 - ar_concentration # fraction of CO2 in the gas
ar_flow = max_gas_flow * ar_concentration # l/h
co2_flow = max_gas_flow * co2_concentration # l/h
ar_flow_air_eq = ar_flow / FACTOR_ARGON # l/h Air equivalent
co2_flow_air_eq = co2_flow / FACTOR_CO2 # l/h Air equivalent    
print("\nMaximal Gas Flows")
print("-----------------------")
print(f"Argon: {ar_flow:.2f} l/h")
print(f"CO2:   {co2_flow:.2f} l/h")
print(f"Argon-Air_eq: {ar_flow_air_eq:.2f} l/h")

print(f"CO2-Air_eq:   {co2_flow_air_eq:.2f} l/h")
print(f"\nArgonpercentage: {100*ar_concentration:.2f} %")
print(f"Total-Gasflow: {max_gas_flow:.2f} l/h")

volume = l * w * h # m^3
l_to_m3 = 1/1000 # l/m^3
htoday = 24 # h/d
safetyfactor = 2 # Safety factor indicating how many times the chamber volume should be flushed
flush_time = volume/(max_gas_flow * l_to_m3) # h
flush_time_safe = flush_time * safetyfactor # h
print("\nFlushing time")
print("----------------")
print(f"Room volume: {volume:.2f} m^3")
print ("\nFlushing Time")
print(f"Flushing-Time without Safety: {volume/(max_gas_flow * l_to_m3):.2f} h")
print(f"Flushing-Time with Safety: {flush_time_safe:.2f} h")

print(f"\nFlushing-Time without Safety in days: {flush_time/htoday:.2f} d")
print (f"Flushing-Time with Safety in days: {flush_time_safe/htoday:.2f} d")



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

days = np.array([0, 4, 18, 20, 21, 24, 27])
ar = np.array([195, 170, 102, 93, 87, 75, 57])
co2 = np.array([61, 60, 59, 59, 58, 56, 55])

# exponentielles Modell
def exp_fit(x, a, b, c):
    return a * np.exp(-b*x) + c

# Ableitung = Druckverlust pro Tag
def pressure_loss(x, a, b):
    return a * b * np.exp(-b*x)

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

plt.figure(figsize=(8,5))
plt.plot(t, loss_ar, label="Argon Druckverlust")
plt.plot(t, loss_co2, label="CO₂ Druckverlust")

plt.xlabel("Zeit [Tage]")
plt.ylabel("Druckverlust [bar/Tag]")
plt.grid()
plt.legend()
plt.show()

# Beispiele:
print(f"Argon Druckverlust am Tag 0: {pressure_loss(0,a_ar,b_ar):.2f} bar/Tag")
print(f"Argon Druckverlust am Tag 27: {pressure_loss(27,a_ar,b_ar):.2f} bar/Tag")
print(f"Argon Druckverlust am Tag 47: {pressure_loss(47,a_ar,b_ar):.2f} bar/Tag")

print(f"CO2 Druckverlust am Tag 0: {pressure_loss(0,a_co2,b_co2):.2f} bar/Tag")
print(f"CO2 Druckverlust am Tag 27: {pressure_loss(27,a_co2,b_co2):.2f} bar/Tag")
print(f"CO2 Druckverlust am Tag 47: {pressure_loss(47,a_co2,b_co2):.2f} bar/Tag")
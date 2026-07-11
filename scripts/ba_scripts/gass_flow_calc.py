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
ar_concentration = 0.82 # fraction of argon in the gas mixture
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


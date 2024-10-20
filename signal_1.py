import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import gausspulse

# Time parameters
T = 0.01  # Total time (s)
fs = 1e6  # Sampling frequency 1 MHz
t = np.linspace(0, T, int(T * fs))  # Time vector
theta = np.radians(45) #45 between transducer and blood vessel

# Signal parameters
A = 1  # Amplitude
f0 = 5e6  # Carrier frequency 5 MHz
B = 5000  # Sweep bandwidth 5 kHz
T_sweep = 0.002  # Sweep time (s)

# Transmitted FMCW signal (triangular modulation)
f_t = f0 + B * (2 * np.abs(np.mod(t, T_sweep) / T_sweep - 0.5) - 0.5) #transmitted frequency 
vT = A * np.cos(2 * np.pi * f_t * t) #Transmitted signal
c = 1540  # Speed of sound in blood (m/s)

# Echo signal top and bottom blood vessel wall
d=abs(6.26*10**(-3) +10**(-4)*(0.1*np.sin(2 * np.pi * t / T)+ 0.05 * np.sin(2 * np.pi * t / T)+0.1*np.cos(2 * np.pi * t / T))) #diameter d=6.26mm with floating

delay=6.26*10**(-3)/c #time delay calculated from t=R/c
# Frequency of the echo signal, top wall and bottom wall with time delay
f_Rt = f0 + B * (2 * np.abs((np.mod(np.linspace(-0.00004, 0.00996, int(0.01 * 1000000)), T_sweep)) / T_sweep - 0.5) - 0.5)
f_Rb = f0 + B * (2 * np.abs((np.mod(np.linspace(-(0.00004+delay), 0.01-(0.00004+delay), int(0.01 * 1000000)), T_sweep)) / T_sweep - 0.5) - 0.5)
#phase shift
phase_shift = 2*np.pi*f0*0.00002
#two echo signals from blood vessel
v_Rt =0.4*A* np.cos(2 * np.pi * f_Rt * (t-0.00004)+phase_shift)
v_Rb =0.2*A* np.cos(2 * np.pi * f_Rb * (t-0.00004-delay)+phase_shift)
#two mixed signals
v_D = vT*v_Rt
v_D2 =vT*v_Rb
# Doppler shift simulation (varying Doppler frequency)
v = 0.4  # Blood flow velocity (m/s)
f_D_max = abs(2 * v * f0*np.cos(theta) / c) #45degree with transducer
# Simulate varying Doppler shift
f_D = abs(f_D_max +f_D_max*10**(-1)*(0.1*np.sin(4 * np.pi * t / T)+ 0.2 * np.sin(4 * np.pi * t / T)+0.5*np.cos(4 * np.pi * t / T) ) )

f_Rd = f_Rt + f_D  # Received frequency with doppler (f_R)
v_Rd = 0.2*A* np.cos(2 * np.pi * f_Rd * (t-0.0002)+phase_shift)#echo signal
v_D3 =vT*v_Rd#mixed signal

A_t= abs((d/2)**2*np.pi)  #area calculation
A0 = (6.1*10**(-3)/2)**2*np.pi  # Reference area (m^2) [Example value]
p0 = 10000  # Reference pressure (Pa) [Example value]
rho = 1060  # Blood density (kg/m^3)
v_t = f_D * c/(2*f0*np.cos(theta)) #speed calculation

Q = v_t*A_t  #flow rate
Q_t_mm3 = Q * 1e9 #m^3 to mm^3

pulse = gausspulse(t, fc=5000, bw=50, retquad=False, retenv=False) #choose gausspulse for blood pressure
k, _ = np.polyfit(A_t * 1e6, Q_t_mm3, 1) #choose a linear polynom for PWV from Q-A function

#PWV calculation
PWV_t= (k+k*10**(-1)*(0.1*np.sin(2 * np.pi * t / T)+ 0.05 * np.sin(4 * np.pi * t / T)+0.1*np.cos(4 * np.pi * t / T)))/100
#choose parameters for the gauss pulse function
num_pulses = 2

blood_pressure = p0 + rho * (PWV_t ** 2) * np.log(A_t / A0) #blood pressure formula

pulse_interval = len(t) // num_pulses  # Interval between pulses
#blood pressure to be more realistic with gauss pulse
for i in range(num_pulses):
    t_shifted = t - i * (T / num_pulses)  # Shift time for each pulse
    pulse = gausspulse(t_shifted, fc=500, bw=1, retquad=False, retenv=False)
    blood_pressure += blood_pressure * 0.6 * pulse 



# Plotting Frequency vs Time (Transmitted and Echo Frequencies)
plt.figure(figsize=(14, 10))
plt.subplot(5, 1, 1)
plt.plot(t, vT, label='Transmitted Signal', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Transmitted Signal')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, v_Rt, label='Echo Signal 1 (Received Signal)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Echo Signal')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t, v_Rb, label='Echo Signal 2 (Received Signal)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Echo Signal')
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, v_D, label='Mixed Signal top', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t, v_D2, label='Mixed Signal bottom', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(t, vT, label='Transmitted Signal', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Transmitted Signal')
plt.grid(True)


plt.subplot(3, 1, 2)
plt.plot(t, v_Rd, label='Echo Signal', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Transmitted Signal')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, v_D3, label='Mixed Signal', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, f_t, label='Transmitted Frequency', color='red')
plt.plot(t, f_Rt, label='Echo Frequency top', color='green')
plt.plot(t, f_Rb, label='Echo Frequency bot', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (MHz)')
plt.title('Frequency vs Time')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, f_t, label='Transmitted Frequency', color='red')
plt.plot(t, f_Rd, label='Echo Frequency velocity', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (MHz)')
plt.title('Frequency vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(t, blood_pressure / 133.322-30, label='Blood Pressure')
plt.xlabel('Time (s)')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Estimated Blood Pressure vs Time')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(t, PWV_t, label='PWV')
plt.xlabel('Time (s)')
plt.ylabel('PWV (m/s)')
plt.title('Estimated PWV vs Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, d * 1000, label="Diameter (mm)", color='r')
plt.xlabel('Time (s)')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, A_t * 1e6, label="Cross-Sectional Area ($mm^2$)", color='g')
plt.xlabel('Time (s)')
plt.ylabel('Area ($mm^2$)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, Q * 1e9, label="Flow Rate ($mm^3$/s)", color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Flow Rate ($mm^3$/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 4))
plt.plot(A_t*1e6, Q_t_mm3, label='PWV')
plt.xlabel('A (mm^2)')
plt.ylabel('Q (mm^3/s)')
plt.title('Q vs A')
plt.grid(True)
plt.show()


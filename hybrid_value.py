import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass


B = 3.0e-9
n = 3.0e6
T = 4.3
v = 400
N = 2048 * 400
# charge exchange rate
gama = 3.4e-4
beta = n * T * 1.6e-19
pressureb = 1 / 2 / mu_0 * B **2
beta = beta / pressureb
wave_parameter = n * e / epsilon_0 / B
print(wave_parameter)

v_a = B / np.sqrt(mu_0*proton_mass*n)
E_n = (proton_mass * v_a ** 2) / 1.6e-19
E = 2*T / E_n
omega_i = e*B / proton_mass
omega_e = e*B/m_e
omega_pi = n * e ** 2 / epsilon_0/ proton_mass
omega_ei = n * e **2/ epsilon_0/m_e
print('omega_e=', omega_e, '  omega_i=', omega_i, ' omega_pe=', omega_ei, ' omega_pi=', omega_pi)
cycle = 2 * np.pi / omega_i
deltaT = 0.025 * cycle
rate = int(deltaT * gama * N)
print('cyclotron frequency is', omega_i)
print('cycle time is', cycle)
print('V_a is', v_a)
print('normalization beta is', E, 'eV')
print(E_n)
print('1 is represent', E_n, 'eV')
print(1 / (3.5e-7 * 3.5 * 0.025))
print('number of particle exchanged per step is', rate)
print('beta is ', beta)




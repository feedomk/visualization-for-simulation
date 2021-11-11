import numpy as np
import matplotlib.pyplot as plt
import math
# from sympy.solvers import solve
# from scipy.optimize import fsolve
#
# from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass
#
# B = 3.0e-9
# n = 3.0e6
# T = 4.3
# v = 400
# # N = 2048 * 400
# # # charge exchange rate
# # gama = 3.4e-4
# # beta = n * T * 1.6e-19
# # pressureb = 1 / 2 / mu_0 * B ** 2
# # beta = beta / pressureb
# omega_i = e * B / proton_mass
# omega_e = e * B / m_e
# omega_pi = n * e ** 2 / epsilon_0 / proton_mass
# omega_pe = n * e ** 2 / epsilon_0 / m_e
#
# func = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
#         1 - omega_pi / (omega * (omega + omega_i)) - omega_pe / (omega * (omega - omega_e)))) \
#                      * c / np.sqrt(omega_pi)
#
# func1 = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
#         1 - omega_pi / (omega * (omega - omega_i)) - omega_pe / (omega * (omega + omega_e)))) \
#                      * c / np.sqrt(omega_pi)
# omega = np.linspace(0.001, 0.4 * np.pi * omega_i, 10001)
# omega1 = omega / omega_i
#
# k = func(omega)
# # k = k[k<2]
# # omega = omega[0:len(k)]
# fig = plt.figure()
# plt.plot(k, omega1, 'r--')
# plt.plot(-k, omega1, 'r--')
# plt.plot(k, -omega1, 'r--')
# plt.plot(-k, -omega1, 'r--')
# k1 = func1(omega)
# k1 = k1[k1<2]
# omega1 = omega1[0:len(k1)]
# plt.plot(k1, omega1, 'r--')
# plt.plot(-k1, omega1, 'r--')
# plt.plot(k1, -omega1, 'r--')
# plt.plot(-k1, -omega1, 'r--')
#
# from scipy.fft import fft
#
# # Number of sample points
# N = 600
# # sample spacing
# T = 1 / 800
# x = np.linspace(0.0, N * T, N)
# y = np.sin(50 * 2. * np.pi * x) + 0.5 * np.sin(80 * 2 * np.pi * x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(xf, (2.0 / N * np.abs(yf[0:N // 2])) ** 2)
# plt.grid()
# plt.figure()
# plt.plot(x, y)
# plt.show()

a = np.arange(32)
a = np.reshape(a, (4,2,4),)
# a.shape=(2,4)
print(a)
# a.shape=(2,2,2)
# a = np.reshape(a,(4,4),order='C')
b = np.transpose(a, (2,1,0)).reshape(2,-1)
c = np.transpose(a, (0,1,2)).reshape(-1,4)
print(c)
# c = np.transpose(c, (1,0)).reshape(4,-1)
# print(a.shape)
# a.shape=(2,4)
c = np.concatenate((c[:4,:],c[4:,:]), axis=1)
print(c)

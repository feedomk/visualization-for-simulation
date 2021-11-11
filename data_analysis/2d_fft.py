import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft, fft2, fftfreq, fftshift
from scipy import signal

path = '/home/kun/Downloads/data/data/'
# path = '/home/ck/Documents/hybrid2D_PUI/data/'
by = 'by'
bz = 'bz'
By = data.read2d_timeseries(path, by)
Bz = data.read2d_timeseries(path, bz)
T = 0.25
dx = 1
By = np.array(By, dtype=float)
By = By[::-1, :]
Bz = np.array(Bz, dtype=float)
Bz = Bz[::-1, :]
leng = int(len(By[:, 0]))
By = By + 1j * Bz
By = By[550:, :]
# Bz = Bz[:250, :]
# By=By[:1000]
# By = By[:2000, :]
Z = fft2(By)
Z = Z / len(By[:, 0]) / len(By[0, :])
# tf = np.linspace(0., 1.0 / (2. * T), len(By[0, :]) // 2) * 2 * np.pi
# xf = np.linspace(0., 1. / (2. * dx), len(By[:, 0]) // 2) * 2 * np.pi
tf = fftfreq(len(By[:, 0]), T) * 2 * np.pi
xf = fftfreq(len(By[0, :]), dx) * 2 * np.pi
xf = fftshift(xf)
tf = fftshift(tf)
By1 = By[:, 0]
f = fft(By1)
omega = np.linspace(0, 2 * np.pi / 2 / T, len(By1) // 2)
# plt.figure()
# plt.plot(omega, (2. / len(By1) * np.abs(f[0:len(By1) // 2])) ** 2)
# plt.xlabel('$\omega\Omega^{-1}$')
# plt.ylabel(r'$nT^2 Hz^(-1)$')
# # plt.savefig('/media/ck/Samsung_T5/data/data/fft.png')

# t20 = By[100, :]
# t40 = By[500, :]
# t80 = By[800, :]
# k20 = fft(t20)
# k40 = fft(t40)
# k80 = fft(t80)
# k = np.linspace(0, 1. / 2 / dx, len(t40) // 2) * 2 * np.pi
# plt.figure()
# plt.plot(k, (2. / len(k) * np.abs(k20[0:len(k20) // 2])) ** 2, label='$t\Omega_p = 20$')
# plt.plot(k, (2. / len(k) * np.abs(k40[0:len(k40) // 2])) ** 2, label='$t\Omega_p = 200$')
# plt.plot(k, (2. / len(k) * np.abs(k80[0:len(k80) // 2])) ** 2, label='$t\Omega_p = 800$')
# plt.xscale("log")
# plt.yscale("log")
# # plt.ylim([10**-12, 10**0])
# # plt.xlim([10**-1, 10**1])
# plt.legend(ncol=1)
# plt.xlabel('$kd_i$')
# plt.ylabel('$\delta B^2/B_0^2(k)$')
# # plt.savefig('/media/ck/Samsung_T5/data/data/k_times.png')

from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass

B = 3.0e-9
n = 3.0e6
T = 4.3
v = 400
N = 2048 * 400
# charge exchange rate
gama = 3.4e-4
beta = n * T * 1.6e-19
pressureb = 1 / 2 / mu_0 * B ** 2
beta = beta / pressureb
print(beta)
omega_i = e * B / proton_mass
omega_e = e * B / m_e
omega_pi = n * e ** 2 / epsilon_0 / proton_mass
omega_pe = n * e ** 2 / epsilon_0 / m_e

func = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
        1 - omega_pi / (omega * (omega + omega_i)) - omega_pe / (omega * (omega - omega_e)))) \
                     * c / np.sqrt(omega_pi)

func1 = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
        1 - omega_pi / (omega * (omega - omega_i)) - omega_pe / (omega * (omega + omega_e)))) \
                      * c / np.sqrt(omega_pi)
omega = np.linspace(0.001, 1. * np.pi * omega_i, 10001)
omega1 = omega / omega_i

k = func(omega)
v = omega1 / k
delta_omega = ((v + 10) / (v)) * omega1

fig = plt.figure(figsize=(8, 6))
# plt.plot(k, delta_omega, 'w', label='whister wave')
plt.plot(-k, -((-v + 10) / (-v)) * omega1, 'w', label='shfited whister')
# plt.plot(k, -omega1-delta_omega, 'r')
# plt.plot(-k, -omega1-delta_omega, 'r')

k1 = func1(omega)
k1 = k1[k1 < 2]
omega1_shf = omega1[0:len(k1)]
v = omega1_shf / k1
delta_omega = ((v + 10) / v) * omega1_shf

# plt.plot(k1, delta_omega, 'p--', label='Alfven wave')
# plt.plot(-k1, ((-v + 10) / -v) * omega1_shf, 'p')
# plt.plot(k1, -omega1_shf-delta_omega, 'b')
# plt.plot(-k1, -omega1_shf-delta_omega, 'b')

# k = k[k<2]
# omega = omega[0:len(k)]
# fig = plt.figure(figsize=(8, 6))
# plt.plot(k, omega1, 'w--', label='whister wave')
# plt.plot(-k, omega1, 'w--')
plt.plot(k, -omega1, 'w--')
plt.plot(-k, -omega1, 'w--')
k1 = func1(omega)
k1 = k1[k1 < 2]
omega1 = omega1[0:len(k1)]
plt.plot(k1, omega1, 'r--')
plt.plot(-k1, omega1, 'r--')
# plt.plot(k1, -omega1, 'p--')
# plt.plot(-k1, -omega1, 'p--')
#
# xf, tf = np.meshgrid(xf, tf)
Z = fftshift(Z)
# xlen = len(xf)
# tlen = len(tf)
# xf = xf[xlen / 2 - xlen / 8:xlen / 2 + xlen / 8]
# tf = tf[tlen / 2: tlen / 2 + tlen / 32]
# Z = Z[xlen / 2 - xlen / 8:xlen / 2 + xlen / 8, tlen / 2: tlen / 2 + tlen / 32]
# plt.figure(figsize=(8, 6))
plt.pcolormesh(xf, tf, np.log10(np.abs(Z) ** 2), cmap='jet', vmin=-10)
plt.xlabel('$k_{||}d_i^{-1}$')
plt.ylabel('$\omega\Omega_i^{-1}$')
plt.xlim(-0.8, 0.8)
plt.ylim(-2., 2.)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1, 0.1)
plt.colorbar()
plt.legend()
# plt.savefig('/home/kun/Downloads/data/data/spectra.png')
plt.show()

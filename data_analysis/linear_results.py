import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import read_data
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

pth = '/media/kun/Samsung_T5/Mathematica/harmonic/0.01beta/50tp/output_beam' + '40.txt'
omegar1, omegai1 = read_data.read_output(pth)
print(121 * 181)
omegar1 = np.array(omegar1)
omegai1 = np.array(omegai1)

pth = '/media/kun/Samsung_T5/Mathematica/harmonic/0.01beta/5tp/output_beam' + '50.txt'
omegar2, omegai2 = read_data.read_output(pth)
print(121 * 181)
omegar2 = np.array(omegar2)
omegai2 = np.array(omegai2)

pth = '/media/kun/Samsung_T5/Mathematica/harmonic/0.01beta/5tp/output_beam' + '60.txt'
omegar3, omegai3 = read_data.read_output(pth)
print(121 * 181)
omegar3 = np.array(omegar3)
omegai3 = np.array(omegai3)

pth = '/media/kun/Samsung_T5/Mathematica/harmonic/0.01beta/50tp/output_beam' + '70.txt'
omegar4, omegai4 = read_data.read_output(pth)
print(121 * 181)
omegar4 = np.array(omegar4)
omegai4 = np.array(omegai4)

# two dimentional

# omegar = omegar.reshape(25, 240)
# omegai = omegai.reshape(25, 240)
# omegai[omegai < 0.00001] = None
# # omegai = omegai[::-1, :]
#
# pth = '/media/kun/Samsung_T5/Mathematica/output_beam' + '22.txt'
# f = open(pth)
# omegai1 = []
# omegar1 = []
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     omegar1.append(float(line[0]))
#     omegai1.append(float(line[1]))
#
# f.close()
# omegai1 = np.array(omegai1).reshape(25, -1)
# # omegai1 = omegai1[::-1, :]
#
# omegai = np.concatenate((omegai, omegai1), axis=0)
# # omegai = omegai[:, ::-1]
# omegai[omegai < 0.00001] = None
#
# # file = pth + 'output4' + '.txt'
# #
# # f = open(file)
# # omegai2 = []
# # omegar2 = []
# # for line in f.readlines():
# #     line = line.split()
# #     if not line:
# #         break
# #     omegar2.append(float(line[0]))
# #     omegai2.append(float(line[1]))
# #
# # f.close()
# #
# # omegai[8, :] = omegai2
# # omegar[8, :] = omegar2
# #
# # file = pth + 'output5' + '.txt'
# #
# # f = open(file)
# # omegai2 = []
# # omegar2 = []
# # for line in f.readlines():
# #     line = line.split()
# #     if not line:
# #         break
# #     omegar2.append(float(line[0]))
# #     omegai2.append(float(line[1]))
# #
# # f.close()
# #
# # omegai[18, :] = omegai2
# # omegar[18, :] = omegar2
#
# # fig = plt.figure(figsize=(16, 8))
# # ax = fig.add_subplot(121, projection='3d')
# # ax.view_init(45, 60)
# # x = np.arange(0.02, 0.200001, 0.001)
# # y = np.arange(0, 0.60001, 0.005)
# # X, Y = np.meshgrid(x, y)
# # # plt.contourf(X, Y, omegar, 10, cmap='jet')
# # # plt.pcolormesh(X, Y, omegar, shading='gouraud', cmap='jet')
# #
# # # Normalize the colors based on omegai
# # scamap = plt.cm.ScalarMappable(cmap='Reds')
# # fcolors = scamap.to_rgba(omegai)
# # fig = ax.scatter(X, Y, omegar, c=omegai, cmap='jet', vmin=-0.01)
# # # fig = ax.plot_surface(X,Y, omegar, facecolors=fcolors)
# # ax.set_xlabel(r'$k_\parallel k_i$')
# # ax.set_ylabel(r'$k_\perp k_i$')
# # ax.set_zlabel(r'$\omega / \Omega_i$')
# # ax.set_xticks([0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21])
# # # cbar = plt.colorbar(scamap)
# # # plt.xlabel(r'k_\parallel')
# # # plt.ylabel(r'k_perp')
# # # cbar = plt.colorbar()
# # # cbar.set_label(r'$\gamma / \Omega_i$')
# #
# #
# # plt.subplot(122)
# # c = plt.contour(X, Y, omegai, np.linspace(0.01, 0.04, 5), cmap='jet', vmin=0)
# # # get data from contour lines
# # c = c.collections[4].get_paths()[0]
# # v = c.vertices
# # x = np.array(v[:, 0] / 0.001, dtype=int)
# # y = np.array(v[:, 1] / 0.005, dtype=int)
# # n = len(x)
# # z = []
# # for i in range(n):
# #     zz = omegar[y[i], x[i]]
# #     z.append(zz)
# # z = np.array(z)
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot3D(v[:, 0], v[:, 1], z, 'k', linewidth=2)
# # # plt.pcolormesh(X, Y, omegai, shading='gouraud', cmap='jet', vmin=0.01)
# # plt.xlabel(r'$k_\parallel$')
# # plt.ylabel(r'$k_\perp$')
# # # cbar = plt.colorbar()
# # # cbar.set_label(r'$\gamma / \Omega_i$')
# # plt.show()
#
# x = np.linspace(0.02, 0.5, 240)
# y = np.linspace(0., 0.49, 50)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# # plt.pcolormesh(X, Y, omegai, shading='gouraud', cmap='jet', vmin=0.01)
# contour = plt.contourf(X, Y, omegai, 10, cmap='Reds', vmin=0.0, vmax=0.12)
# plt.xlabel(r'$k_\parallel / k_i$')
# plt.ylabel(r'$k_\perp / k_i$')
# cbar = fig.colorbar(contour)
# cbar.set_label(r'$\gamma / \Omega_i$')
# plt.annotate('$n=1$', xy=(0.14, 0.2), xytext=(0.2, 0.05), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=15)
# plt.annotate(r'$n=2$', xy=(0.25, 0.35), xytext=(0.31, 0.2), arrowprops=dict(facecolor='black', shrink = 0.05), fontsize=15)
# # cbar.set_clim(0.015, 0.12)
# plt.show()


# one dimentional

# omegar1 = omegar1[::-1]
# omegai1 = omegai1[::-1]
# omegai4 = omegai4[::-1]
# omegar4 = omegar4[::-1]
k = np.arange(0.02, 0.6, 0.002)

fig, ax = plt.subplots(figsize=(8, 6))
plt.title('ion cyclotron instability')

ax2 = ax.twinx()
ax.plot(k, omegar1, 'r', label=r'$\theta=40\degree$')
ax2.plot(k, omegai1, 'r:', )
ax.plot(k, omegar4, 'b', label=r'$\theta = 70 \degree$')
ax2.plot(k, omegai4, 'b:',)

# giving labels to the axises
ax.set_xlabel(r"$k_\parallel k_i^{-1}$", )
ax.set_ylabel(r'$\omega \Omega^{-1}$', color='k')
# ax.set_ylim([0, 1])
# ax.tick_params(axis='y', colors='r')
# ax.spines['left'].set_color('red')
# ax.spines['right'].set_color('blue')
# secondary y-axis label
ax2.set_ylabel(r'$\gamma \Omega^{-1}$', color='k')
# ax2.tick_params(axis='y', colors='b')      #settubg up y-axis tick color to blue
# ax2.spines['right'].set_color('blue')    #settubg up y-axis tick color to blue
# ax2.spines['left'].set_color('red')

# defining display layout
plt.tight_layout()
ax.legend()
plt.show()

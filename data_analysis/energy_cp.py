import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

path = '/home/kun/Downloads/data/maxwell/'
(t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
 back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
 back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
 ex, ey, ez, e_total, bx, by, bz, b_total0, total) = data.read_energy(path)
b_total0 = np.array(b_total0)
b_total0 = b_total0 - 1

fig, ax = plt.subplots(figsize=(12, 4))
axins = zoomed_inset_axes(ax, zoom=9, loc=4)
marker = ('r', 'y', 'g', 'c', 'b', 'm')
marker = ('lightblue', 'cadetblue', 'darkturquoise', 'c', 'darkcyan', 'darkslategray',)
for i in [1, 2, 3, 4, 5, 6]:
    # path = '/home/ck/Documents/hybrid2D_PUI/data/'
    path = '/home/kun/Downloads/data/0.002/' + 'data' + str(i) + '/'

    (t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
     back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
     back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
     ex, ey, ez, e_total, bx, by, bz, b_total, total) = data.read_energy(path)

    T = 0.025
    t = np.array(t) * T
    b_total = np.array(b_total)
    b_total = b_total - 1 - b_total0
    ax.plot(t, b_total, label='$v_b=$' + str(4 + 2 * i) + '$v_A$', color=marker[i - 1])
    ax.plot(t, b_total + b_total0, '--', label='$v_b=$' + str(4 + 2 * i) + '$v_A$', color=marker[i - 1])
    axins.plot(t, b_total, color=marker[i - 1])
    x1, x2, y1, y2 = 52, 64, 6 * 10 ** -6, 3.2e-5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.setp(axins.get_xticklabels(), visible=False)
    # plt.setp(axins.get_yticklabels(), visible=False)
    axins.set_yticks([])
    axins.set_xticks([])

mark_inset(ax, axins, loc1=3, loc2=1, ec='0.5')
ax.set_yscale("log")
axins.set_yscale('log')
ax.set_ylim(10 ** -12, 10 ** 0)
ax.plot(t, b_total0, label='maxwellian', color='k')
ax.set_ylabel(r'$(\delta B/B_0)^2$')
ax.set_xlabel(r"$\Omega_i t$")
ax.legend(ncol=2)
plt.show()

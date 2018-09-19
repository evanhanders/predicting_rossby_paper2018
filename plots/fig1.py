import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,0), 950, 250), ((50, 370), 900, 250), ((50, 750), 900, 250))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))


#READ DATA, PLOT 1
onset_data = np.genfromtxt('../data/eps1e-4_onsets.csv', skip_header=1, delimiter=',', usecols=(0,3))
ax1.plot(onset_data[:,0], onset_data[:,1], c='k', lw=3)

original_co = np.genfromtxt('../data/coprime_data_original_co_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,6)) #ra, ro, ta -- need to double check if this is backwards ra/ta.
co1 = original_co[:,0]/original_co[:,2] == 1
x, y = original_co[co1, 2], original_co[co1, 0]
ax1.plot(x, y, c='r', ls='--', lw=2, label='Co = 1')

sigma = np.genfromtxt('../data/coprime_data_sigma_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,5,6)) #ra, ro, sigma, ta
sig1 = sigma[:,2] == 2

ax1.plot(sigma[sig1, 3], sigma[sig1, 0], c='orange', dashes=(4,1,2,1), lw=2, label=r'$\mathcal{S}=3$')

new_c = np.genfromtxt('../data/ra_ta0.75_AR4_alldata.csv', skip_header=1, delimiter=',', usecols=(0,1,2,5)) #sigma, ra, ta, ro
c957 = new_c[:,0] == 0.957
c158 = new_c[:,0] == 1.58
#ax1.plot(new_c[c957, 2], new_c[c957, 1])
ax1.plot(new_c[c158, 2], new_c[c158, 1], c='blue', lw=2, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')


ax1.set_xlim(1e1, 1e9)
ax1.set_ylim(1e1, 1e6)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Ta')
ax1.set_ylabel('Ra')
ax1.legend(loc='lower right', frameon=False)


#PLOT 2

ax2.plot(new_c[c158, 2], new_c[c158, 3], marker='*', color='blue', lw=0, ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')
ax2.plot(new_c[c957, 2], new_c[c957, 3], marker='o', color='blue', lw=0, ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
ax2.plot(sigma[sig1, 3], sigma[sig1, 1], marker='o', color='orange', lw=0, ms=4)
ax2.plot(original_co[co1, 0], original_co[co1, 1], marker='o', color='red', lw=0, ms=4)
ax2.legend(loc='upper left', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0)

lines, labels = [], []
co1_ra = original_co[co1, 0][original_co[co1, 0] >= 1e3]
co1_ro = original_co[co1, 1][original_co[co1, 0] >= 1e3]
p = np.polyfit(np.log10(co1_ra), np.log10(co1_ro), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
str2 = r'{:s}'.format(str2)
lines += ax2.plot(co1_ra, 10**(p[1])*co1_ra**(p[0]), label=r'{:s}'.format(str2), c='k')
labels += [str2]

p = np.polyfit(np.log10(sigma[sig1, 3]), np.log10(sigma[sig1,1]), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
str2 = r'{:s}'.format(str2)
lines += ax2.plot(sigma[sig1,3], 10**(p[1])*sigma[sig1,3]**(p[0]), label=r'{:s}'.format(str2), c='k', ls='--')
labels += [str2]


from matplotlib.legend import Legend
leg = Legend(ax2, lines, labels, frameon=False, fontsize=8, loc='lower left', borderpad=0.1, handletextpad=0.1, handlelength=1.5)
ax2.add_artist(leg)


ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Ro')



# PLOT 3

c_sig = np.sqrt(sigma[sig1, 0]/sigma[sig1, 3]**(3/4))
ax3.plot(c_sig, sigma[sig1, 1], marker='o', lw=0, c='orange')
p = np.polyfit(np.log10(c_sig), np.log10(sigma[sig1, 1]), deg=1)
label_str = 'Ro' + r'$ \propto $'
str2 = '$\mathcal{P}__{\mathrm{Ro}}^{' + '{:.2f}'.format(p[0]) + '}$'
#ax3.plot(c_sig, 10**(p[1])*c_sig**(p[0]), label=label_str+r'{:s}'.format(str2))

ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlabel(r'$\mathcal{P}_{\mathrm{Ro}}$')
ax3.set_ylabel('Ro')
ax3.legend(loc='upper right', frameon=False)


fig.savefig('../tex/figs/parameter_space.png', dpi=300, bbox_inches='tight')

import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



fig = plt.figure(figsize=(3, 5))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,50), 450, 900), ((500, 50), 450, 900))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))


#READ DATA, PLOT 1
onset_data = np.genfromtxt('../data/eps1e-4_onsets.csv', skip_header=1, delimiter=',', usecols=(0,3))

original_co = np.genfromtxt('../data/coprime_data_original_co_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,6)) #ra, ro, ta
co1 = original_co[:,0]/original_co[:,2] == 1
x, y = original_co[co1, 0], original_co[co1, 2]

sigma = np.genfromtxt('../data/coprime_data_sigma_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,5,6)) #ra, ro, sigma, ta
sig1 = sigma[:,2] == 2

new_c = np.genfromtxt('../data/ra_ta0.75_AR4_alldata.csv', skip_header=1, delimiter=',') 
#co, ra, ta, nu, re, ro
c957 = new_c[:,0] == 0.957
c158 = new_c[:,0] == 1.58
lines, labels = [], []
lines += ax1.plot(new_c[c957, 1], new_c[c957, 3], c='orange', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
lines += ax1.plot(new_c[c158, 1], new_c[c158, 3], c='blue', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')

labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$']
labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$']

for xlabel_i in ax1.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)


p = np.polyfit(np.log10(new_c[c158,1]), np.log10(new_c[c158, 3]), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(new_c[c158, 1], 10**(p[1])*new_c[c158, 1]**(p[0]), label=r'{:s}'.format(str2), color='blue', alpha=0.4)
p = np.polyfit(np.log10(new_c[c957,1]), np.log10(new_c[c957, 3]), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(new_c[c957, 1], 10**(p[1])*new_c[c957, 1]**(p[0]), label=r'{:s}'.format(str2), color='orange', alpha=0.4)

ax1.legend(lines[:2], labels[:2], loc='upper left', frameon=False)
from matplotlib.legend import Legend
leg = Legend(ax1, lines[2:], labels[2:],
             loc='lower right', frameon=False)
ax1.add_artist(leg)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('Nu')


lines, labels = [], []
#PLOT 2
lines += ax2.plot(new_c[c957, 1], new_c[c957, 4], c='orange', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
lines += ax2.plot(new_c[c158, 1], new_c[c158, 4], c='blue', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')
ax2.legend(loc='upper left', frameon=False)

labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$']
labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$']

p = np.polyfit(np.log10(new_c[c158,1]), np.log10(new_c[c158, 4]), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax2.plot(new_c[c158, 1], 10**(p[1])*new_c[c158, 1]**(p[0]), label=r'{:s}'.format(str2), color='blue', alpha=0.4)
p = np.polyfit(np.log10(new_c[c957,1]), np.log10(new_c[c957, 4]), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax2.plot(new_c[c957, 1], 10**(p[1])*new_c[c957, 1]**(p[0]), label=r'{:s}'.format(str2), color='orange', alpha=0.4)

ax2.legend(lines[:2], labels[:2], loc='upper left', frameon=False)
from matplotlib.legend import Legend
leg = Legend(ax2, lines[2:], labels[2:],
             loc='lower right', frameon=False)
ax2.add_artist(leg)



ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Re')


#PLOT 2

fig.savefig('../tex/figs/nu_and_re.png', dpi=300, bbox_inches='tight')

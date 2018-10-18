import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams['font.family'] = 'serif'



fig = plt.figure(figsize=(3, 3.5))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,50), 450, 900), ((500, 50), 450, 900))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))


#READ DATA, PLOT 1
new_c = np.genfromtxt('../data/constant_pro.csv', skip_header=1, delimiter=',') 
#co, ra, ta, nu, re, ro
c957 = np.round(np.sqrt(new_c[:,0]),3) == 0.957
c158 = np.round(np.sqrt(new_c[:,0]),3) == 1.58
start_pt = 4
ra957 = new_c[c957,2][start_pt:]
nu957 = new_c[c957,1][start_pt:]
ra957_full = new_c[c957,2]
nu957_full = new_c[c957,1]
ra158 = new_c[c158,2][start_pt:]
nu158 = new_c[c158,1][start_pt:]
ra158_full = new_c[c158,2]
nu158_full = new_c[c158,1]




lines, labels = [], []
lines += ax1.plot(ra957_full, nu957_full, c='orange', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
lines += ax1.plot(ra158_full, nu158_full, c='blue', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')

#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$']
#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$']
labels += [r'Ro$\sim$0.1']
labels += [r'Ro$\sim$0.4']

for xlabel_i in ax1.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)


p = np.polyfit(np.log10(ra158), np.log10(nu158), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(ra158_full, 10**(p[1])*ra158_full**(p[0]), label=r'{:s}'.format(str2), color='blue', alpha=0.4)
p = np.polyfit(np.log10(ra957), np.log10(nu957), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(ra957_full, 10**(p[1])*ra957_full**(p[0]), label=r'{:s}'.format(str2), color='orange', alpha=0.4)

ax1.legend(lines[:2], labels[:2], loc='upper left', frameon=False, fontsize=8, handletextpad=0)
from matplotlib.legend import Legend
leg = Legend(ax1, lines[2:], labels[2:],
             loc='lower right', frameon=False, fontsize=8)
ax1.add_artist(leg)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('Nu')
ax1.text(2.5e1, 2.5e1, "(a)", ha="center", va="center", size=8)


lines, labels = [], []
#PLOT 2
start_pt = 4
ra957 = new_c[c957,2][start_pt:]
re957 = new_c[c957,3][start_pt:]
ra957_full = new_c[c957,2]
re957_full = new_c[c957,3]
ra158 = new_c[c158,2][start_pt:]
re158 = new_c[c158,3][start_pt:]
ra158_full = new_c[c158,2]
re158_full = new_c[c158,3]

lines += ax2.plot(new_c[c957, 2], new_c[c957, 3], c='orange', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
lines += ax2.plot(new_c[c158, 2], new_c[c158, 3], c='blue', lw=0, marker='o', ms=4, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')

#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$']
#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$']
labels += [r'Ro$\sim$0.1']
labels += [r'Ro$\sim$0.4']

p = np.polyfit(np.log10(ra158), np.log10(re158), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax2.plot(ra158_full, 10**(p[1])*ra158_full**(p[0]), label=r'{:s}'.format(str2), color='blue', alpha=0.4)
p = np.polyfit(np.log10(ra957), np.log10(re957), deg=1)
str2 = 'Ra$^{' + '{:.2f}'.format(p[0]) + '}$'
labels += [r'{:s}'.format(str2)]
lines += ax2.plot(ra957_full, 10**(p[1])*ra957_full**(p[0]), label=r'{:s}'.format(str2), color='orange', alpha=0.4)

#ax2.legend(lines[:2], labels[:2], loc='upper left', frameon=False, fontsize=8, handletextpad=0)
from matplotlib.legend import Legend
leg = Legend(ax2, lines[2:], labels[2:],
             loc='lower right', frameon=False, fontsize=8)
ax2.add_artist(leg)



ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Re')
ax2.text(2.5e1, 6e2, "(b)", ha="center", va="center", size=8)


#PLOT 2

fig.savefig('../tex/figs/nu_and_re.png', dpi=300, bbox_inches='tight')

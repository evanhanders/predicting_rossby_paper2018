import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams['font.family'] = 'serif'


Pro_color= [1.        , 0.49803922, 0.05490196]
Co_color = [0.85882353, 0.85882353, 0.55294118]
S_color  = [0.68235294, 0.78039216, 0.90980392]

Pro_marker = 'o'
Co_marker  = 'v'
S_marker   = 's'



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,0), 950, 250), ((50, 370), 950, 250), ((50, 750), 950, 250))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))


#READ DATA, PLOT 1
onset_data = np.genfromtxt('../data/eps1e-4_onsets.csv', skip_header=1, delimiter=',', usecols=(0,3))
ax1.plot(onset_data[:,0], onset_data[:,1], c='k', lw=3)

original_co = np.genfromtxt('../data/coprime_data_original_co_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,6)) #ra, ro, ta -- need to double check if this is backwards ra/ta.
print( original_co[:,0]/original_co[:,2])
co1 = original_co[:,0]/original_co[:,2] == 1
co03 = np.round(np.sqrt(original_co[:,0]/original_co[:,2]), decimals=1) == 0.3
co01 = np.round(np.sqrt(original_co[:,0]/original_co[:,2]), decimals=1) == 0.1
x, y = original_co[co1, 2], original_co[co1, 0]
x, y = zip(*sorted(zip(x, y)))
ax1.plot(x, y, c=Co_color, lw=2, label='Co = 1', dashes=(3,1))

#sigma = np.genfromtxt('../data/coprime_data_sigma_runs.csv', skip_header=1, delimiter=',', usecols=(2,4,5,6)) #ra, ro, sigma, ta
#print(sigma[:,2])
#sig2 = sigma[:,2] == 2
#sig1_1 = sigma[:,2] == 1
sigma = np.genfromtxt('../data/constant_supercriticality.csv', skip_header=1, delimiter=',', usecols=(2,4,5,6)) #ra, ro, supercrit, ta
sig2 = sigma[:,2] == 3
sig1_1 = sigma[:,2] == 2


ax1.plot(sigma[sig2, 3], sigma[sig2, 0], c=S_color, dashes=(3,0.5,0.5,0.5), lw=2, label=r'$\mathcal{S}=3$')

new_c = np.genfromtxt('../data/ra_ta0.75_AR4_alldata.csv', skip_header=1, delimiter=',', usecols=(0,1,2,5)) #sigma, ra, ta, ro
c957 = new_c[:,0] == 0.957
c158 = new_c[:,0] == 1.58
#ax1.plot(new_c[c957, 2], new_c[c957, 1])
ax1.plot(new_c[c158, 2], new_c[c158, 1], c=Pro_color, lw=3, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')


ax1.set_xlim(1e1, 1e9)
ax1.set_ylim(1e1, 1e6)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='lower right', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0.4)

ax1.set_xlabel('Ta')
ax1.set_ylabel('Ra')


#PLOT 2

ax2.plot(original_co[co1, 0], original_co[co1, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*Co_color, 1))
ax2.plot(original_co[co03, 0], original_co[co03, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=3.5, markeredgecolor=(*Co_color, 1))
ax2.plot(original_co[co01, 0], original_co[co01, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*Co_color, 1))
ax2.plot(sigma[sig2, 0], sigma[sig2, 1], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*S_color, 1))
ax2.plot(sigma[sig1_1, 0], sigma[sig1_1, 1], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*S_color, 1))
ax2.plot(new_c[c158, 1], new_c[c158, 3], marker=Pro_marker, color=Pro_color, lw=0, ms=4.5, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$')
ax2.plot(new_c[c957, 1], new_c[c957, 3], marker=Pro_marker, color=Pro_color, lw=0, ms=2.5, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$')
ax2.legend(loc='upper middle', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0)

lines, labels = [], []
co1_ra = original_co[co1, 0][original_co[co1, 0] >= 1e3]
co1_ro = original_co[co1, 1][original_co[co1, 0] >= 1e3]

from matplotlib.legend import Legend
leg = Legend(ax2, lines, labels, frameon=False, fontsize=8, loc='lower left', borderpad=0.1, handletextpad=0.1, handlelength=1.5)
ax2.add_artist(leg)


ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Ro')



# PLOT 3
print(sigma[sig2,3], sigma[sig2,0]/sigma[sig2,3]**(3/4))
c_sig = np.sqrt(sigma[sig2, 0]/sigma[sig2, 3]**(3/4))
c_sig_1 = np.sqrt(sigma[sig1_1, 0]/sigma[sig1_1, 3]**(3/4))
ax3.plot(c_sig, sigma[sig2, 1], marker='o', lw=0, c=(*S_color, 0.3), ms=4.5, label='$\mathcal{S} = 3$', markeredgecolor=(*S_color, 1))
ax3.plot(c_sig_1, sigma[sig1_1, 1], marker='o', lw=0, c=(*S_color, 0.3), ms=2.5, label='$\mathcal{S} = 2$', markeredgecolor=(*S_color, 1))

#p = np.polyfit(np.log10(c_sig[sigma[sig2,1] <= 0.3]), np.log10(sigma[sig2, 1][sigma[sig2,1] < 0.3]), deg=1)
#p2 = np.polyfit(np.log10(c_sig_1[sigma[sig1_1,1] <= 0.3]), np.log10(sigma[sig1_1, 1][sigma[sig1_1,1] < 0.3]), deg=1)
#label_str = 'Ro' + r'$ \propto $'
#str2 = '$\mathcal{P}__{\mathrm{Ro}}^{' + '{:.2f}'.format(p[0]) + '}$'
#print('{:.4g}p^({:.4g})'.format(10**p[1], p[0]))
#print('{:.4g}p^({:.4g})'.format(10**p2[1], p2[0]))
#ax3.plot(c_sig, 10**(p[1])*c_sig**(p[0]))#, label=label_str+r'{:s}'.format(str2))
#ax3.plot(c_sig_1, 10**(p2[1])*c_sig_1**(p2[0]))#, label=label_str+r'{:s}'.format(str2))


ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlabel(r'$\mathcal{P}_{\mathrm{Ro}}$')
ax3.set_ylabel('Ro')
ax3.legend(loc='lower right', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0)

ax3.set_xticks((1,3,10))
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax3.tick_params(axis='x', which='minor', labelsize=0)



fig.savefig('../tex/figs/parameter_space.png', dpi=300, bbox_inches='tight')

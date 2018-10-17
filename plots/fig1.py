import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, interp2d
from scipy.optimize import curve_fit, leastsq
matplotlib.rcParams['font.family'] = 'serif'


Pro_color= [1.        , 0.49803922, 0.05490196]
Co_color = [0.85882353, 0.85882353, 0.55294118]
S_color  = [0.68235294, 0.78039216, 0.90980392]

Pro_marker = 'o'
Co_marker  = 'v'
S_marker   = 's'



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,0), 950, 250), ((50, 340), 950, 250), ((50, 680), 950, 250), ((50, 950), 950, 50))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))
cax = plt.subplot(gs.new_subplotspec(*gs_info[3]))


#READ DATA, PLOT 1
onset_data = np.genfromtxt('../data/eps1e-4_onsets.csv', skip_header=1, delimiter=',', usecols=(0,3))

ax1.plot(onset_data[:,0], onset_data[:,1], c='k', lw=0.5)
ax1.fill_between(onset_data[:,0], 0, onset_data[:,1], color='k', alpha=0.1)

original_co = np.genfromtxt('../data/constant_co.csv', skip_header=1, delimiter=',', usecols=(2,4,6)) #ra, ro, ta -- need to double check if this is backwards ra/ta.
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
ax1.plot(new_c[c158, 2], new_c[c158, 1], c=Pro_color, lw=3, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$')


ax1.set_xlim(1e1, 1e7)
ax1.set_ylim(1e1, 1e6)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0.4)

ax1.set_xlabel('Ta')
ax1.set_ylabel('Ra')
ax1.text(4e6, 1.5e1, "(a)", ha="center", va="center", size=8)
ax1.text(3e5, 2e2, "stable", ha="center", va="center", rotation=0, size=10, alpha=0.7)
ax1.set_xticks((1e1, 1e3, 1e5, 1e7))


#PLOT 2

ax2.plot(original_co[co1, 0], original_co[co1, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*Co_color, 1))
ax2.plot(original_co[co03, 0], original_co[co03, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=3.5, markeredgecolor=(*Co_color, 1))
ax2.plot(original_co[co01, 0], original_co[co01, 1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*Co_color, 1))
ax2.plot(sigma[sig2, 0], sigma[sig2, 1], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*S_color, 1))
ax2.plot(sigma[sig1_1, 0], sigma[sig1_1, 1], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*S_color, 1))
ax2.plot(new_c[c158, 1], new_c[c158, 3], marker=Pro_marker, color=Pro_color, lw=0, ms=4.5, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$')
ax2.plot(new_c[c957, 1], new_c[c957, 3], marker=Pro_marker, color=Pro_color, lw=0, ms=2.5, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$')
#ax2.legend(loc='upper middle', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0)

lines, labels = [], []
co1_ra = original_co[co1, 0][original_co[co1, 0] >= 1e3]
co1_ro = original_co[co1, 1][original_co[co1, 0] >= 1e3]

#from matplotlib.legend import Legend
#leg = Legend(ax2, lines, labels, frameon=False, fontsize=8, loc='lower left', borderpad=0.1, handletextpad=0.1, handlelength=1.5)
#ax2.add_artist(leg)


ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Ro', labelpad=0)
ax2.set_xticks((1e1, 1e3, 1e5, 1e7))
ax2.text(3e1, 5e-2, "(b)", ha="center", va="center", size=8)
ax2.text(3e6, 6.5e-2, "2", ha="center", va="center", size=8, color=S_color)
ax2.text(1.35e7, 1e-1, "3", ha="center", va="center", size=8, color=S_color)
ax2.text(5e2, 5.5e-2, "0.96", ha="center", va="center", size=8, color=Pro_color)
ax2.text(7e1, 2.2e-1, "1.58", ha="center", va="center", size=8, color=Pro_color)
ax2.text(9e6, 3.4e0, "1.0", ha="center", va="center", size=8, color=Co_color)
ax2.text(4e6, 8e-1, "0.3", ha="center", va="center", size=8, color=Co_color)
ax2.text(8e6, 2.6e-1, "0.1", ha="center", va="center", size=8, color=Co_color)



# PLOT 3
#print(sigma[sig2,3], sigma[sig2,0]/sigma[sig2,3]**(3/4))
#c_sig = np.sqrt(sigma[sig2, 0]/sigma[sig2, 3]**(3/4))
#c_sig_1 = np.sqrt(sigma[sig1_1, 0]/sigma[sig1_1, 3]**(3/4))
#ax3.plot(c_sig, sigma[sig2, 1], marker='o', lw=0, c=(*S_color, 0.3), ms=4.5, label='$\mathcal{S} = 3$', markeredgecolor=(*S_color, 1))
#ax3.plot(c_sig_1, sigma[sig1_1, 1], marker='o', lw=0, c=(*S_color, 0.3), ms=2.5, label='$\mathcal{S} = 2$', markeredgecolor=(*S_color, 1))
#
#cutoff = 0.3
#pros = np.concatenate((c_sig[sigma[sig2,1] < cutoff], c_sig_1[sigma[sig1_1,1] < cutoff]))
#ros  = np.concatenate((sigma[sig2, 1][sigma[sig2,1] < cutoff], sigma[sig1_1, 1][sigma[sig1_1,1] < cutoff]))
#print(pros.shape, ros.shape)
#p  = np.polyfit(np.log10(pros), np.log10(ros), deg=1)
#
#label_str = 'Ro' + r'$ \propto $'
#str2 = '$\mathrm{Ro}_{\mathrm{p}}^{' + '{:.2f}'.format(p[0]) + '}$'
#print('{:.4g}p^({:.4g})'.format(10**p[1], p[0]))
#line = ax3.plot(pros, 10**(p[1])*pros**(p[0]), color=(*S_color, 1))
#label = [r'{:s}'.format(str2),]
#
#pros = np.concatenate((c_sig[sigma[sig2,1] >= cutoff], c_sig_1[sigma[sig1_1,1] >= cutoff]))
#ros  = np.concatenate((sigma[sig2, 1][sigma[sig2,1] >= cutoff], sigma[sig1_1, 1][sigma[sig1_1,1] >= cutoff]))
#pros, ros = zip(*sorted(zip(pros, ros)))
#p2  = np.polyfit(np.log10(pros), np.log10(ros), deg=1)
#
#label_str = 'Ro' + r'$ \propto $'
#str2 = '$\mathrm{Ro}_{\mathrm{p}}^{' + '{:.2f}'.format(p2[0]) + '}$'
#print('{:.4g}p^({:.4g})'.format(10**p2[1], p2[0]))
#line += ax3.plot(pros, 10**(p2[1])*pros**(p2[0]), color=(*S_color, 1), dashes=(3,1,0.5,1))
#label += [r'{:s}'.format(str2)]
#
#
#
#ax3.set_yscale('log')
#ax3.set_xscale('log')
#ax3.set_xlabel(r'$\mathrm{Ro}_{\mathrm{p}}$')
#ax3.set_ylabel('Ro')
#ax3.set_ylim(4e-2, 6e0)
#ax3.legend(loc='lower right', fontsize=8, frameon=False, borderpad=0.1, handletextpad=0)
#
#from matplotlib.legend import Legend
#leg = Legend(ax3, line, label, frameon=False, fontsize=8, loc='upper center', borderpad=0.1, handletextpad=0.1, handlelength=1.5)
#ax3.add_artist(leg)
#
#
#
#ax3.set_xticks((1,3,10))
#ax3.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
#ax3.tick_params(axis='x', which='minor', labelsize=0)
#ax3.text(7.5e-1, 4.5e0, "(c)", ha="center", va="center", size=8)

#READ DATA
const_s = np.genfromtxt('../data/constant_supercriticality.csv', skip_header=1, delimiter=',') 
ta_s = np.log10(const_s[:,6])
ro_c_s = np.log10(const_s[:,2] / const_s[:,6]) #ra/ta
ro_m_s = np.log10(const_s[:,4])
ro_p_s = np.log10(const_s[:, 0])

const_co = np.genfromtxt('../data/constant_co.csv', skip_header=1, delimiter=',') 
ta_co = np.log10(const_co[:,6])
ro_c_co = np.log10(const_co[:,2] / const_co[:,6]) #ra/ta
ro_m_co = np.log10(const_co[:,4])
ro_p_co = np.log10(const_co[:, 0])

const_rop = np.genfromtxt('../data/ra_ta0.75_AR4_alldata.csv', skip_header=1, delimiter=',')
ta_rop = np.log10(const_co[:,2])
ro_c_rop = np.log10(const_co[:,1] / const_co[:,2]) #ra/ta
ro_m_rop = np.log10(const_co[:,4])
ro_p_rop = np.log10(const_co[:,0])

ta_full = np.concatenate((ta_s, ta_co, ta_rop))
ro_c_full = np.concatenate((ro_c_s, ro_c_co, ro_c_rop))
ro_m_full = np.concatenate((ro_m_s, ro_m_co, ro_m_rop))
ro_p_full = np.concatenate((ro_p_s, ro_p_co, ro_p_rop))


def ro_m_calc(params, ro_c, ro_p):
    a, b, c = params
    func = a + b*ro_c + c*ro_p
    return func



#LEAST SQUARES FIT
def run_analysis(ro_c, ro_m, ro_p):
    def ro_m_func(params, xdata, ydata):
        a, b, c = params
        func = a + b*xdata[0,:] + c*xdata[1,:]
        return np.abs(10**(ydata) - 10**(func))

    ro_cp = np.array((ro_c, ro_p))
    x0 = (0.1,0,1)
    fit, cov, infodict, mesg, ier = leastsq(ro_m_func, x0, args=(ro_cp, ro_m), full_output=True)
    print(fit, cov)
    fit_str = 'Best fit: Ro_m = {:.2g}'.format(10**fit[0]) + ' $Ro_c^{' +  '{:.2g}'.format(fit[1]) + '} Ro_p^{' + '{:.2g}'.format(fit[2]) + '}$'

    ### PLOTTING

    ro_c_i = np.linspace(ro_c.min(), ro_c.max(), 100)
    ro_p_i = np.linspace(ro_p.min(), ro_p.max(), 100)
    ro_pp, ro_cc = np.meshgrid(ro_p_i, ro_c_i)


    ro_c_b = np.logspace(ro_c.min(), ro_c.max(), 100)
    ro_p_b = np.logspace(ro_p.min(), ro_p.max(), 100)
    ro_b_pp, ro_b_cc = np.meshgrid(ro_p_b, ro_c_b)


    ro_m_interp = griddata((ro_c, ro_p), ro_m, (np.copy(ro_cc), np.copy(ro_pp)), method='linear')
    ro_m_func = fit[0] + ro_cc*(fit[1]) + ro_pp*(fit[2])

    return ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit, ro_b_cc, ro_b_pp

#full data set
ro_c, ro_m, ro_p = ro_c_full, ro_m_full, ro_p_full
ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit, ro_b_cc, ro_b_pp = run_analysis(ro_c, ro_m, ro_p)


### Plot 1
plot = ax3.pcolormesh(ro_b_cc, ro_b_pp, ro_m_interp, cmap='viridis', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
ax3.plot(10**(ro_c), 10**(ro_p), marker='s', lw=0, markersize=2, markerfacecolor='none', markeredgecolor='black')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xticks((1e-5, 1e-3, 1e-1, 1e1))
ax3.set_xlabel(r'Ro$_\mathrm{c}$')
ax3.set_ylabel(r'Ro$_\mathrm{p}$')

plt.colorbar(plot, cax=cax, orientation='vertical')
cax.annotate('Ro', xy=(.25,1.01), annotation_clip=False)
cax.set_yticklabels((r'$10^{-1.5}$', r'$10^{-1}$', r'$10^{-0.5}$', '1', r'$10^{0.5}$'))
cax.set_xticklabels([])


fig.savefig('../tex/figs/parameter_space.png', dpi=600, bbox_inches='tight')

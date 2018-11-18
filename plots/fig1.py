import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, interp2d
from scipy.optimize import curve_fit, leastsq
import scipy.optimize as scop
matplotlib.rcParams['font.family'] = 'serif'
def linear(x, a, b):
    return a + b*x


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

ax1.plot(onset_data[:,0], onset_data[:,1], c='k', lw=0.5, rasterized=True)
ax1.fill_between(onset_data[:,0], 0, onset_data[:,1], color='k', alpha=0.1, rasterized=True)

runs_data = np.genfromtxt('../data/release_data.csv', skip_header=1, delimiter=',')
Ra = runs_data[:,0]
Ta = runs_data[:,1]
S  = runs_data[:,6]
Ro_p = runs_data[:,7]
Ro_c = runs_data[:,8]
Ro_m = runs_data[:,9]
Re = runs_data[:,11]



original_co = np.genfromtxt('../data/constant_co.csv', skip_header=1, delimiter=',', usecols=(2,4,6)) #ra, ro, ta -- need to double check if this is backwards ra/ta.
co1 = Ro_c == 1
co03 = Ro_c == 0.3
co01 = Ro_c == 0.1
this_ta = np.logspace(1,7,100)
this_ra = this_ta
ax1.plot(this_ta, this_ra, c=Co_color, lw=2, label=r'$\mathrm{Ro}_\mathrm{c} = 1$', dashes=(3,1), rasterized=True)

S3 = S == 3
S2 = S == 2

x, y = Ta[S3], Ra[S3]
x, y = zip(*sorted(zip(x, y)))
ax1.plot(x, y, c=S_color, dashes=(3,0.5,0.5,0.5), lw=2, label=r'$\mathcal{S}=3$', rasterized=True)

Rop06  = Ro_p == 0.6
Rop957 = Ro_p == 0.957
Rop158 = Ro_p == 1.58
this_ta = np.logspace(1,7)
this_ra = 1.58**2*this_ta**(3/4)
ax1.plot(this_ta, this_ra, c=Pro_color, lw=3, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', rasterized=True)


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

ax2.plot(Ra[co1],    Ro_m[co1], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*Co_color, 1), rasterized=True)
ax2.plot(Ra[co03],   Ro_m[co03], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=3.5, markeredgecolor=(*Co_color, 1), rasterized=True)
ax2.plot(Ra[co01],   Ro_m[co01], marker=Co_marker, color=(*Co_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*Co_color, 1), rasterized=True)
ax2.plot(Ra[S3],     Ro_m[S3], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=4.5, markeredgecolor=(*S_color, 1), rasterized=True)
ax2.plot(Ra[S2],     Ro_m[S2], marker=S_marker, color=(*S_color, 0.3), lw=0, ms=2.5, markeredgecolor=(*S_color, 1), rasterized=True)
ax2.plot(Ra[Rop158], Ro_m[Rop158], marker=Pro_marker, color=Pro_color, lw=0, ms=4.5, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', rasterized=True)
ax2.plot(Ra[Rop957], Ro_m[Rop957], marker=Pro_marker, color=Pro_color, lw=0, ms=2.5, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', rasterized=True)
ax2.plot(Ra[Rop06],  Ro_m[Rop06], marker=Pro_marker, color=Pro_color, lw=0, ms=1.5, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.6$', rasterized=True)

lines, labels = [], []
co1_ra = Ra[co1][Ra[co1] >= 1e3]
co1_ro = Ro_m[co1][Ra[co1] >= 1e3]


ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Ra')
ax2.set_ylabel('Ro', labelpad=0)
ax2.set_ylim(1e-2, 6)
ax2.set_xticks((1e1, 1e3, 1e5, 1e7, 1e9))
ax2.text(3e1, 1.2e-2, "(b)", ha="center", va="center", size=8)
ax2.text(7e6, 5.2e-2, "2", ha="center", va="center", size=8, color=S_color)
ax2.text(4e7, 7.8e-2, "3", ha="center", va="center", size=8, color=S_color)
ax2.text(1.5e5, 1.15e-2, "0.6", ha="center", va="center", size=8, color=Pro_color)
ax2.text(8e2, 6e-2, "0.96", ha="center", va="center", size=8, color=Pro_color)
ax2.text(7e1, 2.2e-1, "1.58", ha="center", va="center", size=8, color=Pro_color)
ax2.text(2e7, 3.4e0, "1.0", ha="center", va="center", size=8, color=Co_color)
ax2.text(8e6, 8e-1, "0.3", ha="center", va="center", size=8, color=Co_color)
ax2.text(1.6e7, 2.6e-1, "0.1", ha="center", va="center", size=8, color=Co_color)




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
    print(fit_str)

    ### PLOTTING

    ro_c_i = np.linspace(ro_c.min(), ro_c.max(), 100)
    ro_p_i = np.linspace(ro_p.min(), ro_p.max(), 100)
    ro_pp, ro_cc = np.meshgrid(ro_p_i, ro_c_i)


    ro_c_b = np.logspace(ro_c.min(), ro_c.max(), 100)
    ro_p_b = np.logspace(ro_p.min(), ro_p.max(), 100)
    ro_b_pp, ro_b_cc = np.meshgrid(ro_p_b, ro_c_b)


    ro_m_interp = griddata((ro_c, ro_p), ro_m, (np.copy(ro_cc), np.copy(ro_pp)), method='linear')
    ro_m_func = fit[0] + ro_cc*(fit[1]) + ro_pp*(fit[2])

    print(ro_b_pp.min(), ro_b_pp.max(), ro_p.min(), ro_p.max())

    return ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit, ro_b_cc, ro_b_pp

#full data set
low_ro = Ro_m < 0.2
high_ro = Ro_m > 0.2
high_Re = Re > 5
good_ro_p = Ro_p[high_Re*low_ro]
good_ro_m = Ro_m[high_Re*low_ro]
(a, b), pcov = scop.curve_fit(linear, np.log10(good_ro_p), np.log10(good_ro_m))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = '10^({:.2f} +/- {:.2f})'.format(p[1], perr[1]) + 'Ro_p$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print(str2)


run_analysis(np.log10(Ro_c[high_Re*low_ro]), np.log10(Ro_m[high_Re*low_ro]), np.log10(Ro_p[high_Re*low_ro]))
run_analysis(np.log10(Ro_c[high_Re*high_ro]), np.log10(Ro_m[high_Re*high_ro]), np.log10(Ro_p[high_Re*high_ro]))
ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit, ro_b_cc, ro_b_pp = run_analysis(np.log10(Ro_c), np.log10(Ro_m), np.log10(Ro_p))


### Plot 1
plot = ax3.pcolormesh(ro_b_cc, ro_b_pp, ro_m_interp, cmap='viridis', vmin=np.log10(Ro_m).min(), vmax=np.log10(Ro_m).max(), snap=True, rasterized=True)
plot.set_edgecolor('face')
norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(Ro_m)), vmax=np.max(np.log10(Ro_m)))
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
for i in range(len(Ro_m)):
    if S2[i] or S3[i]:
        marker = S_marker
    elif co1[i] or co03[i] or co01[i]:
        marker = Co_marker
    elif Rop06[i] or Rop957[i] or Rop158[i]:
        marker = Pro_marker
    ax3.plot(Ro_c[i], Ro_p[i], markerfacecolor=sm.to_rgba(np.log10(Ro_m[i])), markeredgecolor='black', markersize=3, lw=0, marker=marker, markeredgewidth=0.5, rasterized=True)


ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xticks((1e-2, 1e-1, 1, 1e1))
ax3.set_xlabel(r'Ro$_\mathrm{c}$')
ax3.set_ylabel(r'Ro$_\mathrm{p}$')

cb = plt.colorbar(plot, cax=cax, orientation='vertical', ticks=[-2, -1.5, -1, -0.5, 0, 0.5])
cb.solids.set_rasterized(True)
cax.annotate('Ro', xy=(.25,1.01), annotation_clip=False)
cax.set_yticklabels((r'$10^{-2}$', r'$10^{-1.5}$', r'$10^{-1}$', r'$10^{-0.5}$', r'$10^0$', r'$10^{0.5}$'))
cax.set_xticklabels([])

onset_ta = onset_data[:,0]
onset_ra = onset_data[:,1]
onset_ta2 = [1e-3, 1e-2, 1e-1]
onset_ra2 = [onset_ra[0]]*3

onset_ta = np.concatenate((onset_ta2, onset_ta))
onset_ra = np.concatenate((onset_ra2, onset_ra))

onset_roc = np.sqrt(onset_ra/onset_ta)
onset_rop = np.sqrt(onset_ra/onset_ta**(0.75))
ax3.plot(onset_roc, onset_rop, c='k', lw=0.5)
ax3.fill_between(onset_roc, 0, onset_rop, color='k', alpha=0.1, rasterized=True)

ax3.set_ylim(0.5, 10)#np.min(ro_b_pp), np.max(ro_b_pp))
ax3.set_xlim(np.min(ro_b_cc), np.max(ro_b_cc))
ax3.text(4e0, 1e0, "stable", ha="center", va="center", rotation=0, size=10, alpha=0.7)



fig.savefig('../tex/figs/parameter_space.png', dpi=600, bbox_inches='tight')
fig.savefig('../tex/figs/parameter_space.pdf', dpi=600, bbox_inches='tight')

import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend import Legend
import scipy.optimize as scop
matplotlib.rcParams['font.family'] = 'serif'

ORANGE   = [1.        , 0.5, 0]
GREEN    = [0, 0.398, 0]
BLUE     = [0, 0.5, 1]
kwargs = {'lw' : 0, 'ms' : 3, 'markeredgewidth' : 0.5}



def linear(x, a, b):
    return a + b*x



fig = plt.figure(figsize=(3, 3.5))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50,50), 450, 900), ((500, 50), 450, 900))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))


#READ DATA, PLOT 1
runs_data = np.genfromtxt('../data/release_data.csv', skip_header=1, delimiter=',')
Ra = runs_data[:,0]
Ta = runs_data[:,1]
S  = runs_data[:,6]
Ro_p = runs_data[:,7]
Ro_c = runs_data[:,8]
Ro_m = runs_data[:,9]
Re_para = runs_data[:,10]
Re_perp = runs_data[:,11]
Nu   = runs_data[:,12]

Rop06  = Ro_p == 0.6
Rop957 = Ro_p == 0.957
Rop158 = Ro_p == 1.58

start_pt = 3
ra957_full = Ra[Rop957]/234.8
nu957_full = Nu[Rop957]
ra158_full = Ra[Rop158]/11
nu158_full = Nu[Rop158]
ra06_full = Ra[Rop06]/7.924e4
nu06_full = Nu[Rop06]
ra957      = ra957_full[start_pt:]
nu957      = nu957_full[start_pt:]
ra158      = ra158_full[start_pt:]
nu158      = nu158_full[start_pt:]
ra06       = ra06_full[start_pt:]
nu06       = nu06_full[start_pt:]

lines, labels = [], []
ra = np.logspace(0.5, 9, 100)
str2 = 'Ra$^{1/3}$'# + '{:.2f}'.format(1/3) + '}$'
lines += ax1.plot(ra, 1.1*ra**(1/3), label=r'{:s}'.format(str2), color='k', lw=0.5)
labels += [r'{:s}'.format(str2)]
#str2 = 'Ra$^{2/7}$'# + '{:.2f}'.format(2/7) + '}$'
#ra = np.logspace(0.5, 9, 100)
#lines += ax1.plot(ra, 1.5*ra**(2/7), label=r'{:s}'.format(str2), color='k', lw=0.5, dashes=(5,1))
#labels += [r'{:s}'.format(str2)]

leg = Legend(ax1, lines[-2:], labels[-2:],
             loc='upper left', frameon=False, fontsize=8)
ax1.add_artist(leg)




lines, labels = [], []
lines += ax1.plot(ra06_full, nu06_full, c=(*GREEN, 0.4), markeredgecolor=(*GREEN, 1),  marker='o',   **kwargs)
lines += ax1.plot(ra957_full, nu957_full, c=(*ORANGE, 0.4), markeredgecolor=(*ORANGE, 1),  marker='o', **kwargs)
lines += ax1.plot(ra158_full, nu158_full, c=(*BLUE, 0.4), markeredgecolor=(*BLUE, 1),  marker='o',  **kwargs)

#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$']
#labels += [r'$\mathcal{P}_{\mathrm{Ro}} = 1.58$']
labels += [r'Ro$\approx$0.03']
labels += [r'Ro$\approx$0.1']
labels += [r'Ro$\approx$0.4']

for xlabel_i in ax1.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)

print('Nu fit, rop_0.6')
p = np.polyfit(np.log10(ra06), np.log10(nu06), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra06), np.log10(nu06))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('nu', str2)
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(ra06_full, 10**(p[1])*ra06_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=GREEN, alpha=0.4)



print('Nu fit, rop_0.957')
p = np.polyfit(np.log10(ra957), np.log10(nu957), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra957), np.log10(nu957))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('nu', str2)
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(ra957_full, 10**(p[1])*ra957_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=ORANGE, alpha=0.4)

print('Nu fit, rop_1.58')
p = np.polyfit(np.log10(ra158), np.log10(nu158), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra158), np.log10(nu158))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('nu', str2)
labels += [r'{:s}'.format(str2)]
lines += ax1.plot(ra158_full, 10**(p[1])*ra158_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=BLUE, alpha=0.4)
print('\n\n')


ax1.legend(lines[:3], labels[:3], loc='lower right', frameon=False, fontsize=8, handletextpad=0)
#leg = Legend(ax1, lines[2:], labels[2:],
#             loc='lower right', frameon=False, fontsize=8)
ax1.add_artist(leg)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('(a) Nu')
#ax1.text(3e-2, 7e1, "(a)", ha="center", va="center", size=8)
ax1.set_yticks([1e1, 1e2])


lines, labels = [], []
#PLOT 2
re06_full = Re_para[Rop06]
re_perp06_full = Re_perp[Rop06]
re957_full = Re_para[Rop957]
re_perp957_full = Re_perp[Rop957]
re158_full = Re_para[Rop158]
re_perp158_full = Re_perp[Rop158]
re06 = re06_full[start_pt:]
re_perp06 = re_perp06_full[start_pt:]
re957 = re957_full[start_pt:]
re_perp957 = re_perp957_full[start_pt:]
re158 = re158_full[start_pt:]
re_perp158 = re_perp158_full[start_pt:]


ra = np.logspace(0.5, 9, 100)
str2 = 'Ra$^{1/2}$'# + '{:.2f}'.format(0.5) + '}$'
lines += ax2.plot(ra, 6*ra**(1/2), label=r'{:s}'.format(str2), color='k', lw=0.5)
labels += [r'{:s}'.format(str2)]
str2 = 'Ra$^{5/18}$'# + '{:.2f}'.format(5/18) + '}$'
ra = np.logspace(0.5, 9, 100)
lines += ax2.plot(ra, 1.8*ra**(5/18), label=r'{:s}'.format(str2), color='k', lw=0.5, dashes=(5,1))
labels += [r'{:s}'.format(str2)]

leg = Legend(ax2, lines[-2:], labels[-2:],
             loc='upper left', frameon=False, fontsize=8)
ax2.add_artist(leg)



lines += ax2.plot(ra06_full, re06_full, c=(*GREEN, 0.4), markeredgecolor=(*GREEN, 1), marker='o', **kwargs)
lines += ax2.plot(ra06_full, re_perp06_full, c=(*GREEN, 0.4), markeredgecolor=(*GREEN, 1), marker='s', **kwargs)
lines += ax2.plot(ra957_full, re957_full, c=(*ORANGE, 0.4), markeredgecolor=(*ORANGE, 1), marker='o', **kwargs)
lines += ax2.plot(ra957_full, re_perp957_full, c=(*ORANGE, 0.4), markeredgecolor=(*ORANGE, 1), marker='s',  **kwargs)
lines += ax2.plot(ra158_full, re158_full, c=(*BLUE, 0.4), markeredgecolor=(*BLUE, 1),  marker='o', **kwargs)
lines += ax2.plot(ra158_full, re_perp158_full, c=(*BLUE, 0.4), markeredgecolor=(*BLUE, 1), marker='s', **kwargs)


leg = Legend(ax2, lines[-2:], (r'$\mathrm{Re}_{\parallel}$', r'$\mathrm{Re}_{\perp}$'),
             loc='lower right', frameon=False, fontsize=8, borderpad=0.05, borderaxespad=0.5, handletextpad=0)
ax2.add_artist(leg)

print('Re fits, rop_0.6')
p = np.polyfit(np.log10(ra06), np.log10(re06), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra06), np.log10(re06))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('para', str2)
lines += ax2.plot(ra06_full, 10**(p[1])*ra06_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=GREEN, alpha=0.4)
p = np.polyfit(np.log10(ra06), np.log10(re_perp06), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra06), np.log10(re_perp06))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('perp', str2)
lines += ax2.plot(ra06_full, 10**(p[1])*ra06_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=GREEN, alpha=0.4)

print('Re fits, rop_0.957')
p = np.polyfit(np.log10(ra957), np.log10(re957), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra957), np.log10(re957))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('para', str2)
lines += ax2.plot(ra957_full, 10**(p[1])*ra957_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=ORANGE, alpha=0.4)
p = np.polyfit(np.log10(ra957), np.log10(re_perp957), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra957), np.log10(re_perp957))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('perp', str2)
lines += ax2.plot(ra957_full, 10**(p[1])*ra957_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=ORANGE, alpha=0.4, dashes=(5,1))

print('Re fits, rop_1.58')
p = np.polyfit(np.log10(ra158), np.log10(re158), deg=1)
p = np.polyfit(np.log10(ra957), np.log10(re_perp957), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra158), np.log10(re158))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('para', str2)
lines += ax2.plot(ra158_full, 10**(p[1])*ra158_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=BLUE, alpha=0.4)
p = np.polyfit(np.log10(ra158), np.log10(re_perp158), deg=1)
(a, b), pcov = scop.curve_fit(linear, np.log10(ra158), np.log10(re_perp158))
(a_err, b_err) = np.sqrt(np.diag(pcov))
p = [b, a]
perr = [b_err, a_err]
str2 = 'Ra$^{' + '{:.2f} \pm {:.2f}'.format(p[0], perr[0]) + '}$'
print('perp', str2)
lines += ax2.plot(ra158_full, 10**(p[1])*ra158_full**(p[0]), label=r'{:s}'.format(str2), lw=0.5, color=BLUE, alpha=0.4, dashes=(5,1))



ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Ra/Ra$_{\mathrm{crit}}$')
ax2.set_ylabel('(b) Re')
#ax2.text(3e-2, 2e3, "(b)", ha="center", va="center", size=8)
ax2.set_yticks([1e1, 1e2, 1e3])



ax1.set_ylim(1, 1e2)
ax2.set_ylim(1, 1e4)
ax1.set_xlim(1, 1e7)
ax2.set_xlim(1, 1e7)

#PLOT 2

fig.savefig('../tex/figs/nu_and_re.png', dpi=600, bbox_inches='tight')
fig.savefig('../tex/figs/nu_and_re.pdf', dpi=600, bbox_inches='tight')

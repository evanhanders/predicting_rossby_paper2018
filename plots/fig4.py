import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import h5py
import scipy.optimize as scop
from scipy.optimize import fmin
from scipy.interpolate import interp1d

ORANGE   = [1.        , 0.5, 0]
GREEN    = [0, 0.398, 0]
BLUE     = [0, 0.5, 1]
kwargs = {'lw' : 0, 'ms' : 3, 'markeredgewidth' : 0.5}



Lz = np.exp(3/(1.5-1e-4)) - 1

cmap = 'viridis_r'

entropy_gradients = dict()
entropy_stds = dict()
rossby_profiles   = dict()
z   = dict()
lines = []
for fname in glob.glob('../profile_data/*.h5'):
    key = ''
    pieces = fname.split('.h5')[0].split('/')[-1].split('_')
    ind_num = -2
    for s in pieces:
        if 'rop' in s:
            key += s + '_'
        if 'ta' in s:
            key += s
    f = h5py.File(fname, 'r')
    for k in f.keys():
        print(k)
    n_profiles = f['grad_s_tot'].value.shape[0]
    entropy_gradients[key] = f['grad_s_tot'].value[-int(n_profiles/3)-1:-1,:]
    entropy_stds[key] = f['s_fluc_std'].value[-int(n_profiles/3)-1:-1,:]
    rossby_profiles[key]  = f['Rossby'].value[-int(n_profiles/3)-1:-1,:]
    z[key]  = f['z'].value
keys = rossby_profiles.keys()
ta = [-float(k.split('_ta')[-1]) for k in keys]
tas, sorted_keys = zip(*sorted(zip(ta, keys)))
tas = -np.array(tas)

maxminta = dict()
for ta, k in zip(tas, sorted_keys):
    key = k.split('_')[0]
    if key in maxminta.keys():
        if ta < maxminta[key][0]:
            maxminta[key][0] = ta
        if ta > maxminta[key][1]:
            maxminta[key][1] = ta
    else:
        maxminta[key] = [ta, ta]

print(maxminta)
        

rops = []
tas = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rops.append(float(rop[3:]))
    tas.append(float(ta[2:]))
rops = np.array(rops)
tas = np.array(tas)
ras = rops**2 * tas**(3/4)

fig_trash = plt.figure()
ax_trash  = fig_trash.add_subplot(1,1,1)



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((0,0), 50, 250), ((50,0), 450, 250), ((500, 0), 450, 250), 
           ((0,370), 50, 250), ((50, 370), 450, 250), ((500, 370), 450, 250),
           ((0, 750), 500, 250), ((500, 750),450, 250))

ax3 = plt.subplot(gs.new_subplotspec(*gs_info[-2]))
ax3_2 = plt.subplot(gs.new_subplotspec(*gs_info[-1]))
ax3_2.axhline(1, c='k')

#COLUMN 1
cax1  = plt.subplot(gs.new_subplotspec(*gs_info[3]))
ax1_1 = plt.subplot(gs.new_subplotspec(*gs_info[4]))
ax1_2 = plt.subplot(gs.new_subplotspec(*gs_info[5]))

norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(maxminta['rop0.957'][0])), vmax=np.max(np.log10(maxminta['rop0.957'][1])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

s_bls = []
ro_bls = []
s_bls_bot = []
ro_bls_bot = []
taylors = []
rayleighs = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])
    ra = rop**2 * ta**(3/4)

    if rop != 0.957:
        continue

    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    ax1_1.plot(z[key], np.mean(entropy_stds[key]   , axis=0), c=sm.to_rgba(np.log10(ta)))
    ax1_2.plot(z[key], np.mean(rossby_profiles[key], axis=0), c=sm.to_rgba(np.log10(ta)))


    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)

#    ax_trash.axvline(Lz - ro_bls[-1], ls='--')
#    ax_trash.axvline(Lz - s_bls[-1])
#    ax_trash.plot(z[key], rossby_profiles[key]/np.max(rossby_profiles[key]))
#    ax_trash.plot(z[key], entropy_stds[key]/np.max(entropy_stds[key]))
#    ax_trash.plot(z[key], entropy_gradients[key]/np.min(entropy_gradients[key]))
#    fig_trash.savefig('ro_{:s}.png'.format(key))
#    ax_trash.cla()


    taylors.append(ta)
    rayleighs.append(ra)



  
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
ro_bls_bot = np.array(ro_bls_bot)
s_bls_bot = np.array(s_bls_bot)
rayleighs = np.array(rayleighs)

lines += ax3_2.plot(rayleighs/270, s_bls/0.66, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', zorder=2, marker='o', color=(*ORANGE, 0.4), markeredgecolor=(*ORANGE, 1), **kwargs)
label096 = r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$'
p = np.polyfit(np.log10(rayleighs/270), np.log10(s_bls/0.66), deg=1)
vert = (1/10**(p[1]))**(1/p[0])
ax3_2.axvline(vert, c=ORANGE, ls='--', lw=0.5)
ax3.axvline(vert, c=ORANGE, ls='--',  lw=0.5)


 
ax3.plot(rayleighs/270, ro_bls/s_bls, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', zorder=2, marker='o', color=(*ORANGE, 0.4), markeredgecolor=(*ORANGE, 1), **kwargs)
ax3.set_xscale('log')

ax1_2.set_xlabel('z')
ax1_2.set_ylabel('Ro')
ax1_1.set_ylabel(r'$\sigma_\mathrm{s} \times 10^{5}$', labelpad=15)
ax1_1.set_xticks(())
ax1_1.set_ylim(0, 4e-5)
ax1_1.set_yticks((1e-5, 2e-5, 3e-5, 4e-5))
ax1_1.set_yticklabels((1, 2, 3, 4))
ax1_1.set_xlim(0, Lz)
ax1_2.set_xlim(0, Lz)

ax1_1.text(1.55, 3e-5, r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', ha='center', va='center')
    

plt.colorbar(sm, cax=cax1, orientation='horizontal')
cax1.set_xticklabels(())
cax1.annotate(r'$10^{0}$', xy=(0,1.2), annotation_clip=False)
cax1.annotate(r'$10^{6}$', xy=(.9,1.2), annotation_clip=False)
cax1.annotate(r'$\mathrm{Ra}/\mathrm{Ra}_{\mathrm{crit}}$', xy=(.35,1.3), annotation_clip=False, style='normal')


#COLUMN 2
cax2  = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2_1 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax2_2 = plt.subplot(gs.new_subplotspec(*gs_info[2]))

norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(maxminta['rop1.58'][0])), vmax=np.max(np.log10(maxminta['rop1.58'][1])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

s_bls = []
ro_bls = []
s_bls_bot = []
ro_bls_bot = []
taylors = []
rayleighs = []
print('_________________')
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])

    if rop != 1.58:
        continue
    
    ra = rop**2 * ta**(3/4)
    ax2_1.plot(z[key], entropy_stds[key][0,:], c=sm.to_rgba(np.log10(ta)))
    ax2_2.plot(z[key], rossby_profiles[key][0,:], c=sm.to_rgba(np.log10(ta)))
    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)


#    ax_trash.axvline(Lz - ro_bls[-1], ls='--')
#    ax_trash.axvline(Lz - s_bls[-1], ls='--')
#    ax_trash.plot(z[key], rossby_profiles[key]/np.max(rossby_profiles[key]))
#    ax_trash.plot(z[key], entropy_stds[key]/np.max(entropy_stds[key]))
#    ax_trash.plot(z[key], entropy_gradients[key]/np.min(entropy_gradients[key]))
#    fig_trash.savefig('ro_{:s}.png'.format(key))
#    ax_trash.cla()


    taylors.append(ta)
    rayleighs.append(ra)

ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
ro_bls_bot = np.array(ro_bls_bot)
s_bls_bot = np.array(s_bls_bot)
rayleighs = np.array(rayleighs)

lines += ax3_2.plot(rayleighs/25, s_bls/0.66, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', zorder=1, marker='o', color=(*BLUE, 0.4), markeredgecolor=(*BLUE, 1), **kwargs)
label158 = r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$'
ax3_2.set_yscale('log')
ax3_2.set_xscale('log')
ax3_2.set_ylabel(r'$\delta_{\mathrm{s}}/\mathrm{H}_\rho$')
p = np.polyfit(np.log10(rayleighs/25), np.log10(s_bls/0.66), deg=1)
vert = (1/10**(p[1]))**(1/p[0])
ax3_2.axvline(vert, c=BLUE, ls='--', lw=0.5)
ax3.axvline(vert, c=BLUE, ls='--',  lw=0.5)





 
ax3.plot(rayleighs/25, ro_bls/s_bls, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', zorder=1, marker='o', color=(*BLUE, 0.4), markeredgecolor=(*BLUE, 1), **kwargs)
#ax3.set_yscale('log')
ax3_2.set_xlabel(r'Ra/Ra$_{\mathrm{crit}}$')
ax3.set_ylabel(r'$\delta_{\mathrm{Ro}}/\delta_{\mathrm{s}}$', labelpad=15)

ax2_2.set_xlabel('z')
ax2_2.set_ylabel('Ro')
ax2_1.set_ylabel(r'$\sigma_\mathrm{s} \times 10^{5}$', labelpad=15)
ax2_1.set_ylim(0, 7e-5)
ax2_1.set_yticks(( 2e-5, 4e-5, 6e-5))
ax2_2.set_yticks(( 0.2, 0.4, 0.6, 0.8))
ax2_1.set_yticklabels((2, 4, 6))
ax2_1.set_xticks(())
ax2_1.set_xlim(0, Lz)
ax2_2.set_xlim(0, Lz)
ax2_1.text(1.55, 5.3e-5, r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', ha='center', va='center')


ax3.set_yticks((0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6))
ax2_1.text(0.4, 6.3e-5,   "(a)", ha="center", va="center", size=8)
ax2_2.text(0.4, 8.2e-1, "(b)", ha="center", va="center", size=8)
ax1_1.text(0.4, 3.6e-5,   "(c)", ha="center", va="center", size=8)
ax1_2.text(0.4, 2.5e-1, "(d)", ha="center", va="center", size=8)
ax3.text(2.5e0, 0.41, "(e)", ha="center", va="center", size=8)
ax3_2.text(2.3e0, 1.1e-1, "(f)", ha="center", va="center", size=8)
 


plt.colorbar(sm, cax=cax2, orientation='horizontal')
cax2.annotate(r'$10^{0}$', xy=(0,1.2), annotation_clip=False)
cax2.annotate(r'$10^{6}$', xy=(.9,1.2), annotation_clip=False)
cax2.annotate(r'$\mathrm{Ra}/\mathrm{Ra}_{\mathrm{crit}}$', xy=(.35,1.3), annotation_clip=False)
cax2.set_xticklabels([])


#low rop
s_bls = []
ro_bls = []
tylors = []
rayleighs = []
for key in sorted_keys:
    rop, ta = key.split('_')
    rop = float(rop[3:])
    ta = float(ta[2:])
    ra = rop**2 * ta**(3/4)

    if rop != 0.6:
        continue

    print('rop {:.2f}, ra {:.4e}, ta {:.4e}'.format(rop,ra, ta))
    n_calcs = 0
    mean_s_bl = 0
    mean_ro_bl = 0

    s_std = np.mean(entropy_stds[key], axis=0)
    ro    = np.mean(rossby_profiles[key], axis=0)

    half_z = z[key][int(len(z[key])/2):]
    half_s = s_std[int(len(s_std)/2):]
    half_ro = ro[int(len(ro)/2):]
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    s = interp1d(half_z, half_s, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)
    big_s  = s(big_z)


    ro_bl = Lz - big_z[np.argmax(big_ro)]
    max_place = big_z[np.argmax(big_s[:-100])]
    s_bl  = Lz - max_place
    mean_s_bl += s_bl
    mean_ro_bl += ro_bl
    n_calcs += 1

    ro_bls.append(mean_ro_bl/n_calcs)
    s_bls.append(mean_s_bl/n_calcs)

    ax_trash.axvline(Lz - ro_bls[-1], ls='--')
    ax_trash.axvline(Lz - s_bls[-1], ls='--')
    ax_trash.plot(z[key], np.mean(rossby_profiles[key], axis=0)/np.max(rossby_profiles[key][-1,:]))
    ax_trash.plot(z[key], np.mean(entropy_stds[key], axis=0)/np.max(entropy_stds[key][-1,:]))
    fig_trash.savefig('ro_{:s}.png'.format(key))
    ax_trash.cla()


    taylors.append(ta)
    rayleighs.append(ra)

rayleighs = np.array(rayleighs)

  
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
lines  += ax3_2.plot(rayleighs/7.924e4, s_bls/0.66, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.6$', zorder=3, marker='o', color=(*GREEN, 0.4), markeredgecolor=(*GREEN, 1), **kwargs)
label06 = r'$\mathrm{Ro}_{\mathrm{p}} = 0.6$'
p = np.polyfit(np.log10(rayleighs/7.924e4), np.log10(s_bls/0.66), deg=1)
vert = (1/10**(p[1]))**(1/p[0])
ax3_2.axvline(vert, c=GREEN, ls='--', lw=0.5)
ax3.axvline(vert, c=GREEN, ls='--',  lw=0.5)
 
ax3.plot(rayleighs/7.924e4, ro_bls/s_bls, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.6$', zorder=3, marker='o', color=(*GREEN, 0.4), markeredgecolor=(*GREEN, 1), **kwargs)
ax3.set_xscale('log')
ax3.set_xticks([])

leg_lines = [lines[2], lines[0], lines[1]]

from matplotlib.legend import Legend
leg = Legend(ax3_2, leg_lines, [label06, label096, label158],
             loc='upper right', frameon=False, fontsize=6, borderpad=0.25, borderaxespad=0.1, handletextpad=0)
ax3_2.add_artist(leg)

fig.savefig('../tex/figs/boundary_layers.png', dpi=600, bbox_inches='tight')
fig.savefig('../tex/figs/boundary_layers.pdf', dpi=600, bbox_inches='tight')

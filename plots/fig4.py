import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import h5py
from scipy.optimize import fmin
from scipy.interpolate import interp1d
matplotlib.rcParams['font.family'] = 'serif'

Lz = np.exp(3/(1.5-1e-4)) - 1

cmap = 'viridis_r'

entropy_gradients = dict()
rossby_profiles   = dict()
z   = dict()
for fname in glob.glob('../profile_data/*.h5'):
    key = ''
    pieces = fname.split('.h5')[0].split('/')[-1].split('_')
    for s in pieces:
        if 'co' in s:
            key += s + '_'
        if 'ta' in s:
            key += s
    f = h5py.File(fname, 'r')
    entropy_gradients[key] = f['grad_s_tot'].value[-2,:]
    rossby_profiles[key]  = f['Rossby'].value[-2,:]
    z[key]  = f['z'].value

cos = []
tas = []
for key in rossby_profiles.keys():
    co, ta = key.split('_')
    cos.append(float(co[2:]))
    tas.append(float(ta[2:]))
cos = np.array(cos)
tas = np.array(tas)
ras = cos**2 * tas**(3/4)

fig_trash = plt.figure()
ax_trash  = fig_trash.add_subplot(1,1,1)



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((0,0), 50, 250), ((50,0), 450, 250), ((500, 0), 450, 250), 
           ((0,370), 50, 250), ((50, 370), 450, 250), ((500, 370), 450, 250),
           ((0, 750), 950, 250))

ax3 = plt.subplot(gs.new_subplotspec(*gs_info[-1]))

#COLUMN 1
cax1  = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax1_1 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax1_2 = plt.subplot(gs.new_subplotspec(*gs_info[2]))

norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(tas[cos==0.96])), vmax=np.max(np.log10(tas[cos==0.96])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

s_bls = []
ro_bls = []
taylors = []
rayleighs = []
for key in rossby_profiles.keys():
    co, ta = key.split('_')
    co = float(co[2:])
    ta = float(ta[2:])
    ra = co**2 * ta**(3/4)

    if co != 0.96:
        continue

    print(key)
    print('ra {:.4e}, ta {:.4e}'.format(ra, ta))
    ax1_1.plot(z[key], entropy_gradients[key], c=sm.to_rgba(np.log10(ta)))
    ax1_2.plot(z[key], rossby_profiles[key], c=sm.to_rgba(np.log10(ta)))

    if ta < 1e5:
        n_pts = 12
        frac = 0.9
    elif ta < 1e7:
        n_pts = 12#int(len(z[key])*5/100)
        frac = 0.985
    else:
        n_pts = 12
        frac = 0.99
    good_zs = z[key][z[key] <= frac*Lz]
    good_grads = entropy_gradients[key][z[key] <= frac*Lz]
    print(z[key][-1], good_zs[-1])
    fit = np.polyfit(good_zs[-n_pts:], good_grads[-n_pts:], deg=1)
    y_fit = np.zeros_like(z[key])
    for i, f in enumerate(fit):
        y_fit += f*z[key]**(len(fit)-1-i)
 

    fit_f = interp1d(y_fit, z[key])
    z0  = fit_f(0)
    s_bl = Lz - z0
    s_bls.append(s_bl)
    
    ax_trash.plot(z[key], entropy_gradients[key])
    ax_trash.plot(z[key], y_fit, c=sm.to_rgba(np.log10(ta)))
    ax_trash.axhline(0)
    ax_trash.set_ylim(-np.max(np.abs(entropy_gradients[key])), np.max(np.abs(entropy_gradients[key])))
    fig_trash.savefig('{:s}.png'.format(key))
    ax_trash.cla()
    

    half_z = z[key][int(len(z[key])/2):]
    half_ro = rossby_profiles[key][int(len(rossby_profiles[key])    /2):] 
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)

    ro_bl_guess_i = np.argmax(big_ro)
    ro_bl = Lz - big_z[ro_bl_guess_i]
    ro_bls.append(ro_bl)
    taylors.append(ta)
    rayleighs.append(ra)


  
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
 
ax3.plot(rayleighs, ro_bls/s_bls, label=r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', marker='o', lw=0, color='orange')
ax3.set_xscale('log')

ax1_2.set_xlabel('z')
ax1_2.set_ylabel('Ro')
ax1_1.set_ylabel(r'$(\partial_z\,s) \times 10^{-4}$')
ax1_1.set_xticks(())
ax1_1.set_yticks((0, -0.00025, -0.0005))
ax1_1.set_yticklabels(('0', r'$-$2.5', r'$-$5.0'))
ax1_1.set_xlim(0, Lz)
ax1_2.set_xlim(0, Lz)

ax1_1.text(2.2, -7e-4, r'$\mathrm{Ro}_{\mathrm{p}} = 0.96$', ha='center', va='center')
ax1_1.text(0.4, -7e-4, "(a)", ha="center", va="center", size=8)
ax1_2.text(0.4, 0.03, "(b)", ha="center", va="center", size=8)
    

plt.colorbar(sm, cax=cax1, orientation='horizontal')
cax1.set_xticklabels(())
cax1.annotate(r'$10^{2.75}$', xy=(0,1.2), annotation_clip=False)
cax1.annotate(r'$10^6$', xy=(.9,1.2), annotation_clip=False)
cax1.annotate('Ra', xy=(.45,1.2), annotation_clip=False)


#COLUMN 2
cax2  = plt.subplot(gs.new_subplotspec(*gs_info[3]))
ax2_1 = plt.subplot(gs.new_subplotspec(*gs_info[4]))
ax2_2 = plt.subplot(gs.new_subplotspec(*gs_info[5]))

norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(tas[cos!=0.96])), vmax=np.max(np.log10(tas[cos!=0.96])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

s_bls = []
ro_bls = []
taylors = []
rayleighs = []
print('_________________')
for key in rossby_profiles.keys():
    co, ta = key.split('_')
    co = float(co[2:])
    ta = float(ta[2:])

    if co == 0.96:
        continue
    
    ra = co**2 * ta**(3/4)
    print('{:.4e}'.format(ra))
    ax2_1.plot(z[key], entropy_gradients[key], c=sm.to_rgba(np.log10(ta)))
    ax2_2.plot(z[key], rossby_profiles[key], c=sm.to_rgba(np.log10(ta)))

    if ta < 1e3:
        n_pts = 12
        frac = 0.9
    elif ta < 1e6:
        n_pts = 12#int(len(z[key])*5/100)
        frac = 0.985
    else:
        n_pts = 12
        frac = 0.995
    good_zs = z[key][z[key] <= frac*Lz]
    good_grads = entropy_gradients[key][z[key] <= frac*Lz]
    fit = np.polyfit(good_zs[-n_pts:], good_grads[-n_pts:], deg=1)
    y_fit = np.zeros_like(z[key])
    for i, f in enumerate(fit):
        y_fit += f*z[key]**(len(fit)-1-i)
 

    fit_f = interp1d(y_fit, z[key])
    z0  = fit_f(0)
    s_bl = Lz - z0
    s_bls.append(s_bl)
    
    ax_trash.plot(z[key], entropy_gradients[key])
    ax_trash.plot(z[key], y_fit, c=sm.to_rgba(np.log10(ta)))
    ax_trash.axhline(0)
    ax_trash.set_ylim(-np.max(np.abs(entropy_gradients[key])), np.max(np.abs(entropy_gradients[key])))
    fig_trash.savefig('{:s}.png'.format(key))
    ax_trash.cla()
    

    half_z = z[key][int(len(z[key])/2):]
    half_ro = rossby_profiles[key][int(len(rossby_profiles[key])    /2):] 
    ro = interp1d(half_z, half_ro, fill_value='extrapolate')
    big_z = np.linspace(Lz/2, Lz, 1e4)
    big_ro = ro(big_z)

    ro_bl_guess_i = np.argmax(big_ro)
    ro_bl = Lz - big_z[ro_bl_guess_i]
    ro_bls.append(ro_bl)
    taylors.append(ta)
    rayleighs.append(ra)
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
 
ax3.plot(rayleighs, ro_bls/s_bls, label=r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', marker='o', lw=0, color='blue')
ax3.set_xlabel('Ra')
ax3.set_ylabel(r'$\delta_{\mathrm{Ro}}/\delta_{\mathrm{s}}$')

ax2_2.set_xlabel('z')
ax2_2.set_ylabel('Ro')
ax2_1.set_ylabel(r'$(\partial_z\,s) \times 10^{-3}$')
ax2_1.set_xticks(())
ax2_1.set_yticks((0, -0.001, -0.002))
ax2_1.set_yticklabels(('0', r'$-$1', r'$-$2'))
ax2_1.set_xlim(0, Lz)
ax2_2.set_xlim(0, Lz)
ax2_1.text(2.2, -2.22e-3, r'$\mathrm{Ro}_{\mathrm{p}} = 1.58$', ha='center', va='center')
ax2_1.text(0.4, -2.22e-3, "(c)", ha="center", va="center", size=8)
ax2_2.text(0.4, 0.125, "(d)", ha="center", va="center", size=8)


ax3.text(1.2e2, 0.55, "(e)", ha="center", va="center", size=8)
ax3.legend(loc='center right', frameon=True, borderpad=0.2, handletextpad=0.2, fontsize=8)
 


plt.colorbar(sm, cax=cax2, orientation='horizontal')
cax2.annotate(r'$10^{2}$', xy=(0,1.2), annotation_clip=False)
cax2.annotate(r'$10^{6.5}$', xy=(.9,1.2), annotation_clip=False)
cax2.annotate('Ra', xy=(.45,1.2), annotation_clip=False)
cax2.set_xticklabels([])


fig.savefig('../tex/figs/boundary_layers.png', dpi=600, bbox_inches='tight')
fig.savefig('../tex/figs/boundary_layers.pdf', dpi=600, bbox_inches='tight')

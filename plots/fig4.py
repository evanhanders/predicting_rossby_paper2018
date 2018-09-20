import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import h5py

Lz = np.exp(3/(1.5-1e-4)) - 1

cmap = 'summer_r'

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



fig = plt.figure(figsize=(8.5, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((0,0), 50, 250), ((50,0), 450, 250), ((500, 0), 450, 250), 
           ((0,370), 50, 250), ((50, 370), 450, 250), ((500, 370), 450, 250),
           ((50, 750), 900, 250))

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
for key in rossby_profiles.keys():
    co, ta = key.split('_')
    co = float(co[2:])
    ta = float(ta[2:])

    if co != 0.96:
        continue
    ax1_1.plot(z[key], entropy_gradients[key], c=sm.to_rgba(np.log10(ta)))
    ax1_2.plot(z[key], rossby_profiles[key], c=sm.to_rgba(np.log10(ta)))
  
    n_pts = int(len(z[key])*5/100)
    good_zs = z[key][z[key] <= 0.998*Lz]
    good_grads = entropy_gradients[key][z[key] <= 0.998*Lz]
    fit = np.polyfit(good_zs[-n_pts:], good_grads[-n_pts:], deg=1)
#    ax1_1.plot(z[key], fit[0]*z[key] + fit[1], c=sm.to_rgba(np.log10(ta)))
    z0  = -fit[1]/fit[0]
    s_bl = Lz - z0
    s_bls.append(s_bl)
    
    ro_bl = np.argmax(rossby_profiles[key][int(len(rossby_profiles[key])/2):])
    ro_bl = Lz - z[key][int(len(z[key])/2+ro_bl)]
    ro_bls.append(ro_bl)
    taylors.append(ta)
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
 
ax3.plot(taylors, ro_bls/s_bls, label=r'$\mathcal{P}_{\mathrm{Ro}} = 0.96$', marker='o', lw=0)
ax3.set_xscale('log')

ax1_2.set_xlabel('z')
ax1_2.set_ylabel('Ro')
ax1_1.set_ylabel(r'$\nabla s$')
    

plt.colorbar(sm, cax=cax1, orientation='horizontal')
cax1.set_xticklabels(())
cax1.annotate(r'$10^{3.5}$', xy=(0,1.2), annotation_clip=False)
cax1.annotate(r'$10^8$', xy=(.9,1.2), annotation_clip=False)
cax1.annotate('Ta', xy=(.45,1.2), annotation_clip=False)


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
for key in rossby_profiles.keys():
    co, ta = key.split('_')
    co = float(co[2:])
    ta = float(ta[2:])

    if co == 0.96:
        continue
    ax2_1.plot(z[key], entropy_gradients[key], c=sm.to_rgba(np.log10(ta)))
    ax2_2.plot(z[key], rossby_profiles[key], c=sm.to_rgba(np.log10(ta)))

    n_pts = int(len(z[key])*5/100)
    good_zs = z[key][z[key] <= 0.998*Lz]
    good_grads = entropy_gradients[key][z[key] <= 0.998*Lz]
    fit = np.polyfit(good_zs[-n_pts:], good_grads[-n_pts:], deg=1)
#    ax2_1.plot(z[key], fit[0]*z[key] + fit[1], c=sm.to_rgba(np.log10(ta)))
#    ax2_1.axvline(Lz*0.99)
    z0  = -fit[1]/fit[0]
    s_bl = Lz - z0
    s_bls.append(s_bl)
    
    ro_bl = np.argmax(rossby_profiles[key][int(len(rossby_profiles[key])/2):])
    ro_bl = Lz - z[key][int(len(z[key])/2+ro_bl)]
    ro_bls.append(ro_bl)
    taylors.append(ta)
ro_bls = np.array(ro_bls)
s_bls = np.array(s_bls)
 
ax3.plot(taylors, ro_bls/s_bls, label=r'$\mathcal{P}_{\mathrm{Ro}} = 1.6$', marker='o', lw=0)
ax3.set_xlabel('Ta')
ax3.set_ylabel(r'$\delta_{\mathrm{Ro}}/\delta_{\mathrm{\nabla s}}$')

ax2_2.set_xlabel('z')
ax2_2.set_ylabel('Ro')
ax2_1.set_ylabel(r'$\nabla s$')


ax3.legend(loc='upper right', frameon=False, borderpad=0.2, handletextpad=0.2, fontsize=8)
 


plt.colorbar(sm, cax=cax2, orientation='horizontal')
cax2.annotate(r'$10^{2}$', xy=(0,1.2), annotation_clip=False)
cax2.annotate(r'$10^8$', xy=(.9,1.2), annotation_clip=False)
cax2.annotate('Ta', xy=(.45,1.2), annotation_clip=False)
cax2.set_xticklabels([])


fig.savefig('../tex/figs/boundary_layers.png', dpi=300, bbox_inches='tight')
#plt.show()
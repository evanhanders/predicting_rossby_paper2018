import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
matplotlib.rcParams['font.family'] = 'serif'

import dedalus.public as de

def nice_exp(number):
    string = '{:.2e}'.format(number)
    pre, exp = string.split('e')
    if exp[0] == '+': exp = exp[1:]
    exp = str(int(exp))
    full = '{} \\times 10^'.format(pre) + '{' + '{}'.format(exp) + '}'
    e_str = full.split('\\times ')[-1]
    return full, pre, e_str
    


fig = plt.figure(figsize=(8, 2))
gs     = gridspec.GridSpec(*(1000,1000))
#gs_info = (((100, 0), 900, 320), ((100, 340), 900, 320), ((100, 680),900, 320))
gs_info = (((100, 0), 900, 230), ((100, 250), 900, 230), ((100, 500),900, 230), ((100, 750), 900, 230))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))
ax4 = plt.subplot(gs.new_subplotspec(*gs_info[3]))

#lowF = h5py.File('../data/slices/pro0.96_ra9.16e5_ta1e8_slices_s60.h5', 'r')
very_lowF = h5py.File('../data/slices/pro0.6_ra1.24e9_ta5.2e12_slices_s230.h5', 'r')
lowF = h5py.File('../data/slices/pro0.96_ra7.67e7_ta3.66e10_slices_s58.h5', 'r')
#highF = h5py.File('../data/slices/pro1.58_ra2.49e5_ta4.64e6_slices_s60.h5', 'r')
highF = h5py.File('../data/slices/pro1.58_ra1.40e6_ta4.64e7.h5', 'r')
coF = h5py.File('../data/slices/co1_ta1e5.h5', 'r')

highres_n = 1024

count = 0
for f, ax in zip( [very_lowF, lowF, highF, coF], [ax4, ax3, ax2, ax1] ):
    print('plotting {}/3'.format(count+1))
    x, y = f['scales']['x']['1.0'], f['scales']['y']['1.0']
    x_basis = de.Fourier(  'x', len(x), interval=[0., 1], dealias=3/2)
    y_basis = de.Fourier(  'y', len(y), interval=[0., 1], dealias=3/2)
    bases = [x_basis, y_basis]
    domain = de.Domain(bases, grid_dtype=np.float64)
    base_scale=1
    big_scale=int(highres_n/len(x))
    s_field = domain.new_field()
        

    s = f['tasks']['s near top'][0,:,:,0]
    s -= s.mean()
    s_field.set_scales(base_scale, keep_data=False)
    s_field['g'] = s

    big_x, big_y = domain.grid(0, scales=big_scale), domain.grid(1, scales=big_scale)
    s_field.set_scales(big_scale, keep_data=True)
    big_s = s_field['g']
    mn, mx = big_s.min(), big_s.max()
    mn *= 0.9
    mx *= 0.9
    
    yy, xx = np.meshgrid(big_y, big_x)
    c = ax.pcolormesh(xx, yy, big_s, cmap='RdBu_r', rasterized=True, vmin=mn, vmax=mx)
    c.set_edgecolor('face')

    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xticks(())
    ax.set_yticks(())

    mn *= 1e5
    mx *= 1e5
    ax.annotate(r'$(S_{\mathrm{min}}, S_{\mathrm{max}}) = $' \
              + r'$({:.1f}, {:.1f})$'.format(mn, mx), 
              xy=(0.03, -0.09), fontsize=8, annotation_clip=False)
    count += 1

cax = plt.subplot(gs.new_subplotspec((0, 350), 75, 300))
bar = plt.colorbar(c, cax=cax, orientation='horizontal')
cax.set_xticklabels(())
bar.set_ticks(())
cax.annotate(r'$S_{\mathrm{min}} \times 10^{-5}$', fontsize=8,  xy=(-0.35, 0.25), annotation_clip=False)
cax.annotate(r'$S_{\mathrm{max}} \times 10^{-5}$', fontsize=8,  xy=(1.02, 0.25), annotation_clip=False)

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax4.text(0.23, 0.93, "Ro = 0.03", ha="center", va="center", size=8, bbox=bbox_props)
ax3.text(0.23, 0.93, "Ro = 0.13", ha="center", va="center", size=8, bbox=bbox_props)
ax2.text(0.23, 0.93, "Ro = 0.42", ha="center", va="center", size=8, bbox=bbox_props)
ax1.text(0.23, 0.93, "Ro = 2.01", ha="center", va="center", size=8, bbox=bbox_props)

print('saving png')
fig.savefig('../tex/figs/dynamics_plot.png', dpi=600, bbox_inches='tight')
print('saving pdf')
fig.savefig('../tex/figs/dynamics_plot.pdf', dpi=600, bbox_inches='tight')

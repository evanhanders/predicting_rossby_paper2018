import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

import dedalus.public as de

def nice_exp(number):
    string = '{:.2e}'.format(number)
    pre, exp = string.split('e')
    if exp[0] == '+': exp = exp[1:]
    exp = str(int(exp))
    full = '{} \\times 10^'.format(pre) + '{' + '{}'.format(exp) + '}'
    e_str = full.split('\\times ')[-1]
    return full, pre, e_str
    


fig = plt.figure(figsize=(8, 4.5))
gs     = gridspec.GridSpec(*(1000,1000))
#gs_info = (((100, 0), 900, 320), ((100, 340), 900, 320), ((100, 680),900, 320))
gs_info = (((100, 0), 400, 230), ((100, 250), 400, 230), ((100, 500), 400, 230), ((100, 750), 400, 230),
           ((510, 0), 400, 230), ((510, 250), 400, 230), ((510, 500), 400, 230), ((510, 750), 400, 230))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))
ax4 = plt.subplot(gs.new_subplotspec(*gs_info[3]))
ax5 = plt.subplot(gs.new_subplotspec(*gs_info[4]))
ax6 = plt.subplot(gs.new_subplotspec(*gs_info[5]))
ax7 = plt.subplot(gs.new_subplotspec(*gs_info[6]))
ax8 = plt.subplot(gs.new_subplotspec(*gs_info[7]))

#lowF = h5py.File('../data/slices/pro0.96_ra9.16e5_ta1e8_slices_s60.h5', 'r')
very_lowF = h5py.File('../data/slices/pro0.6_ra1.24e9_ta5.2e12_slices_s230.h5', 'r')
lowF = h5py.File('../data/slices/pro0.96_ra4.76e6_ta9.00e8_slices_s130.h5', 'r')
#highF = h5py.File('../data/slices/pro1.58_ra2.49e5_ta4.64e6_slices_s60.h5', 'r')
highF = h5py.File('../data/slices/pro1.58_ra7.89e4_ta1.00e6_slices_s80.h5', 'r')
#coF = h5py.File('../data/slices/rop6_ra6.77e2_slices_s80.h5', 'r')
coF = h5py.File('../data/slices/co1_ta4.39e3.h5', 'r')

highres_n = 1024

count = 0
for i, f, ax in zip( range(8), [very_lowF, lowF, highF, coF]*2, [ax4, ax3, ax2, ax1, ax8, ax7, ax6, ax5] ):
    print('plotting {}/8'.format(count+1))
    x, y = f['scales']['x']['1.0'], f['scales']['y']['1.0']
    x_basis = de.Fourier(  'x', len(x), interval=[0., 1], dealias=3/2)
    y_basis = de.Fourier(  'y', len(y), interval=[0., 1], dealias=3/2)
    bases = [x_basis, y_basis]
    domain = de.Domain(bases, grid_dtype=np.float64)
    base_scale=1
    big_scale=int(highres_n/len(x))
    s_field = domain.new_field()
        
    w_num = 10
    if i < 4:
        s = f['tasks']['s near top'][w_num,:,:,0]
    else:
        s = f['tasks']['s midplane'][w_num,:,:,0]
    s -= s.mean()
    s_field.set_scales(base_scale, keep_data=False)
    s_field['g'] = s

    big_x, big_y = domain.grid(0, scales=big_scale), domain.grid(1, scales=big_scale)
    s_field.set_scales(big_scale, keep_data=True)
    big_s = s_field['g']
    mn, mx = big_s.min(), big_s.max()
    mn *= 0.9
    mx *= 0.9

    max_abs = np.max([np.abs(mn), np.abs(mx)])
    min_abs = np.min([np.abs(mn), np.abs(mx)])
    mx = min_abs + (max_abs-min_abs)*0.25
    mn = -min_abs - (max_abs - min_abs)*0.25
    
    yy, xx = np.meshgrid(big_y, big_x)
    c = ax.pcolormesh(xx, yy, big_s, cmap='RdBu_r', rasterized=True, vmin=mn, vmax=mx)
    c.set_edgecolor('face')

    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xticks(())
    ax.set_yticks(())

    mn *= 1e5
    mx *= 1e5
#    ax.annotate(r'$(S_{\mathrm{min}}, S_{\mathrm{max}}) = $' \
#              + r'$({:.1f}, {:.1f})$'.format(mn, mx), 
#              xy=(0.03, -0.09), fontsize=8, annotation_clip=False)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
    ax.text(0.77, 0.09, r'$|S| = $' \
              + r'${:.2f}$'.format(mx), 
              fontsize=8, ha='center', va='center', bbox=bbox_props)
    count += 1

cax = plt.subplot(gs.new_subplotspec((35, 350), 43, 300))
bar = plt.colorbar(c, cax=cax, orientation='horizontal')
cax.set_xticklabels(())
bar.set_ticks(())
cax.annotate(r'$-|S| \times 10^{-5}$', fontsize=8,  xy=(-0.37, 0.5), va='center', annotation_clip=False)
cax.annotate(r'$|S| \times 10^{-5}$', fontsize=8,  xy=(1.02, 0.5),  va='center',  annotation_clip=False)

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
ax4.text(0.26, 0.91, "Ro = 0.026", ha="center", va="center", size=8, bbox=bbox_props)
ax3.text(0.24, 0.91, "Ro = 0.13", ha="center", va="center", size=8, bbox=bbox_props)
ax2.text(0.24, 0.91, "Ro = 0.42", ha="center", va="center", size=8, bbox=bbox_props)
ax1.text(0.24, 0.91, "Ro = 1.21", ha="center", va="center", size=8, bbox=bbox_props)
#ax8.text(0.26, 0.91, "Ro = 0.026", ha="center", va="center", size=8, bbox=bbox_props)
#ax7.text(0.24, 0.91, "Ro = 0.13", ha="center", va="center", size=8, bbox=bbox_props)
#ax6.text(0.24, 0.91, "Ro = 0.42", ha="center", va="center", size=8, bbox=bbox_props)
#ax5.text(0.24, 0.91, "Ro = 1.21", ha="center", va="center", size=8, bbox=bbox_props)
ax1.text(0.11, 0.09, "top", ha="center", va="center", size=8, bbox=bbox_props)
ax5.text(0.23, 0.91, "midplane", ha="center", va="center", size=8, bbox=bbox_props)

print('saving png')
fig.savefig('../tex/figs/dynamics_plot.png', dpi=400, bbox_inches='tight')
print('saving pdf')
fig.savefig('../tex/figs/dynamics_plot.pdf', dpi=400, bbox_inches='tight')

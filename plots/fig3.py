import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py


fig = plt.figure(figsize=(8, 3))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((50, 0), 900, 335), ((50, 335), 900, 335), ((50, 670),900, 330))
ax1 = plt.subplot(gs.new_subplotspec(*gs_info[0]))
ax2 = plt.subplot(gs.new_subplotspec(*gs_info[1]))
ax3 = plt.subplot(gs.new_subplotspec(*gs_info[2]))

lowF = h5py.File('../data/slices/pro0.96_ra9.16e5_ta1e8_slices_s60.h5', 'r')
highF = h5py.File('../data/slices/pro1.58_ra2.49e5_ta4.64e6_slices_s60.h5', 'r')
coF = h5py.File('../data/slices/co1_ta1e5.h5', 'r')

xL, yL = lowF['scales']['x']['1.0'], lowF['scales']['y']['1.0']
xH, yH = highF['scales']['x']['1.0'], highF['scales']['y']['1.0']
xC, yC = coF['scales']['x']['1.0'], coF['scales']['y']['1.0']

yyL, xxL = np.meshgrid(yL, xL)
yyH, xxH = np.meshgrid(yH, xH)
yyC, xxC = np.meshgrid(yC, xC)

ax3.pcolormesh(xxL, yyL,  lowF['tasks']['s near top'][0,:,:,0], cmap='RdBu_r')
ax2.pcolormesh(xxH, yyH, highF['tasks']['s near top'][0,:,:,0], cmap='RdBu_r')
ax1.pcolormesh(xxC, yyC,   coF['tasks']['s near top'][0,:,:,0], cmap='RdBu_r')

for ax in [ax1, ax2, ax3]:
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xticks(())
    ax.set_yticks(())

fig.savefig('../tex/figs/dynamics_plot.png', dpi=300, bbox_inches='tight')

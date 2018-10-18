import numpy as np
import matplotlib
#matplotlib settings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, interp2d, interp1d
from scipy.optimize import curve_fit, leastsq
matplotlib.rcParams['font.family'] = 'serif'

small_thresh = np.log10(0.3) #threshold for high/low ro
big_thresh = np.log10(0.3) #threshold for high/low ro
S_thresh = 2
Re_thresh = 1e2

#READ DATA
onset_data = np.genfromtxt('../data/eps1e-4_onsets.csv', skip_header=1, delimiter=',', usecols=(0,3))

onset_ta = onset_data[:,0]
onset_ra = onset_data[:,1]
onset_ra_func = interp1d(onset_ta, onset_ra, fill_value='extrapolate')



const_s = np.genfromtxt('constant_supercriticality.csv', skip_header=1, delimiter=',') 
ta_s = np.log10(const_s[:,6])
ra_s = np.log10(const_s[:,2])
re_s = const_s[:,3]
good_s = re_s >= Re_thresh
#good_s = 10**ra_s/onset_ra_func(10**ta_s) >= S_thresh
ro_c_s = np.log10(np.sqrt(const_s[good_s,2] / const_s[good_s,6])) #ra/ta
ro_m_s = np.log10(const_s[good_s,4])
ro_p_s = np.log10(np.sqrt(const_s[good_s, 0]))

const_co = np.genfromtxt('constant_co.csv', skip_header=1, delimiter=',') 
ta_co = np.log10(const_co[:,6])
ra_co = np.log10(const_co[:,2])
re_co = const_co[:,3]
good_co = re_co >= Re_thresh
#good_co = 10**ra_co/onset_ra_func(10**ta_co) >= S_thresh
ro_c_co = np.log10(np.sqrt(const_co[good_co,2] / const_co[good_co,6])) #ra/ta
ro_m_co = np.log10(const_co[good_co,4])
ro_p_co = np.log10(np.sqrt(const_co[good_co, 0]))

const_rop = np.genfromtxt('ra_ta0.75_AR4_alldata.csv', skip_header=1, delimiter=',')
ta_rop = np.log10(const_rop[:,2])
ra_rop = np.log10(const_rop[:,1])
re_rop = const_rop[:,4]
#good_rop = 10**ra_rop/onset_ra_func(10**ta_rop) >= S_thresh
good_rop = re_rop >= Re_thresh
ro_c_rop = np.log10(np.sqrt(const_rop[good_rop,1] / const_rop[good_rop,2])) #ra/ta
ro_m_rop = np.log10(const_rop[good_rop,5])
ro_p_rop = np.log10(const_rop[good_rop,0])

ta_full = np.concatenate((ta_s[good_s], ta_co[good_co], ta_rop[good_rop]))
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


    ro_m_interp = griddata((ro_c, ro_p), ro_m, (np.copy(ro_cc), np.copy(ro_pp)), method='linear')
    ro_m_func = fit[0] + ro_cc*(fit[1]) + ro_pp*(fit[2])

    return ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit

#full data set
ro_c, ro_m, ro_p = ro_c_full, ro_m_full, ro_p_full
ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit = run_analysis(ro_c, ro_m, ro_p)


fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


ro_pp2, ro_cc2 = np.meshgrid(ro_p_full, ro_c_full)
ro_m2 = np.zeros(ro_pp2.shape)
ro_m2 *= np.nan
ro_m2_mask = np.ones(ro_pp2.shape, dtype=bool)
for i in range(ro_pp2.shape[0]):
    for j in range(ro_pp2.shape[1]):
        if True in (ro_p_full == ro_pp2[i,j])*(ro_c_full == ro_cc2[i,j]):
            ro_m2[i,j] = ro_m_full[(ro_p_full == ro_pp2[i,j])*(ro_c_full == ro_cc2[i,j])]
            ro_m2_mask[i,j] = 0
ro_m2_true = np.ma.array(ro_m2, mask=ro_m2_mask)
interp = interp2d(ro_c_full, ro_p_full, ro_m2_true)

### Plot 1
plot = ax1.pcolormesh(ro_cc, ro_pp, ro_m_interp, cmap='viridis', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
ax1.plot(ro_c, ro_p, marker='s', lw=0, markersize=3, markerfacecolor='none', markeredgecolor='black')
ax1.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax1.set_ylabel(r'log$_{10}$(Ro$_p$)')


### Plot 2
plot = ax2.pcolormesh(ro_cc, ro_pp, ro_m_func,  cmap='viridis', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
bar = plt.colorbar(plot)
bar.set_label(r'log$_{10}$(Ro$_m$)')
ax2.set_title(r'{:s}'.format(fit_str))
ax2.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax2.set_ylabel(r'log$_{10}$(Ro$_p$)')

fig.savefig('ro_scaling_3d.png', dpi=300, bbox_inches='tight')

errors = []

for rc, rp, rm in zip(ro_c, ro_p, ro_m):
    calculated = ro_m_calc(fit, rc, rp)
    print("% err is {:.4e} for rc: {:.4g} / rp: {:.4g}".format(np.abs(100*(10**calculated - 10**rm)/10**(rm)), rc, rp))
    errors += [(rm, np.abs(100*(10**calculated - 10**rm)/10**(rm)))]

errors = np.array(errors)
plt.figure()
plt.scatter(10**errors[:,0], errors[:,1])
plt.yscale('log')
plt.xscale('log')
#plt.show()




fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

#low ro
ro_c, ro_m, ro_p, ta = ro_c_full[ro_m_full < small_thresh], ro_m_full[ro_m_full < small_thresh], ro_p_full[ro_m_full < small_thresh], ta_full[ro_m_full < small_thresh]
ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit = run_analysis(ro_c, ro_m, ro_p)

### Plot 1
plot = ax1.pcolormesh(ro_cc, ro_pp, ro_m_interp, cmap='plasma', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
ax1.plot(ro_c, ro_p, marker='s', lw=0, markersize=3, markerfacecolor='none', markeredgecolor='black')
ax1.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax1.set_ylabel(r'log$_{10}$(Ro$_p$)')


### Plot 2
plot = ax2.pcolormesh(ro_cc, ro_pp, ro_m_func,  cmap='plasma', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
bar = plt.colorbar(plot)
bar.set_label(r'log$_{10}$(Ro$_m$)')
ax2.set_title(r'{:s}'.format(fit_str))
ax2.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax2.set_ylabel(r'log$_{10}$(Ro$_p$)')

fig.savefig('ro_scaling_3d_lowro.png', dpi=300, bbox_inches='tight')


errors = []

for rc, rp, rm, t in zip(ro_c, ro_p, ro_m, ta):
    if rm >= small_thresh:
        continue

    calculated = ro_m_calc(fit, rc, rp)
    print("% err is {:.4e} for rc: {:.4g} / rp: {:.4g} / ta: {:.4g}".format(np.abs(100*(10**calculated - 10**rm)/10**(rm)), rc, rp, 10**t))
    errors += [(rm, np.abs(100*(10**calculated - 10**rm)/10**(rm)))]

errors = np.array(errors)
plt.figure()
plt.scatter(10**errors[:,0], errors[:,1])
plt.yscale('log')
plt.xscale('log')
#plt.show()

    



fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#high ro
ro_c, ro_m, ro_p = ro_c_full[ro_m_full >= big_thresh], ro_m_full[ro_m_full >= big_thresh], ro_p_full[ro_m_full >= big_thresh]
ro_cc, ro_pp, ro_m_interp, ro_m_func, fit_str, fit = run_analysis(ro_c, ro_m, ro_p)

### Plot 1
plot = ax1.pcolormesh(ro_cc, ro_pp, ro_m_interp, cmap='inferno', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
ax1.plot(ro_c, ro_p, marker='s', lw=0, markersize=3, markerfacecolor='none', markeredgecolor='black')
ax1.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax1.set_ylabel(r'log$_{10}$(Ro$_p$)')


### Plot 2
plot = ax2.pcolormesh(ro_cc, ro_pp, ro_m_func,  cmap='inferno', vmin=ro_m.min(), vmax=ro_m.max(), snap=True)
bar = plt.colorbar(plot)
bar.set_label(r'log$_{10}$(Ro$_m$)')
ax2.set_title(r'{:s}'.format(fit_str))
ax2.set_xlabel(r'log$_{10}$(Ro$_c$)')
ax2.set_ylabel(r'log$_{10}$(Ro$_p$)')

fig.savefig('ro_scaling_3d_highro.png', dpi=300, bbox_inches='tight')

errors = []

for rc, rp, rm in zip(ro_c, ro_p, ro_m):
    if rm < big_thresh:
        continue

    calculated = ro_m_calc(fit, rc, rp)
    print("% err is {:.4e} for rc: {:.4g} / rp: {:.4g}".format(np.abs(100*(10**calculated - 10**rm)/10**(rm)), rc, rp))
    errors += [(rm, np.abs(100*(10**calculated - 10**rm)/10**(rm)))]

errors = np.array(errors)
plt.figure()
plt.scatter(10**errors[:,0], errors[:,1])
plt.yscale('log')
plt.xscale('log')
#plt.show()



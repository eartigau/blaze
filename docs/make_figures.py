"""Build the figures used by README.md.

PNG, not the PDF the pipeline writes, because GitHub does not render PDF inline.

The model is rebuilt from the header of blaze_model.fits rather than re-fitted,
which doubles as a check that the header fully describes the model: the
reconstruction is compared to the stored array before anything is plotted.

    python docs/make_figures.py        (from the top of the repository)
"""
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLAZE_FILE = 'FDFCD50E5Ff_pp_blaze_AB.fits'
WAVE_FILE = 'FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits'
MODEL_FILE = 'blaze_model.fits'
HC_K = 1.438776877e7

SURFACE = '#fcfcfb'
INK = '#1c1c1a'
INK_SOFT = '#52514e'
GRID = '#e2e1dc'
OBS = '#2b2b29'          # the data, neutral ink
MODEL = '#eb6834'        # the model, one accent
BLUE = '#2a78d6'
RAMP = LinearSegmentedColormap.from_list('orders', ['#9fc6ee', '#0d3b66'])

mpl.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.size': 10,
    'text.color': INK, 'axes.labelcolor': INK_SOFT, 'axes.edgecolor': GRID,
    'xtick.color': INK_SOFT, 'ytick.color': INK_SOFT,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'axes.linewidth': .8,
    'grid.color': GRID, 'grid.linewidth': .7, 'legend.frameon': False,
    'legend.fontsize': 9, 'figure.dpi': 140,
})


def tidy(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis=grid, zorder=0)
    ax.set_axisbelow(True)


blaze = fits.getdata(BLAZE_FILE)
wave = fits.getdata(WAVE_FILE)
model, hdr = fits.getdata(MODEL_FILE, header=True)
index = np.arange(wave.shape[0])

cst, beta = hdr['MODCST'], hdr['MODBETA']
theta_b, teff = np.radians(hdr['MODTHETA']), hdr['MODTEFF']
npoly, m0 = hdr['MODNPOLY'], hdr['MODORD0']
coef = np.array([hdr['MODTR{}'.format(k)] for k in range(npoly + 1)])

valid = np.isfinite(blaze) & np.isfinite(wave) & (blaze > 0)
step = int(-np.sign(np.median(np.diff(wave[:, wave.shape[1] // 2]))))
morder = m0 + step * index
mm = np.repeat(morder[:, None], wave.shape[1], axis=1).astype(float)

# the four terms of the model, rebuilt from the header alone
photon = 1.0 / (wave ** 4 * np.expm1(HC_K / (wave * teff)))
ss = mm * wave / cst
arg = beta * mm * np.cos(theta_b) / ss * (
    ss * np.cos(theta_b) - np.sqrt(np.clip(1 - (ss * np.sin(theta_b)) ** 2, 1e-8, None)))
env = np.sinc(arg) ** 2
dwave = np.abs(np.gradient(wave, axis=1))
wmin, wmax = np.min(wave[valid]), np.max(wave[valid])
uu = 2 * (wave - wmin) / (wmax - wmin) - 1
trans = np.exp(np.sum([coef[k] * uu ** k for k in range(npoly + 1)], axis=0))

rebuilt = photon * np.maximum(env, 1e-8) * dwave * trans
assert np.allclose(rebuilt, model, rtol=2e-6), 'header does not describe the model'
print('model rebuilt from the header, max relative difference {:.1e}'.format(
    np.max(np.abs(rebuilt / model - 1))))

resid = 100 * (blaze / model - 1)
colour = RAMP(np.linspace(0, 1, index.size))

# --------------------------------------------------------------- 1. the fit
fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.3, 1], hspace=.12))
for i in index:
    ax[0].plot(wave[i], blaze[i], color=OBS, lw=1.5, alpha=.85, zorder=3)
    ax[0].plot(wave[i], model[i], color=MODEL, lw=1.0, zorder=4)
    ax[1].plot(wave[i], resid[i], color=OBS, lw=.8, alpha=.8, zorder=3)
ax[0].plot([], [], color=OBS, lw=1.5, label='observed blaze')
ax[0].plot([], [], color=MODEL, lw=1.2, label='model')
ax[0].set(ylabel='flux [ADU]', ylim=(0, 1.12 * np.max(blaze[valid])),
          xlim=(wave.min(), wave.max()))
ax[0].set_title('49 orders of a SPIRou flat lamp, and the model fitted to them',
                loc='left', color=INK)
ax[0].legend(loc='upper left', ncol=2)
ax[1].axhline(0, color=MODEL, lw=1)
ax[1].set(xlabel='wavelength [nm]', ylabel='obs / model - 1  [%]', ylim=(-25, 25))
for a in ax:
    tidy(a)
fig.savefig('docs/fit.png', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------- 2. diffraction orders
lam = wave[:, wave.shape[1] // 2]
trial = np.arange(m0 - 12, m0 + 13)
scat = np.array([np.std((t + step * index) * lam) / np.mean((t + step * index) * lam)
                 for t in trial])
peak_obs = morder * wave[index, np.nanargmax(np.nan_to_num(blaze), axis=1)]
peak_mod = morder * wave[index, np.nanargmax(np.nan_to_num(model), axis=1)]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(trial, scat, 'o-', color=BLUE, ms=4, lw=1.2, zorder=3)
ax[0].plot([m0], [scat[trial == m0]], 'o', color=MODEL, ms=9, zorder=4)
ax[0].annotate('m = {}'.format(m0), (m0, scat[trial == m0][0]),
               textcoords='offset points', xytext=(10, 6), color=MODEL, fontsize=10)
ax[0].set(yscale='log', xlabel='trial order of the first spectral order',
          ylabel=r'scatter of $m\lambda$ across orders')
ax[0].set_title('the order number is unambiguous', loc='left', color=INK)
ax[1].plot(morder, peak_obs, 'o-', color=OBS, ms=3.5, lw=1.2, label='observed', zorder=3)
ax[1].plot(morder, peak_mod, 'o-', color=MODEL, ms=3.5, lw=1.2,
           label='model, C held constant', zorder=4)
ax[1].axhline(cst, color=BLUE, ls='--', lw=1, label='fitted C')
ax[1].set(xlabel='diffraction order', ylabel=r'$m\,\lambda_{\rm peak}$ [nm]')
ax[1].set_title('the drift of the peak is the lamp, not the grating', loc='left', color=INK)
ax[1].legend(loc='lower right')
for a in ax:
    tidy(a)
fig.savefig('docs/orders.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------- 3. the components
srt = np.argsort(wave[valid])
fig, ax = plt.subplots(4, 1, figsize=(9, 8.4), sharex=True,
                       gridspec_kw=dict(hspace=.42))
ax[0].plot(wave[valid][srt], photon[valid][srt] / photon[valid].max(), color=BLUE, lw=1.6)
ax[0].set(ylabel='photons / nm')
ax[0].set_title('blackbody at {:.0f} K, in photon density'.format(teff),
                loc='left', color=INK)
ax[1].plot(wave[valid][srt], trans[valid][srt] / trans[valid].max(), color=BLUE, lw=1.6)
ax[1].set(ylabel='transmission', yscale='log')
ax[1].set_title('instrument transmission, exp of a degree {} polynomial'.format(npoly),
                loc='left', color=INK)
for i in index:
    ax[2].plot(wave[i], env[i], color=colour[i], lw=1.1)
ax[2].set(ylabel='blaze envelope', ylim=(0, 1.05))
ax[2].set_title(r'grating blaze, sinc$^2$ of the offset to the peak in orders',
                loc='left', color=INK)
for i in index:
    ax[3].plot(wave[i], dwave[i], color=colour[i], lw=1.1)
ax[3].set(xlabel='wavelength [nm]', ylabel=r'|d$\lambda$/dpix| [nm]')
ax[3].set_title('pixel width, the wavelength solution is not linear',
                loc='left', color=INK)
for a in ax:
    tidy(a)
fig.savefig('docs/components.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------ 4. a few orders
show = [np.argmin(np.abs(morder - m)) for m in (70, 55, 40)]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, i in zip(ax, show):
    g = valid[i]
    a.plot(wave[i][g], blaze[i][g] / 1e5, color=OBS, lw=2.2, alpha=.85, zorder=3)
    a.plot(wave[i][g], model[i][g] / 1e5, color=MODEL, lw=1.2, zorder=4)
    a.set_title('order m = {}'.format(morder[i]), loc='left', color=INK)
    a.set_xlabel('wavelength [nm]')
    tidy(a)
ax[0].set_ylabel(r'flux [10$^5$ ADU]')
ax[0].plot([], [], color=OBS, lw=2.2, label='observed')
ax[0].plot([], [], color=MODEL, lw=1.2, label='model')
ax[0].legend(loc='lower center')
fig.savefig('docs/orders_zoom.png', bbox_inches='tight')
plt.close(fig)
print('wrote docs/fit.png, docs/orders.png, docs/components.png, docs/orders_zoom.png')

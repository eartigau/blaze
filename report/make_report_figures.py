"""Figures of the report, from the results written by run_analysis.py.
All figures are PDF, in figures/."""
import json
import os

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from blaze_like import BlazeLikelihood
from run_analysis import INSTRUMENTS, REPO, full_residual

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

SURFACE = 'white'
INK = '#1c1c1a'
INK_SOFT = '#52514e'
GRID = '#e2e1dc'
OBS = '#2b2b29'
MODEL = '#eb6834'
BLUE = '#2a78d6'
AQUA = '#1baf7a'
RAMP = LinearSegmentedColormap.from_list('orders', ['#9fc6ee', '#0d3b66'])
LABELS = {'c0': r'$C(\lambda_{\rm ref})$ [nm]', 'c1': r'$c_1 = dC/d\lambda$',
          'beta': r'$\beta$', 'asym': r'$\alpha$', 'lns': r'$\ln s$'}
COLOUR = {'SPIRou': BLUE, 'NIRPS': MODEL}

mpl.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK_SOFT, 'axes.edgecolor': '#bdbcb6',
    'xtick.color': INK_SOFT, 'ytick.color': INK_SOFT, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'axes.linewidth': .7,
    'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'axes.axisbelow': True, 'pdf.fonttype': 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def load(inst):
    work = os.path.join(HERE, 'work', inst)
    path = os.path.join(work, 'results.json')
    if not os.path.exists(path):
        return None
    js = json.load(open(path))
    npz = dict(np.load(os.path.join(work, 'results.npz')))
    cfg = INSTRUMENTS[inst]
    L = BlazeLikelihood(os.path.join(REPO, cfg['blaze']), os.path.join(REPO, cfg['wave']),
                        os.path.join(work, 'blaze_model.fits'))
    r, _ = full_residual(L, np.append(L.best, 0.0))
    L.set_use(L.fitted & (np.abs(r) < 5 * np.std(r[L.fitted])))
    L.corr_length = js['corr_length']
    L.nu = js['nu']
    return dict(js=js, npz=npz, L=L)


data = {inst: load(inst) for inst in INSTRUMENTS}
data = {k: v for k, v in data.items() if v is not None}

# ------------------------------------------------------ order identification
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for inst, d in data.items():
    z = d['npz']
    m0 = int(z['ident_m_first'])
    sel = np.abs(z['ident_trial'] - m0) <= 15
    ax[0].plot(z['ident_trial'][sel] - m0, z['ident_scatter'][sel], 'o-', ms=3, lw=1,
               color=COLOUR[inst], label='{}, $m_0$ = {}'.format(inst, m0))
    ax[1].plot(z['ident_trial_c'] - m0, z['ident_scatter_c'], 'o-', ms=3, lw=1,
               color=COLOUR[inst], label=inst)
ax[0].set(yscale='log', xlabel=r'trial $m_0$ minus adopted value',
          ylabel=r'rms/mean of $m\,\lambda_{\rm peak}$')
ax[0].set_title(r'mkblaze.py: $\lambda$ at the blaze maximum', loc='left')
ax[1].set(yscale='log', xlabel=r'trial $m_0$ minus adopted value',
          ylabel=r'rms/mean of $m\,\lambda_{\rm centre}$')
ax[1].set_title(r'fit_blaze.py: $\lambda$ at the central pixel', loc='left')
ax[0].legend()
save(fig, 'orders_scan.pdf')

fig, ax = plt.subplots(len(data), 1, figsize=(7.2, 2.3 * len(data)), squeeze=False)
for a, (inst, d) in zip(ax[:, 0], data.items()):
    z = d['npz']
    m = z['ident_morder']
    a.plot(m, z['ident_m_overlap'] - m, 'o', ms=3, color=BLUE,
           label=r'overlap, $\lambda_{k+1}/(\lambda_{k+1}-\lambda_k)$')
    a.plot(m, z['ident_m_gradient'] - m, 's', ms=2.6, color=MODEL,
           label=r'gradient, $\lambda/|d\lambda/dk|$')
    a.axhline(0, color=INK_SOFT, lw=.8)
    for y in (-0.5, 0.5):
        a.axhline(y, color=INK_SOFT, lw=.6, ls=':')
    a.set(ylabel=r'estimate $-\ m$', ylim=(-1.5, 1.5))
    a.set_title('{}: direct estimates of the order number'.format(inst), loc='left')
    a.legend(loc='lower left', ncol=2)
ax[-1, 0].set_xlabel('diffraction order $m$')
save(fig, 'orders_direct.pdf')

# ----------------------------------------------------------- the fit itself
for inst, d in data.items():
    L, js, z = d['L'], d['js'], d['npz']
    th = z['theta_map']
    model, kx, ky = L.full_model(th)
    rel = 100 * (L.blaze / model - 1)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1], hspace=.08))
    for i in range(L.nord):
        ax[0].plot(L.wave[i], L.blaze[i], color=OBS, lw=1.3, alpha=.85)
        ax[0].plot(L.wave[i], model[i], color=MODEL, lw=.7)
        ax[1].plot(L.wave[i], rel[i], color=OBS, lw=.6, alpha=.8)
    ax[0].plot([], [], color=OBS, lw=1.3, label='observed blaze')
    ax[0].plot([], [], color=MODEL, lw=1, label='model (MAP)')
    ax[0].set(ylabel='flux [ADU]', ylim=(0, 1.12 * np.nanmax(L.blaze)))
    ax[0].legend(loc='upper left', ncol=2)
    ax[1].axhline(0, color=MODEL, lw=.8)
    ax[1].set(xlabel='wavelength [nm]', ylabel=r'obs/model $-$ 1 [%]', ylim=(-15, 15),
              xlim=(np.nanmin(L.wave), np.nanmax(L.wave)))
    save(fig, 'fit_{}.pdf'.format(inst))

    # components
    photon = L.photon
    m = L.morder.astype(float)[:, None]
    env = np.exp(L.log_envelope_2d(L.wave, m, th))
    spl = L.spline(th)[0]
    trans = np.exp(spl(L.wave.ravel()).reshape(L.wave.shape))
    col = RAMP(np.linspace(0, 1, L.nord))
    v = L.valid
    srt = np.argsort(L.wave[v])
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(hspace=.3))
    ax[0].plot(L.wave[v][srt], photon[v][srt] / photon[v].max(), color=BLUE, lw=1.4)
    ax[0].set(ylabel='photons / nm')
    ax[0].set_title('blackbody at {:.0f} K, photon density'.format(js['teff']), loc='left')
    tn = trans[v].max()
    ax[1].plot(L.wave[v][srt], trans[v][srt] / tn, color=BLUE, lw=1.4)
    ax[1].plot(kx, np.exp(ky) / tn, 'o', color=MODEL, ms=3, label='knots, one per order peak')
    ax[1].set(ylabel='transmission', yscale='log')
    ax[1].set_title('transmission, linear spline in log through the order peaks', loc='left')
    ax[1].legend(loc='lower right')
    for i in range(L.nord):
        ax[2].plot(L.wave[i], env[i], color=col[i], lw=.8)
    ax[2].set(ylabel='envelope', ylim=(0, 1.05))
    ax[2].set_title(r'grating envelope, sinc$^2[\beta m(e+\alpha e^2)]$', loc='left')
    for i in range(L.nord):
        ax[3].plot(L.wave[i], L.dwave[i], color=col[i], lw=.8)
    ax[3].set(ylabel=r'$|d\lambda/dx|$ [nm]', xlabel='wavelength [nm]')
    ax[3].set_title('pixel width', loc='left')
    save(fig, 'components_{}.pdf'.format(inst))

    # three orders close up
    picks = [int(np.nanpercentile(L.morder, q)) for q in (80, 50, 20)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for a, mm in zip(ax, picks):
        i = int(np.argmin(np.abs(L.morder - mm)))
        g = L.valid[i]
        a.plot(L.wave[i][g], L.blaze[i][g] / 1e5, color=OBS, lw=2, alpha=.85)
        a.plot(L.wave[i], model[i] / 1e5, color=MODEL, lw=1)
        a.set_xlim(L.wave[i][g].min() - 2, L.wave[i][g].max() + 2)
        a.set_title('order {}'.format(L.morder[i]), loc='left')
        a.set_xlabel('wavelength [nm]')
    ax[0].set_ylabel(r'flux [10$^5$ ADU]')
    save(fig, 'zoom_{}.pdf'.format(inst))

    # peaks: position in m*lambda and offsets in pixels
    chain = z['chain']
    wpk = L.wave[np.arange(L.nord), L.peaks]
    samp = chain[np.random.default_rng(1).choice(len(chain), 400, replace=False)]
    cband = np.array([s[0] + s[1] * (wpk - L.lref) for s in samp])
    model_const = L.full_model(z['theta_const'])[0]
    idx = np.arange(L.nord)
    pk_obs = L.morder * L.wave[idx, np.nanargmax(np.where(L.valid, L.blaze, np.nan), axis=1)]
    pk_map = L.morder * L.wave[idx, np.nanargmax(np.where(L.valid, model, np.nan), axis=1)]
    pk_con = L.morder * L.wave[idx, np.nanargmax(np.where(L.valid, model_const, np.nan), axis=1)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax[0].fill_between(L.morder, *np.percentile(cband, [2.3, 97.7], axis=0), color=BLUE,
                       alpha=.25, lw=0, label=r'$C(\lambda_{\rm peak})$, 95% band')
    ax[0].plot(L.morder, pk_obs, 'o-', color=OBS, ms=2.5, lw=.9, label='observed')
    ax[0].plot(L.morder, pk_map, 'o-', color=MODEL, ms=2.5, lw=.9, label=r'model, chromatic $C$')
    ax[0].plot(L.morder, pk_con, 'o-', color=AQUA, ms=2.5, lw=.9, label=r'model, constant $C$')
    ax[0].set(xlabel='diffraction order', ylabel=r'$m\,\lambda_{\rm peak}$ [nm]')
    lo = np.nanpercentile(pk_obs, 5)
    ax[0].set_ylim(lo - 0.3 * (np.nanmax(pk_obs) - lo), None)
    ax[0].legend(loc='lower right')
    ax[1].axhline(0, color=INK_SOFT, lw=.8)
    ax[1].plot(L.morder, z['p_obs'] - z['p_const'], 'o-', color=AQUA, ms=2.5, lw=.9,
               label=r'constant $C$')
    ax[1].plot(L.morder, z['p_obs'] - z['p_map'], 'o-', color=MODEL, ms=2.5, lw=.9,
               label=r'chromatic $C$')
    ax[1].set(xlabel='diffraction order', ylabel='observed $-$ model peak [pixels]',
              ylim=(-260, 120))
    ax[1].legend(loc='lower right')
    save(fig, 'peaks_{}.pdf'.format(inst))

    # per-order quality
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.2))
    ax.bar(L.morder, 100 * z['per_order'], color=COLOUR[inst], width=.8)
    ax.set(xlabel='diffraction order', ylabel=r'rms of obs/model $-$ 1 [%]', yscale='log')
    ax.axhline(js['median_per_order'], color=INK_SOFT, ls='--', lw=.8)
    save(fig, 'per_order_{}.pdf'.format(inst))

    # corner
    names = ['c0', 'c1', 'beta', 'asym', 'lns']
    c_off = 10 * np.floor(np.median(chain[:, 0]) / 10)
    shifted = chain.copy()
    shifted[:, 0] -= c_off
    truths = np.array(z['theta_map'], float)
    truths[0] -= c_off
    labels = [r'$C(\lambda_{{\rm ref}}) - {:.0f}$ [nm]'.format(c_off)] + [LABELS[n] for n in names[1:]]
    fig = corner.corner(shifted, labels=labels, rasterized=True,
                        truths=truths, truth_color=MODEL, color=COLOUR[inst],
                        quantiles=[0.15865, 0.5, 0.84135], show_titles=False,
                        label_kwargs=dict(fontsize=9), max_n_ticks=3, smooth=1.0)
    fig.set_size_inches(7.2, 7.2)
    axes = np.array(fig.axes).reshape(len(names), len(names))
    for k in range(len(names)):
        q = np.percentile(shifted[:, k], [15.865, 50, 84.135])
        err = 0.5 * (q[2] - q[0])
        nd = max(0, int(-np.floor(np.log10(err))) + 1)
        axes[k, k].set_title('${0:.{3}f}^{{+{1:.{3}f}}}_{{-{2:.{3}f}}}$'.format(
            q[1], q[2] - q[1], q[1] - q[0], nd), fontsize=8)
    save(fig, 'corner_{}.pdf'.format(inst))

    # chains
    full = z['full_chain']
    fig, ax = plt.subplots(5, 1, figsize=(7.2, 6.0), sharex=True, gridspec_kw=dict(hspace=.15))
    for k, n in enumerate(names):
        steps = np.arange(full.shape[0])[::5]
        ax[k].plot(steps, full[::5, :, k], color=COLOUR[inst], lw=.3, alpha=.35, rasterized=True)
        ax[k].axvline(js['burn'], color=INK_SOFT, ls='--', lw=.8)
        ax[k].set_ylabel(LABELS[n])
        ax[k].grid(False)
    ax[-1].set_xlabel('step')
    save(fig, 'chains_{}.pdf'.format(inst))

    # jackknife
    jack = z['jack']
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
    for a, k in zip(ax.ravel(), range(4)):
        a.plot(L.morder, jack[:, k] - z['theta_map'][k], 'o', ms=2.8, color=COLOUR[inst])
        a.axhline(0, color=MODEL, lw=.9)
        a.set_ylabel([r'$\Delta C(\lambda_{\rm ref})$ [nm]', r'$\Delta c_1$',
                      r'$\Delta\beta$', r'$\Delta\alpha$'][k])
        a.ticklabel_format(axis='y', useOffset=False)
    for a in ax[-1]:
        a.set_xlabel('order left out')
    save(fig, 'jackknife_{}.pdf'.format(inst))

# ------------------------------------------------------ residual statistics
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for inst, d in data.items():
    z, js = d['npz'], d['js']
    ax[0].plot(z['lags'], z['acf'], 'o-', ms=3, lw=1, color=COLOUR[inst],
               label='{}, $\\ell$ = {:.0f} px'.format(inst, js['corr_length']))
    ax[0].axvline(js['corr_length'], color=COLOUR[inst], ls=':', lw=.9)
    r = z['res_d']
    bins = np.linspace(-0.12, 0.12, 121)
    h, e = np.histogram(r, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax[1].step(c, h, where='mid', color=COLOUR[inst], lw=1, label=inst)
    ax[1].plot(c, stats.t.pdf(c, js['nu'], 0, js['t_scale']), color=COLOUR[inst], lw=.8, ls='--')
    ax[1].plot(c, stats.norm.pdf(c, 0, js['res_std']), color=COLOUR[inst], lw=.8, ls=':')
ax[0].axhline(np.exp(-1), color=INK_SOFT, lw=.7, ls='--')
ax[0].set(xscale='log', xlabel='lag [pixels]', ylabel='autocorrelation', ylim=(-0.6, 1.2))
ax[0].legend(loc='lower left')
ax[1].plot([], [], color=INK_SOFT, ls='--', lw=.8, label='Student-t fit')
ax[1].plot([], [], color=INK_SOFT, ls=':', lw=.8, label='Gaussian, same rms')
ax[1].set(yscale='log', xlabel=r'$\ln$(obs/model)', ylabel='density', ylim=(1e-2, None))
ax[1].legend(loc='upper right')
save(fig, 'residual_stats.pdf')

# --------------------------------------------------------------- Teff profile
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
for inst, d in data.items():
    tp = d['npz']['teff_prof']
    js = d['js']
    ref = tp[np.argmin(np.abs(tp[:, 0] - js['teff']))]
    sig = lambda k: 0.5 * (js['lo'][k] + js['hi'][k])
    ax[0].plot(tp[:, 0], tp[:, 1] - js['lp_map'], 'o-', ms=3, color=COLOUR[inst], label=inst)
    ax[1].plot(tp[:, 0], (tp[:, 3] - ref[3]) / sig('c1'), 'o-', ms=3, color=COLOUR[inst])
    ax[2].plot(tp[:, 0], (tp[:, 5] - ref[5]) / sig('asym'), 'o-', ms=3, color=COLOUR[inst])
for a in ax:
    a.set(xscale='log', xlabel=r'$T_{\rm eff}$ [K]')
ax[0].set_ylabel(r'$\ln\mathcal{L}(T) - \ln\mathcal{L}(5000\,{\rm K})$')
for a in ax[1:]:
    a.ticklabel_format(axis='y', useOffset=False)
ax[1].set_ylabel(r'shift of $c_1$ [posterior $\sigma$]')
ax[2].set_ylabel(r'shift of $\alpha$ [posterior $\sigma$]')
ax[0].legend(loc='upper left')
save(fig, 'teff_profile.pdf')

# ------------------------------------------------------------ C(lambda) bands
fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))
for inst, d in data.items():
    L, z = d['L'], d['npz']
    lam = np.linspace(L.wmin, L.wmax, 200)
    samp = z['chain'][np.random.default_rng(2).choice(len(z['chain']), 500, replace=False)]
    rel = np.array([1e2 * ((s[0] + s[1] * (lam - L.lref)) / s[0] - 1) for s in samp])
    ax.fill_between(lam, *np.percentile(rel, [2.3, 97.7], axis=0), color=COLOUR[inst], alpha=.3,
                    lw=0)
    ax.plot(lam, np.median(rel, axis=0), color=COLOUR[inst], lw=1.2,
            label=r'{}, $\lambda_{{\rm ref}}$ = {:.0f} nm'.format(inst, L.lref))
ax.axhline(0, color=INK_SOFT, lw=.7)
ax.set(xlabel='wavelength [nm]', ylabel=r'$C(\lambda)/C(\lambda_{\rm ref}) - 1$ [%]')
ax.legend()
save(fig, 'c_lambda.pdf')

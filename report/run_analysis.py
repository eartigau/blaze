"""Everything the report needs, for every instrument whose files are present:

  1. the order identification of mkblaze.py, in full, plus the gradient
     estimator used by fit_blaze.py
  2. the best fit, by running fit_blaze.py itself
  3. the residual statistics the likelihood is built on
  4. an MCMC of c0, c1, beta, asym and the residual scale
  5. a jackknife over orders, profiles in Teff, c1 = 0 and the Littrow asym,
     and the peak offsets with a constant and a chromatic C

Results go to work/<instrument>/results.npz and results.json.

    python run_analysis.py [instrument ...]
"""
import json
import os
import subprocess
import sys
import time

import emcee
import numpy as np
from astropy.io import fits
from scipy import stats
from scipy.optimize import minimize

from blaze_like import BlazeLikelihood, PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INSTRUMENTS = {
    'SPIRou': dict(blaze='FDFCD50E5Ff_pp_blaze_AB.fits',
                   wave='FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits',
                   littrow_tan=2.0),
    'NIRPS': dict(blaze='NIRPS_2023-04-01T12_07_06_942_pp_blaze_A.fits',
                  wave='NIRPS_2025-09-13T21_30_32_350_pp_e2dsff_A_wave_night_A.fits',
                  littrow_tan=4.0),
}
TEFF_GRID = [2500., 3000., 3500., 4000., 5000., 6000., 8000., 12000., 30000.]
NWALKERS = 32
CHUNK = 2000
MAX_STEPS = 30000
SCALE = np.array([1.0, 1e-3, 1e-3, 0.05, 0.01])      # typical parameter steps


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- mkblaze
def order_identification(blaze, wave):
    """mkblaze.py, step by step, plus the gradient estimator of fit_blaze.py"""
    index = np.arange(wave.shape[0])
    wpeak = wave[index, np.nanargmax(np.nan_to_num(blaze, nan=-np.inf), axis=1)]
    trial = np.arange(index.size + 1, 200)
    cst = (trial[:, None] - index[None, :]) * wpeak[None, :]
    scatter = np.std(cst, axis=1) / np.mean(cst, axis=1)
    m_first = int(trial[np.argmin(scatter)])
    morder = m_first - index
    lam = wave[:, wave.shape[1] // 2]
    m_overlap = lam[1:] / (lam[1:] - lam[:-1])
    m_overlap = np.append(m_overlap, m_overlap[-1] - 1)
    m_gradient = lam / np.abs(np.gradient(lam))
    # fit_blaze.py: same flatness test, on the centre-pixel wavelength
    step = int(-np.sign(np.median(np.diff(lam))))
    start = int(np.round(np.median(m_gradient - step * index)))
    trial_c = np.arange(start - 12, start + 13)
    scatter_c = np.array([np.std((t + step * index) * lam) / np.mean((t + step * index) * lam)
                          for t in trial_c])
    srt = np.sort(scatter)
    return dict(index=index, wpeak=wpeak, trial=trial, scatter=scatter, m_first=m_first,
                morder=morder, mlam=morder * wpeak, lam_centre=lam, m_overlap=m_overlap,
                m_gradient=m_gradient, trial_c=trial_c, scatter_c=scatter_c,
                scatter_best=srt[0], scatter_second=srt[1],
                runner_up=int(trial[np.argsort(scatter)[1]]),
                start_gradient=start, step=step,
                mlam_mean=float(np.mean(morder * wpeak)))


# ---------------------------------------------------------------- helpers
def run_fit(inst, cfg, workdir):
    os.makedirs(workdir, exist_ok=True)
    out = subprocess.run([sys.executable, os.path.join(REPO, 'fit_blaze.py'),
                          os.path.join(REPO, cfg['blaze']), os.path.join(REPO, cfg['wave'])],
                         capture_output=True, text=True, cwd=workdir, check=True).stdout
    open(os.path.join(workdir, 'fit_blaze.log'), 'w').write(out)
    return out


def full_residual(L, theta):
    model = L.full_model(theta)[0]
    r = np.full(L.blaze.shape, np.nan)
    r[L.valid] = np.log(L.blaze[L.valid] / model[L.valid])
    return r, model


def correlation(r, lags):
    """autocorrelation of the residual along the orders, each order's mean
    removed, pooled over orders. Pearson normalisation, over the pairs that
    actually overlap at each lag, so it stays within [-1, 1]."""
    out = []
    for lag in lags:
        num = da = db = 0.0
        for row in r:
            x = row - np.nanmean(row)
            a, b = x[:-lag], x[lag:]
            ok = np.isfinite(a) & np.isfinite(b)
            num += np.sum(a[ok] * b[ok])
            da += np.sum(a[ok] ** 2)
            db += np.sum(b[ok] ** 2)
        out.append(num / np.sqrt(da * db))
    return np.array(out)


def optimise(L, theta0, fixed=None):
    """maximum of the posterior, optionally with some parameters held fixed.

    Nelder-Mead in coordinates scaled by SCALE, which is of the order of the
    posterior widths. The simplex is set explicitly and large: the knots are
    medians, so the posterior is flat on very small scales, and scipy's
    default simplex (0.025% of the start) would never leave the start point.
    Restarted with shrinking simplices until the optimum stops moving."""
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    base = np.array(theta0, float)
    for i, v in fixed.items():
        base[i] = v

    def full(x):
        th = base.copy()
        th[free] = base[free] + x * SCALE[free]
        return th

    def neg(y):
        v = -L.log_post(full(y))
        return v if np.isfinite(v) else 1e30

    x = np.zeros(len(free))
    best = neg(x)
    for size in (3.0, 1.0, 1.0, 0.3, 0.3, 0.1):
        simplex = np.vstack([x] + [x + size * np.eye(len(free))[k] for k in range(len(free))])
        res = minimize(neg, x, method='Nelder-Mead',
                       options=dict(maxiter=6000, xatol=1e-4, fatol=1e-6, adaptive=True,
                                    initial_simplex=simplex))
        if res.fun < best:
            x, best = res.x, res.fun
    return full(x), -best


def run_mcmc(L, theta0):
    ndim = 5
    rng = np.random.default_rng(42)
    p0 = theta0 + SCALE * rng.normal(size=(NWALKERS, ndim))
    sampler = emcee.EnsembleSampler(NWALKERS, ndim, L.log_post)
    state = p0
    tau_hist = []
    while sampler.iteration < MAX_STEPS:
        state = sampler.run_mcmc(state, CHUNK, progress=False)
        tau = sampler.get_autocorr_time(tol=0)
        tau_hist.append((sampler.iteration, tau.copy()))
        log('   {} steps, autocorrelation times {}'.format(
            sampler.iteration, np.array2string(tau, precision=0)))
        if len(tau_hist) > 1:
            prev = tau_hist[-2][1]
            if sampler.iteration > 50 * np.max(tau) and np.all(np.abs(prev - tau) / tau < 0.02):
                break
    tau = sampler.get_autocorr_time(tol=0)
    burn = int(3 * np.max(tau))
    thin = max(1, int(np.min(tau) / 2))
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    return dict(chain=chain, logp=logp, full_chain=sampler.get_chain(),
                tau=tau, burn=burn, thin=thin, nsteps=sampler.iteration,
                acceptance=float(np.mean(sampler.acceptance_fraction)),
                tau_hist_steps=np.array([t[0] for t in tau_hist]),
                tau_hist=np.array([t[1] for t in tau_hist]))


def peak_positions(blaze, model, valid):
    """sub-pixel peak of each order, parabola on log flux over the top 15%"""
    out = np.full(blaze.shape[0], np.nan)
    for i in range(blaze.shape[0]):
        g = valid[i]
        top = g & (blaze[i] > 0.85 * np.nanmax(np.where(g, blaze[i], np.nan)))
        x = np.where(top)[0]
        if x.size < 20:
            continue
        p = np.polyfit(x - x.mean(), np.log(model[i][x]), 2)
        out[i] = -p[1] / (2 * p[0]) + x.mean()
    return out


# ---------------------------------------------------------------- analysis
def analyse(inst, cfg):
    workdir = os.path.join(HERE, 'work', inst)
    log('{}: order identification'.format(inst))
    blaze = fits.getdata(os.path.join(REPO, cfg['blaze']))
    wave = fits.getdata(os.path.join(REPO, cfg['wave']))
    ident = order_identification(blaze, wave)

    log('{}: best fit with fit_blaze.py'.format(inst))
    fit_log = run_fit(inst, cfg, workdir)
    model_file = os.path.join(workdir, 'blaze_model.fits')
    L = BlazeLikelihood(os.path.join(REPO, cfg['blaze']), os.path.join(REPO, cfg['wave']),
                        model_file)

    # clipping as in fit_blaze.py, at the best fit
    r, _ = full_residual(L, np.append(L.best, 0.0))
    use = L.fitted & (np.abs(r) < 5 * np.std(r[L.fitted]))
    L.set_use(use)
    r, model_best = full_residual(L, np.append(L.best, 0.0))

    lags = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700,
                     800, 900, 1000, 1200, 1500, 2000])
    rr = np.where(use, r, np.nan)
    acf = correlation(rr, lags)
    below = np.where(acf < np.exp(-1))[0][0]
    corr_length = float(np.interp(np.exp(-1), acf[[below, below - 1]], lags[[below, below - 1]]))
    L.corr_length = corr_length
    n_eff = use.sum() / corr_length
    log('{}: residual correlation length {:.0f} px, {:.0f} effective samples'.format(
        inst, corr_length, n_eff))

    res_d = L.residuals(np.append(L.best, 0.0))
    nu, loc, scale = stats.t.fit(res_d, floc=0)
    L.nu = nu
    log('{}: Student-t nu = {:.2f}, scale = {:.4f}'.format(inst, nu, scale))

    theta0 = np.append(L.best, np.log(scale))
    log('{}: maximum a posteriori'.format(inst))
    theta_map, lp_map = optimise(L, theta0)

    log('{}: MCMC'.format(inst))
    t0 = time.time()
    mc = run_mcmc(L, theta_map)
    log('{}: MCMC done in {:.0f} s, {} steps, acceptance {:.2f}'.format(
        inst, time.time() - t0, mc['nsteps'], mc['acceptance']))
    chain = mc['chain']
    der = L.derived(chain.T)
    med = np.median(chain, axis=0)
    lo, hi = np.percentile(chain, [15.865, 84.135], axis=0)

    log('{}: jackknife over orders'.format(inst))
    jack = []
    for j in range(L.nord):
        L.set_drop([j])
        th, _ = optimise(L, theta_map)
        jack.append(th)
    L.set_drop([])
    jack = np.array(jack)
    n = len(jack)
    jack_sigma = np.sqrt((n - 1) / n * np.sum((jack - jack.mean(0)) ** 2, axis=0))
    jack_der = L.derived(jack.T)
    jack_der_sigma = {k: float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
                      for k, v in jack_der.items()}

    log('{}: profiles'.format(inst))
    _, lp_const = optimise(L, theta_map, fixed={1: 0.0})
    theta_const, _ = optimise(L, theta_map, fixed={1: 0.0})
    if cfg.get('littrow_tan') is not None:
        asym_littrow = cfg['littrow_tan'] ** 2 / 2 - 1
        theta_littrow, lp_littrow = optimise(L, theta_map, fixed={3: asym_littrow})
    else:
        asym_littrow, theta_littrow, lp_littrow = np.nan, theta_map * np.nan, np.nan
    teff_prof = []
    for teff in TEFF_GRID:
        L.set_teff(teff)
        th, lp = optimise(L, theta_map)
        teff_prof.append((teff, lp, *th))
    L.set_teff(L.hdr['MODTEFF'])
    teff_prof = np.array(teff_prof)

    log('{}: peak offsets'.format(inst))
    model_map = L.full_model(theta_map)[0]
    model_const = L.full_model(theta_const)[0]
    p_obs = peak_positions(L.blaze, L.blaze, L.valid)
    p_map = peak_positions(L.blaze, model_map, L.valid)
    p_const = peak_positions(L.blaze, model_const, L.valid)
    rel = L.blaze / model_map - 1
    per_order = np.array([np.std(rel[i][L.fitted[i]]) if L.fitted[i].any() else np.nan
                          for i in range(L.nord)])

    np.savez(os.path.join(workdir, 'results.npz'),
             chain=chain, logp=mc['logp'], full_chain=mc['full_chain'], tau=mc['tau'],
             tau_hist=mc['tau_hist'], tau_hist_steps=mc['tau_hist_steps'],
             jack=jack, teff_prof=teff_prof, lags=lags, acf=acf, res_d=res_d,
             p_obs=p_obs, p_map=p_map, p_const=p_const, morder=L.morder,
             per_order=per_order, theta_map=theta_map, theta_const=theta_const,
             theta_littrow=theta_littrow,
             **{'ident_' + k: np.asarray(v) for k, v in ident.items()})
    summary = dict(
        instrument=inst, nord=int(L.nord), npix=int(L.npix),
        orders=[int(L.morder[0]), int(L.morder[-1])],
        wmin=float(L.wmin), wmax=float(L.wmax), lref=float(L.lref),
        teff=float(L.hdr['MODTEFF']), n_fitted=int(L.fitted.sum()), n_use=int(use.sum()),
        n_valid=int(L.valid.sum()), n_total=int(L.blaze.size),
        n_likelihood=int(L.d_wave.size), n_window=int(np.isfinite(L.w_wave).sum()),
        decimate=L.decimate, corr_length=corr_length, n_eff=float(n_eff),
        nu=float(nu), t_scale=float(scale),
        mad_sigma=float(1.4826 * np.median(np.abs(res_d - np.median(res_d)))),
        res_std=float(np.std(res_d)), kurtosis=float(stats.kurtosis(res_d)),
        skew=float(stats.skew(res_d)),
        best=dict(zip(PARAMS[:4], map(float, L.best))),
        best_derived={k: float(v) for k, v in L.derived(L.best).items()},
        map=dict(zip(PARAMS, map(float, theta_map))), lp_map=float(lp_map),
        map_derived={k: float(v) for k, v in L.derived(theta_map).items()},
        median=dict(zip(PARAMS, map(float, med))),
        lo=dict(zip(PARAMS, map(float, med - lo))), hi=dict(zip(PARAMS, map(float, hi - med))),
        derived_median={k: float(np.median(v)) for k, v in der.items()},
        derived_lo={k: float(np.median(v) - np.percentile(v, 15.865)) for k, v in der.items()},
        derived_hi={k: float(np.percentile(v, 84.135) - np.median(v)) for k, v in der.items()},
        corr=np.corrcoef(chain.T).tolist(),
        tau=mc['tau'].tolist(), nsteps=int(mc['nsteps']), burn=int(mc['burn']),
        thin=int(mc['thin']), nsamples=int(len(chain)), acceptance=mc['acceptance'],
        nwalkers=NWALKERS,
        jack_sigma=dict(zip(PARAMS, map(float, jack_sigma))), jack_der_sigma=jack_der_sigma,
        jack_mean=dict(zip(PARAMS, map(float, jack.mean(0)))),
        lp_const=float(lp_const), dlp_const=float(lp_map - lp_const),
        const_c0=float(theta_const[0]),
        asym_littrow=float(asym_littrow), lp_littrow=float(lp_littrow),
        dlp_littrow=float(lp_map - lp_littrow),
        teff_dlp=dict(zip(map(str, TEFF_GRID), map(float, lp_map - teff_prof[:, 1]))),
        teff_c1=dict(zip(map(str, TEFF_GRID), map(float, teff_prof[:, 3]))),
        teff_asym=dict(zip(map(str, TEFF_GRID), map(float, teff_prof[:, 5]))),
        teff_beta=dict(zip(map(str, TEFF_GRID), map(float, teff_prof[:, 4]))),
        peak_offset_map=dict(mean=float(np.nanmean((p_obs - p_map)[L.morder > 35 if inst == 'SPIRou' else L.morder > 0])),
                             rms=float(np.nanstd((p_obs - p_map)[L.morder > 35 if inst == 'SPIRou' else L.morder > 0]))),
        peak_offset_const=dict(mean=float(np.nanmean((p_obs - p_const)[L.morder > 35 if inst == 'SPIRou' else L.morder > 0])),
                               rms=float(np.nanstd((p_obs - p_const)[L.morder > 35 if inst == 'SPIRou' else L.morder > 0]))),
        median_rel=float(100 * np.median(np.abs(rel[L.fitted]))),
        rms_rel=float(100 * np.std(rel[L.fitted])),
        median_per_order=float(100 * np.nanmedian(per_order)),
        ident=dict(m_first=ident['m_first'], runner_up=ident['runner_up'],
                   scatter_best=float(ident['scatter_best']),
                   scatter_second=float(ident['scatter_second']),
                   mlam_mean=ident['mlam_mean'], start_gradient=ident['start_gradient'],
                   step=ident['step']),
        fit_log=fit_log,
    )
    json.dump(summary, open(os.path.join(workdir, 'results.json'), 'w'), indent=1)
    log('{}: done'.format(inst))


if __name__ == '__main__':
    names = sys.argv[1:] or list(INSTRUMENTS)
    for name in names:
        cfg = INSTRUMENTS[name]
        if not all(os.path.exists(os.path.join(REPO, cfg[k])) for k in ('blaze', 'wave')):
            log('{}: input files not found, skipped'.format(name))
            continue
        analyse(name, cfg)

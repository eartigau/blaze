"""Write numbers.tex (one LaTeX macro per quoted number) and the order tables,
from work/<instrument>/results.json and results.npz, so that the text of the
report cannot drift away from the analysis."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PREFIX = {'SPIRou': 'sp', 'NIRPS': 'ni'}
# published SPIRou grating (Donati et al. 2020): R2, 23.2 grooves/mm
SPIROU_GROOVES = 23.2
SPIROU_TANB = 2.0

macros = []
ratios = []
changes = {}


def mac(name, value):
    macros.append('\\newcommand{{\\{}}}{{{}}}'.format(name, value))


def num(x, nd):
    return '{:.{}f}'.format(x, nd)


def sig_digits(err):
    """decimals so that the error keeps two significant digits"""
    if err <= 0 or not np.isfinite(err):
        return 2
    return max(0, int(-np.floor(np.log10(err))) + 1)


def pm(val, lo, hi, nd=None, scale=1.0):
    val, lo, hi = val * scale, lo * scale, hi * scale
    if nd is None:
        nd = sig_digits(min(lo, hi))
    return '{}^{{+{}}}_{{-{}}}'.format(num(val, nd), num(hi, nd), num(lo, nd))


for inst, p in PREFIX.items():
    path = os.path.join(HERE, 'work', inst, 'results.json')
    if not os.path.exists(path):
        continue
    js = json.load(open(path))
    z = np.load(os.path.join(HERE, 'work', inst, 'results.npz'))
    mac(p + 'Nord', js['nord'])
    mac(p + 'Npix', js['npix'])
    mac(p + 'Mfirst', js['orders'][0])
    mac(p + 'Mlast', js['orders'][1])
    mac(p + 'Wmin', num(js['wmin'], 1))
    mac(p + 'Wmax', num(js['wmax'], 1))
    mac(p + 'Lref', num(js['lref'], 1))
    mac(p + 'Teff', num(js['teff'], 0))
    mac(p + 'Ntotal', '{:,}'.format(js['n_total']).replace(',', '\\,'))
    mac(p + 'Nvalid', '{:,}'.format(js['n_valid']).replace(',', '\\,'))
    mac(p + 'ValidPct', num(100 * js['n_valid'] / js['n_total'], 1))
    mac(p + 'Nfitted', '{:,}'.format(js['n_fitted']).replace(',', '\\,'))
    mac(p + 'Nuse', '{:,}'.format(js['n_use']).replace(',', '\\,'))
    mac(p + 'Nlike', '{:,}'.format(js['n_likelihood']).replace(',', '\\,'))
    mac(p + 'Nwin', '{:,}'.format(js['n_window']).replace(',', '\\,'))
    mac(p + 'Dec', js['decimate'])
    mac(p + 'Corr', num(js['corr_length'], 0))
    mac(p + 'Neff', num(js['n_eff'], 0))
    mac(p + 'Nu', num(js['nu'], 2))
    mac(p + 'Tscale', num(100 * js['t_scale'], 2))
    mac(p + 'Mad', num(100 * js['mad_sigma'], 2))
    mac(p + 'ResStd', num(100 * js['res_std'], 2))
    mac(p + 'Kurt', num(js['kurtosis'], 1))
    mac(p + 'Skew', num(js['skew'], 2))
    mac(p + 'Temper', '1/{:.0f}'.format(js['corr_length'] / js['decimate']))

    b, bd = js['best'], js['best_derived']
    mac(p + 'BestCzero', num(b['c0'], 1))
    mac(p + 'BestCone', num(b['c1'], 4))
    mac(p + 'BestBeta', num(b['beta'], 4))
    mac(p + 'BestAsym', num(b['asym'], 3))
    mac(p + 'BestAzero', num(bd['a0'], 2))
    mac(p + 'BestAone', num(bd['a1'], 5))
    mac(p + 'BestChange', num(bd['change'], 3))

    md, lo, hi = js['median'], js['lo'], js['hi']
    mac(p + 'Czero', pm(md['c0'], lo['c0'], hi['c0']))
    mac(p + 'Cone', pm(md['c1'], lo['c1'], hi['c1']))
    mac(p + 'Beta', pm(md['beta'], lo['beta'], hi['beta']))
    mac(p + 'Asym', pm(md['asym'], lo['asym'], hi['asym']))
    mac(p + 'Lns', pm(md['lns'], lo['lns'], hi['lns']))
    dm, dl, dh = js['derived_median'], js['derived_lo'], js['derived_hi']
    mac(p + 'Azero', pm(dm['a0'], dl['a0'], dh['a0']))
    mac(p + 'Aone', pm(dm['a1'], dl['a1'], dh['a1']))
    mac(p + 'Dlnc', pm(dm['dlnc'], dl['dlnc'], dh['dlnc'], scale=1e3))
    mac(p + 'Change', pm(dm['change'], dl['change'], dh['change']))
    mac(p + 'ChangeVal', num(dm['change'], 3))
    mac(p + 'ChangeAbs', num(abs(dm['change']), 3))
    mac(p + 'ChangeSig', num(abs(dm['change']) / max(dl['change'], dh['change']), 0))
    mac(p + 'ChangeSigJack', num(abs(dm['change']) / js['jack_der_sigma']['change'], 0))
    ratios.extend(js['jack_sigma'][k] / (0.5 * (lo[k] + hi[k])) for k in ('c0', 'c1', 'beta', 'asym'))
    changes[inst] = dm['change']

    js_ = js['jack_sigma']
    jd = js['jack_der_sigma']
    for key, name in [('c0', 'Czero'), ('c1', 'Cone'), ('beta', 'Beta'), ('asym', 'Asym')]:
        err = 0.5 * (lo[key] + hi[key])
        nd = sig_digits(min(err, js_[key]))
        mac(p + 'Sig' + name, num(err, nd))
        mac(p + 'Jack' + name, num(js_[key], nd))
        mac(p + 'Ratio' + name, num(js_[key] / err, 1))
    for key, name in [('a0', 'Azero'), ('a1', 'Aone'), ('change', 'Change')]:
        err = 0.5 * (dl[key] + dh[key])
        nd = sig_digits(min(err, jd[key]))
        mac(p + 'Sig' + name, num(err, nd))
        mac(p + 'Jack' + name, num(jd[key], nd))
    err = 0.5 * (dl['dlnc'] + dh['dlnc']) * 1e3
    mac(p + 'SigDlnc', num(err, sig_digits(min(err, 1e3 * jd['dlnc']))))
    mac(p + 'JackDlnc', num(1e3 * jd['dlnc'], sig_digits(min(err, 1e3 * jd['dlnc']))))

    corr = np.array(js['corr'])
    mac(p + 'CorrCzeroCone', num(corr[0, 1], 2))
    mac(p + 'CorrBetaAsym', num(corr[2, 3], 2))
    mac(p + 'CorrCzeroAsym', num(corr[0, 3], 2))
    mac(p + 'CorrConeAsym', num(corr[1, 3], 2))
    mac(p + 'MaxCorr', num(np.max(np.abs(corr - np.eye(len(corr)))), 2))
    sym = ['$c_0$', '$c_1$', '$\\beta$', '$\\alpha$', '$\\ln s$']
    off = np.abs(corr - np.eye(len(corr)))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    mac(p + 'CorrPair', '{} and {}'.format(sym[min(i, j)], sym[max(i, j)]))
    mac(p + 'CorrPairVal', num(corr[i, j], 2))

    mac(p + 'TauMax', num(max(js['tau']), 0))
    mac(p + 'TauMin', num(min(js['tau']), 0))
    mac(p + 'Nsteps', '{:,}'.format(js['nsteps']).replace(',', '\\,'))
    mac(p + 'StepsOverTau', num(js['nsteps'] / max(js['tau']), 0))
    mac(p + 'Burn', '{:,}'.format(js['burn']).replace(',', '\\,'))
    mac(p + 'Thin', js['thin'])
    mac(p + 'Nsamples', '{:,}'.format(js['nsamples']).replace(',', '\\,'))
    mac(p + 'Acc', num(js['acceptance'], 2))
    mac(p + 'Nwalkers', js['nwalkers'])

    mac(p + 'DlpConst', num(js['dlp_const'], 0))
    mac(p + 'ConstCzero', num(js['const_c0'], 1))
    mac(p + 'DlpLittrow', num(js['dlp_littrow'], 0))
    mac(p + 'AsymLittrow', num(js['asym_littrow'], 2))
    td = js['teff_dlp']
    mid = {k: v for k, v in td.items() if 3000 <= float(k) <= 8000}
    mac(p + 'TeffDlpMid', num(max(abs(v) for v in mid.values()), 2))
    mac(p + 'TeffDlpMax', num(max(abs(v) for v in td.values()), 2))
    tp = z['teff_prof']
    ref = tp[np.argmin(np.abs(tp[:, 0] - js['teff']))]
    sig = {k: 0.5 * (js['lo'][k] + js['hi'][k]) for k in ('c0', 'c1', 'beta', 'asym')}
    shift = max(np.max(np.abs(tp[:, 2 + i] - ref[2 + i])) / sig[k]
                for i, k in enumerate(('c0', 'c1', 'beta', 'asym')))
    mac(p + 'TeffShift', num(shift, 3))
    # Littrow asymmetry against the posterior, on the side where it lies
    if np.isfinite(js['asym_littrow']):
        side = js['hi']['asym'] if js['asym_littrow'] > md['asym'] else js['lo']['asym']
        mac(p + 'LittrowSig', num(abs(js['asym_littrow'] - md['asym']) / side, 1))
        mac(p + 'LittrowSigJack', num(abs(js['asym_littrow'] - md['asym']) / js_['asym'], 1))
        mac(p + 'DlpLittrowDec', num(js['dlp_littrow'], 1))

    mac(p + 'PkMapMean', num(js['peak_offset_map']['mean'], 0))
    mac(p + 'PkMapRms', num(js['peak_offset_map']['rms'], 0))
    mac(p + 'PkConstMean', num(js['peak_offset_const']['mean'], 0))
    mac(p + 'PkConstRms', num(js['peak_offset_const']['rms'], 0))
    mac(p + 'MedRel', num(js['median_rel'], 2))
    mac(p + 'RmsRel', num(js['rms_rel'], 2))
    mac(p + 'MedPerOrder', num(js['median_per_order'], 2))
    per = z['per_order']
    mac(p + 'PerOrderQlo', num(100 * np.nanpercentile(per, 25), 2))
    mac(p + 'PerOrderQhi', num(100 * np.nanpercentile(per, 75), 2))
    worst = np.argsort(-np.nan_to_num(per))[:3]
    mac(p + 'Worst', ', '.join('{} ({:.0f}\\%)'.format(int(z['morder'][i]), 100 * per[i]) for i in worst))

    idn = js['ident']
    mac(p + 'ScatBest', '{:.2e}'.format(idn['scatter_best']).replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}')
    mac(p + 'ScatSecond', '{:.2e}'.format(idn['scatter_second']).replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}')
    mac(p + 'ScatRatio', num(idn['scatter_second'] / idn['scatter_best'], 1))
    mac(p + 'RunnerUp', idn['runner_up'])
    mac(p + 'MlamMean', num(idn['mlam_mean'], 1))
    mac(p + 'GradStart', idn['start_gradient'])
    sc = z['ident_scatter_c']
    tc = z['ident_trial_c']
    srt = np.argsort(sc)
    mac(p + 'ScatCBest', '{:.1e}'.format(sc[srt[0]]).replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}')
    mac(p + 'ScatCRatio', num(sc[srt[1]] / sc[srt[0]], 0))
    mo = z['ident_m_overlap'] - z['ident_morder']
    mg = z['ident_m_gradient'] - z['ident_morder']
    mac(p + 'OverlapMaxDev', num(np.max(np.abs(mo)), 2))
    mac(p + 'GradMaxDev', num(np.max(np.abs(mg[1:-1])), 2))
    mac(p + 'GradEdgeFirst', num(mg[0], 2))
    mac(p + 'GradEdgeLast', '{:+.2f}'.format(mg[-1]))
    mac(p + 'OverlapMeanDev', num(np.mean(mo), 2))

    # order table: everything mkblaze.py prints, plus the gradient estimator
    rows = []
    for i in range(len(z['ident_index'])):
        rows.append('{} & {} & {:.2f} & {:.1f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(
            int(z['ident_index'][i]), int(z['ident_morder'][i]), z['ident_wpeak'][i],
            z['ident_mlam'][i], z['ident_lam_centre'][i], z['ident_m_overlap'][i],
            z['ident_m_gradient'][i]))
    open(os.path.join(HERE, 'orders_{}.tex'.format(inst)), 'w').write('\n'.join(rows) + '\n')

    # model comparison table and extrapolation test
    cpath = os.path.join(HERE, 'work', inst, 'comparison.json')
    if os.path.exists(cpath):
        comp = json.load(open(cpath))
        rows = []
        for v in comp['variants']:
            bold = v['name'].startswith('linear spline, chromatic')
            fmt = '\\textbf{{{}}}' if bold else '{}'
            cells = [v['name'].replace(' C', ' $C$'), '{:.2f}'.format(v['median']),
                     '{:.2f}'.format(v['rms']), '{:.2f}'.format(v['per_order']),
                     '{:+.3f}'.format(v['change']) if abs(v['change']) > 1e-4 else '0',
                     '{:+.2f}'.format(v['asym']), '{:.4f}'.format(v['beta'])]
            rows.append(' & '.join(fmt.format(c) for c in cells) + ' \\\\')
        open(os.path.join(HERE, 'comparison_{}.tex'.format(inst)), 'w').write('\n'.join(rows) + '\n')
        names = {'linear spline': 'Lin', 'cubic spline': 'Cub', 'polynomial 21': 'Poly'}
        for x in comp['extrapolation']:
            val = x['beyond']
            txt = '{:.0f}'.format(val) if val < 1e4 else '{:.0e}'.format(val).replace('e+0', '\\times10^{').replace('e+', '\\times10^{') + '}'
            mac(p + 'Ext' + names[x['name']], txt)
        chg = [v['change'] for v in comp['variants'] if 'chromatic' in v['name']]
        mac(p + 'ChangeRangeLo', num(min(chg), 3))
        mac(p + 'ChangeRangeHi', num(max(chg), 3))
        by = {v['name']: v for v in comp['variants']}
        mac(p + 'CubDiffMed', num(abs(by['cubic spline, chromatic C']['median'] - by['linear spline, chromatic C']['median']), 2))

    # the numbers fit_blaze.py itself prints, for the section on the fit
    import re
    log_ = js['fit_log']
    m_ = re.search(r'fitted range : median \|\.\| = ([\d.]+)%\s+rms = ([\d.]+)%', log_)
    mac(p + 'FitMed', m_.group(1))
    mac(p + 'FitRms', m_.group(2))
    m_ = re.search(r'median of the per-order rms ([\d.]+)%, worst orders (.*)', log_)
    mac(p + 'FitPerOrder', m_.group(1))
    mac(p + 'FitWorst', m_.group(2).replace('m=', '').replace('%', '\\%'))
    m_ = re.search(r'drift of m\*lambda_peak over the \d+ fully fitted orders: (\d+) observed', log_)
    mac(p + 'PeakDrift', m_.group(1))

    # the fit_blaze.py log, verbatim
    open(os.path.join(HERE, 'fitlog_{}.txt'.format(inst)), 'w').write(js['fit_log'])

    # peak offsets, observed minus model, in pixels
    red = z['morder'] > 35 if inst == 'SPIRou' else np.ones(len(z['morder']), bool)
    dconst = (z['p_obs'] - z['p_const'])[red]
    dmap = z['p_obs'] - z['p_map']
    k = int(np.nanargmax(np.abs(dconst)))
    mac(p + 'PkConstMax', num(abs(dconst[k]), 0))
    mac(p + 'PkConstMaxOrder', int(z['morder'][red][k]))
    mac(p + 'PkMapFirst', num(dmap[0], 0))
    k = int(np.nanargmax(np.where(red, dmap, -np.inf)))
    mac(p + 'PkMapMaxPos', num(dmap[k], 0))
    mac(p + 'PkMapMaxPosOrder', int(z['morder'][k]))

    # checks on the model itself, for the text
    from blaze_like import BlazeLikelihood
    from astropy.io import fits
    from scipy.interpolate import InterpolatedUnivariateSpline
    import time
    cfg = {'SPIRou': ('FDFCD50E5Ff_pp_blaze_AB.fits', 'FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits'),
           'NIRPS': ('NIRPS_2023-04-01T12_07_06_942_pp_blaze_A.fits',
                     'NIRPS_2025-09-13T21_30_32_350_pp_e2dsff_A_wave_night_A.fits')}[inst]
    mfile = os.path.join(HERE, 'work', inst, 'blaze_model.fits')
    L = BlazeLikelihood(os.path.join(REPO, cfg[0]), os.path.join(REPO, cfg[1]), mfile)
    kn = fits.getdata(mfile, 'TRANS_KNOTS')
    spl = InterpolatedUnivariateSpline(kn['WAVE'], kn['LOGTRANS'], k=L.k, ext=0)
    logenv = L.log_envelope_2d(L.wave, L.morder.astype(float)[:, None], L.best)
    rebuilt = L.photon * L.dwave * np.exp(logenv + spl(L.wave.ravel()).reshape(L.wave.shape))
    stored = fits.getdata(mfile)
    mac(p + 'LikeCheck', '{:.0e}'.format(np.max(np.abs(rebuilt / stored - 1))).replace('e-0', '\\times10^{-') + '}')
    ky = L.spline(L.best)[2]
    mac(p + 'KnotCheck', '10^{{{:.0f}}}'.format(np.floor(np.log10(max(np.max(np.abs(ky - kn['LOGTRANS'])), 1e-16)))))
    th = np.append(L.best, np.log(js['t_scale']))
    L.corr_length, L.nu = js['corr_length'], js['nu']
    t0 = time.time()
    for _ in range(200):
        L.log_post(th)
    mac(p + 'LikeMs', num((time.time() - t0) / 200 * 1e3, 1))
    var = [L.dwave[i].max() / L.dwave[i].min() - 1 for i in range(L.nord)]
    mac(p + 'DwaveVar', num(100 * np.median(var), 0))
    tr = np.exp(kn['LOGTRANS'])
    mac(p + 'TransRange', num(tr.max() / tr.min(), 0))
    if os.path.exists(cpath):
        mac(p + 'MedRatioConst', num(by['linear spline, constant C']['median'] /
                                     by['linear spline, chromatic C']['median'], 1))

    if inst == 'SPIRou':
        # out-of-plane angle implied by the published grating
        c2d = 2 * (1e6 / SPIROU_GROOVES) * np.sin(np.arctan(SPIROU_TANB))
        a0, a1 = bd['a0'], bd['a1']
        cb, cr = dm['a0'] + dm['a1'] * js['wmin'], dm['a0'] + dm['a1'] * js['wmax']
        mac('spTwoDsin', num(c2d, 1))
        mac('spCblue', num(cb, 1))
        mac('spCred', num(cr, 1))
        mac('spGammaBlue', num(np.degrees(np.arccos(cb / c2d)), 2))
        mac('spGammaRed', num(np.degrees(np.arccos(cr / c2d)), 2))
        mac('spGammaDelta', num(np.degrees(np.arccos(cr / c2d)) - np.degrees(np.arccos(cb / c2d)), 2))
        mac('spGrooves', num(SPIROU_GROOVES, 1))
        mac('spCdrop', num(cb - cr, 0))

# across instruments
if ratios:
    mac('MaxRatio', num(max(ratios), 1))
    mac('MinRatio', num(min(ratios), 1))
if len(changes) == 2:
    mac('ChangeRatio', num(changes['SPIRou'] / changes['NIRPS'], 1))

# accuracy of the second-order envelope against the exact Littrow form, over
# one order (m = 55, argument out to about 0.8), and the photon/energy peaks
m, cst = 55, 76847.0
lam = np.linspace(cst / m * (1 - 0.8 / m), cst / m * (1 + 0.8 / m), 2001)
for tag, tbd in [('ExpAccA', 30.0), ('ExpAccB', np.degrees(np.arctan(2))), ('ExpAccC', 76.0)]:
    tb = np.radians(tbd)
    ss = m * lam / cst
    ee = ss - 1
    exact = m * np.cos(tb) / ss * (ss * np.cos(tb) - np.sqrt(1 - (ss * np.sin(tb)) ** 2))
    expan = m * (ee + (np.tan(tb) ** 2 / 2 - 1) * ee * ee)
    d = np.max(np.abs(exact - expan))
    mant, expo = '{:.1e}'.format(d).split('e')
    mac(tag, '{}\\times10^{{{}}}'.format(mant, int(expo)))
from scipy.optimize import brentq
x_energy = brentq(lambda x: 5 * np.expm1(-x) + x, 1, 10)     # h c / (lambda k T) at the peak
x_photon = brentq(lambda x: 4 * np.expm1(-x) + x, 1, 10)
mac('PeakRatio', num(x_energy / x_photon, 2))

# the verbatim output of mkblaze.py, run on the SPIRou files in a scratch folder
work = os.path.join(HERE, 'work', 'mkblaze')
os.makedirs(work, exist_ok=True)
for f in ['FDFCD50E5Ff_pp_blaze_AB.fits', 'FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits']:
    dst = os.path.join(work, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(REPO, f), dst)
env = dict(os.environ, MPLBACKEND='Agg')
out = subprocess.run([sys.executable, os.path.join(REPO, 'mkblaze.py')], capture_output=True,
                     text=True, cwd=work, env=env).stdout
open(os.path.join(HERE, 'mkblaze_log.txt'), 'w').write(out)

open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(macros) + '\n')
print('wrote {} macros'.format(len(macros)))

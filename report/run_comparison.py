"""Model comparison: fit_blaze.py re-run with one setting changed at a time.

Each variant is a copy of fit_blaze.py with a single edit, run in its own
folder. The edits are checked to apply, so a change to fit_blaze.py that
breaks one fails loudly instead of silently comparing the default twice.

    python run_comparison.py [instrument ...]
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from astropy.io import fits

from run_analysis import INSTRUMENTS, REPO, log

HERE = os.path.dirname(os.path.abspath(__file__))

CONST_C = ('slope_max = cst_start / lref', 'slope_max = 1e-9 * cst_start / lref')
POLY = ("TRANSMISSION = 'spline'", "TRANSMISSION = 'poly'")
CUBIC = ('SPLINE_K = 1 ', 'SPLINE_K = 3 ')
FIT_2400 = ('WAVE_FIT_MAX = 2500.0', 'WAVE_FIT_MAX = 2400.0')


def npoly(n):
    return ('NPOLY = 21 ', 'NPOLY = {} '.format(n))


VARIANTS = [
    ('linear spline, chromatic C', []),
    ('linear spline, constant C', [CONST_C]),
    ('cubic spline, chromatic C', [CUBIC]),
    ('polynomial 11, chromatic C', [POLY, npoly(11)]),
    ('polynomial 21, chromatic C', [POLY]),
    ('polynomial 21, constant C', [POLY, CONST_C]),
]
EXTRAPOLATION = [
    ('linear spline', [FIT_2400]),
    ('cubic spline', [FIT_2400, CUBIC]),
    ('polynomial 21', [FIT_2400, POLY]),
]


def run_variant(src, edits, files, workdir):
    code = src
    for old, new in edits:
        if code.count(old) != 1:
            raise RuntimeError('edit {!r} does not apply to fit_blaze.py'.format(old))
        code = code.replace(old, new)
    code = code.replace("\n\ndebug_plots('blaze_model_debug.pdf', res)\n", '\n')
    os.makedirs(workdir, exist_ok=True)
    script = os.path.join(workdir, 'fit_variant.py')
    open(script, 'w').write(code)
    out = subprocess.run([sys.executable, script] + files, capture_output=True, text=True,
                         cwd=workdir, check=True).stdout

    def grab(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    return dict(
        median=grab(r'fitted range : median \|\.\| = ([\d.]+)%'),
        rms=grab(r'fitted range : median \|\.\| = [\d.]+%\s+rms = ([\d.]+)%'),
        per_order=grab(r'median of the per-order rms ([\d.]+)%'),
        change=grab(r'C changes by ([-+\d.]+)%'),
        asym=grab(r'asym = ([-+\d.]+)\n'),
        beta=grab(r'blaze width beta = ([\d.]+)'),
        beyond=grab(r'beyond \d+\s*: median \|\.\| = ([\d.e+]+)%'),
        log=out)


def compare(inst, cfg):
    src = open(os.path.join(REPO, 'fit_blaze.py')).read()
    files = [os.path.join(REPO, cfg['blaze']), os.path.join(REPO, cfg['wave'])]
    base = os.path.join(HERE, 'work', inst, 'comparison')
    out = dict(variants=[], extrapolation=[])
    for k, (name, edits) in enumerate(VARIANTS):
        log('{}: {}'.format(inst, name))
        res = run_variant(src, edits, files, os.path.join(base, 'v{}'.format(k)))
        res['name'] = name
        out['variants'].append(res)
    blaze, wave = fits.getdata(files[0]), fits.getdata(files[1])
    wmax = np.nanmax(np.where(np.isfinite(blaze), wave, np.nan))
    if wmax > 2400:
        for k, (name, edits) in enumerate(EXTRAPOLATION):
            log('{}: extrapolation, {}'.format(inst, name))
            res = run_variant(src, edits, files, os.path.join(base, 'x{}'.format(k)))
            res['name'] = name
            out['extrapolation'].append(res)
    for item in out['variants'] + out['extrapolation']:
        item.pop('log')
    json.dump(out, open(os.path.join(HERE, 'work', inst, 'comparison.json'), 'w'), indent=1)
    log('{}: comparison done'.format(inst))


if __name__ == '__main__':
    for name in sys.argv[1:] or list(INSTRUMENTS):
        cfg = INSTRUMENTS[name]
        if all(os.path.exists(os.path.join(REPO, cfg[k])) for k in ('blaze', 'wave')):
            compare(name, cfg)

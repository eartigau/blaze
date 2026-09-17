"""Full characterisation of an echelle blaze.

    python characterize.py                 analysis, figures and the PDF report
    python characterize.py --no-report     analysis only, with a summary printed
    python characterize.py --report-only   figures and PDF from existing results

The analysis identifies the diffraction orders, fits the model with
fit_blaze.py, samples the grating parameters with an MCMC, runs a jackknife
over orders and profile tests, and compares transmission models. Results go
to report/work/<instrument>/. The report is report/blaze_report.pdf.

By default the two configured instruments are used, SPIRou and NIRPS, whose
files are in the repository. Any other instrument can be characterised
without a report:

    python characterize.py --no-report --name MYINST --blaze b.fits --wave w.fits
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(REPO, 'report')
sys.path.insert(0, REPORT)


def banner(text):
    print('\n' + '=' * 78 + '\n' + text + '\n' + '=' * 78, flush=True)


def missing_files(cfg):
    return [cfg[k] for k in ('blaze', 'wave') if not os.path.exists(os.path.join(REPO, cfg[k]))]


def summarize(name):
    path = os.path.join(REPORT, 'work', name, 'results.json')
    js = json.load(open(path))
    md, lo, hi = js['median'], js['lo'], js['hi']
    dm, dl, dh = js['derived_median'], js['derived_lo'], js['derived_hi']
    jk, jd = js['jack_sigma'], js['jack_der_sigma']

    def row(label, val, a, b, jack, fmt):
        err = 0.5 * (a + b)
        print('  {:<24s} {:>14s} +/- {:<10s} jackknife +/- {}'.format(
            label, fmt.format(val), fmt.format(err), fmt.format(jack) if jack is not None else '-'))

    banner('{}: diffraction orders {} to {}'.format(name, *js['orders']))
    print('  wavelength {:.1f} to {:.1f} nm, reference {:.1f} nm, Teff {:.0f} K (fixed)'.format(
        js['wmin'], js['wmax'], js['lref'], js['teff']))
    print('  residuals: correlation length {:.0f} px, {:.0f} effective samples, '
          'Student-t nu {:.2f}'.format(js['corr_length'], js['n_eff'], js['nu']))
    print('  MCMC: {} steps, tau <= {:.0f}, {} samples, acceptance {:.2f}\n'.format(
        js['nsteps'], max(js['tau']), js['nsamples'], js['acceptance']))
    row('C(lambda_ref) [nm]', md['c0'], lo['c0'], hi['c0'], jk['c0'], '{:.2f}')
    row('dC/dlambda', md['c1'], lo['c1'], hi['c1'], jk['c1'], '{:.5f}')
    row('beta', md['beta'], lo['beta'], hi['beta'], jk['beta'], '{:.5f}')
    row('asym', md['asym'], lo['asym'], hi['asym'], jk['asym'], '{:.3f}')
    row('a0 [nm]', dm['a0'], dl['a0'], dh['a0'], jd['a0'], '{:.2f}')
    row('C change across [%]', dm['change'], dl['change'], dh['change'], jd['change'], '{:.4f}')
    print('\n  constant C excluded by      dlnL = {:.0f}'.format(js['dlp_const']))
    if js['dlp_littrow'] == js['dlp_littrow']:
        print('  Littrow asymmetry ({:+.2f}) excluded by dlnL = {:.0f}'.format(
            js['asym_littrow'], js['dlp_littrow']))
    print('  obs/model - 1: median {:.2f}%, rms {:.2f}%, median per-order rms {:.2f}%'.format(
        js['median_rel'], js['rms_rel'], js['median_per_order']))
    print('  peak offset obs - model: {:+.0f} +/- {:.0f} px (constant C: {:+.0f} +/- {:.0f} px)'.format(
        js['peak_offset_map']['mean'], js['peak_offset_map']['rms'],
        js['peak_offset_const']['mean'], js['peak_offset_const']['rms']))
    print('  results in', os.path.relpath(os.path.dirname(path), REPO))


def build_pdf():
    for tool in ('pdflatex', 'bibtex'):
        if shutil.which(tool) is None:
            sys.exit('{} not found, the report cannot be compiled'.format(tool))
    steps = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'blaze_report.tex'],
             ['bibtex', 'blaze_report'],
             ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'blaze_report.tex'],
             ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'blaze_report.tex']]
    for cmd in steps:
        out = subprocess.run(cmd, cwd=REPORT, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-3000:])
            sys.exit('{} failed, see report/blaze_report.log'.format(cmd[0]))
    print('report written to report/blaze_report.pdf')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--no-report', action='store_true', help='run the analysis, skip figures and PDF')
    mode.add_argument('--report-only', action='store_true', help='rebuild figures and PDF from existing results')
    ap.add_argument('--instrument', action='append', help='restrict to a configured instrument (repeatable)')
    ap.add_argument('--skip-comparison', action='store_true', help='skip the model comparison')
    ap.add_argument('--name', help='name of a custom instrument (with --blaze and --wave, --no-report only)')
    ap.add_argument('--blaze', help='blaze FITS file of a custom instrument')
    ap.add_argument('--wave', help='wavelength solution FITS file of a custom instrument')
    ap.add_argument('--littrow-tan', type=float, help='tan(blaze angle) of a custom instrument, for the Littrow test')
    args = ap.parse_args()

    import run_analysis
    import run_comparison

    instruments = dict(run_analysis.INSTRUMENTS)
    if args.blaze or args.wave or args.name:
        if not (args.blaze and args.wave and args.name):
            ap.error('--name, --blaze and --wave go together')
        if not args.no_report:
            ap.error('a custom instrument is characterised with --no-report; the report is written for SPIRou and NIRPS')
        instruments = {args.name: dict(blaze=os.path.abspath(args.blaze), wave=os.path.abspath(args.wave),
                                       littrow_tan=args.littrow_tan)}
        run_analysis.INSTRUMENTS[args.name] = instruments[args.name]
    elif args.instrument:
        unknown = set(args.instrument) - set(instruments)
        if unknown:
            ap.error('unknown instrument(s) {}; configured: {}'.format(sorted(unknown), sorted(instruments)))
        instruments = {k: instruments[k] for k in args.instrument}

    report = not args.no_report
    if report:
        absent = {k: missing_files(v) for k, v in run_analysis.INSTRUMENTS.items() if missing_files(v)}
        if absent or set(instruments) != set(run_analysis.INSTRUMENTS):
            for k, files in absent.items():
                print('{}: missing {}'.format(k, ', '.join(files)))
            sys.exit('the report needs every configured instrument ({}); run with --no-report '
                     'to characterise what is available'.format(', '.join(run_analysis.INSTRUMENTS)))

    if not args.report_only:
        for name, cfg in instruments.items():
            files = missing_files(cfg)
            if files:
                print('{}: missing {}, skipped'.format(name, ', '.join(files)))
                continue
            banner('{}: analysis'.format(name))
            run_analysis.analyse(name, cfg)
            if not args.skip_comparison:
                run_comparison.compare(name, cfg)
        if args.no_report:
            for name in instruments:
                if os.path.exists(os.path.join(REPORT, 'work', name, 'results.json')):
                    summarize(name)
            return

    banner('report')
    for script in ('make_numbers.py', 'make_report_figures.py'):
        subprocess.run([sys.executable, script], cwd=REPORT, check=True,
                       env=dict(os.environ, MPLBACKEND='Agg'))
    build_pdf()


if __name__ == '__main__':
    main()

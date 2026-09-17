"""Fit an echelle blaze with a physical model of the spectrograph.

    model(m, pixel) = BB_photon(lambda, Teff) * Trans(lambda)
                      * sinc^2(blaze) * dlambda/dpixel

  * BB_photon is a blackbody in PHOTON density, not in energy, because the
    detector counts photons. Dividing B_lambda by the photon energy hc/lambda
    gives  1 / (lambda^4 * (exp(hc/lambda.k.T) - 1)).
  * Trans(lambda) = exp(S(lambda)): log(transmission) is either a spline
    through the peaks of the orders (default) or a polynomial of wavelength.
    It carries the global amplitude.
  * sinc^2 is the single-groove diffraction envelope of the grating. Its
    natural variable is the distance to the blaze peak counted in orders,
    e = m * lambda / C - 1, with C = m * lambda_blaze. The argument is
    beta * m * (e + asym * e^2). beta = 1 means a fully illuminated groove,
    i.e. first zeros exactly one free spectral range from the peak. asym is
    the second-order term that makes the envelope lopsided in wavelength; a
    grating in Littrow would give asym = tan(theta_b)^2 / 2 - 1, but it is left
    free because the data do not follow that relation.
  * C is slightly chromatic, C(lambda) = a0 + a1 * lambda. C = 2 d sin(theta_b)
    cos(gamma), and when the beam crosses a dispersive element before the
    grating (a cross-disperser used in double pass, as in SPIRou) the
    out-of-plane angle gamma, hence C, depends on wavelength.
  * dlambda/dpixel turns the photon density per unit wavelength into photons
    per pixel. The wavelength solution is not linear, so this term varies
    along an order and tilts it.

Nothing here is specific to one instrument. C(lambda), beta and asym are
fitted, and the diffraction orders are read off the wavelength solution itself.

usage: python fit_blaze.py [blaze.fits [wave.fits]]
"""
import sys

from astropy.io import fits
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares

TEFF = 5000.0                    # K, temperature of the flat lamp
TRANSMISSION = 'spline'          # model of log(transmission), 'spline' or 'poly'
SPLINE_K = 1                     # spline degree, 1 = linear between order peaks
PEAK_HALF_WIDTH = 50             # pixels. The spline knot of an order is the
                                 # median over +- this around its observed peak
NPOLY = 21                       # polynomial order, for TRANSMISSION = 'poly'
HC_K = 1.438776877e7             # h*c/k_B in nm.K
SIGMA_CLIP = 5.0
WAVE_FIT_MAX = 2500.0            # do not let anything redder than this drive the
                                 # fit: the detector cut-off is far sharper than a
                                 # polynomial can follow. The model is still
                                 # evaluated there, as an extrapolation.

BLAZE_FILE = sys.argv[1] if len(sys.argv) > 1 else 'FDFCD50E5Ff_pp_blaze_AB.fits'
WAVE_FILE = sys.argv[2] if len(sys.argv) > 2 else 'FEF450D367a_pp_e2dsff_AB_wave_night_AB.fits'

blaze = fits.getdata(BLAZE_FILE)
wave = fits.getdata(WAVE_FILE)
index = np.arange(wave.shape[0])


def diffraction_orders(wave):
    """Diffraction order of every spectral order, from the wavelength solution
    alone. At a given pixel all orders share the same diffraction angle, so
    m*lambda is the same constant for all of them, which gives both a direct
    estimate m = lambda / |dlambda/dorder| and the integer offset that makes
    m*lambda the flattest. Works whichever way round the orders are stored and
    on a subset of the array."""
    lam = wave[:, wave.shape[1] // 2]
    step = int(-np.sign(np.median(np.diff(lam))))     # +1 if m grows with index
    guess = lam / np.abs(np.gradient(lam))
    start = int(np.round(np.median(guess - step * index)))
    trial = start + np.arange(-2, 3)
    scatter = [np.std((t + step * index) * lam) / np.mean((t + step * index) * lam)
               for t in trial]
    return trial[np.argmin(scatter)] + step * index


morder = diffraction_orders(wave)
mm = np.repeat(morder[:, None], wave.shape[1], axis=1).astype(float)
dwave = np.abs(np.gradient(wave, axis=1))            # |dlambda/dpixel|, nm per pixel
photon = 1.0 / (wave ** 4 * np.expm1(HC_K / (wave * TEFF)))   # blackbody, photons


def grating_constant(c0, c1):
    """C(lambda) = c0 + c1 * (lambda - lref): c0 is C at the reference
    wavelength and c1 = dC/dlambda. Same line as a0 + a1 * lambda, written
    around lref so that the two parameters are not correlated in the fit."""
    return c0 + c1 * (wave - lref)


def envelope(c0, c1, beta, asym):
    """Grating blaze function. e = m*lambda/C - 1 is 0 at the blaze peak and
    runs to about +-1/m at the edges of the order."""
    ee = mm * wave / grating_constant(c0, c1) - 1
    return np.sinc(beta * mm * (ee + asym * ee ** 2)) ** 2


def backbone(c0, c1, beta, asym):
    """Everything in the model except the transmission."""
    return photon * np.maximum(envelope(c0, c1, beta, asym), 1e-8) * dwave


# valid is where there is data, fitted is what the fit is allowed to see, and
# use is what is left of fitted once the sigma clipping has had its say
valid = np.isfinite(blaze) & np.isfinite(wave) & (blaze > 0)
fitted = valid & (wave < WAVE_FIT_MAX)
use = fitted.copy()
logobs = np.full(blaze.shape, np.nan)
logobs[valid] = np.log(blaze[valid])
# polynomial variable normalised to [-1, 1] over the whole array, so that the
# unconstrained red end stays a mild extrapolation rather than a runaway one
wmin, wmax = np.min(wave[valid]), np.max(wave[valid])
uu = 2 * (wave - wmin) / (wmax - wmin) - 1
lref = np.median(wave[fitted])       # reference wavelength of C(lambda)
# the spline knot of every order comes from a fixed window around its peak
peaks = np.argmax(np.where(fitted, blaze, -np.inf), axis=1)
pixel = np.arange(wave.shape[1])
window = fitted & (np.abs(pixel[None, :] - peaks[:, None]) <= PEAK_HALF_WIDTH)


def solve_trans(c0, c1, beta, asym):
    """log(transmission) for a given grating, with the residual map. It never
    goes through the non-linear solver, which only sees the grating.

    spline: one knot per order, at its peak, worth the median of
    log(observed) - log(backbone) over +-PEAK_HALF_WIDTH pixels. The backbone
    comes out first because the observed peak also carries the blackbody, the
    sinc^2 and the pixel width, which would otherwise be counted twice.
    poly: log(transmission) enters linearly, solved by least squares.

    Returns log(transmission) and the residual, both on the full array, and
    the transmission parameters: knot wavelengths and values stacked for the
    spline, coefficients in u for the polynomial."""
    yy = logobs - np.log(backbone(c0, c1, beta, asym))
    if TRANSMISSION == 'spline':
        win = window & use
        rows = np.where(win.any(axis=1))[0]
        kx = np.array([np.median(wave[i][win[i]]) for i in rows])
        ky = np.array([np.median(yy[i][win[i]]) for i in rows])
        srt = np.argsort(kx)
        spline = InterpolatedUnivariateSpline(kx[srt], ky[srt], k=SPLINE_K, ext=0)
        logt = spline(wave.ravel()).reshape(wave.shape)
        par = np.array([kx[srt], ky[srt]])
    else:
        vander = np.array([uu[use] ** k for k in range(NPOLY + 1)]).T
        par = np.linalg.lstsq(vander, yy[use], rcond=None)[0]
        logt = np.sum([par[k] * uu ** k for k in range(NPOLY + 1)], axis=0)
    return logt, yy - logt, par


# C = m * lambda_blaze = 2 * d * sin(theta_b) * cos(gamma) is the grating
# spacing as the beam sees it: the optical path difference between two adjacent
# grooves at the blaze peak, i.e. the groove spacing d of the ruling projected
# along the blaze direction. The grooves are the same for every order, but the
# angle gamma at which the beam meets them can drift with wavelength, so C gets
# a linear term. It starts flat, at the mean of m*lambda over the observed blaze
# peaks, and the envelope starts symmetric; everything then floats.
inrange = fitted.any(axis=1)
cst_start = np.mean(morder[inrange] * wave[index[inrange], peaks[inrange]])
guess = np.array([cst_start, 0.0, 1.0, 0.0])
slope_max = cst_start / lref
bounds = ([cst_start / 10, -slope_max, 0.1, -100], [cst_start * 10, slope_max, 5.0, 100])
scale = [cst_start, 1e-3 * slope_max, 1.0, 1.0]
for loop in range(3):
    fit = least_squares(lambda p: solve_trans(*p)[1][use], guess, bounds=bounds,
                        loss='soft_l1', f_scale=0.05, x_scale=scale)
    guess = fit.x
    logt, res, trans_par = solve_trans(*fit.x)
    rms = np.std(res[use])
    use = fitted & (np.abs(res) < SIGMA_CLIP * rms)
    print('pass {}: C({:.0f}) = {:9.1f}   dC/dlambda = {:+.4f}   beta = {:.4f}   '
          'asym = {:+.3f}   rms = {:5.2f}%   {} points kept'.format(
              loop + 1, lref, *fit.x, 100 * rms, use.sum()))

c0, c1, beta, asym = fit.x
cst_a0, cst_a1 = c0 - c1 * lref, c1          # C(lambda) = a0 + a1 * lambda
logt, res, trans_par = solve_trans(c0, c1, beta, asym)
# the transmission is evaluated as it is everywhere, including past
# WAVE_FIT_MAX where nothing constrains it. Clamping it there would be smooth in
# value but would put a kink in the slope at the boundary, so it is left free.
# Beyond its end knots the spline carries on with its end pieces (straight
# lines for SPLINE_K = 1); a polynomial can run away much faster.
trans = np.exp(logt)

# the model is evaluated on every pixel of every order, including the ones the
# pipeline threw away when it thresholded the blaze. Only the fit is restricted.
model = backbone(c0, c1, beta, asym) * trans
resid = blaze / model - 1

print('')
print('diffraction orders {} to {}'.format(morder[0], morder[-1]))
print('fit restricted to lambda < {:.1f}: {} points used, {} left free'.format(
    WAVE_FIT_MAX, fitted.sum(), (valid & ~fitted).sum()))
print('model evaluated on all {} pixels, {:.1f}% of which have no observed blaze'.format(
    model.size, 100 * np.mean(~valid)))
print('C = m * lambda_blaze = 2.d.sin(theta_b).cos(gamma) = a0 + a1 * lambda')
print('    a0 = {:.2f}   a1 = {:+.5f}'.format(cst_a0, cst_a1))
print('    {:.1f} from the blaze peaks; after the fit {:.1f} at {:.0f}, {:.1f} at {:.0f}, '
      '{:.1f} at {:.0f}'.format(cst_start, cst_a0 + cst_a1 * wmin, wmin, c0, lref,
                                cst_a0 + cst_a1 * wmax, wmax))
print('    i.e. dlnC/dlnlambda = {:+.5f}, C changes by {:+.3f}% across the array'.format(
    c1 * lref / c0, 100 * c1 * (wmax - wmin) / c0))
print('blaze width beta = {:.4f}   (1 = first zeros one FSR from the peak)'.format(beta))
print('envelope asymmetry asym = {:+.3f}'.format(asym))
if TRANSMISSION == 'spline':
    print('transmission: degree {} spline through {} order peaks, knot = median over '
          '+-{} px'.format(SPLINE_K, trans_par.shape[1], PEAK_HALF_WIDTH))
else:
    print('log(transmission) coefficients in u = 2*(lambda-{:.1f})/{:.1f}-1:'.format(
        wmin, wmax - wmin))
    print('   ' + np.array2string(trans_par, precision=4))
print('obs/model - 1 over the fitted range : median |.| = {:.2f}%   rms = {:.2f}%'.format(
    100 * np.median(np.abs(resid[fitted])), 100 * np.std(resid[fitted])))
if (valid & ~fitted).any():
    print('                 beyond {:.0f}          : median |.| = {:.2f}%   rms = {:.2f}%'.format(
        WAVE_FIT_MAX, 100 * np.median(np.abs(resid[valid & ~fitted])),
        100 * np.std(resid[valid & ~fitted])))
offset = np.array([np.nanmedian(resid[i]) for i in index])
print('   of which order-to-order offsets {:.2f}% rms, within-order {:.2f}% rms'.format(
    100 * np.std(offset), 100 * np.nanstd(resid - offset[:, None])))
# a handful of orders where the transmission changes within one free spectral
# range (filter edges) dominate the global rms, the per-order view is fairer
per_order = np.array([np.std(resid[i][fitted[i]]) if fitted[i].any() else np.nan
                      for i in index])
worst = np.argsort(-np.nan_to_num(per_order))[:3]
print('   median of the per-order rms {:.2f}%, worst orders {}'.format(
    100 * np.nanmedian(per_order),
    ', '.join('m={} ({:.0f}%)'.format(morder[i], 100 * per_order[i]) for i in worst)))

# the peak of each order sits slightly off the blaze wavelength because the SED
# is not flat across the order. A constant C therefore still produces a drift.
peak_obs = morder * wave[index, np.nanargmax(np.nan_to_num(blaze), axis=1)]
peak_mod = morder * wave[index, np.nanargmax(np.nan_to_num(model), axis=1)]
full = fitted.sum(axis=1) == valid.sum(axis=1)          # orders entirely fitted
print('drift of m*lambda_peak over the {} fully fitted orders: {:.0f} observed, '
      '{:.0f} reproduced by the model'.format(
          full.sum(), np.ptp(peak_obs[full]), np.ptp(peak_mod[full])))

hdu = fits.PrimaryHDU(model.astype('float32'))
hdu.header['MODTEFF'] = (TEFF, 'blackbody temperature of the lamp [K]')
hdu.header['MODCA0'] = (cst_a0, 'm*lambda_blaze = MODCA0 + MODCA1 * lambda')
hdu.header['MODCA1'] = (cst_a1, 'dC/dlambda')
hdu.header['MODCST0'] = (cst_start, 'm*lambda_blaze, flat, from the peaks')
hdu.header['MODBETA'] = (beta, 'blaze width, 1 = first zero one FSR away')
hdu.header['MODASYM'] = (asym, 'envelope asymmetry, arg = beta m (e + asym e^2)')
hdu.header['MODWMAX'] = (WAVE_FIT_MAX, 'red limit of the fitted range')
hdu.header['MODORD0'] = (morder[0], 'diffraction order of the first spectral order')
hdu.header['MODTRANS'] = (TRANSMISSION, 'model of log(transmission)')
hdul = fits.HDUList([hdu])
if TRANSMISSION == 'spline':
    # the knots go in a table extension, one row per order peak
    hdu.header['MODSPLK'] = (SPLINE_K, 'degree of the transmission spline')
    hdu.header['MODPKHW'] = (PEAK_HALF_WIDTH, 'half width of the knot window [pix]')
    hdul.append(fits.BinTableHDU.from_columns(
        [fits.Column(name='WAVE', format='D', array=trans_par[0]),
         fits.Column(name='LOGTRANS', format='D', array=trans_par[1])],
        name='TRANS_KNOTS'))
else:
    hdu.header['MODNPOLY'] = (NPOLY, 'order of the log(transmission) polynomial')
    hdu.header['MODWLO'] = (wmin, 'wavelength at u = -1')
    hdu.header['MODWHI'] = (wmax, 'wavelength at u = +1')
    for k in range(NPOLY + 1):
        hdu.header['MODTR{}'.format(k)] = (trans_par[k],
                                          'log(trans) coefficient of u^{}'.format(k))
hdul.writeto('blaze_model.fits', overwrite=True)
print('model spectrum written to blaze_model.fits')

res = dict(wave=wave, blaze=blaze, model=model, morder=morder, valid=valid,
           fitted=fitted, resid=resid, offset=offset, photon=photon, trans=trans,
           dwave=dwave, envelope=envelope(c0, c1, beta, asym),
           cst_peak=cst_a0 + cst_a1 * wave[index, peaks], wmax=wmax,
           peak_obs=peak_obs, peak_mod=peak_mod, teff=TEFF,
           wave_fit_max=WAVE_FIT_MAX,
           knots=trans_par if TRANSMISSION == 'spline' else None,
           trans_label=('degree {} spline'.format(SPLINE_K) if TRANSMISSION == 'spline'
                        else 'P{}'.format(NPOLY)))



# --------------------------------------------------------------------------
# Everything below is debug output only. The model above needs nothing from it,
# and matplotlib is imported inside the function so the fitting part carries no
# plotting dependency.
# --------------------------------------------------------------------------


def debug_plots(filename, res):
    """Observed versus model pages. res is the dictionary assembled after the
    fit, it holds the arrays and the fitted parameters the plots need."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    wave, blaze, model = res['wave'], res['blaze'], res['model']
    morder, valid, fitted = res['morder'], res['valid'], res['fitted']
    resid, offset = res['resid'], res['offset']
    photon, trans, dwave = res['photon'], res['trans'], res['dwave']
    env, cst_peak, wmax = res['envelope'], res['cst_peak'], res['wmax']
    peak_obs, peak_mod = res['peak_obs'], res['peak_mod']
    teff, wfit = res['teff'], res['wave_fit_max']
    knots, trans_label = res['knots'], res['trans_label']
    index = np.arange(wave.shape[0])

    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
        for i in index:
            ax[0].plot(wave[i], blaze[i], 'k-', lw=1.4, alpha=.55)
            ax[0].plot(wave[i], model[i], 'r-', lw=.8)
        ax[0].plot([], [], 'k-', label='observed blaze')
        ax[0].plot([], [], 'r-', label='model')
        ax[0].set(ylabel='flux',
                  title=r'{:.0f} K blackbody $\times$ transmission ({}) $\times$ '
                        r'sinc$^2$ blaze $\times$ d$\lambda$/dpix'.format(teff, trans_label))
        ax[0].legend(loc='lower right')
        ax[0].set_ylim(0, 1.1 * np.max(blaze[valid]))
        for i in index:
            ax[1].plot(wave[i], 100 * resid[i], 'k-', lw=.7, alpha=.75)
        ax[1].axhline(0, color='r', lw=.8)
        ax[1].set(xlabel='wavelength', ylabel='obs/model - 1 [%]', ylim=[-25, 25])
        for ax1 in ax:
            if wmax > wfit:
                ax1.axvspan(wfit, wmax, color='b', alpha=.08, lw=0)
                ax1.axvline(wfit, color='b', ls=':', lw=.9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        srt = np.argsort(wave[valid])
        fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        ax[0].plot(wave[valid][srt], photon[valid][srt], 'k-')
        ax[0].set(ylabel='photons / nm', title='model components, {:.0f} K'.format(teff))
        ax[1].plot(wave[valid][srt], trans[valid][srt], 'k-')
        if knots is not None:
            ax[1].plot(knots[0], np.exp(knots[1]), 'o', color='r', ms=3,
                       label='knots, one per order peak')
            ax[1].legend(loc='lower right')
        ax[1].set(ylabel='transmission', yscale='log')
        for i in index:
            ax[2].plot(wave[i], env[i], '-', lw=.9)
        ax[2].set(ylabel=r'sinc$^2$ blaze')
        for i in index:
            ax[3].plot(wave[i], dwave[i], 'k-', lw=.9)
        ax[3].set(xlabel='wavelength', ylabel=r'|d$\lambda$/dpix|')
        for ax1 in ax:
            if wmax > wfit:
                ax1.axvspan(wfit, wmax, color='b', alpha=.08, lw=0)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        nrow = int(np.ceil(np.sqrt(index.size)))
        ncol = int(np.ceil(index.size / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 1.8 * nrow), squeeze=False)
        for ax1 in axes.ravel()[index.size:]:
            ax1.axis('off')
        for i, ax1 in zip(index, axes.ravel()):
            ax1.plot(wave[i], blaze[i], 'k-', lw=1.6, alpha=.55)
            ax1.plot(wave[i], model[i], 'r-', lw=.9)
            ax1.set_title('m = {}   {:+.1f}%'.format(morder[i], 100 * offset[i]), fontsize=8,
                          color='b' if not fitted[i].all() else 'k')
            ax1.tick_params(labelsize=6)
            ax1.set_yticks([])
        fig.suptitle('observed (black) vs model (red), median offset per order', y=1.0)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(2, 1, figsize=(9, 7))
        ax[0].plot(morder, peak_obs, 'ko-', ms=4, label='observed')
        ax[0].plot(morder, peak_mod, 'ro-', ms=4, label='model')
        ax[0].plot(morder, cst_peak, 'b--', lw=.8, label=r'fitted C($\lambda$) at the peak')
        ax[0].set(xlabel='diffraction order', ylabel=r'$m\lambda_{peak}$',
                  title=r'peak position: observed, model, and C($\lambda$)')
        ax[0].legend()
        ax[1].plot(morder, 100 * offset, 'ko-', ms=4)
        ax[1].axhline(0, color='r', lw=.8)
        ax[1].set(xlabel='diffraction order', ylabel='median obs/model - 1 [%]',
                  title='median offset per order, transmission: {}'.format(trans_label))
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print('debug plots written to {}'.format(filename))


debug_plots('blaze_model_debug.pdf', res)

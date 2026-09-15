"""Fit an echelle blaze with a physical model of the spectrograph.

    model(m, pixel) = BB_photon(lambda, Teff) * Trans(lambda)
                      * sinc^2(blaze) * dlambda/dpixel

  * BB_photon is a blackbody in PHOTON density, not in energy, because the
    detector counts photons. Dividing B_lambda by the photon energy hc/lambda
    gives  1 / (lambda^4 * (exp(hc/lambda.k.T) - 1)).
  * Trans(lambda) = exp(P(lambda)): log(transmission) is a polynomial of
    wavelength. Its constant term carries the global amplitude.
  * sinc^2 is the single-groove diffraction envelope of the grating. Its
    natural variable is the distance to the blaze peak counted in orders,
    x = m * (lambda - lambda_blaze) / lambda_blaze, with m * lambda_blaze = C
    assumed to be the same constant for every order. beta = 1 means a fully
    illuminated groove, i.e. first zeros exactly one free spectral range from
    the peak. The form below is the un-linearised one, which is slightly
    asymmetric in wavelength and involves the blaze angle theta_b.
  * dlambda/dpixel turns the photon density per unit wavelength into photons
    per pixel. The wavelength solution is not linear, so this term varies
    along an order and tilts it.

Nothing here is specific to one instrument. C, beta and theta_b are fitted,
and the diffraction orders are read off the wavelength solution itself.

usage: python fit_blaze.py [blaze.fits [wave.fits]]
"""
import sys

from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares

TEFF = 5000.0                    # K, temperature of the flat lamp
NPOLY = 11                       # polynomial order of log(transmission)
HC_K = 1.438776877e7             # h*c/k_B in nm.K
SIGMA_CLIP = 5.0
THETA_B0 = np.radians(45.0)      # neutral start for the blaze angle, it is fitted
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


def envelope(cst, beta, theta_b):
    """Grating blaze function. s = m*lambda/C is 1 at the blaze peak; the
    argument reduces to beta*m*(lambda-lambda_blaze)/lambda_blaze near it."""
    ss = mm * wave / cst
    arg = beta * mm * np.cos(theta_b) / ss * (
        ss * np.cos(theta_b) - np.sqrt(np.clip(1 - (ss * np.sin(theta_b)) ** 2, 1e-8, None)))
    return np.sinc(arg) ** 2


def backbone(cst, beta, theta_b):
    """Everything in the model except the transmission polynomial."""
    return photon * np.maximum(envelope(cst, beta, theta_b), 1e-8) * dwave


# valid is where there is data, fitted is what the fit is allowed to see
valid = np.isfinite(blaze) & np.isfinite(wave) & (blaze > 0)
fitted = valid & (wave < WAVE_FIT_MAX)
# polynomial variable normalised to [-1, 1] over the whole array, so that the
# unconstrained red end stays a mild extrapolation rather than a runaway one
wmin, wmax = np.min(wave[valid]), np.max(wave[valid])
uu = 2 * (wave - wmin) / (wmax - wmin) - 1
vander = np.array([uu[fitted] ** k for k in range(NPOLY + 1)]).T
logobs = np.log(blaze[fitted])
keep = np.ones(logobs.size, dtype=bool)


def solve_poly(cst, beta, theta_b):
    """log(transmission) enters linearly, so solve it exactly for a given
    grating and leave only C, beta and theta_b to the non-linear solver."""
    yy = logobs - np.log(backbone(cst, beta, theta_b)[fitted])
    coef = np.linalg.lstsq(vander[keep], yy[keep], rcond=None)[0]
    return coef, yy - vander @ coef


# C = m * lambda_blaze = 2 * d * sin(theta_b) * cos(gamma) is the grating
# spacing as the beam sees it: the optical path difference between two adjacent
# grooves at the blaze peak, i.e. the groove spacing d of the ruling projected
# along the blaze direction. One number for the whole array, since every order
# is blazed by the same grooves. It starts from the mean of m*lambda over the
# observed blaze peaks, then floats in the fit.
inrange = fitted.any(axis=1)
peaks = np.argmax(np.where(fitted, blaze, -np.inf), axis=1)
cst_start = np.mean(morder[inrange] * wave[index[inrange], peaks[inrange]])
guess = np.array([cst_start, 1.0, THETA_B0])
bounds = ([cst_start / 10, 0.1, np.radians(5)], [cst_start * 10, 5.0, np.radians(89)])
for loop in range(3):
    fit = least_squares(lambda p: solve_poly(*p)[1][keep], guess, bounds=bounds,
                        loss='soft_l1', f_scale=0.05, x_scale=[cst_start, 1.0, 1.0])
    guess = fit.x
    coef, res = solve_poly(*fit.x)
    rms = np.std(res[keep])
    keep = np.abs(res) < SIGMA_CLIP * rms
    print('pass {}: C = {:9.1f}   beta = {:.4f}   theta_b = {:5.2f} deg   '
          'rms = {:5.2f}%   {} points kept'.format(
              loop + 1, fit.x[0], fit.x[1], np.degrees(fit.x[2]), 100 * rms, keep.sum()))

cst, beta, theta_b = fit.x
coef, res = solve_poly(cst, beta, theta_b)
# the polynomial is evaluated as it is everywhere, including past WAVE_FIT_MAX
# where nothing constrains it. Clamping it there would be smooth in value but
# would put a kink in the slope at the boundary, so it is left free instead. The
# model is therefore an extrapolation beyond that point, and can run high.
trans = np.exp(np.sum([coef[k] * uu ** k for k in range(NPOLY + 1)], axis=0))

# the model is evaluated on every pixel of every order, including the ones the
# pipeline threw away when it thresholded the blaze. Only the fit is restricted.
model = backbone(cst, beta, theta_b) * trans
resid = blaze / model - 1

print('')
print('diffraction orders {} to {}'.format(morder[0], morder[-1]))
print('fit restricted to lambda < {:.1f}: {} points used, {} left free'.format(
    WAVE_FIT_MAX, fitted.sum(), (valid & ~fitted).sum()))
print('model evaluated on all {} pixels, {:.1f}% of which have no observed blaze'.format(
    model.size, 100 * np.mean(~valid)))
print('C = m * lambda_blaze = 2.d.sin(theta_b).cos(gamma)')
print('    {:.1f} from the blaze peaks, {:.1f} after the fit ({:+.1f}, {:+.3f}%)'.format(
    cst_start, cst, cst - cst_start, 100 * (cst / cst_start - 1)))
print('blaze width beta = {:.4f}   (1 = first zeros one FSR from the peak)'.format(beta))
print('blaze angle theta_b = {:.2f} deg   (R{:.1f} grating)'.format(
    np.degrees(theta_b), np.tan(theta_b)))
print('    implied groove spacing d = C/(2.sin(theta_b)) = {:.0f}, i.e. {:.2f} grooves/mm'
      ' if lambda is in nm'.format(cst / (2 * np.sin(theta_b)),
                                   1e6 / (cst / (2 * np.sin(theta_b)))))
print('log(transmission) coefficients in u = 2*(lambda-{:.1f})/{:.1f}-1:'.format(
    wmin, wmax - wmin))
print('   ' + np.array2string(coef, precision=4))
print('obs/model - 1 over the fitted range : median |.| = {:.2f}%   rms = {:.2f}%'.format(
    100 * np.median(np.abs(resid[fitted])), 100 * np.std(resid[fitted])))
if (valid & ~fitted).any():
    print('                 beyond {:.0f}          : median |.| = {:.2f}%   rms = {:.2f}%'.format(
        WAVE_FIT_MAX, 100 * np.median(np.abs(resid[valid & ~fitted])),
        100 * np.std(resid[valid & ~fitted])))
offset = np.array([np.nanmedian(resid[i]) for i in index])
print('   of which order-to-order offsets {:.2f}% rms, within-order {:.2f}% rms'.format(
    100 * np.std(offset), 100 * np.nanstd(resid - offset[:, None])))

# the peak of each order sits slightly off the blaze wavelength because the SED
# is not flat across the order. A constant C therefore still produces a drift.
peak_obs = morder * wave[index, np.nanargmax(np.nan_to_num(blaze), axis=1)]
peak_mod = morder * wave[index, np.nanargmax(np.nan_to_num(model), axis=1)]
full = fitted.sum(axis=1) == valid.sum(axis=1)          # orders entirely fitted
print('drift of m*lambda_peak over the {} fully fitted orders: {:.0f} observed, '
      '{:.0f} reproduced by the model at constant C'.format(
          full.sum(), np.ptp(peak_obs[full]), np.ptp(peak_mod[full])))

hdu = fits.PrimaryHDU(model.astype('float32'))
hdu.header['MODTEFF'] = (TEFF, 'blackbody temperature of the lamp [K]')
hdu.header['MODCST'] = (cst, 'm*lambda_blaze, fitted')
hdu.header['MODCST0'] = (cst_start, 'm*lambda_blaze, from the peaks')
hdu.header['MODBETA'] = (beta, 'blaze width, 1 = first zero one FSR away')
hdu.header['MODTHETA'] = (np.degrees(theta_b), 'blaze angle [deg]')
hdu.header['MODNPOLY'] = (NPOLY, 'order of the log(transmission) polynomial')
hdu.header['MODWMAX'] = (WAVE_FIT_MAX, 'red limit of the fitted range')
hdu.header['MODORD0'] = (morder[0], 'diffraction order of the first spectral order')
for k in range(NPOLY + 1):
    hdu.header['MODTR{}'.format(k)] = (coef[k], 'log(trans) coefficient of u^{}'.format(k))
hdu.writeto('blaze_model.fits', overwrite=True)
print('model spectrum written to blaze_model.fits')

res = dict(wave=wave, blaze=blaze, model=model, morder=morder, valid=valid,
           fitted=fitted, resid=resid, offset=offset, photon=photon, trans=trans,
           dwave=dwave, envelope=envelope(cst, beta, theta_b), cst=cst, wmax=wmax,
           peak_obs=peak_obs, peak_mod=peak_mod, teff=TEFF, npoly=NPOLY,
           wave_fit_max=WAVE_FIT_MAX)



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
    env, cst, wmax = res['envelope'], res['cst'], res['wmax']
    peak_obs, peak_mod = res['peak_obs'], res['peak_mod']
    teff, npoly, wfit = res['teff'], res['npoly'], res['wave_fit_max']
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
                  title=r'{:.0f} K blackbody $\times$ exp(P$_{{{}}}$) transmission $\times$ '
                        r'sinc$^2$ blaze $\times$ d$\lambda$/dpix'.format(teff, npoly))
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
        ax[0].plot(morder, peak_mod, 'ro-', ms=4, label='model, C constant')
        ax[0].axhline(cst, color='b', ls='--', lw=.8, label='fitted C')
        ax[0].set(xlabel='diffraction order', ylabel=r'$m\lambda_{peak}$',
                  title='the drift of the peak comes from the SED, not from C')
        ax[0].legend()
        ax[1].plot(morder, 100 * offset, 'ko-', ms=4)
        ax[1].axhline(0, color='r', lw=.8)
        ax[1].set(xlabel='diffraction order', ylabel='median obs/model - 1 [%]',
                  title='left over order-to-order structure, the P{} transmission '
                        'cannot follow it'.format(npoly))
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print('debug plots written to {}'.format(filename))


debug_plots('blaze_model_debug.pdf', res)

"""Fast likelihood for the blaze model, used by the MCMC of the report.

The model is the one of fit_blaze.py:

    model = BB_photon(lambda, Teff) * exp(S(lambda)) * sinc^2(env) * dlambda/dpixel
    env   = beta * m * (e + asym * e^2),   e = m * lambda / C(lambda) - 1
    C     = c0 + c1 * (lambda - lref)

and S is the linear spline through one knot per order, each knot the median of
log(observed) - log(everything else) over a window around the order peak. The
spline is recomputed at every call, i.e. the transmission is profiled out.

Only two sets of pixels are ever evaluated: the fitted pixels, decimated along
each order, for the likelihood, and the peak windows, at full resolution, for
the knots. That is a few thousand pixels instead of a few hundred thousand.
"""
import numpy as np
from astropy.io import fits
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline

HC_K = 1.438776877e7          # h*c/k_B in nm.K
PARAMS = ['c0', 'c1', 'beta', 'asym', 'lns']


def photon_bb(wave, teff):
    return 1.0 / (wave ** 4 * np.expm1(HC_K / (wave * teff)))


class BlazeLikelihood:
    def __init__(self, blaze_file, wave_file, model_file, decimate=16):
        self.blaze = fits.getdata(blaze_file).astype(float)
        self.wave = fits.getdata(wave_file).astype(float)
        self.model_file = model_file
        hdr = fits.getheader(model_file)
        self.hdr = hdr
        self.teff = hdr['MODTEFF']
        self.half = hdr['MODPKHW']
        self.k = hdr['MODSPLK']
        self.wave_fit_max = hdr['MODWMAX']
        nord, npix = self.wave.shape
        self.nord, self.npix = nord, npix
        lam = self.wave[:, npix // 2]
        step = int(-np.sign(np.median(np.diff(lam))))
        self.morder = hdr['MODORD0'] + step * np.arange(nord)
        self.valid = np.isfinite(self.blaze) & np.isfinite(self.wave) & (self.blaze > 0)
        self.fitted = self.valid & (self.wave < self.wave_fit_max)
        self.lref = np.median(self.wave[self.fitted])
        self.wmin = np.min(self.wave[self.valid])
        self.wmax = np.max(self.wave[self.valid])
        self.dwave = np.abs(np.gradient(self.wave, axis=1))
        self.logobs = np.full(self.blaze.shape, np.nan)
        self.logobs[self.valid] = np.log(self.blaze[self.valid])
        self.peaks = np.argmax(np.where(self.fitted, self.blaze, -np.inf), axis=1)
        pix = np.arange(npix)
        self.window = self.fitted & (np.abs(pix[None, :] - self.peaks[:, None]) <= self.half)
        self.decimate = decimate
        self.corr_length = float(decimate)       # set by measure_correlation()
        self.nu = 1e6                            # set by fit_student()
        self.drop = np.zeros(nord, bool)         # orders left out (jackknife)
        self.set_teff(self.teff)
        self.set_use(self.fitted)
        c0 = hdr['MODCA0'] + hdr['MODCA1'] * self.lref
        self.best = np.array([c0, hdr['MODCA1'], hdr['MODBETA'], hdr['MODASYM']])

    # ------------------------------------------------------------ pixel sets
    def set_teff(self, teff):
        self.teff = teff
        self.photon = photon_bb(self.wave, teff)
        if hasattr(self, 'use'):
            self.set_use(self.use)

    def set_use(self, use):
        """use: fitted pixels kept by the clipping. Builds the decimated
        likelihood set and the padded window arrays."""
        self.use = use
        dec = use & (np.arange(self.npix)[None, :] % self.decimate == 0)
        dec &= ~self.drop[:, None]
        o, p = np.where(dec)
        self.d_ord, self.d_pix = o, p
        self.d_wave = self.wave[o, p]
        self.d_m = self.morder[o].astype(float)
        self.d_base = np.log(self.photon[o, p] * self.dwave[o, p])
        self.d_logobs = self.logobs[o, p]
        width = 2 * self.half + 1
        win = self.window & use
        self.w_wave = np.full((self.nord, width), np.nan)
        self.w_logobs = np.full((self.nord, width), np.nan)
        self.w_base = np.full((self.nord, width), np.nan)
        for i in range(self.nord):
            if self.drop[i]:
                continue
            p = np.where(win[i])[0]
            if p.size == 0:
                continue
            sl = p - self.peaks[i] + self.half
            self.w_wave[i, sl] = self.wave[i, p]
            self.w_logobs[i, sl] = self.logobs[i, p]
            self.w_base[i, sl] = np.log(self.photon[i, p] * self.dwave[i, p])
        self.w_rows = np.where(np.isfinite(self.w_wave).any(axis=1))[0]
        self.w_m = self.morder.astype(float)[:, None]

    def set_drop(self, orders):
        self.drop = np.zeros(self.nord, bool)
        self.drop[list(orders)] = True
        self.set_use(self.use)

    # ------------------------------------------------------------ the model
    def spline(self, theta):
        rows = self.w_rows
        logenv = self.log_envelope_rows(theta)
        yy = self.w_logobs[rows] - self.w_base[rows] - logenv
        kx = np.nanmedian(self.w_wave[rows], axis=1)
        ky = np.nanmedian(yy, axis=1)
        srt = np.argsort(kx)
        return InterpolatedUnivariateSpline(kx[srt], ky[srt], k=self.k, ext=0), kx[srt], ky[srt]

    def log_envelope_rows(self, theta):
        rows = self.w_rows
        return self.log_envelope_2d(self.w_wave[rows], self.w_m[rows], theta)

    def log_envelope_2d(self, wave, m, theta):
        c0, c1, beta, asym = theta[:4]
        ee = m * wave / (c0 + c1 * (wave - self.lref)) - 1
        return np.log(np.maximum(np.sinc(beta * m * (ee + asym * ee * ee)) ** 2, 1e-8))

    def residuals(self, theta):
        """log(observed / model) on the decimated likelihood pixels"""
        spl = self.spline(theta)[0]
        logenv = self.log_envelope_2d(self.d_wave, self.d_m, theta)
        return self.d_logobs - self.d_base - logenv - spl(self.d_wave)

    def full_model(self, theta):
        """model on every pixel, with knots from the current windows"""
        spl, kx, ky = self.spline(theta)
        m = self.morder.astype(float)[:, None]
        logenv = self.log_envelope_2d(self.wave, m, theta)
        logt = spl(self.wave.ravel()).reshape(self.wave.shape)
        return np.exp(np.log(self.photon * self.dwave) + logenv + logt), kx, ky

    # ------------------------------------------------------------ likelihood
    def log_like(self, theta):
        """Student-t on log residuals, tempered by decimate / corr_length so
        that the pixels count as the independent samples they really are."""
        lns = theta[4]
        r = self.residuals(theta)
        s = np.exp(lns)
        nu = self.nu
        z = r / s
        lp = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
              - 0.5 * np.log(nu * np.pi) - lns
              - (nu + 1) / 2 * np.log1p(z * z / nu))
        return self.decimate / self.corr_length * np.sum(lp)

    def log_prior(self, theta):
        c0, c1, beta, asym, lns = theta
        c00 = self.best[0]
        if not (0.9 * c00 < c0 < 1.1 * c00):
            return -np.inf
        if not (-c00 / self.lref < c1 < c00 / self.lref):
            return -np.inf
        if not (0.1 < beta < 5) or not (-100 < asym < 100) or not (-12 < lns < 2):
            return -np.inf
        return 0.0

    def log_post(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_like(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    # ------------------------------------------------------------ helpers
    def derived(self, theta):
        """a0, a1, dlnC/dlnlambda, and the change of C across the array in %"""
        c0, c1 = theta[0], theta[1]
        return dict(a0=c0 - c1 * self.lref, a1=c1,
                    dlnc=c1 * self.lref / c0,
                    change=100 * c1 * (self.wmax - self.wmin) / c0)


"""
Convolve irregularly sampled data (x, y, yerr) onto a regular grid x2 with a
parametric kernel, propagating uncertainties.

Each input sample i owns a "cell" [left_i, right_i] bounded by the midpoints to
its neighbours. By default, the weight of sample i in output pixel j is the
kernel mass that falls inside that cell,

    w_ij = CDF(right_i - x2_j) - CDF(left_i - x2_j),

which is the exact convolution of the piecewise-constant (nearest-neighbour)
interpolant of the data. This is robust to clumpy sampling (a dense clump does
not get more weight than the same x range sampled sparsely). The output is

    y2_j   = sum_i w_ij y_i / sum_i w_ij
    err2_j = sqrt(sum_i w_ij**2 err_i**2) / sum_i w_ij

and coverage_j = sum_i w_ij is the fraction of the kernel mass backed by data
(about 0.5 at the edges of the data, < 1 inside gaps if max_half_cell is set).

When the samples are several interleaved samplings of different functions of
x (e.g. detector rows across a slit, each with its own illumination), pass
`group` (e.g. the row index) so that cells are built within each row; with
normalize=False the result is then the sum over rows, the properly resampled
equivalent of a plain column sum.

Only samples whose cell overlaps the window where the kernel is above
`threshold` times its peak (1e-6 by default, i.e. +/-5.26 sigma for a Gaussian)
are ever touched, so the cost scales as n_out * (samples per kernel window).

Note: neighbouring output pixels share input samples, so their errors are
correlated. Use return_matrix=True to get the sparse linear operator W
(y2 = W @ y) and build the full covariance as W @ diag(err**2) @ W.T.
"""
import numpy as np
from scipy import sparse
from scipy.special import erf


class Kernel:
    """
    Interface for a parametric kernel. Distances dx are in x units.

    profile(dx)          : kernel shape normalized to 1 at its peak
    cdf(dx)              : integral of the unit-area kernel from -inf to dx
    half_width(threshold): |dx| beyond which profile(dx) < threshold
    """

    def profile(self, dx):
        raise NotImplementedError

    def cdf(self, dx):
        raise NotImplementedError

    def half_width(self, threshold):
        raise NotImplementedError


class GaussianKernel(Kernel):
    def __init__(self, fwhm):
        self.fwhm = float(fwhm)
        self.sigma = self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def profile(self, dx):
        return np.exp(-0.5 * (dx / self.sigma) ** 2)

    def cdf(self, dx):
        return 0.5 * (1.0 + erf(dx / (self.sigma * np.sqrt(2.0))))

    def half_width(self, threshold):
        return self.sigma * np.sqrt(-2.0 * np.log(threshold))


def _cell_edges(x, max_half_cell=None):
    """Cell boundaries at midpoints between sorted samples; end cells symmetric."""
    if x.size == 1:
        half = np.inf if max_half_cell is None else max_half_cell
        return x - half, x + half
    mid = 0.5 * (x[1:] + x[:-1])
    left = np.concatenate([[x[0] - (mid[0] - x[0])], mid])
    right = np.concatenate([mid, [x[-1] + (x[-1] - mid[-1])]])
    if max_half_cell is not None:
        left = np.maximum(left, x - max_half_cell)
        right = np.minimum(right, x + max_half_cell)
    return left, right


def _window_pairs(left, right, x2, half_width):
    """(output pixel, sample) pairs whose cell overlaps x2_j +/- half_width."""
    # left and right are both sorted, so this is a pair of binary searches
    lo = np.searchsorted(right, x2 - half_width, side='right')
    hi = np.searchsorted(left, x2 + half_width, side='left')
    counts = np.clip(hi - lo, 0, None)
    rows = np.repeat(np.arange(x2.size), counts)
    starts = np.cumsum(counts) - counts
    cols = lo[rows] + (np.arange(counts.sum()) - starts[rows])
    return rows, cols


def convolve_irregular(x, y, yerr, x2, kernel=None, fwhm_pix=2.0,
                       threshold=1e-6, weighting='integral',
                       max_half_cell=None, min_coverage=0.5,
                       group=None, normalize=True, return_matrix=False):
    """
    Convolve irregularly sampled (x, y, yerr) onto the regular grid x2.

    Parameters
    ----------
    x, y, yerr : 1d arrays, input samples (any order). Non-finite values and
                 yerr <= 0 are ignored.
    x2         : 1d regular, strictly increasing output grid.
    kernel     : Kernel instance (FWHM etc. in x units). If None, a Gaussian
                 with FWHM = fwhm_pix output pixels is used.
    fwhm_pix   : FWHM in units of x2 pixels for the default Gaussian kernel.
    threshold  : ignore samples where the kernel is below threshold * peak.
    weighting  : 'integral' -> w_ij = kernel mass within the sample's cell
                               (true convolution of the sampled function)
                 'ivar'     -> w_ij = kernel(x_i - x2_j) / yerr_i**2
                               (minimum-variance kernel-weighted mean)
    max_half_cell : optional cap (x units) on each cell half-width, so a
                 sample next to a gap does not claim half the gap. With it,
                 coverage drops inside gaps and min_coverage masks them.
    min_coverage  : output pixels with a smaller fraction of kernel mass
                 backed by data are set to NaN (0.5 ~ the data edges).
    group      : optional 1d array of labels (e.g. detector row). Cells are
                 then built within each group only. Use it when the samples
                 are several interleaved samplings of *different* functions
                 of x (e.g. rows across a slit with different illumination):
                 without it, cell widths come from the interleaving of
                 unrelated samples and are essentially random. If a group
                 has a gap (e.g. a row crossing a curved order twice), set
                 max_half_cell, or the samples bordering the gap will each
                 claim half of it.
    normalize  : True  -> y2 = sum w y / sum w (weighted mean; with groups,
                          the mean over groups)
                 False -> y2 = sum w y (with groups and 'integral', the sum
                          over groups, i.e. the resampled equivalent of a
                          plain sum across rows). Requires 'integral'.
    return_matrix : also return the sparse (len(x2), len(x)) operator W
                 such that y2 = W @ y (with NaN inputs set to 0).

    Returns
    -------
    dict with 'y', 'err', 'coverage' (mean fraction of the kernel mass backed
    by data, per contributing group), 'sum_w' (sum of the cell weights, i.e.
    the effective number of groups covering the pixel), 'npts' (samples used
    per output pixel) and, if requested, 'matrix'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n_in, n_out = x.size, x2.size

    step = np.diff(x2)
    if n_out < 2 or np.any(step <= 0):
        raise ValueError('x2 must be strictly increasing with >= 2 points')
    dx2 = step.mean()
    if not np.allclose(step, dx2, rtol=1e-6, atol=0):
        raise ValueError('x2 is not a regular grid')
    if weighting not in ('integral', 'ivar'):
        raise ValueError("weighting must be 'integral' or 'ivar'")
    if not normalize and weighting != 'integral':
        raise ValueError("normalize=False requires weighting='integral'")

    if kernel is None:
        kernel = GaussianKernel(fwhm_pix * dx2)
    half_width = kernel.half_width(threshold)

    if group is None:
        group = np.zeros(n_in, dtype=int)
    group = np.asarray(group)

    # keep valid samples, sorted by group then x; `idx` maps back to inputs
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    idx = np.flatnonzero(valid)
    idx = idx[np.lexsort((x[idx], group[idx]))]
    xs, ys, es, gs = x[idx], y[idx], yerr[idx], group[idx]

    # cells and window pairs, group by group
    bounds = np.flatnonzero(np.diff(gs) != 0) + 1
    all_rows, all_cols, all_cell_w = [], [], []
    ngroups = np.zeros(n_out)
    for i0, i1 in zip(np.r_[0, bounds], np.r_[bounds, xs.size]):
        if i1 == i0:
            continue
        left, right = _cell_edges(xs[i0:i1], max_half_cell)
        rows, cols = _window_pairs(left, right, x2, half_width)
        xc = x2[rows]
        cell_w = kernel.cdf(right[cols] - xc) - kernel.cdf(left[cols] - xc)
        ngroups += np.bincount(rows, weights=cell_w, minlength=n_out) > 0
        all_rows.append(rows)
        all_cols.append(cols + i0)
        all_cell_w.append(cell_w)
    rows = np.concatenate(all_rows) if all_rows else np.zeros(0, dtype=int)
    cols = np.concatenate(all_cols) if all_cols else np.zeros(0, dtype=int)
    cell_w = np.concatenate(all_cell_w) if all_cell_w else np.zeros(0)

    if weighting == 'integral':
        w = cell_w
    else:
        prof = kernel.profile(xs[cols] - x2[rows])
        # the cell-overlap selection can include centers just past the cutoff
        keep = prof >= threshold
        rows, cols, prof, cell_w = rows[keep], cols[keep], prof[keep], cell_w[keep]
        w = prof / es[cols] ** 2

    sum_w = np.bincount(rows, weights=w, minlength=n_out)
    sum_wy = np.bincount(rows, weights=w * ys[cols], minlength=n_out)
    sum_w2e2 = np.bincount(rows, weights=(w * es[cols]) ** 2, minlength=n_out)
    sum_cell = np.bincount(rows, weights=cell_w, minlength=n_out)
    coverage = np.zeros(n_out)
    np.divide(sum_cell, ngroups, out=coverage, where=ngroups > 0)

    norm = sum_w if normalize else np.ones(n_out)
    good = (sum_w > 0) & (coverage >= min_coverage)
    out = dict(y=np.full(n_out, np.nan), err=np.full(n_out, np.nan))
    out['y'][good] = sum_wy[good] / norm[good]
    out['err'][good] = np.sqrt(sum_w2e2[good]) / norm[good]
    out['coverage'] = coverage
    out['sum_w'] = sum_cell
    out['npts'] = np.bincount(rows, minlength=n_out)

    if return_matrix:
        keep = good[rows]
        vals = w[keep] / norm[rows[keep]]
        out['matrix'] = sparse.csr_matrix((vals, (rows[keep], idx[cols[keep]])),
                                          shape=(n_out, n_in))
    return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # irregular sampling: random positions, denser clump, and a gap
    x = np.concatenate([rng.uniform(0, 200, 1500), rng.uniform(60, 80, 1500)])
    x = x[(x < 120) | (x > 135)]
    period = 15.0
    truth = lambda xx: np.sin(2 * np.pi * xx / period)
    yerr = rng.uniform(0.05, 0.3, x.size)
    y_clean = truth(x)
    y = y_clean + rng.normal(0, yerr)

    x2 = np.arange(0, 200, 0.5)
    fwhm_pix = 2.0
    kern = GaussianKernel(fwhm_pix * 0.5)

    # analytic convolution of a sine with a unit-area Gaussian
    k = 2 * np.pi / period
    y2_true = truth(x2) * np.exp(-0.5 * (k * kern.sigma) ** 2)

    for mode in ('integral', 'ivar'):
        res_clean = convolve_irregular(x, y_clean, yerr, x2, fwhm_pix=fwhm_pix,
                                       weighting=mode, max_half_cell=1.0)
        ok = np.isfinite(res_clean['y'])
        full = res_clean['coverage'] > 0.999
        diff = (res_clean['y'] - y2_true)[full]
        print(f'[{mode}] noiseless - analytic, full-coverage pixels: '
              f'rms {np.sqrt(np.mean(diff ** 2)):.2e}, '
              f'max {np.max(np.abs(diff)):.2e}')

        # Monte Carlo check of the propagated errors
        res = convolve_irregular(x, y, yerr, x2, fwhm_pix=fwhm_pix,
                                 weighting=mode, max_half_cell=1.0,
                                 return_matrix=True)
        mc = np.array([convolve_irregular(x, y_clean + rng.normal(0, yerr),
                                          yerr, x2, fwhm_pix=fwhm_pix,
                                          weighting=mode,
                                          max_half_cell=1.0)['y']
                       for _ in range(300)])
        ratio = np.std(mc[:, ok], axis=0) / res['err'][ok]
        print(f'[{mode}] median MC std / propagated err = '
              f'{np.median(ratio):.3f}, '
              f'W @ y matches: {np.allclose((res["matrix"] @ y)[ok], res["y"][ok])}')

    res = convolve_irregular(x, y, yerr, x2, fwhm_pix=fwhm_pix,
                             max_half_cell=1.0)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].errorbar(x, y, yerr, fmt='.', ms=2, alpha=0.2, color='grey',
                   label='input (irregular)')
    ax[0].errorbar(x2, res['y'], res['err'], fmt='o', ms=3, color='C0',
                   label='convolved onto x2')
    ax[0].plot(x2, y2_true, color='C3', lw=1, label='analytic')
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel('y')
    ax[1].plot(x2, res['coverage'], color='C2')
    ax[1].axhline(0.5, ls=':', color='k')
    ax[1].set_ylabel('coverage')
    ax[1].set_xlabel('x')
    fig.tight_layout()
    fig.savefig('convolve_irregular_demo.pdf')
    print('saved convolve_irregular_demo.pdf')

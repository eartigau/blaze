"""
1D profile of one order of a flat, resampled onto a regular x grid by
convolution with a Gaussian kernel, from the detector pixels and their x map.

Each detector pixel has an x position (XMAP, along the dispersion, tilted with
respect to the columns) and a flux (PROFILE_AB). The pixels of one order are an
irregular sampling of x, made of ~15 interleaved rows that see different parts
of the slit. The profile is built row by row (group=row) and summed over rows,
which is the resampled equivalent of a plain column sum.

    python order_profile.py                      # shipped extract, order 20
    python order_profile.py flatblaze.fits       # full frame (XMAP, ORDER_MAP,
    python order_profile.py flatblaze.fits --order 48 --fwhm 20
                                                 # PROFILE_AB), writes an extract

Writes order_profile.pdf and order_profile.svg (the latter for the README).
"""
import argparse

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import median_filter

from convolve_irregular import convolve_irregular

OVERSAMPLING = 2       # output grid, in steps per detector pixel
RON = 8.0              # read-out noise, in counts
MAX_HALF_CELL = 0.75   # cap on the cell half-width, in detector pixels
EXTRACT = 'order{:02d}_pixels.fits'

# categorical slots (validated palette), text inks
C_ROWS = '#2a78d6'     # per-row convolution, the answer
C_NAIVE = '#eb6834'    # convolution without groups, the failure
C_COLSUM = '#1baf7a'   # plain column sum, the reference
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load_pixels(path, order):
    """Pixels of one order: detector row and column, x (detector px), flux."""
    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        if 'XMAP' not in names:
            tbl = Table(hdul[1].data)
            return (np.asarray(tbl['row']), np.asarray(tbl['col']),
                    np.asarray(tbl['x']), np.asarray(tbl['flux']))
        in_order = hdul['ORDER_MAP'].data == order
        row, col = np.where(in_order)
        x = hdul['XMAP'].data[in_order]
        flux = hdul['PROFILE_AB'].data[in_order]
    tbl = Table(dict(row=row.astype(np.int16), col=col.astype(np.int16),
                     x=x, flux=flux.astype(np.float32)))
    tbl.meta['ORDER'] = order
    tbl.write(EXTRACT.format(order), overwrite=True)
    print(f'wrote {EXTRACT.format(order)} ({row.size} pixels)')
    return row, col, x, flux


def hf_noise(profile, width=51, edge=200):
    """rms of profile / median after removing a running median."""
    p = profile / np.nanmedian(profile)
    resid = p - median_filter(np.nan_to_num(p), width)
    return np.nanstd(resid[edge:-edge])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('file', nargs='?', default=EXTRACT.format(20))
    parser.add_argument('--order', type=int, default=20)
    parser.add_argument('--fwhm', type=float, default=1.0,
                        help='kernel FWHM, in output (oversampled) pixels')
    args = parser.parse_args()

    row, col, x_det, flux = load_pixels(args.file, args.order)
    ncol = col.max() + 1
    x = x_det * OVERSAMPLING
    yerr = np.sqrt(np.abs(flux) + RON ** 2)
    x2 = np.arange(ncol * OVERSAMPLING, dtype=float)

    common = dict(fwhm_pix=args.fwhm, weighting='integral', min_coverage=0.5)
    naive = convolve_irregular(x, flux, yerr, x2, **common)
    rows = convolve_irregular(x, flux, yerr, x2, group=row, normalize=False,
                              max_half_cell=MAX_HALF_CELL * OVERSAMPLING,
                              **common)

    # plain column sum, placed at the mean x of the column
    colsum = np.bincount(col, weights=np.nan_to_num(flux), minlength=ncol)
    npix = np.bincount(col, minlength=ncol)
    xcol = np.bincount(col, weights=x, minlength=ncol) / np.maximum(npix, 1)
    ok = npix > 0
    colsum, xcol = colsum[ok], xcol[ok]

    ratio = np.interp(xcol, x2, rows['y']) / colsum
    inner = slice(100, -100)
    nrows = np.median(rows['sum_w'][rows['sum_w'] > 0])
    stats = dict(naive=hf_noise(naive['y']), rows=hf_noise(rows['y']),
                 colsum=hf_noise(colsum))
    print(f'order {args.order}, {row.size} pixels, FWHM {args.fwhm} output px')
    print(f'rows covering an output pixel (median): {nrows:.2f}')
    print('high-frequency noise: without groups {naive:.4f}, per row '
          '{rows:.4f}, column sum {colsum:.4f}'.format(**stats))
    print('per-row / column sum, percentiles 5/50/95: ' + ' / '.join(
        f'{v:.3f}' for v in np.nanpercentile(ratio[inner], [5, 50, 95])))

    make_figure(args, row, col, x_det, flux, x2, naive, rows, xcol, colsum,
                stats)


def make_figure(args, row, col, x_det, flux, x2, naive, rows, xcol, colsum,
                stats):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    plt.rcParams.update({
        'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'legend.frameon': False})
    blues = LinearSegmentedColormap.from_list('blues', ['#b7d3f2', '#0d3b73'])

    fig = plt.figure(figsize=(10, 11.5), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.2, 1.2, 1.2])
    ax_img = fig.add_subplot(gs[0, :])
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_cell = fig.add_subplot(gs[1, 1])
    ax_prof = fig.add_subplot(gs[2, :])
    ax_zoom = fig.add_subplot(gs[3, :])

    # (a) the order on the detector
    r0, r1 = row.min(), row.max()
    img = np.full((r1 - r0 + 1, col.max() + 1), np.nan)
    img[row - r0, col] = flux
    ax_img.imshow(img, origin='lower', aspect='auto', cmap='Greys',
                  norm=LogNorm(*np.nanpercentile(flux[flux > 0], [2, 99.5])),
                  extent=(-0.5, col.max() + 0.5, r0 - 0.5, r1 + 0.5),
                  interpolation='nearest')
    # a row that crosses the curved order twice
    twice = [r for r in np.unique(row)
             if np.any(np.diff(np.sort(col[row == r])) > 1)]
    r_ex = twice[len(twice) // 2] if twice else None
    if r_ex is not None:
        cc = np.sort(col[row == r_ex])
        ax_img.axhline(r_ex, color=C_NAIVE, lw=1, ls=(0, (4, 3)))
        cut = np.flatnonzero(np.diff(cc) > 1)[0]
        ax_img.annotate('', xy=(cc[cut + 1], r_ex + 3), xytext=(cc[cut], r_ex + 3),
                        arrowprops=dict(arrowstyle='<->', color=C_NAIVE, lw=1))
        ax_img.text(0.5 * (cc[cut] + cc[cut + 1]), 0.5 * (r0 + r_ex),
                    f'row {r_ex} (dashed) crosses the order twice, with a gap '
                    f'of {cc[cut + 1] - cc[cut]} columns\nin between: '
                    'max_half_cell keeps the cells from bridging it',
                    color=INK, ha='center', va='center', fontsize=8.5)
    ax_img.set(xlabel='detector column', ylabel='detector row',
               title=f'(a) Order {args.order} on the detector: '
                     f'{np.median(np.bincount(col)[np.bincount(col) > 0]):.0f} '
                     'rows per column, curved trace')
    ax_img.grid(False)

    # (b) raw samples around the middle of the order, coloured by slit position
    xc = np.median(x_det)
    sel = np.abs(x_det - xc) < 4
    slit = np.zeros(row.size)
    for c in np.unique(col[sel]):
        m = col == c
        slit[m] = row[m] - row[m].min()
    for r in np.unique(row[sel]):
        m = sel & (row == r)
        o = np.argsort(x_det[m])
        ax_raw.plot(x_det[m][o], flux[m][o], lw=0.6, zorder=2,
                    color=blues(slit[m].mean() / max(slit[sel].max(), 1)))
    sc = ax_raw.scatter(x_det[sel], flux[sel], c=slit[sel], cmap=blues, s=16,
                        edgecolors='white', linewidths=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax_raw, pad=0.01)
    cb.set_label('position in the slit (row within the column)', color=INK2)
    ax_raw.set(xlabel='x (detector px)', ylabel='flux (counts)',
               title='(b) Pixels are not one function of x')
    ax_raw.text(0.02, 0.97, 'lines join pixels of the same detector row',
                transform=ax_raw.transAxes, va='top', color=INK2, fontsize=8.5)

    # (c) cell widths: from all rows interleaved vs from each row
    xs_sel = np.sort(x_det[sel])
    w_all = np.diff(xs_sel)
    w_row = np.concatenate([np.diff(np.sort(x_det[sel & (row == r)]))
                            for r in np.unique(row[sel])])
    bins = np.logspace(-3, 0.5, 40)
    ax_cell.hist(w_all, bins=bins, color=C_NAIVE, alpha=0.85,
                 label='all rows interleaved')
    ax_cell.hist(w_row, bins=bins, color=C_ROWS, alpha=0.85,
                 label='within each row')
    ax_cell.set_xscale('log')
    ax_cell.set(xlabel='spacing to the next sample (detector px)',
                ylabel='number of samples',
                title='(c) Cell widths set the weights')
    ax_cell.legend(loc='upper left')

    # (d) the profiles along the whole order
    def norm(p):
        return p / np.nanmedian(p)
    xd = x2 / OVERSAMPLING
    ax_prof.plot(xd, norm(naive['y']), color=C_NAIVE, lw=0.8,
                 label=f'convolution, all pixels pooled '
                       f'(noise {stats["naive"]:.3f})')
    ax_prof.plot(xcol / OVERSAMPLING, norm(colsum), color=C_COLSUM, lw=1.2,
                 label=f'plain column sum (noise {stats["colsum"]:.3f})')
    ax_prof.plot(xd, norm(rows['y']), color=C_ROWS, lw=1.2,
                 label=f'convolution per row, summed over rows '
                       f'(noise {stats["rows"]:.3f})')
    ax_prof.set(xlabel='x (detector px)', ylabel='flux / median',
                title='(d) Profile along the order')
    ax_prof.legend(loc='upper left', fontsize=8)
    ax_prof.set_ylim(0, 1.25 * np.nanpercentile(norm(rows['y']), 99.5))

    # (e) zoom, with the propagated 1-sigma envelope of the per-row profile
    lo = np.nanmedian(xd) - 60
    zoom = (xd > lo) & (xd < lo + 120)
    zc = (xcol / OVERSAMPLING > lo) & (xcol / OVERSAMPLING < lo + 120)
    med = np.nanmedian(rows['y'])
    ax_zoom.plot(xd[zoom], norm(naive['y'])[zoom], color=C_NAIVE, lw=0.8)
    ax_zoom.plot(xcol[zc] / OVERSAMPLING, norm(colsum)[zc], color=C_COLSUM,
                 lw=1.2, marker='o', ms=3)
    ax_zoom.fill_between(xd[zoom], (rows['y'] - rows['err'])[zoom] / med,
                         (rows['y'] + rows['err'])[zoom] / med,
                         color=C_ROWS, alpha=0.25, lw=0)
    ax_zoom.plot(xd[zoom], norm(rows['y'])[zoom], color=C_ROWS, lw=1.4)
    for label, color, y in (('pooled', C_NAIVE, 0.95), ('column sum', C_COLSUM, 0.87),
                            ('per row', C_ROWS, 0.79)):
        ax_zoom.text(1.0, y, label, color=INK, transform=ax_zoom.transAxes,
                     ha='right', fontsize=8.5,
                     bbox=dict(boxstyle='square,pad=0.2', fc='white', ec=color,
                               lw=1.5))
    ax_zoom.set(xlabel='x (detector px)', ylabel='flux / median',
                title='(e) Zoom on 120 px, per-row profile with its '
                      'propagated 1-sigma envelope')
    fig.suptitle(f'Resampling order {args.order} of a flat onto a regular x '
                 f'grid (Gaussian kernel, FWHM {args.fwhm / OVERSAMPLING:g} '
                 'detector px)', fontsize=11, color=INK, x=0.01, ha='left')
    for ext in ('pdf', 'svg'):
        fig.savefig(f'order_profile.{ext}')
    print('wrote order_profile.pdf and order_profile.svg')


if __name__ == '__main__':
    main()
